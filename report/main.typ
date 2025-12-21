// #import "@preview/bubble-zju:0.1.0": *
#import "bubble-zju/lib.typ": *
// and you can import any Typst package you want!
#import "@preview/note-me:0.5.0": *
// #import "@preview/cetz:0.4.1": canvas, draw, matrix, vector
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: circle as fletcher_circle, hexagon, house


#show: bubble.with(
  title: "中国空间站机械臂运动仿真",
  subtitle: "机器人技术与实践 实验报告",
  author: "罗建明 程韬 王昱涵 陈廷峰 魏云翔 李浩博",
  affiliation: "浙江大学",
  date: datetime.today().display("2025 年 12 月 21 日"),
  year: "机器人技术与实践, 2025",
  // class: "Class",
  // other: ("Made with Typst", "https://typst.com")
)

#outline(title: "目录")

#pagebreak()

#counter(page).update(1)


= DH参数

在本实验中，我们实际上并未直接使用 DH 参数进行建模。相反，我们通过定义符合 IK 求解器要求的一般关节树结构来构建 IK 链，具体定义可见 `data/A-base-arm.json` 和 `data/B-base-arm.json` 文件。

需要注意的是，在定义坐标系和旋转轴时，必须与仿真环境中机械臂的定义保持一致，不能随意定义，否则会导致机械臂驱动错误。

图 @fig:coordinate-system 展示了我们使用的坐标系定义示意图：

#figure(
  image("/assets/typst/main/坐标系定义.jpg", width: 60%),
  caption: [坐标系定义示意图],
) <fig:coordinate-system>

出于实验完整性的考量，我们在下面给出推导的 MDH（Modified Denavit-Hartenberg）参数表。参数列顺序为：$[a_(i-1), alpha_(i-1), d_i, theta_i]$，其中长度单位为 mm，角度单位为度。

```python
# Modified DH 参数表
# 列顺序: [a(i-1), alpha(i-1), d(i), theta(i)]
# 单位: 长度(mm), 角度(deg)
dh_params_deg = [
    [0,      0,    120,    0],
    [0,     90,    100,   90],
    [0,    -90,      0,    0],
    [400,    0,    150,    0],
    [-400,  180,  -150,    0],
    [0,    -90,   -100,  -90],
    [0,    -90,   -120,    0],
]
```

= 逆运动学解析解

注：我们最后实际上并没有使用到逆运动学解析解，仅仅出于实验的完整性考量（PPT中要求），我们在这里给出逆运动学解析解的推导过程。因为这段推导有点长且不重要，详细的推导过程请参见附录 @appendix-ik。


= 轨迹规划

轨迹规划模块（`traj.py`）的核心任务是：定义末端执行器的关键位置和姿态，通过逆运动学求解得到每个关键点对应的关节角度，并将结果保存为 `trajectory_results.json` 文件，供后续运动仿真阶段使用。

== 核心思路

我们的核心思路是：定义几个末端执行器关键点，然后通过IK求解得到每个关键点的关节角度。轨迹规划阶段的产出就是这些关节角度 `trajectory_results.json`，运动仿真阶段将会基于这些关节角进行插值，从而得到每个时刻的控制关节角。

这里我们实际上并没有使用逆运动学解析解的IK，而是使用了我们此前实现的基于逆向雅可比方法的IK求解器。我们每次求解关键点IK时，都会使用上一个关键点的关节角度作为初始值，辅以该IK方法的"变化连续性"，我们期望能够得到一个相对更平滑的关节角度变化轨迹。另外通过在部分不重要的关键点通过放宽姿态容差和容许未完全收敛的解，从而减少 因关键点取用不佳、可解性不佳导致的规划失败。

== 实现细节

`traj.py` 的实现主要包括以下几个部分：

*关键点定义*：我们为两个运动步骤分别定义了关键点序列。Step 1 以 B 为基座、A 为末端执行器，定义了 5 个关键点的相对位置 `POS_A_REL` 和欧拉角 `EULER_A`。Step 2 以 A 为基座、B 为末端执行器，同样定义了 5 个关键点的相对位置 `POS_B_REL` 和欧拉角 `EULER_B`。

*IK求解流程*：`solve_trajectory` 函数是通用的轨迹求解函数，其工作流程如下：
1. 根据基座类型（"A-base" 或 "B-base"）加载对应的骨骼文件
2. 查找末端执行器（通过 `find_effector` 函数寻找没有子节点的 `FixedJoint`）
3. 构建IK链，初始化T-Pose
4. 对每个关键点进行IK求解：
   - 如果关键点已有预设关节角度，则直接使用
   - 否则，将上一个关键点的关节角度作为初始值（保证连续性）
   - 调用 `solve_ik` 函数求解，得到满足目标位姿的关节角度
5. 将所有关键点的关节角度保存到结果列表中

*换基处理*：这是本实验的一个关键难点。当从 Step 1（B-base）切换到 Step 2（A-base）时，由于两个基座的关节顺序完全相反，且链的方向相反，需要进行角度映射：

- B-base 的关节顺序：`[J7-B, J6, J5, J4, J3, J2, J1-A]`
- A-base 的关节顺序：`[J1-A, J2, J3, J4, J5, J6, J7-B]`

因此，当使用 Step 1 最后一个关键点的关节角度作为 Step 2 第 0 个关键点的初始值时，需要：
1. 反转关节顺序：`angles[::-1]`
2. 取反每个角度：`[-a for a in angles]`

这样可以得到：`initial_angles_for_step2 = [-a for a in result_a[-1][::-1]]`

*结果输出*：所有关键点的关节角度（已归一化到 `[0, 2π)` 范围）被保存到 `trajectory_results.json` 文件中，包含每个关键点的索引、相对位置、欧拉角和关节角度信息。

= 运动仿真

运动仿真模块（`main.py`）负责在 CoppeliaSim 中控制机械臂按照规划好的轨迹进行运动。我们使用 `coppeliasim_zmqremoteapi_client.RemoteAPIClient` 通过外部 Python 脚本对仿真机械臂进行控制。

== 核心思路

核心思路是：基于 `trajectory_results.json` 中的关键点关节角度，再设定每一段从一个关键点到另一个关键点的运动时长`FRAME_TIME`，通过五次多项式插值，得到中间时刻的关节角度，然后通过`sim.setJointPosition()`直接设置到机械臂的关节角度中。

== 实现细节

=== 初始化
程序首先连接到 CoppeliaSim 仿真环境，获取所有关节和链对象的句柄，并定义 A-base 和 B-base 两种基座下的关节列表和链结构。

=== 轨迹数据加载
从 `trajectory_results.json` 文件中读取 Step 1 和 Step 2 的关键点关节角度数据，转换为 numpy 数组格式（单位为弧度）。

=== 时间规划
为每个关键点段设定运动时长 `FRAME_TIME`，并计算累积时间点 `cumulative_times`，用于确定当前时刻处于哪个时间段。

=== 插值策略
- 对于大部分关键点段，使用 `angle_interpolate_2` 函数进行两点间的五次多项式插值，保证起点和终点的速度、加速度均为零，实现平滑运动
- 对于某些特殊段（如 Step 1 的关键点 1-2-3 段和 Step 2 的关键点 6-7-8 段），使用 `angle_interpolate_3` 函数进行三点间的五次多项式插值，经过中间点，从而使运动更加平滑。

=== 换基处理
这是运动仿真阶段的关键难点。在 Step 1 完成后、Step 2 开始前，需要切换基座，我们将其封装在一个`switch_base`函数中。具体处理方式为：

1. `main.py`调用 `switch_base` 函数，传入新的链结构 `A_base_chain`、旧基座关节列表 `B_base_joints` 和新基座关节列表 `A_base_joints`
2. `switch_base` 函数内部会：
   - 记录新基座的位置（确保换基前后位置不变）
   - 读取旧基座下的当前关节角度
   - 将所有关节角度清零（避免 `inplace=True` 在非零角度下的问题）
   - 断开所有对象的父子关系，重新建立新的父子关系
   - 将旧基座下的关节角度映射到新基座：反转顺序并取反每个角度
   - 设置新基座下的关节角度
   - 恢复新基座位置


=== 换基处理的原理与必要性
换基操作之所以如此复杂，不能简单地直接更改父子关系，推测根本原因在于 CoppeliaSim 的关节实现机制。

在 CoppeliaSim 中，每个子关节存储的（也是我们通过 `sim.setJointPosition()` 能够操作的）应该实际上是相对于其父关节的相对位姿。当父子关系断开时，这个相对位姿信息就失去了参考系，变得无意义。即便使用 `inplace=True` 参数强制子关节的世界位姿保持不变，这样的操作也会更改关节的零位（T-Pose），导致换基后机械臂的 T-Pose 发生变化。一旦 T-Pose 改变，我们关于关节角度的建模和一切基于该模型的 IK 计算都将静默失效，因为所有的关节角度都是相对于 T-Pose 定义的。

因此，我们必须在断开父子关系前将所有关节角度归零，回到 T-Pose 状态。但为了保持机械臂的当前姿态，我们需要先读取旧基座下的当前关节角度，以便在换基后恢复。在恢复关节角度时，由于链的顺序是反的（B-base 的关节顺序 `[J7-B, J6, J5, J4, J3, J2, J1-A]` 与 A-base 的关节顺序 `[J1-A, J2, J3, J4, J5, J6, J7-B]` 完全相反），且链的方向相反，所以我们需要对旧关节角度进行反转并取反：`[-a for a in angles[::-1]]`。

另外，由于清空关节角度时是基于旧基座的，这会导致新基座的位姿发生变化（因为新基座在旧基座下是作为子链的一部分存在的）。因此，我们还需要在换基操作的最后恢复新基座的位置，确保换基前后新基座在世界坐标系中的位置保持不变，从而保证整个换基过程的连续性和正确性。

综上所述，`switch_base` 函数的复杂流程是为了在 CoppeliaSim 的关节实现机制下，既保持机械臂的当前姿态，又确保 T-Pose 的一致性，同时还要处理基座位置的变化，这是一个多约束条件下的精确操作。


=== 实时控制循环
在主循环中，根据当前仿真时间 `t` 确定处于哪个时间段，然后：
- 如果是 Step 1，使用 `B_base_joints` 设置关节角度
- 如果是 Step 2，首先检查是否需要换基（仅在 Step 1 刚完成时执行一次），然后使用 `A_base_joints` 设置关节角度
- 对每个关节调用相应的插值函数，计算当前时刻的关节角度并设置到仿真环境中
- 调用 `client.step()` 触发下一步仿真

通过这种方式，机械臂能够平滑地从 Step 1 的关键点运动到 Step 2 的关键点，并在换基时保持运动的连续性。

#pagebreak()

= 附录

== 逆运动学解析解 <appendix-ik>

#include "IK.typ"