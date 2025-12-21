# 中国空间站机械臂运动仿真项目

## 主要代码文件

- `traj.py` - 轨迹规划模块，用于生成关键点的关节角度
  - 运行方法：`python traj.py`
  - 结果保存在 `trajectory_results.json` 文件中

- `main.py` - 运动仿真模块，在 CoppeliaSim 中控制机械臂运动
  - 运行方法：先在 CoppeliaSim 中打开 `attachment/SpaceRobot.ttt` 文件，然后运行 `python main.py` 即可查看仿真

## 仿真结果

- `video/demo.mp4`

## 工具文件

- `utils.py` - 工具函数库，包含角度插值、换基处理等辅助函数
- `rad2deg.py` - 角度单位转换工具脚本

## 目录结构

- `IK_Solver/` - IK 求解器模块, 使用方法见对应目录README
- `data/` - 机械臂关节关系定义文件，用于 IK 求解
  - `A-base-arm.json` - A 基座下的关节树结构
  - `B-base-arm.json` - B 基座下的关节树结构


## 仿真文件

- `attachment/SpaceRobot.ttt` - CoppeliaSim 仿真场景文件，最终仿真在此场景中的机械臂上运行

## 其他文件

- `DHverified.m` - 用于验证 MDH 参数的 MATLAB 脚本
