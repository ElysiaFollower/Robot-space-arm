#set text(font: ("Roboto Mono", "Noto Sans CJK SC"), lang: "zh")
#set math.equation(numbering: none)

= 七自由度机械臂逆运动学报告

== 1. 方程建立

=== 1.1 正运动学：构造各级变换矩阵 $attach(T, tl: i-1, br: i)$

根据 Craig-MDH 参数表（略），对每个关节建立齐次变换：

$ attach(T, tl: i-1, br: i) = R_x (alpha_(i-1)) T_x (a_(i-1)) R_z (theta_i) T_z (d_i) $

各级相乘得到末端正运动学：

$ attach(T, tl: 0, br: 7) = attach(T, tl: 0, br: 1) attach(T, tl: 1, br: 2) attach(T, tl: 2, br: 3) attach(T, tl: 3, br: 4) attach(T, tl: 4, br: 5) attach(T, tl: 5, br: 6) attach(T, tl: 6, br: 7) $

期望末端位姿写成：

$ attach(T, tl: 0, br: 7, tr: d) = mat(
  n_x, a_x, o_x, p_x;
  n_y, a_y, o_y, p_y;
  n_z, a_z, o_z, p_z;
  0, 0, 0, 1;
) $

其中 $bold(n), bold(a), bold(o)$ 为旋转矩阵三列，$bold(p)$ 为位置向量。

=== 1.2 变换矩阵的两种表达式（选取 $T_(57), T_(17), T_(16)$）

为了从矩阵等式中构造标量方程，选取三段子链并写出两种表达：

(1) 末端子链 $attach(T, tl: 5, br: 7)$

+ 正运动学表达（关节变量侧）： $attach(T, tl: 5, br: 7) = attach(T, tl: 5, br: 6) attach(T, tl: 6, br: 7)$

+ 由期望位姿反推（目标侧）： $attach(T, tl: 5, br: 7) = (attach(T, tl: 0, br: 5))^(-1) attach(T, tl: 0, br: 7, tr: d)$

(2) 基座到末端 $attach(T, tl: 1, br: 7)$

+ 正运动学表达： $attach(T, tl: 1, br: 7) = attach(T, tl: 1, br: 2) attach(T, tl: 2, br: 3) attach(T, tl: 3, br: 4) attach(T, tl: 4, br: 5) attach(T, tl: 5, br: 6) attach(T, tl: 6, br: 7)$

+ 目标侧表达： $attach(T, tl: 1, br: 7) = (attach(T, tl: 0, br: 1))^(-1) attach(T, tl: 0, br: 7, tr: d)$

(3) 基座到第 6 坐标系 $attach(T, tl: 1, br: 6)$

+ 正运动学表达： $attach(T, tl: 1, br: 6) = attach(T, tl: 1, br: 2) attach(T, tl: 2, br: 3) attach(T, tl: 3, br: 4) attach(T, tl: 4, br: 5) attach(T, tl: 5, br: 6)$

+ 目标侧表达（由 $attach(T, tl: 1, br: 7)$ 去掉末端一节）： $attach(T, tl: 1, br: 6) = attach(T, tl: 1, br: 7) (attach(T, tl: 6, br: 7))^(-1) = (attach(T, tl: 0, br: 1))^(-1) attach(T, tl: 0, br: 7, tr: d) (attach(T, tl: 6, br: 7))^(-1)$

=== 1.3 标量方程 E1–E7（由矩阵元素对齐得到）

来自 $attach(T, tl: 5, br: 7)$ 的姿态与位置分量（用于求 $q_1, q_2, q_6, q_7$）

(E1) $(attach(T, tl: 5, br: 7)[2,0])$ $ cos(q_6) cos(q_7) = n_x cos(q_1) cos(q_2) + n_y sin(q_1) cos(q_2) + n_z sin(q_2) $

(E2) $(attach(T, tl: 5, br: 7)[2,1])$ $ -cos(q_6) sin(q_7) = a_x cos(q_1) cos(q_2) + a_y sin(q_1) cos(q_2) + a_z sin(q_2) $

(E3) $(attach(T, tl: 5, br: 7)[2,2])$ $ -sin(q_6) = o_x cos(q_1) cos(q_2) + o_y sin(q_1) cos(q_2) + o_z sin(q_2) $

(E4) $(attach(T, tl: 5, br: 7)[2,3])$ $ 120 sin(q_6) = p_x cos(q_1) cos(q_2) + p_y sin(q_1) cos(q_2) + p_z sin(q_2) - 120 sin(q_2) + 300 $

说明：E1–E4 是“末端子链”约束，主要用于先解出 $q_1, q_2, q_6, q_7$

来自 $attach(T, tl: 1, br: 7)$ 的姿态分量（用于组合角）

(E5) $(attach(T, tl: 1, br: 7)[1,2])$ $ sin(q_3 + q_4 - q_5) cos(q_6) = -o_x sin(q_1) + o_y cos(q_1) $

该式给出组合角 $phi = q_3 + q_4 - q_5$ 的约束。

来自 $attach(T, tl: 1, br: 6)$ 的位置分量（用于 $q_3, q_4, q_5$）

(E6) $(attach(T, tl: 1, br: 6)[1,3])$ $ 400 sin(q_3) - 400 sin(q_3 + q_4) + 100 cos(q_3 + q_4 - q_5) - 100 = -120 o_x sin(q_1) + 120 o_y cos(q_1) - p_x sin(q_1) + p_y cos(q_1) $

(E7) $(attach(T, tl: 1, br: 6)[2,3])$ $ -300 sin(q_2) - 100 sin(q_3 + q_4 - q_5) cos(q_2) + 400 cos(q_2) cos(q_3) - 400 cos(q_2) cos(q_3 + q_4) = 120 o_z + p_z - 120 $

== 2. 方程求解

=== 0. 记号与问题结构

==== 0.1 关节变量

$ bold(q) = (q_1, q_2, q_3, q_4, q_5, q_6, q_7) $

定义简写： $ c_i = cos q_i, quad s_i = sin q_i $

==== 0.2 末端位姿输入

+ 旋转矩阵列向量： $ bold(n) = vec(n_x, n_y, n_z), quad bold(a) = vec(a_x, a_y, a_z), quad bold(o) = vec(o_x, o_y, o_z) $

+ 末端位置： $ bold(p) = vec(p_x, p_y, p_z) $

假设 $[bold(n) bold(a) bold(o)] in "SO"(3)$。

==== 0.3 基座方向向量（关键中间变量）

#box(
  stroke: 1pt + black,
  inset: 8pt,
  $ bold(u) eq.def vec(c_1 c_2, s_1 c_2, s_2) $
)

显然有 $|bold(u)| = 1$

$bold(u)$ 是一个仅由 $q_1, q_2$ 决定的单位向量，它将在后续成为冗余自由度的载体。

=== 1. (E1–E3)：姿态投影表达

原始方程组前三项可统一写为内积形式：

$
  bold(n)^T bold(u) &= cos q_6 cos q_7 quad ("E1") \
  bold(a)^T bold(u) &= -cos q_6 sin q_7 quad ("E2") \
  bold(o)^T bold(u) &= -sin q_6 quad ("E3")
$

==== 1.1 几何解释

+ $(cos q_6 cos q_7, -cos q_6 sin q_7, -sin q_6)$ 是一个单位向量；

+ $(bold(n)^T bold(u), bold(a)^T bold(u), bold(o)^T bold(u))$ 是 $bold(u)$ 在末端坐标系下的坐标。

因此 (E1–E3) 表示：

#box(
  stroke: 1pt + black,
  inset: 8pt,
  $ bold(u) "在" bold(n), bold(a), bold(o) "坐标系下的表示由" (q_6, q_7) "决定" $
)

这三式并不会完全约束 $bold(u)$，而只是把 $bold(u)$ 与 $(q_6, q_7)$ 绑定。

=== 2. (E4)：圆解集

第四个方程为： $ 120 sin q_6 = bold(p)^T bold(u) - 120 sin q_2 + 300 $

利用 $ sin q_6 = -bold(o)^T bold(u), quad sin q_2 = u_z $

代入并整理得： $(bold(p) + 120 bold(o) - 120 bold(e)_z)^T bold(u) = -300 $

其中 $ bold(e)_z = vec(0, 0, 1) $

==== 2.1 关键几何结论

前四个方程等价于：

#box(
  stroke: 1pt + black,
  inset: 8pt,
  $
    cases(
      |bold(u)| = 1,
      bold(k)^T bold(u) = -300
    ) quad bold(k) eq.def bold(p) + 120 bold(o) - 120 bold(e)_z
  $
)

这是：

+ 一个单位球面

+ 与一个平面

的交集。

==== 2.2 冗余自由度

+ 若 $|-300| < |bold(k)|$：解集是一个圆

+ 若等于：唯一解

+ 若大于：无解

因此在一般非退化情况下：

#box(
  stroke: 1pt + black,
  inset: 8pt,
  $ bold(u) "具有 1 维自由度，可用圆参数" psi "表达" $
)

这是七自由度系统的结构性冗余来源。

=== 3. 已知 $bold(u)$：求解 $q_1, q_2, q_6, q_7$

此时 $bold(u)$ 被视为已知输入条件（由冗余参数选定）。

==== 3.1 解 $q_1, q_2$

由定义直接反解：

$ q_1 = "atan2"(u_y, u_x) $

$ q_2 = "atan2"(u_z, sqrt(u_x^2 + u_y^2)) $

奇异性说明：若 $u_x = u_y = 0$，即 $c_2 = 0$，则 $q_1$ 不可观（基座竖直奇异）。

==== 3.2 解 $q_6$

由 (E3)： $ sin q_6 = - (bold(o)^T bold(u)) $

因此： $ q_6 = arcsin(- bold(o)^T bold(u)) $

存在标准两分支： $ q_6 in {theta, pi - theta} $

==== 3.3 解 $q_7$

由 (E1–E2)：

$
  cos q_7 &= (bold(n)^T bold(u)) / cos q_6 \
  sin q_7 &= - (bold(a)^T bold(u)) / cos q_6
$

于是： $ q_7 = "atan2"(- bold(a)^T bold(u), bold(n)^T bold(u)) $

腕部奇异：若 $cos q_6 = 0$，则 (E1–E2) 同时退化为 0，此时 $q_7$ 自由。

=== 4. 已知 $q_1, q_2, q_6, q_7$：求解组合角

定义组合角：

#box(
  stroke: 1pt + black,
  inset: 8pt,
  $ phi eq.def q_3 + q_4 - q_5 $
)

==== 4.1 由第五个方程求 $sin phi$

原式： $ sin(q_3 + q_4 - q_5) cos q_6 = -o_x sin q_1 + o_y cos q_1 $

整理得： $ sin phi = frac(-o_x sin q_1 + o_y cos q_1, cos q_6) $

因此： $ phi in {arcsin(·), pi - arcsin(·)} $

并可同时确定： $ cos phi = plus.minus sqrt(1 - sin^2 phi) $

=== 5. 已知组合角：求解 $q_3, q_4$（和差法，非奇异）

定义： $ beta eq.def q_3 + q_4 $

==== 5.1 构造两条仅含 $q_3, beta$ 的方程

由第六、七式整理得：

$ 400 (sin q_3 - sin beta) = A $

$ 400 (cos q_3 - cos beta) = B $

其中 $A, B$ 是已知标量（由 $q_1, q_2, phi, bold(p), bold(o)$ 决定）。

==== 5.2 和差变量替换

设： $ m = frac(q_3 + beta, 2), quad d = frac(q_3 - beta, 2) $

利用恒等式：

$ sin q_3 - sin beta = 2 cos m sin d $

$ cos q_3 - cos beta = -2 sin m sin d $

代入得：

$
  cases(
    800 cos m sin d = A,
    -800 sin m sin d = B
  )
$

==== 5.3 求解流程（非奇异 $sin d ≠ 0$）

1. 取比值得： $ tan m = -B / A $ → 解出 $m$（模 $pi$）

2. 回代任一式求： $ sin d = A / (800 cos m) $

3. 得： $ q_3 = m + d, quad beta = m - d $

==== 5.4 回代求 $q_4, q_5$

$ q_4 = beta - q_3 $

$ q_5 = beta - phi $