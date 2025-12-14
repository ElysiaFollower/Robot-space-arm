## 使用说明

修改config.json文件，确定输入的骨骼和目标轨迹路径。
然后指定导出关键帧关节角序列的路径。
可以调参：
- solve_mode: 用于区分关键帧求解+关节插值 和 关键帧插值+逐帧求解。但是在这里没区别
- max_iterations: 每次求解关键帧时最大迭代次数
- position_tolerance: 位置容差。使用 p_{target} - p _{current} 的向量模长定义
- orientation_tolerance: 旋转容差。使用 —— R_target * R_current^(-1)， 后转为轴角表示(模长为旋转角) —— 的向量模长定义
- damping: 阻尼参数，越大阻尼越强 —— 体现为，允许的误差增大，但奇异情况受到约束


设定好参数后直接运行 `run_solver.py` 即可。