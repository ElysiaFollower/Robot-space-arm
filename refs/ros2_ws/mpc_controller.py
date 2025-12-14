"""
MPC控制器 - 用于动态避障和轨迹跟踪

特点:
1. 预测时域内的未来状态
2. 考虑动态障碍物的预测位置
3. 优化控制序列(vx, vw)
4. 处理约束(速度、加速度、避障)
5. 滚动时域优化
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import time


class MPCController:
    """
    模型预测控制器
    
    差速驱动模型:
    x_{k+1} = x_k + vx * cos(θ_k) * dt
    y_{k+1} = y_k + vx * sin(θ_k) * dt
    θ_{k+1} = θ_k + vw * dt
    """
    
    def __init__(self,
                 prediction_horizon=10,      # 预测步数
                 control_horizon=5,          # 控制步数
                 dt=0.1,                     # 时间步长(秒)
                 max_linear_vel=1000,        # mm/s
                 max_angular_vel=3.0,        # rad/s
                 max_linear_acc=2000,        # mm/s^2
                 max_angular_acc=6.0,        # rad/s^2
                 robot_radius=150,           # mm (考虑安全裕度)
                 safety_distance=100):       # mm 额外安全距离
        
        self.N = prediction_horizon
        self.M = control_horizon
        self.dt = dt
        
        # 速度约束
        self.vx_max = max_linear_vel
        self.vw_max = max_angular_vel
        
        # 加速度约束
        self.ax_max = max_linear_acc
        self.aw_max = max_angular_acc
        
        # 安全参数
        self.robot_radius = robot_radius
        self.safety_distance = safety_distance
        
        # 权重矩阵
        self.Q_position = 100.0      # 位置跟踪权重
        self.Q_orientation = 10.0    # 朝向权重
        self.R_vx = 0.1              # 线速度代价
        self.R_vw = 0.1              # 角速度代价
        self.R_dvx = 1.0             # 线速度变化代价
        self.R_dvw = 1.0             # 角速度变化代价
        self.Q_obstacle = 1000.0     # 障碍物惩罚权重
        
        # 轨迹参数
        self.tck = None
        self.trajectory_length = 0
        
        # 上一次的控制量(用于计算加速度)
        self.last_vx = 0.0
        self.last_vw = 0.0
        
        # 动态障碍物列表
        self.dynamic_obstacles = []  # [(x, y, vx, vy, radius), ...]
        
        print(f"MPC Controller initialized: N={self.N}, M={self.M}, dt={self.dt}s")
    
    def set_trajectory(self, path, k=3, s=0):
        """设置参考轨迹"""
        if len(path) < k + 1:
            raise ValueError(f"Path needs at least {k+1} points")
        
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        
        self.tck, u = splprep([x, y], s=s, k=k)
        
        # 计算轨迹长度
        u_dense = np.linspace(0, 1, 200)
        points = np.array(splev(u_dense, self.tck)).T
        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs**2).sum(axis=1))
        self.trajectory_length = distances.sum()
        
        print(f"Trajectory set: {len(path)} points, {self.trajectory_length:.1f}mm length")
        
        return self.tck  # 返回B样条参数用于可视化
    
    def update_dynamic_obstacles(self, obstacles):
        """
        更新动态障碍物信息
        
        Args:
            obstacles: [(x, y, vx, vy, radius), ...] 
                      位置、速度、半径
        """
        self.dynamic_obstacles = obstacles
    
    def find_closest_point_on_trajectory(self, robot_pos, start_u=0.0):
        """在轨迹上找最近点"""
        u_samples = np.linspace(max(0, start_u - 0.1), min(1, start_u + 0.3), 50)
        trajectory_points = np.array(splev(u_samples, self.tck)).T
        
        distances = np.sqrt(((trajectory_points - robot_pos)**2).sum(axis=1))
        min_idx = distances.argmin()
        
        return u_samples[min_idx], trajectory_points[min_idx]
    
    def get_reference_trajectory(self, current_u, N):
        """
        获取未来N步的参考轨迹
        
        Returns:
            ref_states: (N+1, 3) [x, y, θ]
        """
        # 估算沿轨迹前进的速度
        avg_velocity = self.vx_max * 0.7  # 假设平均速度
        ds_per_step = avg_velocity * self.dt  # 每步前进的距离
        
        # 转换为B样条参数的增量
        du_per_step = ds_per_step / self.trajectory_length
        
        ref_states = []
        u = current_u
        
        for i in range(N + 1):
            u = min(u, 1.0)
            
            # 当前位置
            x, y = splev(u, self.tck)
            
            # 计算切线方向(朝向)
            if u < 1.0:
                dx, dy = splev(u, self.tck, der=1)
                theta = np.arctan2(dy, dx)
            else:
                # 终点使用最后的方向
                dx, dy = splev(0.999, self.tck, der=1)
                theta = np.arctan2(dy, dx)
            
            ref_states.append([x, y, theta])
            u += du_per_step
        
        return np.array(ref_states)
    
    def predict_state(self, state, vx, vw):
        """
        预测下一步状态
        
        Args:
            state: [x, y, θ]
            vx, vw: 控制量
            
        Returns:
            next_state: [x, y, θ]
        """
        x, y, theta = state
        
        x_next = x + vx * np.cos(theta) * self.dt
        y_next = y + vx * np.sin(theta) * self.dt
        theta_next = theta + vw * self.dt
        
        # 角度归一化
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.array([x_next, y_next, theta_next])
    
    def predict_obstacle_position(self, obstacle, time_step):
        """
        预测障碍物在未来time_step后的位置
        
        Args:
            obstacle: (x, y, vx, vy, radius)
            time_step: 时间(秒)
            
        Returns:
            (x_pred, y_pred, radius)
        """
        x, y, vx, vy, radius = obstacle
        
        x_pred = x + vx * time_step
        y_pred = y + vy * time_step
        
        return (x_pred, y_pred, radius)
    
    def compute_cost(self, u_flat, current_state, ref_trajectory):
        """
        计算代价函数
        
        Args:
            u_flat: 扁平化的控制序列 [vx_0, vw_0, vx_1, vw_1, ...]
            current_state: 当前状态 [x, y, θ]
            ref_trajectory: 参考轨迹 (N+1, 3)
            
        Returns:
            cost: 总代价
        """
        M = self.M  # 控制时域
        N = self.N  # 预测时域
        
        # 重塑控制序列
        U = u_flat.reshape((M, 2))  # (M, 2) [vx, vw]
        
        # 预测状态序列
        states = [current_state]
        state = current_state.copy()
        
        for i in range(N):
            # 使用对应的控制量(如果超出控制时域,使用最后一个)
            if i < M:
                vx, vw = U[i]
            else:
                vx, vw = U[-1]
            
            state = self.predict_state(state, vx, vw)
            states.append(state)
        
        states = np.array(states)  # (N+1, 3)
        
        # 1. 跟踪代价
        position_error = states[:, :2] - ref_trajectory[:, :2]
        position_cost = self.Q_position * np.sum(position_error**2)
        
        # 角度误差
        angle_error = states[:, 2] - ref_trajectory[:, 2]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        orientation_cost = self.Q_orientation * np.sum(angle_error**2)
        
        # 2. 控制代价
        control_cost = self.R_vx * np.sum(U[:, 0]**2) + self.R_vw * np.sum(U[:, 1]**2)
        
        # 3. 控制变化代价(平滑性)
        dvx = np.diff(U[:, 0], prepend=self.last_vx)
        dvw = np.diff(U[:, 1], prepend=self.last_vw)
        smoothness_cost = self.R_dvx * np.sum(dvx**2) + self.R_dvw * np.sum(dvw**2)
        
        # 4. 障碍物代价
        obstacle_cost = 0.0
        if len(self.dynamic_obstacles) > 0:
            for i, state in enumerate(states):
                time_ahead = i * self.dt
                
                for obstacle in self.dynamic_obstacles:
                    # 预测障碍物位置
                    obs_x, obs_y, obs_r = self.predict_obstacle_position(obstacle, time_ahead)
                    
                    # 计算距离
                    dist = np.sqrt((state[0] - obs_x)**2 + (state[1] - obs_y)**2)
                    
                    # 最小安全距离
                    min_safe_dist = self.robot_radius + obs_r + self.safety_distance
                    
                    # 障碍物惩罚(指数型,距离越近惩罚越大)
                    if dist < min_safe_dist:
                        obstacle_cost += self.Q_obstacle * (min_safe_dist - dist)**2
                    elif dist < min_safe_dist * 1.5:
                        # 软约束区域
                        obstacle_cost += self.Q_obstacle * 0.1 * (min_safe_dist * 1.5 - dist)**2
        
        total_cost = position_cost + orientation_cost + control_cost + smoothness_cost + obstacle_cost
        
        return total_cost
    
    def compute_constraints(self, u_flat, current_state):
        """
        计算约束违反量(用于约束优化)
        
        Returns:
            violations: 约束违反量数组,应该都 >= 0
        """
        M = self.M
        N = self.N
        
        U = u_flat.reshape((M, 2))
        
        violations = []
        
        # 预测状态
        states = [current_state]
        state = current_state.copy()
        
        for i in range(N):
            if i < M:
                vx, vw = U[i]
            else:
                vx, vw = U[-1]
            
            state = self.predict_state(state, vx, vw)
            states.append(state)
        
        # 障碍物避障约束
        if len(self.dynamic_obstacles) > 0:
            for i, state in enumerate(states):
                time_ahead = i * self.dt
                
                for obstacle in self.dynamic_obstacles:
                    obs_x, obs_y, obs_r = self.predict_obstacle_position(obstacle, time_ahead)
                    dist = np.sqrt((state[0] - obs_x)**2 + (state[1] - obs_y)**2)
                    min_safe_dist = self.robot_radius + obs_r + self.safety_distance
                    
                    # 约束: dist - min_safe_dist >= 0
                    violations.append(dist - min_safe_dist)
        
        return np.array(violations) if violations else np.array([1.0])
    
    def compute_control(self, current_pos, current_orientation, current_u=0.0):
        """
        计算MPC控制量
        
        Args:
            current_pos: [x, y]
            current_orientation: θ
            current_u: 轨迹参数(可选)
            
        Returns:
            (vx, vw, finished, debug_info)
        """
        if self.tck is None:
            raise ValueError("Trajectory not set! Call set_trajectory() first.")
        
        # 当前状态
        current_state = np.array([current_pos[0], current_pos[1], current_orientation])
        
        # 找到轨迹上的最近点
        closest_u, _ = self.find_closest_point_on_trajectory(current_pos, current_u)
        
        # 检查是否到达终点
        if closest_u >= 0.98:
            end_point = np.array(splev(1.0, self.tck)).flatten()
            dist_to_end = np.linalg.norm(current_pos - end_point)
            if dist_to_end < 150:
                return 0.0, 0.0, True, {}
        
        # 获取参考轨迹
        ref_trajectory = self.get_reference_trajectory(closest_u, self.N)
        
        # 初始猜测(使用上一次的控制量)
        u0 = np.zeros(self.M * 2)
        for i in range(self.M):
            u0[2*i] = self.last_vx
            u0[2*i+1] = self.last_vw
        
        # 控制量的边界
        bounds = []
        for i in range(self.M):
            # vx 边界
            vx_min = max(0, self.last_vx - self.ax_max * self.dt)
            vx_max = min(self.vx_max, self.last_vx + self.ax_max * self.dt)
            bounds.append((vx_min, vx_max))
            
            # vw 边界
            vw_min = max(-self.vw_max, self.last_vw - self.aw_max * self.dt)
            vw_max = min(self.vw_max, self.last_vw + self.aw_max * self.dt)
            bounds.append((vw_min, vw_max))
        
        # 定义约束
        constraints = []
        
        # 障碍物避障约束
        if len(self.dynamic_obstacles) > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda u: self.compute_constraints(u, current_state)
            })
        
        # 优化
        start_time = time.time()
        
        result = minimize(
            fun=lambda u: self.compute_cost(u, current_state, ref_trajectory),
            x0=u0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints if constraints else None,
            options={'maxiter': 50, 'ftol': 1e-3}
        )
        
        opt_time = time.time() - start_time
        
        # 提取第一步的控制量
        U_opt = result.x.reshape((self.M, 2))
        vx_opt = float(U_opt[0, 0])
        vw_opt = float(U_opt[0, 1])
        
        # 更新历史控制量
        self.last_vx = vx_opt
        self.last_vw = vw_opt
        
        # 预测未来状态(用于可视化)
        predicted_states = []
        state = current_state.copy()
        for i in range(self.M):
            predicted_states.append([state[0], state[1], state[2]])
            state = self.predict_state(state, U_opt[i, 0], U_opt[i, 1])
        # 添加剩余预测时域
        for i in range(self.M, self.N):
            predicted_states.append([state[0], state[1], state[2]])
            state = self.predict_state(state, U_opt[-1, 0], U_opt[-1, 1])  # 使用最后一个控制量
        
        # 调试信息
        debug_info = {
            'closest_u': closest_u,
            'progress': closest_u * 100,
            'optimization_time': opt_time * 1000,  # ms
            'cost': result.fun,
            'success': result.success,
            'n_obstacles': len(self.dynamic_obstacles),
            'predicted_states': predicted_states  # 添加预测轨迹用于可视化
        }
        
        return vx_opt, vw_opt, False, debug_info


class DynamicObstaclePredictor:
    """
    动态障碍物预测器
    
    功能:
    1. 跟踪障碍物历史位置
    2. 估计速度
    3. 预测未来轨迹
    """
    
    def __init__(self, history_length=5):
        self.history_length = history_length
        self.obstacle_history = {}  # {id: [(t, x, y), ...]}
        
    def update(self, obstacles_dict):
        """
        更新障碍物信息
        
        Args:
            obstacles_dict: {id: (x, y, radius), ...}
        """
        current_time = time.time()
        
        for obs_id, (x, y, radius) in obstacles_dict.items():
            if obs_id not in self.obstacle_history:
                self.obstacle_history[obs_id] = []
            
            self.obstacle_history[obs_id].append((current_time, x, y))
            
            # 只保留最近的N个记录
            if len(self.obstacle_history[obs_id]) > self.history_length:
                self.obstacle_history[obs_id].pop(0)
    
    def estimate_velocity(self, obs_id):
        """
        估计障碍物速度(使用线性拟合)
        
        Returns:
            (vx, vy) mm/s
        """
        if obs_id not in self.obstacle_history:
            return 0.0, 0.0
        
        history = self.obstacle_history[obs_id]
        
        if len(history) < 2:
            return 0.0, 0.0
        
        # 最小二乘拟合速度
        times = np.array([h[0] for h in history])
        xs = np.array([h[1] for h in history])
        ys = np.array([h[2] for h in history])
        
        # 归一化时间
        times = times - times[0]
        
        if times[-1] < 0.1:  # 时间间隔太短
            return 0.0, 0.0
        
        # 速度 = dx/dt
        vx = (xs[-1] - xs[0]) / times[-1]
        vy = (ys[-1] - ys[0]) / times[-1]
        
        return vx, vy
    
    def get_predicted_obstacles(self, obstacles_dict):
        """
        获取带速度预测的障碍物列表
        
        Args:
            obstacles_dict: {id: (x, y, radius), ...}
            
        Returns:
            [(x, y, vx, vy, radius), ...]
        """
        predicted = []
        
        for obs_id, (x, y, radius) in obstacles_dict.items():
            vx, vy = self.estimate_velocity(obs_id)
            predicted.append((x, y, vx, vy, radius))
        
        return predicted


if __name__ == '__main__':
    # 测试MPC控制器
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("MPC控制器测试")
    print("="*60)
    
    # 创建测试轨迹
    path = [
        [0, 0],
        [1000, 0],
        [2000, 500],
        [3000, 1000],
        [4000, 1000],
        [5000, 500],
        [6000, 0]
    ]
    
    # 创建MPC控制器
    mpc = MPCController(
        prediction_horizon=10,
        control_horizon=5,
        dt=0.1,
        max_linear_vel=1000,
        robot_radius=150
    )
    
    mpc.set_trajectory(path, k=3, s=0)
    
    # 模拟动态障碍物
    dynamic_obstacles = [
        (2500, 800, 100, 0, 150),    # 静止障碍物
        (3500, 600, -200, 100, 150)  # 移动障碍物
    ]
    
    mpc.update_dynamic_obstacles(dynamic_obstacles)
    
    # 模拟轨迹跟踪
    robot_pos = np.array([0.0, 0.0])
    robot_orientation = 0.0
    current_u = 0.0
    
    trajectory_history = [robot_pos.copy()]
    
    print("\n开始模拟...")
    for step in range(30):
        vx, vw, finished, debug_info = mpc.compute_control(
            robot_pos, 
            robot_orientation,
            current_u
        )
        
        if finished:
            print(f"Step {step}: 到达终点!")
            break
        
        # 更新机器人状态
        robot_orientation += vw * mpc.dt
        robot_pos[0] += vx * np.cos(robot_orientation) * mpc.dt
        robot_pos[1] += vx * np.sin(robot_orientation) * mpc.dt
        
        trajectory_history.append(robot_pos.copy())
        
        # 更新动态障碍物
        dynamic_obstacles = [
            (obs[0] + obs[2] * mpc.dt, obs[1] + obs[3] * mpc.dt, obs[2], obs[3], obs[4])
            for obs in dynamic_obstacles
        ]
        mpc.update_dynamic_obstacles(dynamic_obstacles)
        
        if step % 5 == 0:
            print(f"Step {step}: pos=({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), "
                  f"vx={vx:.1f}, vw={vw:.2f}, "
                  f"cost={debug_info['cost']:.1f}, "
                  f"opt_time={debug_info['optimization_time']:.1f}ms")
    
    # 可视化
    trajectory_history = np.array(trajectory_history)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制轨迹
    u_dense = np.linspace(0, 1, 200)
    ref_traj = np.array(splev(u_dense, mpc.tck)).T
    
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'b--', linewidth=2, label='Reference Trajectory')
    plt.plot(trajectory_history[:, 0], trajectory_history[:, 1], 'r-', linewidth=2, label='MPC Trajectory')
    
    # 绘制控制点
    plt.plot([p[0] for p in path], [p[1] for p in path], 'go', markersize=8, label='Control Points')
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('MPC Controller Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('mpc_test.png', dpi=150, bbox_inches='tight')
    
    print("\n测试完成! 结果已保存: mpc_test.png")
