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
import math
import time
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev


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
                 max_linear_vel=0.5,         # m/s
                 max_angular_vel=1.0,        # rad/s
                 max_linear_acc=1.0,         # m/s^2
                 max_angular_acc=2.0,        # rad/s^2
                 robot_radius=0.2,           # m (考虑安全裕度)
                 safety_distance=0.1):       # m 额外安全距离
        
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
        self.Q_position = 20.0       # 位置跟踪权重
        self.Q_orientation = 10.0    # 朝向权重
        self.R_vx = 0.1              # 线速度代价
        self.R_vw = 0.1              # 角速度代价
        self.R_dvx = 1.0             # 线速度变化代价
        self.R_dvw = 1.0             # 角速度变化代价
        self.Q_obstacle = 100.0      # 障碍物惩罚权重
        
        # 轨迹参数
        self.tck = None
        self.trajectory_length = 0
        
        # 上一次的控制量(用于计算加速度)
        self.last_vx = 0.0
        self.last_vw = 0.0
        
        # 动态障碍物列表
        self.dynamic_obstacles = []  # [(x, y, vx, vy, radius), ...]
        
        print(f"MPC Controller initialized: N={self.N}, M={self.M}, dt={self.dt}s")
    
    def set_trajectory(self, tck):
        """设置参考轨迹(B样条参数)"""
        self.tck = tck
        
        # 计算轨迹长度
        u_dense = np.linspace(0, 1, 200)
        points = np.array(splev(u_dense, self.tck)).T
        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs**2).sum(axis=1))
        self.trajectory_length = distances.sum()
        
        return self.tck
    
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
        # 局部搜索
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
        avg_velocity = self.vx_max * 0.8  # 假设平均速度
        ds_per_step = avg_velocity * self.dt  # 每步前进的距离
        
        # 转换为B样条参数的增量
        if self.trajectory_length > 0:
            du_per_step = ds_per_step / self.trajectory_length
        else:
            du_per_step = 0.01
        
        ref_states = []
        u = current_u
        
        for i in range(N + 1):
            u = min(u, 1.0)
            
            # 当前位置
            x, y = splev(u, self.tck)
            
            # 计算切线方向(朝向)
            dx, dy = splev(u, self.tck, der=1)
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
                # 预测障碍物位置 (假设匀速直线运动)
                time_step = (i + 1) * self.dt
                for obs in self.dynamic_obstacles:
                    ox, oy, ovx, ovy, oradius = obs
                    pred_ox = ox + ovx * time_step
                    pred_oy = oy + ovy * time_step
                    
                    dist = np.sqrt((state[0] - pred_ox)**2 + (state[1] - pred_oy)**2)
                    safe_dist = self.robot_radius + oradius + self.safety_distance
                    
                    if dist < safe_dist:
                        obstacle_cost += self.Q_obstacle * (safe_dist - dist)**2
                        if dist < self.robot_radius + oradius:
                            obstacle_cost += 10000  # 碰撞惩罚
        
        total_cost = position_cost + orientation_cost + control_cost + smoothness_cost + obstacle_cost
        
        return total_cost
    
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
            return 0.0, 0.0, True, {}
        
        # 当前状态
        current_state = np.array([current_pos[0], current_pos[1], current_orientation])
        
        # 找到轨迹上的最近点
        closest_u, _ = self.find_closest_point_on_trajectory(current_pos, current_u)
        
        # 检查是否到达终点
        if closest_u >= 0.98:
            end_point = np.array(splev(1.0, self.tck)).flatten()
            dist_to_end = np.linalg.norm(current_pos - end_point)
            if dist_to_end < 0.15: # 15cm
                return 0.0, 0.0, True, {'closest_u': 1.0}
        
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
        
        # 优化
        start_time = time.time()
        
        # 使用SLSQP求解
        result = minimize(
            fun=lambda u: self.compute_cost(u, current_state, ref_trajectory),
            x0=u0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 20, 'ftol': 1e-3, 'disp': False}
        )
        
        opt_time = time.time() - start_time
        
        # 提取第一步的控制量
        if result.success:
            U_opt = result.x.reshape((self.M, 2))
            vx_opt = float(U_opt[0, 0])
            vw_opt = float(U_opt[0, 1])
        else:
            # 优化失败，减速停车
            vx_opt = max(0, self.last_vx - self.ax_max * self.dt)
            vw_opt = 0.0
            print("MPC optimization failed!")
        
        # 更新历史控制量
        self.last_vx = vx_opt
        self.last_vw = vw_opt
        
        # 调试信息
        debug_info = {
            'closest_u': closest_u,
            'progress': closest_u * 100,
            'optimization_time': opt_time * 1000,  # ms
            'cost': result.fun,
            'success': result.success,
            'n_obstacles': len(self.dynamic_obstacles)
        }
        
        return vx_opt, vw_opt, False, debug_info
