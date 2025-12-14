"""
轨迹跟踪控制器 - 基于B样条参数化的连续轨迹跟踪

相比点对点控制的优势:
1. 连续跟踪,无停顿
2. 考虑轨迹曲率,预判转向
3. 速度平滑变化
4. 支持前馈+反馈控制
"""

import numpy as np
from scipy.interpolate import splprep, splev, splrep


class TrajectoryController:
    """
    B样条轨迹跟踪控制器
    
    特点:
    - 使用B样条参数 u(t) 而不是离散点
    - 计算曲率来调整速度
    - 前馈+反馈控制
    """
    
    def __init__(self, 
                 max_linear_vel=1000,      # mm/s
                 max_angular_vel=3.0,       # rad/s
                 lookahead_distance=300,    # mm 前视距离
                 kp_lateral=2.0,            # 横向误差增益
                 kp_angular=4.0):           # 角度误差增益
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.lookahead_distance = lookahead_distance
        self.kp_lateral = kp_lateral
        self.kp_angular = kp_angular
        
        # 轨迹参数
        self.tck = None           # B样条参数
        self.u_values = None      # 参数范围
        self.total_length = 0     # 轨迹总长度
        
        # 当前跟踪状态
        self.current_u = 0.0      # 当前B样条参数
        self.target_u = 0.0       # 目标参数(前视点)
        
    def set_trajectory(self, path, k=3, s=0):
        """
        设置要跟踪的轨迹
        
        Args:
            path: 路径点列表 [(x1,y1), (x2,y2), ...]
            k: B样条阶数
            s: 平滑参数
        """
        if len(path) < k + 1:
            raise ValueError(f"Path needs at least {k+1} points for order {k} B-spline")
        
        # 提取坐标
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        
        # 生成B样条
        self.tck, u = splprep([x, y], s=s, k=k)
        self.u_values = u
        
        # 计算轨迹总长度(用于进度估计)
        u_dense = np.linspace(0, 1, 200)
        points = np.array(splev(u_dense, self.tck)).T
        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs**2).sum(axis=1))
        self.total_length = distances.sum()
        
        # 初始化跟踪参数
        self.current_u = 0.0
        self.target_u = 0.0
        
        return self.tck  # 返回B样条参数用于可视化
        
        print(f"Trajectory set: {len(path)} control points, {self.total_length:.1f}mm total length")
    
    def find_closest_point_on_trajectory(self, robot_pos):
        """
        在轨迹上找到最接近机器人的点
        
        Returns:
            (closest_u, closest_point, distance)
        """
        # 在轨迹上采样多个点
        u_samples = np.linspace(max(0, self.current_u - 0.2), 
                                min(1, self.current_u + 0.2), 50)
        
        trajectory_points = np.array(splev(u_samples, self.tck)).T
        
        # 计算到机器人的距离
        distances = np.sqrt(((trajectory_points - robot_pos)**2).sum(axis=1))
        
        # 找到最近点
        min_idx = distances.argmin()
        closest_u = u_samples[min_idx]
        closest_point = trajectory_points[min_idx]
        min_distance = distances[min_idx]
        
        return closest_u, closest_point, min_distance
    
    def find_lookahead_point(self, start_u, lookahead_dist):
        """
        从start_u开始,沿轨迹前进lookahead_dist距离,找到前视点
        
        Returns:
            (lookahead_u, lookahead_point)
        """
        if start_u >= 1.0:
            # 已到达终点
            target_point = np.array(splev(1.0, self.tck)).flatten()
            return 1.0, target_point
        
        # 从当前点开始,逐步增加u直到累计距离达到lookahead_dist
        u_current = start_u
        accumulated_dist = 0
        du = 0.01  # 步长
        
        while u_current < 1.0 and accumulated_dist < lookahead_dist:
            u_next = min(u_current + du, 1.0)
            
            # 计算这一小段的长度
            p1 = np.array(splev(u_current, self.tck)).flatten()
            p2 = np.array(splev(u_next, self.tck)).flatten()
            segment_length = np.linalg.norm(p2 - p1)
            
            accumulated_dist += segment_length
            u_current = u_next
        
        lookahead_point = np.array(splev(u_current, self.tck)).flatten()
        return u_current, lookahead_point
    
    def compute_curvature(self, u):
        """
        计算轨迹在参数u处的曲率
        
        曲率 κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        
        Returns:
            curvature (1/mm)
        """
        # 计算一阶导数
        dx_du, dy_du = splev(u, self.tck, der=1)
        
        # 计算二阶导数
        ddx_ddu, ddy_ddu = splev(u, self.tck, der=2)
        
        # 曲率公式
        numerator = abs(dx_du * ddy_ddu - dy_du * ddx_ddu)
        denominator = (dx_du**2 + dy_du**2)**(3/2)
        
        if denominator < 1e-6:
            return 0
        
        curvature = numerator / denominator
        return curvature
    
    def compute_control(self, robot_pos, robot_orientation):
        """
        计算控制指令
        
        Args:
            robot_pos: [x, y] 机器人位置
            robot_orientation: 机器人朝向 (rad)
            
        Returns:
            (vx, vw, is_finished)
        """
        # 1. 找到机器人在轨迹上的最近点
        closest_u, closest_point, lateral_error = self.find_closest_point_on_trajectory(robot_pos)
        self.current_u = closest_u
        
        # 2. 找到前视点
        lookahead_u, lookahead_point = self.find_lookahead_point(
            closest_u, self.lookahead_distance
        )
        self.target_u = lookahead_u
        
        # 3. 判断是否完成
        if lookahead_u >= 0.99:
            # 接近终点,检查是否真的到达
            end_point = np.array(splev(1.0, self.tck)).flatten()
            dist_to_end = np.linalg.norm(robot_pos - end_point)
            if dist_to_end < 150:
                return 0, 0, True  # 已完成
        
        # 4. 计算前视点的方向
        dx = lookahead_point[0] - robot_pos[0]
        dy = lookahead_point[1] - robot_pos[1]
        target_angle = np.arctan2(dy, dx)
        
        # 5. 计算角度误差
        angle_error = self._normalize_angle(target_angle - robot_orientation)
        
        # 6. 计算轨迹曲率,用于速度调整
        curvature = self.compute_curvature(closest_u)
        
        # 7. 根据曲率调整线速度
        # 曲率大 → 转弯半径小 → 速度慢
        # v_max = sqrt(a_max / κ) 离心加速度限制
        if curvature > 1e-3:  # 避免除零
            curvature_limited_vel = min(
                self.max_linear_vel,
                np.sqrt(2000 / curvature)  # 假设最大横向加速度 2000 mm/s^2
            )
        else:
            curvature_limited_vel = self.max_linear_vel
        
        # 8. 根据角度误差调整速度(反馈控制)
        angle_factor = np.cos(angle_error)
        angle_factor = max(0, angle_factor)
        
        # 根据横向误差调整速度
        if lateral_error > 200:
            lateral_factor = 200 / lateral_error
        else:
            lateral_factor = 1.0
        
        # 综合速度
        vx = curvature_limited_vel * angle_factor * lateral_factor
        
        # 如果角度误差太大,先原地转向
        if abs(angle_error) > np.pi / 3:
            vx = 0
        
        # 9. 计算角速度(比例控制 + 前馈)
        # 前馈项:根据轨迹曲率
        feedforward_omega = curvature * vx if curvature > 0 else 0
        
        # 反馈项:角度误差
        feedback_omega = self.kp_angular * angle_error
        
        vw = feedforward_omega + feedback_omega
        vw = np.clip(vw, -self.max_angular_vel, self.max_angular_vel)
        
        return vx, vw, False
    
    def get_progress(self):
        """
        获取轨迹跟踪进度 0~1
        """
        return self.current_u
    
    @staticmethod
    def _normalize_angle(angle):
        """将角度归一化到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_debug_info(self):
        """
        获取调试信息
        """
        if self.tck is None:
            return {}
        
        current_point = np.array(splev(self.current_u, self.tck)).flatten()
        target_point = np.array(splev(self.target_u, self.tck)).flatten()
        curvature = self.compute_curvature(self.current_u)
        
        return {
            'current_u': self.current_u,
            'target_u': self.target_u,
            'progress': self.current_u * 100,
            'current_point': current_point,
            'target_point': target_point,
            'curvature': curvature,
            'turn_radius': 1/curvature if curvature > 1e-6 else float('inf')
        }


if __name__ == '__main__':
    # 测试轨迹控制器
    import matplotlib.pyplot as plt
    
    # 创建测试路径(S型曲线)
    path = [
        [0, 0],
        [1000, 0],
        [2000, 500],
        [3000, 1000],
        [4000, 1000],
        [5000, 500],
        [6000, 0]
    ]
    
    # 创建控制器
    controller = TrajectoryController(
        max_linear_vel=1000,
        lookahead_distance=300
    )
    
    # 设置轨迹
    controller.set_trajectory(path, k=3, s=0)
    
    # 可视化轨迹和曲率
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制轨迹
    u_dense = np.linspace(0, 1, 200)
    trajectory = np.array(splev(u_dense, controller.tck)).T
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='B样条轨迹')
    ax1.plot([p[0] for p in path], [p[1] for p in path], 'ro-', markersize=8, label='控制点')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('轨迹可视化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 绘制曲率分布
    curvatures = [controller.compute_curvature(u) for u in u_dense]
    arc_lengths = []
    s = 0
    for i in range(len(trajectory)-1):
        s += np.linalg.norm(trajectory[i+1] - trajectory[i])
        arc_lengths.append(s)
    arc_lengths = [0] + arc_lengths
    
    ax2.plot(arc_lengths, curvatures, 'g-', linewidth=2)
    ax2.set_xlabel('弧长 (mm)')
    ax2.set_ylabel('曲率 (1/mm)')
    ax2.set_title('轨迹曲率分布')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trajectory_controller_demo.png', dpi=150, bbox_inches='tight')
    print("演示图已保存: trajectory_controller_demo.png")
    
    # 模拟跟踪
    print("\n模拟轨迹跟踪:")
    robot_pos = np.array([0.0, 0.0])
    robot_orientation = 0.0
    dt = 0.05  # 50ms
    
    for i in range(20):
        vx, vw, finished = controller.compute_control(robot_pos, robot_orientation)
        
        if finished:
            print(f"Step {i}: 轨迹跟踪完成!")
            break
        
        # 简单的运动学更新
        robot_orientation += vw * dt
        robot_pos[0] += vx * np.cos(robot_orientation) * dt
        robot_pos[1] += vx * np.sin(robot_orientation) * dt
        
        debug_info = controller.get_debug_info()
        print(f"Step {i}: pos=({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), "
              f"orient={np.degrees(robot_orientation):.1f}°, "
              f"vx={vx:.1f}, vw={vw:.2f}, "
              f"progress={debug_info['progress']:.1f}%, "
              f"curvature={debug_info['curvature']:.6f}")
