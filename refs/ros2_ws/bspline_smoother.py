"""
B样条路径平滑模块
用于将RRT*生成的离散路径平滑为连续可导的B样条曲线
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize


class BSplineSmoother:
    """B样条路径平滑器"""
    
    def __init__(self, robot_radius=100):
        """
        初始化平滑器
        
        Args:
            robot_radius: 机器人半径（用于碰撞检测）
        """
        self.robot_radius = robot_radius
        
    def smooth_path(self, path, obstacles, s=0, k=3, num_points=100):
        """
        使用B样条平滑路径
        
        Args:
            path: 原始路径点列表 [[x1,y1], [x2,y2], ...]
            obstacles: 障碍物列表 [(x, y, radius), ...]
            s: 平滑因子，0表示插值，越大越平滑
            k: B样条阶数，通常为3（三次样条）
            num_points: 输出路径点数量
            
        Returns:
            平滑后的路径点列表
        """
        if len(path) < 4:
            # 路径点太少，无法生成B样条，直接返回
            print(f"Warning: Path too short ({len(path)} points), returning original path")
            return path
            
        # 将路径转换为numpy数组
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        # 确保k不超过点数-1
        k = min(k, len(path) - 1)
        
        try:
            # 使用splprep生成B样条
            # s=0表示插值（通过所有点），s>0表示平滑拟合
            tck, u = splprep([x, y], s=s, k=k)
            
            # 生成平滑路径点
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            
            smoothed_path = [[x_new[i], y_new[i]] for i in range(len(x_new))]
            
            # 检查平滑后的路径是否碰撞（使用宽松的检测）
            collision_count = self._count_collisions(smoothed_path, obstacles)
            
            # 如果碰撞点太多（>10%），尝试降低平滑度
            if collision_count > len(smoothed_path) * 0.1:
                print(f"Warning: Smoothed path has {collision_count} collisions, trying less smoothing")
                if s > 0:
                    # 降低平滑因子
                    return self.smooth_path(path, obstacles, s=0, k=k, num_points=num_points)
                else:
                    # 插值模式下仍有碰撞，使用更少的采样点
                    print(f"Using fewer sample points to avoid obstacles")
                    return self.smooth_path(path, obstacles, s=0, k=k, num_points=len(path)*2)
            
            return smoothed_path
            
        except Exception as e:
            print(f"B-spline smoothing failed: {e}, returning original path")
            return path
    
    def _count_collisions(self, path, obstacles):
        """
        统计路径中有多少点碰撞
        
        Args:
            path: 路径点列表
            obstacles: 障碍物列表
            
        Returns:
            碰撞点的数量
        """
        collision_count = 0
        for point in path:
            for obs in obstacles:
                obs_x, obs_y, obs_radius = obs
                dist = np.sqrt((point[0] - obs_x)**2 + (point[1] - obs_y)**2)
                # 使用稍微宽松的碰撞检测（0.9倍的安全距离）
                if dist < (self.robot_radius + obs_radius) * 0.9:
                    collision_count += 1
                    break  # 这个点已经碰撞了，不需要检查其他障碍物
        return collision_count
    
    def optimize_path(self, path, obstacles, max_iterations=50):
        """
        优化路径以避免碰撞并最小化长度
        
        Args:
            path: 初始路径
            obstacles: 障碍物列表
            max_iterations: 最大迭代次数
            
        Returns:
            优化后的路径
        """
        if len(path) < 3:
            return path
        
        # 固定起点和终点，只优化中间点
        start = path[0]
        end = path[-1]
        middle_points = np.array(path[1:-1]).flatten()
        
        def objective(x):
            """目标函数：路径长度 + 碰撞惩罚"""
            points = np.concatenate([[start], x.reshape(-1, 2), [end]])
            
            # 计算路径长度
            length = 0
            for i in range(len(points) - 1):
                length += np.linalg.norm(points[i+1] - points[i])
            
            # 计算碰撞惩罚
            collision_penalty = 0
            for point in points:
                for obs in obstacles:
                    obs_x, obs_y, obs_radius = obs
                    dist = np.sqrt((point[0] - obs_x)**2 + (point[1] - obs_y)**2)
                    safe_dist = self.robot_radius + obs_radius
                    if dist < safe_dist:
                        collision_penalty += 1000 * (safe_dist - dist)
            
            return length + collision_penalty
        
        # 优化
        result = minimize(objective, middle_points, method='BFGS', 
                         options={'maxiter': max_iterations})
        
        if result.success:
            optimized_middle = result.x.reshape(-1, 2)
            return [start] + optimized_middle.tolist() + [end]
        else:
            return path


class VelocityProfiler:
    """速度规划器 - 为路径生成时间最优的速度曲线"""
    
    def __init__(self, max_vel=1000, max_acc=500, max_angular_vel=3.0):
        """
        初始化速度规划器
        
        Args:
            max_vel: 最大线速度 (mm/s)
            max_acc: 最大加速度 (mm/s^2)
            max_angular_vel: 最大角速度 (rad/s)
        """
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_angular_vel = max_angular_vel
    
    def compute_velocity_profile(self, path, dt=0.01):
        """
        计算路径的速度曲线
        
        Args:
            path: 路径点列表
            dt: 时间步长
            
        Returns:
            [(x, y, v, t), ...] 每个点的位置、速度、时间
        """
        if len(path) < 2:
            return []
        
        # 计算每段的长度
        segments = []
        total_length = 0
        for i in range(len(path) - 1):
            seg_length = np.linalg.norm(
                np.array(path[i+1]) - np.array(path[i])
            )
            segments.append(seg_length)
            total_length += seg_length
        
        # 简单的梯形速度曲线
        # 加速 -> 匀速 -> 减速
        acc_dist = self.max_vel**2 / (2 * self.max_acc)
        
        profile = []
        current_dist = 0
        current_time = 0
        
        for i, point in enumerate(path):
            if current_dist < acc_dist:
                # 加速阶段
                v = np.sqrt(2 * self.max_acc * current_dist)
            elif current_dist > total_length - acc_dist:
                # 减速阶段
                remaining = total_length - current_dist
                v = np.sqrt(2 * self.max_acc * remaining)
            else:
                # 匀速阶段
                v = self.max_vel
            
            v = min(v, self.max_vel)
            profile.append((*point, v, current_time))
            
            if i < len(path) - 1:
                current_dist += segments[i]
                if v > 0:
                    current_time += segments[i] / v
        
        return profile
