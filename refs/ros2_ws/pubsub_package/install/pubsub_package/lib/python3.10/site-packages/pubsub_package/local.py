#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
局部规划节点 - 轨迹跟踪与动态避障

功能:
1. 接收全局路径，生成B样条平滑轨迹
2. 使用Pure Pursuit进行轨迹跟踪
3. 使用激光雷达进行动态避障
"""
import sys
sys.path.append("/home/ros2_ws/pubsub_package/pubsub_package/")
import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_pose, PoseStamped
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
import math
import numpy as np
import time
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from threading import Lock, Thread
from mpc_controller import MPCController


class BSplineSmoother:
    """B样条路径平滑器"""
    
    def __init__(self, robot_radius=0.2):
        self.robot_radius = robot_radius
        
    def smooth_path(self, path, obstacles=None, s=0, k=3, num_points=100):
        """
        使用B样条平滑路径
        
        Args:
            path: 原始路径点列表 [[x1,y1], [x2,y2], ...]
            obstacles: 障碍物列表 [(x, y, radius), ...] (可选)
            s: 平滑因子，0表示插值，越大越平滑
            k: B样条阶数，通常为3（三次样条）
            num_points: 输出路径点数量
            
        Returns:
            (smoothed_path, tck): 平滑后的路径和B样条参数
        """
        if len(path) < 4:
            return path, None
            
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]
        
        k = min(k, len(path) - 1)
        
        try:
            tck, u = splprep([x, y], s=s, k=k)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            smoothed_path = [[x_new[i], y_new[i]] for i in range(len(x_new))]
            return smoothed_path, tck
        except Exception as e:
            print(f"B-spline smoothing failed: {e}")
            return path, None


class LocalPlanner(Node):
    def __init__(self, real):
        super().__init__('local_planner')
        self.real = real
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vw = 0.0
        self.path = Path()
        self.arrive = 0.2  # 到达判定距离 (m)
        self.threshold = 1.5  # 激光雷达阈值 (m)
        self.robot_size = 0.2  # 机器人半径 (m)
        self.V_X = 0.5  # 最大线速度 (m/s)
        self.V_W = 1.0  # 最大角速度 (rad/s)

        # 初始化B样条平滑器和MPC控制器
        self.smoother = BSplineSmoother(robot_radius=self.robot_size)
        self.controller = MPCController(
            prediction_horizon=10,
            control_horizon=5,
            dt=0.05,  # 50ms
            max_linear_vel=self.V_X,
            max_angular_vel=self.V_W,
            robot_radius=self.robot_size,
            safety_distance=0.1
        )
        
        # 轨迹是否已设置
        self.trajectory_ready = False
        self.current_u = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_sub = self.create_subscription(Path, '/course_agv/global_path', self.path_callback, 1)
        self.midpose_pub = self.create_publisher(PoseStamped, '/course_agv/mid_goal', 1)
        
        if self.real:
            self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 1)
            self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        else:
            self.laser_sub = self.create_subscription(LaserScan, '/course_agv/laser/scan', self.laser_callback, 1)
            self.vel_pub = self.create_publisher(Twist, '/course_agv/velocity', 1)

        self.planning_thread = None
        self.lock = Lock()
        self.laser_lock = Lock()
        
        # 障碍物数据
        self.ob = []

        self.traj_pub = self.create_publisher(Path, '/course_agv/trajectory', 1)
        self.smooth_path_pub = self.create_publisher(Path, '/course_agv/smooth_path', 1)
        self.traj = Path()
        self.traj.header.frame_id = 'map'
        
        self.get_logger().info("Local Planner initialized with MPC trajectory tracking")

    def path_callback(self, msg):
        """接收全局路径并生成平滑轨迹"""
        self.lock.acquire()
        self.path = msg
        
        # 将Path消息转换为点列表
        path_points = []
        for pose in msg.poses:
            path_points.append([pose.pose.position.x, pose.pose.position.y])
        
        # 1. 简单的可视点连接平滑 (Greedy Path Smoothing)
        # 注意：这里没有全局地图，只能假设RRT*生成的路径是安全的，
        # 或者使用局部障碍物进行有限的检查。这里仅作为示例实现，
        # 实际应用中应在Global Planner中结合地图进行。
        if len(path_points) > 2:
            path_points = self._greedy_smoothing(path_points)
            self.get_logger().info(f"Greedy smoothing: reduced to {len(path_points)} points")

        if len(path_points) >= 4:
            # 2. B样条平滑
            smoothed_path, tck = self.smoother.smooth_path(
                path_points, 
                s=0,  # 插值模式
                k=3,  # 三次样条
                num_points=max(50, len(path_points) * 3)
            )
            
            if tck is not None:
                self.controller.set_trajectory(tck)
                self.trajectory_ready = True
                self.current_u = 0.0
                self.get_logger().info(f"Trajectory set: length: {self.controller.trajectory_length:.2f}m")
                
                # 发布平滑路径用于可视化
                self._publish_smooth_path(smoothed_path)
            else:
                self.trajectory_ready = False
        else:
            # 路径点太少
            self.trajectory_ready = False
            self.get_logger().warn(f"Path too short: {len(path_points)} points")
        
        self.update_global_pose(init=True)
        self.lock.release()
        
        if self.planning_thread is None and self.trajectory_ready:
            self.planning_thread = Thread(target=self.plan_thread_func)
            self.planning_thread.start()

    def _greedy_smoothing(self, path):
        """
        贪婪路径平滑：尝试连接不相邻的点，如果连线无碰撞则替换中间路径
        注意：由于缺少全局地图，这里仅做简单的直线连接，不进行严格的碰撞检测
        (假设RRT*生成的路径在宽阔区域是冗余的)
        """
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # 尝试连接尽可能远的点
            next_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx + 1, -1):
                # 这里应该检查 collision_check(path[current_idx], path[i])
                # 由于没有地图，我们保守一点，只在局部范围内尝试跳过点
                # 或者假设如果两点距离较近且中间点不多，可以直接连接
                dist = math.hypot(path[i][0] - path[current_idx][0], 
                                  path[i][1] - path[current_idx][1])
                
                # 简化的逻辑：如果距离小于一定阈值，或者中间点都在连线附近(这里省略复杂计算)
                # 这里仅演示结构，实际需要 map 数据
                # 作为一个简单的启发式：每隔几个点取一个，减少路径点数量
                if i == current_idx + 1:
                    next_idx = i
                    break
                
                # 模拟检查通过
                # next_idx = i
                # break
            
            smoothed.append(path[next_idx])
            current_idx = next_idx
            
        return path # 暂时返回原路径，避免在无地图情况下产生碰撞

    def _publish_smooth_path(self, smoothed_path):
        """发布平滑路径用于RViz可视化"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for point in smoothed_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.smooth_path_pub.publish(path_msg)

    def plan_thread_func(self):
        self.get_logger().info("Running MPC tracking thread!")
        rate = 20  # 20Hz 控制频率
        
        while True:
            start_time = time.time()
            
            self.lock.acquire()
            finished = self.plan_once()
            self.lock.release()
            
            if finished:
                self.lock.acquire()
                self.publish_velocity(zero=True)
                self.lock.release()
                self.get_logger().info("Arrived at goal!")
                break
            
            # 控制循环频率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0/rate - elapsed)
            time.sleep(sleep_time)
            
        self.planning_thread = None
        self.trajectory_ready = False
        self.get_logger().info("Exiting MPC tracking thread!")
        self.get_logger().info("----------------------------------------------------")

    def plan_once(self):
        """执行一次MPC控制"""
        self.update_global_pose(init=False)
        
        if not self.trajectory_ready:
            return True
        
        # 获取障碍物（在机器人坐标系下）
        self.update_obstacle()
        
        # 构造MPC需要的障碍物格式: [(x, y, vx, vy, radius), ...]
        # 这里假设障碍物是静止的，或者速度未知(设为0)
        mpc_obstacles = []
        if hasattr(self, 'plan_ob') and len(self.plan_ob) > 0:
            for obs in self.plan_ob:
                # 假设障碍物半径0.1m，速度0
                mpc_obstacles.append((obs[0], obs[1], 0.0, 0.0, 0.1))
        
        self.controller.update_dynamic_obstacles(mpc_obstacles)
        
        # 计算控制指令
        robot_pos = np.array([self.x, self.y])
        vx, vw, finished, debug_info = self.controller.compute_control(
            robot_pos, 
            self.yaw,
            self.current_u
        )
        
        # 更新当前轨迹参数
        if 'closest_u' in debug_info:
            self.current_u = debug_info['closest_u']
        
        # 限幅
        self.vx = max(min(vx, self.V_X), -self.V_X)
        self.vw = max(min(vw, self.V_W), -self.V_W)
        
        # 发布速度
        self.publish_velocity(zero=False)
        
        # 打印调试信息
        if int(self.current_u * 100) % 10 == 0:
            self.get_logger().info(f"MPC Progress: {self.current_u*100:.1f}%, "
                                   f"vx={self.vx:.2f}, vw={self.vw:.2f}, "
                                   f"cost={debug_info.get('cost', 0):.1f}")
        
        return finished

        self.yaw = 0.0
        self.vx = 0.0
        self.vw = 0.0
        self.path = Path()
        self.arrive = 0.2  # 到达判定距离 (m)
        self.threshold = 1.5  # 激光雷达阈值 (m)
        self.robot_size = 0.2  # 机器人半径 (m)
        self.V_X = 0.5  # 最大线速度 (m/s)
        self.V_W = 1.0  # 最大角速度 (rad/s)

        # 初始化B样条平滑器和轨迹控制器
        self.smoother = BSplineSmoother(robot_radius=self.robot_size)
        self.controller = TrajectoryController(
            max_linear_vel=self.V_X,
            max_angular_vel=self.V_W,
            lookahead_distance=0.4,  # 前视距离 0.4m
            kp_angular=2.0
        )
        
        # 轨迹是否已设置
        self.trajectory_ready = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_sub = self.create_subscription(Path, '/course_agv/global_path', self.path_callback, 1)
        self.midpose_pub = self.create_publisher(PoseStamped, '/course_agv/mid_goal', 1)
        
        if self.real:
            self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 1)
            self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        else:
            self.laser_sub = self.create_subscription(LaserScan, '/course_agv/laser/scan', self.laser_callback, 1)
            self.vel_pub = self.create_publisher(Twist, '/course_agv/velocity', 1)

        self.planning_thread = None
        self.lock = Lock()
        self.laser_lock = Lock()
        
        # 障碍物数据
        self.ob = []

        self.traj_pub = self.create_publisher(Path, '/course_agv/trajectory', 1)
        self.smooth_path_pub = self.create_publisher(Path, '/course_agv/smooth_path', 1)
        self.traj = Path()
        self.traj.header.frame_id = 'map'
        
        self.get_logger().info("Local Planner initialized with B-spline trajectory tracking")

    def path_callback(self, msg):
        """接收全局路径并生成平滑轨迹"""
        self.lock.acquire()
        self.path = msg
        
        # 将Path消息转换为点列表
        path_points = []
        for pose in msg.poses:
            path_points.append([pose.pose.position.x, pose.pose.position.y])
        
        if len(path_points) >= 4:
            # B样条平滑
            smoothed_path, tck = self.smoother.smooth_path(
                path_points, 
                s=0,  # 插值模式
                k=3,  # 三次样条
                num_points=max(50, len(path_points) * 3)
            )
            
            if tck is not None:
                self.controller.set_trajectory(tck)
                self.trajectory_ready = True
                self.get_logger().info(f"Trajectory set: {len(path_points)} -> {len(smoothed_path)} points, "
                                       f"length: {self.controller.total_length:.2f}m")
                
                # 发布平滑路径用于可视化
                self._publish_smooth_path(smoothed_path)
            else:
                # B样条失败，使用原始路径
                success = self.controller.set_trajectory_from_path(path_points, k=min(3, len(path_points)-1), s=0)
                self.trajectory_ready = success
                if success:
                    self.get_logger().info(f"Using original path: {len(path_points)} points")
        else:
            # 路径点太少
            self.trajectory_ready = False
            self.get_logger().warn(f"Path too short: {len(path_points)} points")
        
        self.update_global_pose(init=True)
        self.lock.release()
        
        if self.planning_thread is None and self.trajectory_ready:
            self.planning_thread = Thread(target=self.plan_thread_func)
            self.planning_thread.start()

    def _publish_smooth_path(self, smoothed_path):
        """发布平滑路径用于RViz可视化"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for point in smoothed_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.smooth_path_pub.publish(path_msg)

    def plan_thread_func(self):
        self.get_logger().info("Running trajectory tracking thread!")
        rate = 20  # 20Hz 控制频率
        
        while True:
            start_time = time.time()
            
            self.lock.acquire()
            finished = self.plan_once()
            self.lock.release()
            
            if finished:
                self.lock.acquire()
                self.publish_velocity(zero=True)
                self.lock.release()
                self.get_logger().info("Arrived at goal!")
                break
            
            # 控制循环频率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0/rate - elapsed)
            time.sleep(sleep_time)
            
        self.planning_thread = None
        self.trajectory_ready = False
        self.get_logger().info("Exiting trajectory tracking thread!")
        self.get_logger().info("----------------------------------------------------")

    def plan_once(self):
        """执行一次轨迹跟踪控制"""
        self.update_global_pose(init=False)
        
        if not self.trajectory_ready:
            return True
        
        # 获取障碍物（在机器人坐标系下）
        self.update_obstacle()
        obstacles = self.plan_ob.tolist() if hasattr(self, 'plan_ob') and len(self.plan_ob) > 0 else None
        
        # 计算控制指令
        robot_pos = np.array([self.x, self.y])
        vx, vw, finished = self.controller.compute_control(
            robot_pos, 
            self.yaw,
            obstacles=obstacles
        )
        
        # 限幅
        self.vx = max(min(vx, self.V_X), -self.V_X)
        self.vw = max(min(vw, self.V_W), -self.V_W)
        
        # 发布速度
        self.publish_velocity(zero=False)
        
        # 打印调试信息
        progress = self.controller.get_progress() * 100
        if int(progress) % 10 == 0:
            self.get_logger().info(f"Progress: {progress:.1f}%, pos=({self.x:.2f}, {self.y:.2f}), "
                                   f"vx={self.vx:.2f}, vw={self.vw:.2f}")
        
        return finished

    def update_global_pose(self, init=False):
        try:
            trans = self.tf_buffer.lookup_transform(
                'map', 
                'base_footprint',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )

            self.x = trans.transform.translation.x
            self.y = trans.transform.translation.y

            rotation = R.from_quat([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
            self.yaw = yaw

        except TransformException as e:
            self.get_logger().error(f"TF error: {e}")
            return

        # 发布轨迹
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        self.traj.poses.append(pose)
        self.traj_pub.publish(self.traj)

        if init:
            self.traj.poses = []
            
        # 计算到终点的距离
        if len(self.path.poses) > 0:
            self.goal_dis = math.hypot(
                self.x - self.path.poses[-1].pose.position.x,
                self.y - self.path.poses[-1].pose.position.y
            )
        else:
            self.goal_dis = float('inf')

    def laser_callback(self, msg):
        """处理激光雷达数据"""
        self.laser_lock.acquire()
        self.ob = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        for i, r in enumerate(msg.ranges):
            # 过滤无效数据
            if r < msg.range_min or r > msg.range_max or math.isnan(r) or math.isinf(r):
                continue
                
            if r < self.threshold:
                a = angle_min + angle_increment * i
                # 转换到机器人坐标系
                x = math.cos(a) * r
                y = math.sin(a) * r
                self.ob.append((x, y))
                
        self.laser_lock.release()

    def update_obstacle(self):
        """更新障碍物数据"""
        self.laser_lock.acquire()
        if len(self.ob) > 0:
            self.plan_ob = np.array(self.ob)
        else:
            self.plan_ob = np.array([])
        self.laser_lock.release()

    def publish_velocity(self, zero=False):
        cmd = Twist()
        if zero:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = float(self.vx)
            cmd.angular.z = float(self.vw)
        self.vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    real = True  # True: 真实机器人, False: 仿真
    local_planner = LocalPlanner(real)
    rclpy.spin(local_planner)
    local_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
