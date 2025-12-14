import numpy as np
from vision import Vision
from action import Action
from debug import Debugger
from zss_debug_pb2 import Debug_Msgs
import time
from time import sleep
from rrt_star import RRTStar
from rrt_with_pathsmoothing import path_smoothing
from bspline_smoother import BSplineSmoother, VelocityProfiler
from trajectory_controller import TrajectoryController
from mpc_controller import MPCController, DynamicObstaclePredictor
from zss_debug_pb2 import Debug_Msg

goals=[[-4500,-3000],[4500,3000]]
curgoal=0
rndcnt=0

# 机器人参数
MAX_LINEAR_VEL = 1000   #最大线速度
MAX_ANGULAR_VEL = 3.0   #最大角速度
ROBOT_RADIUS = 100      #机器人半径
POSITION_TOLERANCE = 150#到达目标点的容差
ANGLE_TOLERANCE = 0.1   #角度容差

if __name__ == '__main__':
	vision = Vision()
	action = Action()
	debugger = Debugger()
	
	bspline_smoother = BSplineSmoother(robot_radius=ROBOT_RADIUS)
    controller = MPCController(
        prediction_horizon=10,     # 预测10步
        control_horizon=5,         # 优化5步控制量
        dt=0.1,                    # 100ms步长
        max_linear_vel=MAX_LINEAR_VEL,
        max_angular_vel=MAX_ANGULAR_VEL,
        robot_radius=ROBOT_RADIUS + 50,  # 额外安全裕度
        safety_distance=100
    )
    obstacle_predictor = DynamicObstaclePredictor(history_length=5)
    print("Using MPC Controller with dynamic obstacle avoidance")
	
	sleep(1)  # wait for vision start
	
	while rndcnt<=1:
		# 1. path planning & velocity planning
		# Do something
		# 枚举除蓝色robot0外的所有机器人位置
		print(vision.my_robot.orientation)
		while curgoal<=1:
			robot_obstacles = []
			for robot in vision.blue_robot[1:]:
				robot_obstacles.append((robot.x, robot.y, 200))
			for robot in vision.yellow_robot:
				robot_obstacles.append((robot.x, robot.y, 200))
			planning = RRTStar(
				start=goals[curgoal],
				goal=[vision.my_robot.x, vision.my_robot.y],
				rand_area=[[-5000, +5000],[-3500, 3500]],
				obstacle_list=robot_obstacles,
				expand_dis=800,
				robot_radius=100,
				max_iter=2000
			)
			original_path = planning.planning(animation=False)
			
			# 可视化RRT*树和最终路径
			vis_package = Debug_Msgs()
			
			if original_path is None:
				print("No Path Found")
				break
			else:
				print("found path!!")
				print(f"Original path length: {len(original_path)} points")
				
				debugger.draw_rrt_tree(vis_package, planning)
				print(f"Visualized RRT* tree with {len(planning.node_list)} nodes")
				
				debugger.draw_path(vis_package, original_path, color=Debug_Msg.RED)
				print(f"Visualized RRT* final path (RED)")
				
				# B样条平滑路径
				smoothed_path = path_smoothing(original_path, 100, robot_obstacles, robot_radius=ROBOT_RADIUS)
				print(f"After simple smoothing: {len(smoothed_path)} points")
				
				bspline_path = bspline_smoother.smooth_path(
					smoothed_path, 
					robot_obstacles, 
					s=0,
					k=3,  # 三次样条
					num_points=max(50, len(smoothed_path) * 2)  # 适度增加点数
				)
				print(f"After B-spline smoothing: {len(bspline_path)} points")
				
				path = bspline_path
				
				# 设置控制器轨迹
				tck = controller.set_trajectory(path, k=3, s=0)
				
				# 3. 绘制B样条参考轨迹(蓝色)
				debugger.draw_trajectory(vis_package, tck, num_points=200, color=Debug_Msg.BLUE)
				print(f"Visualized B-spline reference trajectory (BLUE)")
				
				# 发送初始可视化
				debugger.send(vis_package)
				print("Initial visualization sent (RRT tree + final path + B-spline trajectory)")
				

			
			current_u = 0.0
			
			while True:
				# 获取当前机器人状态
				current_pos = np.array([vision.my_robot.x, vision.my_robot.y])
				current_orientation = vision.my_robot.orientation
				
				obstacles_dict = {}
				
				for i, robot in enumerate(vision.blue_robot[1:], start=1):
					obs_id = f'blue_{i}'
					obstacles_dict[obs_id] = (robot.x, robot.y, 150)
				
				for i, robot in enumerate(vision.yellow_robot):
					obs_id = f'yellow_{i}'
					obstacles_dict[obs_id] = (robot.x, robot.y, 150)
				

                obstacle_predictor.update(obstacles_dict)
                
                # 获取带速度预测的障碍物列表
                dynamic_obstacles = obstacle_predictor.get_predicted_obstacles(obstacles_dict)
                
                # 更新MPC控制器
                controller.update_dynamic_obstacles(dynamic_obstacles)
                
                # 计算MPC控制指令
                vx, vw, finished, debug_info = controller.compute_control(
                    current_pos,
                    current_orientation,
                    current_u
                )
                
                # 更新轨迹参数
                if 'closest_u' in debug_info:
                    current_u = debug_info['closest_u']
                
				
				if finished:
					print(f"Reached goal {curgoal+1}")
					break
				
				# 发送控制指令
				action.sendCommand(vx=vx, vy=0, vw=vw)
				
				# 调试可视化
				package = Debug_Msgs()
				
				# 重绘静态背景(RRT树、最终路径、B样条轨迹)
				debugger.draw_rrt_tree(package, planning)
				debugger.draw_path(package, original_path, color=Debug_Msg.RED)
				debugger.draw_trajectory(package, tck, num_points=200, color=Debug_Msg.BLUE)
				
				# 绘制机器人位置(绿色)
				debugger.draw_circle(package, vision.my_robot.x, vision.my_robot.y, 100, color=Debug_Msg.GREEN)
				
				# 绘制MPC预测路径(黄色)
				if USE_MPC and 'predicted_states' in debug_info:
					predicted = debug_info['predicted_states']
					if len(predicted) > 1:
						debugger.draw_prediction(package, predicted, color=Debug_Msg.YELLOW)
				
				# 绘制跟踪点(Pure Pursuit)
				if not USE_MPC and 'current_point' in debug_info:
					cp = debug_info['current_point']
					tp = debug_info['target_point']
					debugger.draw_point(package, int(cp[0]), int(cp[1]), color=Debug_Msg.CYAN)
					debugger.draw_point(package, int(tp[0]), int(tp[1]), color=Debug_Msg.ORANGE)
				
				# 绘制障碍物(紫色圆圈)
				for obs_id, (ox, oy, radius) in obstacles_dict.items():
					debugger.draw_circle(package, ox, oy, radius, color=Debug_Msg.PURPLE)
				
				debugger.send(package)
				
				# 打印调试信息
				if USE_MPC:
					if 'progress' in debug_info and int(debug_info['progress']) % 10 == 0:
						print(f"MPC - Progress: {debug_info['progress']:.1f}%, "
						      f"pos=({current_pos[0]:.0f},{current_pos[1]:.0f}), "
						      f"vx={vx:.0f}, vw={vw:.2f}, "
						      f"cost={debug_info.get('cost', 0):.1f}, "
						      f"obs={debug_info.get('n_obstacles', 0)}, "
						      f"opt_time={debug_info.get('optimization_time', 0):.1f}ms")
				else:
					progress = controller.get_progress()
					if int(progress * 100) % 10 == 0:
						print(f"PurePursuit - Progress: {progress*100:.1f}%, "
						      f"pos=({current_pos[0]:.0f},{current_pos[1]:.0f}), "
						      f"vx={vx:.0f}, vw={vw:.2f}")
				
				time.sleep(0.01)
			
			# 到达目标点,停止
			action.sendCommand(vx=0, vy=0, vw=0)
			curgoal += 1
			# 2. send command
			# action.sendCommand(vx=100, vy=0, vw=0)

			# 3. draw debug msg
			
		# time.sleep(0.01)
		rndcnt+=1
		curgoal=0
