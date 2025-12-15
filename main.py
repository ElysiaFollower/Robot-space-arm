# Before running this example, make sure you have installed the following package.
# pip install coppeliasim-zmqremoteapi-client numpy
# You can find more information about ZeroMQ remote API 
# in the file path <Coppeliasim_install_path>\programming\zmqRemoteApi
# or on https://github.com/CoppeliaRobotics/zmqRemoteApi
#
# You can get more API about coppleliasim on https://coppeliarobotics.com/helpFiles/en/apiFunctions.htm

import time
import numpy as np
import json
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import utils

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
response = client.call('sim.setStepping', [True])
print('Response from sim.setStepping:', response)
client.setStepping(True)
sim.startSimulation()

# Get object handle
l_joint1 = sim.getObject('./L_Joint1')
l_joint2 = sim.getObject('./L_Joint2')
l_joint3 = sim.getObject('./L_Joint3')
joint4 = sim.getObject('./Joint4')
r_joint1 = sim.getObject('./R_Joint1')
r_joint2 = sim.getObject('./R_Joint2')
r_joint3 = sim.getObject('./R_Joint3')

l_base = sim.getObject('./L_Base')
l_link1 = sim.getObject('./L_Link1')
l_link2 = sim.getObject('./L_Link2')
l_link3 = sim.getObject('./L_Link3')
r_base = sim.getObject('./R_Base')
r_link1 = sim.getObject('./R_Link1')
r_link2 = sim.getObject('./R_Link2')
r_link3 = sim.getObject('./R_Link3')

A_base_joints = [l_joint1, l_joint2, l_joint3, joint4, r_joint3, r_joint2, r_joint1]
B_base_joints = A_base_joints[::-1]

A_base_chain = [l_base, l_joint1, l_link1, l_joint2, l_link2, l_joint3, l_link3, joint4, r_link3, r_joint3, r_link2, r_joint2, r_link1, r_joint1, r_base]
B_base_chain = A_base_chain[::-1]


# =================================
# main control logic begin
# =================================
# 读取轨迹数据
with open('trajectory_results.json', 'r', encoding='utf-8') as f:
    trajectory_data = json.load(f)

# 提取关键点关节角数据
step1_keypoints = trajectory_data['step1']['keypoints']
step2_keypoints = trajectory_data['step2']['keypoints']

# 将关键点关节角转换为numpy数组（JSON中已经是弧度制，单位：rad）
step1_angles = []
for kp in step1_keypoints:
    angles_rad = [float(angle) for angle in kp['joint_angles']]  # JSON中已经是弧度
    step1_angles.append(angles_rad)

step2_angles = []
for kp in step2_keypoints:
    angles_rad = [float(angle) for angle in kp['joint_angles']]  # JSON中已经是弧度
    step2_angles.append(angles_rad)

NORMAL_FRAME_TIME = 3.0
STEP1_FRAME_NUM = 5
FRAME_TIME = [ # 从前一个关键点到关键点i所需要的时间
    # step1: 
    1.0,
    1.0,
    NORMAL_FRAME_TIME,
    NORMAL_FRAME_TIME,
    1.0,

    # step2:
    1.0,
    1.0,
    NORMAL_FRAME_TIME,
    NORMAL_FRAME_TIME,
    1.0,
]
TOTAL_TIME = sum(FRAME_TIME)

# 计算累积时间点
cumulative_times = [0] # 到第i-1个关键点的累积时间
for i, frame_time in enumerate(FRAME_TIME):
    cumulative_times.append(cumulative_times[-1] + frame_time)

# step1的总时间
step1_total_time = sum(FRAME_TIME[:STEP1_FRAME_NUM])
step1_finished = False

# 在开始前先将所有关节角设置为0
for joint in A_base_joints:
    sim.setJointPosition(joint, 0.0)

# 在开始前切换基座到step1需要的基座（B-base/R基座）
utils.switch_base(sim, B_base_chain, A_base_joints, B_base_joints)

while (t := sim.getSimulationTime()) < TOTAL_TIME:
    # 确定当前处于哪个时间段
    segment_idx = 0
    for i in range(len(cumulative_times) - 1):
        if cumulative_times[i] <= t < cumulative_times[i + 1]:
            segment_idx = i
            break
    else:
        # 如果t超出了所有时间段，使用最后一个时间段
        segment_idx = len(cumulative_times) - 2
    
    # 判断是step1还是step2
    if segment_idx < STEP1_FRAME_NUM:  # step1: 0-STEP1_FRAME_NUM-1
        # step1的关键点索引
        # segment_idx=0: 到关键点0
        # segment_idx=1: 从关键点0到关键点1
        # segment_idx=2: 从关键点1到关键点2
        # segment_idx=3: 从关键点2到关键点3
        # segment_idx=4: 从关键点3到关键点4
        
        # 关键点 1-2-3 使用 interpolate_3 插值（从关键点1到关键点3，经过关键点2）
        if segment_idx >= 2 and segment_idx <= 3:
            # 使用关键点1、2、3进行插值
            start_angles = step1_angles[1]  # 关键点1
            mid_angles = step1_angles[2]    # 关键点2
            end_angles = step1_angles[3]     # 关键点3
            
            t_start = cumulative_times[2]   # 关键点1的时间
            t_mid = cumulative_times[3]     # 关键点2的时间
            t_end = cumulative_times[4]     # 关键点3的时间
            
            # 对每个关节进行三次插值
            for joint_idx in range(7):
                interpolated_angle = utils.angle_interpolate_3(
                    start_angles[joint_idx],
                    mid_angles[joint_idx],
                    end_angles[joint_idx],
                    t_start,
                    t_mid,
                    t_end,
                    t
                )
                sim.setJointPosition(B_base_joints[joint_idx], interpolated_angle)
        else:
            # 其他关键点使用原来的 interpolate_2 插值
            if segment_idx < len(step1_angles):
                start_angles = [0.0] * 7 if segment_idx == 0 else step1_angles[segment_idx - 1]
                end_angles = step1_angles[segment_idx]
                t_start = cumulative_times[segment_idx]
                t_end = cumulative_times[segment_idx+1]
                
                #对每个关节进行五次多项式插值
                for joint_idx in range(7):
                    interpolated_angle = utils.angle_interpolate_2(
                        start_angles[joint_idx],
                        end_angles[joint_idx],
                        t_start,
                        t_end,
                        t
                    )
                    sim.setJointPosition(B_base_joints[joint_idx], interpolated_angle)
    else:  # step2: 5-9
        # 检查step1是否完成，如果完成则切换基座到step2需要的基座（A-base/L基座）
        if not step1_finished:
            # 切换基座（内置了角度读取、清零、映射和设置的逻辑）
            utils.switch_base(sim, A_base_chain, B_base_joints, A_base_joints)
            step1_finished = True

        # step2的关键点索引（相对于step2）
        step2_segment_idx = segment_idx - STEP1_FRAME_NUM
        # step2_segment_idx=0: 到关键点0
        # step2_segment_idx=1: 从关键点0到关键点1
        # step2_segment_idx=2: 从关键点1到关键点2
        # step2_segment_idx=3: 从关键点2到关键点3
        # step2_segment_idx=4: 从关键点3到关键点4
        
        # 关键点 6-7-8 使用 interpolate_3 插值（从关键点6到关键点8，经过关键点7）
        if segment_idx >= 7 and segment_idx <= 8:
            # 使用关键点6、7、8进行插值
            start_angles = step2_angles[1]  # 关键点6 (step2的第2个点)
            mid_angles = step2_angles[2]    # 关键点7 (step2的第3个点)
            end_angles = step2_angles[3]     # 关键点8 (step2的第4个点)
            
            t_start = cumulative_times[7]   # 关键点6的时间
            t_mid = cumulative_times[8]     # 关键点7的时间
            t_end = cumulative_times[9]     # 关键点8的时间
            
            # 对每个关节进行三次插值
            for joint_idx in range(7):
                interpolated_angle = utils.angle_interpolate_3(
                    start_angles[joint_idx],
                    mid_angles[joint_idx],
                    end_angles[joint_idx],
                    t_start,
                    t_mid,
                    t_end,
                    t
                )
                sim.setJointPosition(A_base_joints[joint_idx], interpolated_angle)
        else:
            # 其他关键点使用原来的 interpolate_2 插值
            if step2_segment_idx < len(step2_angles):
                # 当从Step1切换到Step2的第一个关键点时，直接使用step2的第一个关键点角度作为起始角度
                # 注意：在CoppeliaSim中，关节角度是相对于关节本身的局部坐标系定义的
                # 切换基座后，虽然关节顺序变了，但每个关节的角度值应该保持不变
                # 所以这里直接使用step2的角度，确保start_angles == end_angles，插值不会产生不必要的运动
                start_angles = step2_angles[0] if step2_segment_idx == 0 else step2_angles[step2_segment_idx - 1]
                end_angles = step2_angles[step2_segment_idx]
                t_start = cumulative_times[segment_idx]
                t_end = cumulative_times[segment_idx + 1]

                # 对每个关节进行五次多项式插值
                for joint_idx in range(7):
                    interpolated_angle = utils.angle_interpolate_2(
                        start_angles[joint_idx],
                        end_angles[joint_idx],
                        t_start,
                        t_end,
                        t
                    )
                    sim.setJointPosition(A_base_joints[joint_idx], interpolated_angle)



    client.step()  # triggers next simulation step
    time.sleep(0.01)

END_TIME = 1.0 # 结束之后保持的时间
time.sleep(END_TIME)
# =================================
# main control logic end
# =================================

# Stop simulation
sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

print('Program ended')