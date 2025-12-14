# 该文件通过定义关键末端位置和欧拉角，通过IK计算出每个对应关键点的关节角度

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# 添加IK_Solver路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ik_solver_dir = os.path.join(current_dir, 'IK_Solver')
sys.path.append(ik_solver_dir)

from data_io import load_skeleton, initialize_tpose, euler_to_transform
from solver.solve_ik import solve_ik
from solver.ik_core import build_ik_chain
from model.joint import FixedJoint


# step 1: based on B [0.35, 0, 0.2], [0, 0, 0]； J7-B为基座, J1-A为末端
POS_A_ABS = [
    [0.65, 0, 0.2],
    [0.65, 0, 0.3],
    [0.2, 0.3, 0.3],  # 圆弧顶点，坐标可以修改
    [-0.25, 0, 0.3],
    [-0.25, 0, 0.2],
]

POS_A_REL = [
    [0.3, 0, 0],
    [0.3, 0, 0.1],
    [-0.15, 0, 0.1],
    [-0.6, 0, 0.1],
    [-0.6, 0, 0],
]

EULER_A = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

JOINT_ANGLE_A = [
    [0, 0, 0, 0, 0, 0, 0],
    None,
    None,
    None,
    None,
]

JOINT_ANGLE_A_TOLERANCE=[
    1e-2,
    1,
    1,
    1,
    1e-2,
]

# ------------------------------------------------------------


# step 2: A [-0.25, 0, 0.2], [0, 0, 0]； J1-A为基座, J7-B为末端
POS_B_ABS = [
    [0.35, 0, 0.2],
    [0.35, 0, 0.3],
    [-0.15, 0.4, 0],  # 圆弧顶点，坐标可以修改
    [-0.65, 0.2, 0.1],
    [-0.65, 0.1, 0.1],
]

POS_B_REL = [
    [0.6, 0, 0],
    [0.6, 0, 0.1],
    [0.1, 0.4, -0.2],
    [-0.4, 0.2, -0.1],
    [-0.4, 0.1, -0.1],
]

EULER_B = [
    [0, 0, 0],
    [0, 0, 0],
    [0, -45, 0],
    [0, -90, 0],
    [0, -90, 0],
]

JOINT_ANGLE_B = [None, None, None, None, None]

JOINT_ANGLE_B_TOLERANCE=[
    1e-2,
    1,
    1,
    1,
    1e-2,
]

def find_effector(node):
    """查找末端执行器：寻找没有子节点的 FixedJoint"""
    if isinstance(node, FixedJoint) and len(node.children) == 0:
        return node
    for child in node.children:
        result = find_effector(child)
        if result is not None:
            return result
    return None

def solve_trajectory(type, pos_rel, euler, joint_angles, joint_tolerance, step_name):
    """
    通用轨迹求解函数
    
    :param type: "A-base" 或 "B-base"，决定使用哪个骨骼文件
    :param pos_rel: 相对位置列表（米，相对于基座）
    :param euler: 欧拉角列表（度）
    :param joint_angles: 关节角度列表（会被修改）
    :param joint_tolerance: 姿态容差列表（弧度）
    :param step_name: 步骤名称（用于打印）
    """
    print("=" * 50)
    print(f"开始求解 {step_name}")
    print("=" * 50)
    
    # 加载骨骼
    skeleton_path = os.path.join(current_dir, 'data', f'{type}-arm.json')
    print(f"加载骨骼: {skeleton_path}")
    root, joint_map = load_skeleton(skeleton_path)
    
    # 查找末端执行器
    effector = find_effector(root)
    if effector is None:
        print("❌ 无法找到末端执行器")
        return
    print(f"末端执行器: {effector.name}")
    
    # 构建IK链
    ik_chain = build_ik_chain(root, effector)
    print(f"IK链构建成功，包含 {len(ik_chain)} 个关节")
    joint_names = [joint.name for joint in ik_chain]
    print(f"关节顺序: {joint_names}")
    
    # 初始化T-Pose
    initialize_tpose(root)
    root.update_global_transform()
    
    # 设置第一个关键点的初始关节角度（如果已给出）
    if joint_angles[0] is not None:
        print(f"\n设置第一个关键点的初始关节角度: {joint_angles[0]}")
        for i, joint in enumerate(ik_chain):
            if i < len(joint_angles[0]):
                joint.q = np.deg2rad(joint_angles[0][i])
        root.update_global_transform()
    
    # 求解每个关键点
    num_keypoints = len(pos_rel)
    for i in range(num_keypoints):
        if joint_angles[i] is not None:
            print(f"\n--- 关键点 {i+1}/{num_keypoints} 已有值，跳过求解 ---")
            for j, joint in enumerate(ik_chain):
                joint.q = np.deg2rad(joint_angles[i][j])
            root.update_global_transform()
            continue
        
        print(f"\n--- 求解关键点 {i+1}/{num_keypoints} ---")
        pos_mm = np.array(pos_rel[i]) * 1000.0
        euler_deg = np.array(euler[i])
        target_transform = euler_to_transform(pos_mm, euler_deg)
        
        if i > 0:
            for j, joint in enumerate(ik_chain):
                joint.q = np.deg2rad(joint_angles[i-1][j])
            root.update_global_transform()
        
        ik_params = {
            'max_iterations': 100,
            'position_tolerance': 1e-3,
            'orientation_tolerance': joint_tolerance[i],
            'damping': 0.001,
            'enable_line_search': True
        }
        
        # 求解IK
        success = solve_ik(
            root=root,
            effector=effector,
            target_transform=target_transform,
            ik_chain=ik_chain,
            **ik_params
        )
        
        joint_angles[i] = [np.rad2deg(joint.q) for joint in ik_chain]
        print(f"✅ 求解成功！" if success else "❌ 求解失败！")
    
    print("\n" + "=" * 50)
    print(f"{step_name} 求解完成！")
    print("=" * 50)
    print("\n关节角度结果:")
    for i, angles in enumerate(joint_angles):
        if angles is not None:
            print(f"关键点 {i+1}: {[f'{a:.4f}' for a in angles]}")
        else:
            print(f"关键点 {i+1}: None")


# 执行求解
if __name__ == "__main__":
    solve_trajectory(
        type="B-base",
        pos_rel=POS_A_REL,
        euler=EULER_A,
        joint_angles=JOINT_ANGLE_A,
        joint_tolerance=JOINT_ANGLE_A_TOLERANCE,
        step_name="Step 1: B为基座，A为末端执行器"
    )
    print("\n")
    solve_trajectory(
        type="A-base",
        pos_rel=POS_B_REL,
        euler=EULER_B,
        joint_angles=JOINT_ANGLE_B,
        joint_tolerance=JOINT_ANGLE_B_TOLERANCE,
        step_name="Step 2: A为基座，B为末端执行器"
    )