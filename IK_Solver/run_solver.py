import json
import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- 动态添加路径，确保能导入子模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model.joint import JointNode, FixedJoint, RevoluteJoint, PrismaticJoint, SphericalJoint
    from solver.solve_ik import solve_ik
    from solver.ik_core import build_ik_chain
    from data_io import (
        load_skeleton,
        load_targets,
        interpolate_targets,
        initialize_tpose,
        interpolate_joint_states
    )
except ImportError as e:
    print("❌ 导入模块失败。请检查目录结构是否完整（特别是 utils 文件夹）。")
    print(f"错误详情: {e}")
    sys.exit(1)


def find_effector(node: JointNode) -> JointNode:
    """
    自动查找末端执行器：寻找没有子节点的 FixedJoint
    """
    if isinstance(node, FixedJoint) and len(node.children) == 0:
        return node
    for child in node.children:
        result = find_effector(child)
        if result is not None:
            return result
    return None


def run_solver(config_path="config.json"):
    # 1. 加载配置
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("----------- IK Solver Headless -----------")
    print(f"配置加载: {config_path}")

    skeleton_path = config.get('skeleton_path')
    targets_path = config.get('targets_path')
    output_path = config.get('output_path', 'animation.json')
    solve_mode = config.get('solve_mode', 1)  # 默认模式2 (逐帧)

    # 求解参数
    params = {
        'max_iterations': config.get('max_iterations', 100),
        'position_tolerance': config.get('position_tolerance', 1e-3),
        'orientation_tolerance': config.get('orientation_tolerance', 1e-2),
        'damping': config.get('damping', 0.0001),
        'enable_line_search': config.get('enable_line_search', True),
        'line_search_alpha': 1.0,
        'line_search_alpha_min': 1e-2
    }

    # 2. 加载骨骼
    print(f"正在加载骨骼: {skeleton_path} ...")
    try:
        root, joint_map = load_skeleton(skeleton_path)
    except Exception as e:
        print(f"❌ 骨骼加载失败: {e}")
        return

    # 3. 查找末端执行器并构建 IK 链
    effector = find_effector(root)
    if effector is None:
        print("❌ 无法自动找到末端执行器 (无子节点的 FixedJoint)")
        return
    print(f"末端执行器: {effector.name}")

    try:
        ik_chain = build_ik_chain(root, effector)
        print(f"IK 链构建成功，包含 {len(ik_chain)} 个关节")
    except Exception as e:
        print(f"❌ IK 链构建失败: {e}")
        return

    # 4. 加载目标轨迹
    print(f"正在加载目标轨迹: {targets_path} ...")
    try:
        keyframes = load_targets(targets_path)
        total_frames = keyframes[-1]['frame']
        print(f"轨迹加载成功，共 {len(keyframes)} 个关键帧，总长 {total_frames} 帧")
    except Exception as e:
        print(f"❌ 目标轨迹加载失败: {e}")
        return

    # 5. 开始求解
    solved_frames = []
    start_time = time.time()
    
    # 初始化 T-Pose
    initialize_tpose(root)
    root.update_global_transform()

    if solve_mode == 0:
        print(">>> 模式 1: 关键帧求解 + 关节插值")
        solved_frames = solve_mode_1(root, effector, ik_chain, joint_map, keyframes, total_frames, params)
    else:
        print(">>> 模式 2: 目标插值 + 逐帧求解")
        solved_frames = solve_mode_2(root, effector, ik_chain, joint_map, keyframes, total_frames, params)

    duration = time.time() - start_time
    print(f"求解完成，耗时: {duration:.2f} 秒")

    # 6. 导出结果
    print(f"正在导出到: {output_path} ...")
    export_result(solved_frames, joint_map, output_path)
    print("✅ 任务完成！")


def extract_current_joint_states(frame_idx, joint_map):
    """提取当前所有关节的状态，存为字典"""
    frame_data = {
        'frame': frame_idx,
        'joint_states': {}
    }
    for name, joint in joint_map.items():
        if isinstance(joint, FixedJoint):
            continue
        
        state = {}
        if hasattr(joint, 'q'):
            state['q'] = float(joint.q)
        elif hasattr(joint, 'quaternion'):
            state['quaternion'] = joint.quaternion.tolist()
        
        frame_data['joint_states'][name] = state
    return frame_data


def solve_mode_2(root, effector, ik_chain, joint_map, keyframes, total_frames, params):
    """模式2：逐帧求解"""
    solved_frames = []
    
    for frame in range(total_frames + 1):
        # 打印进度
        if frame % 10 == 0:
            sys.stdout.write(f"\r进度: {frame}/{total_frames}")
            sys.stdout.flush()

        # 插值目标
        target_transform = interpolate_targets(keyframes, frame)
        
        # 求解
        solve_ik(
            root=root,
            effector=effector,
            target_transform=target_transform,
            ik_chain=ik_chain,
            **params
        )
        
        # 记录状态
        solved_frames.append(extract_current_joint_states(frame, joint_map))
    
    print() # 换行
    return solved_frames


def solve_mode_1(root, effector, ik_chain, joint_map, keyframes, total_frames, params):
    """模式1：关键帧求解 + 插值"""
    keyframe_results = {}
    prev_frame = None
    
    keyframe_indices = [kf['frame'] for kf in keyframes]

    # 1. 求解关键帧
    for kf in keyframes:
        frame = kf['frame']
        sys.stdout.write(f"\r正在求解关键帧: {frame}")
        sys.stdout.flush()

        target_transform = interpolate_targets(keyframes, frame)

        # Warm start: 如果不是第一帧，使用上一帧的结果作为初值
        if prev_frame is not None:
            prev_data = keyframe_results[prev_frame]
            for name, state in prev_data['joint_states'].items():
                joint = joint_map[name]
                if 'q' in state: joint.q = state['q']
                elif 'quaternion' in state: joint.quaternion = np.array(state['quaternion'])
            root.update_global_transform()
        else:
            initialize_tpose(root)
            root.update_global_transform()

        solve_ik(
            root=root,
            effector=effector,
            target_transform=target_transform,
            ik_chain=ik_chain,
            **params
        )

        keyframe_results[frame] = extract_current_joint_states(frame, joint_map)
        prev_frame = frame

    print("\n正在进行插值...")

    # 2. 插值中间帧
    solved_frames = []
    for frame in range(total_frames + 1):
        # 找到区间
        start_kf_frame = keyframe_indices[0]
        end_kf_frame = keyframe_indices[-1]
        
        if frame <= start_kf_frame:
            frame_data = keyframe_results[start_kf_frame].copy()
            frame_data['frame'] = frame
        elif frame >= end_kf_frame:
            frame_data = keyframe_results[end_kf_frame].copy()
            frame_data['frame'] = frame
        else:
            for i in range(len(keyframe_indices) - 1):
                if keyframe_indices[i] <= frame < keyframe_indices[i+1]:
                    start_kf_frame = keyframe_indices[i]
                    end_kf_frame = keyframe_indices[i+1]
                    break
            
            frame_data = interpolate_joint_states(
                joint_map,
                keyframe_results[start_kf_frame],
                keyframe_results[end_kf_frame],
                frame
            )
        
        solved_frames.append(frame_data)
    
    return solved_frames


def export_result(solved_frames, joint_map, output_path):
    """
    导出动画 JSON
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    frames_output = []
    for frame_data in solved_frames:
        frame_num = frame_data['frame']
        joint_states = frame_data['joint_states']
        
        frame_out = {
            'frame': frame_num,
            'joints': {}
        }
        
        for name, joint in joint_map.items():
            if isinstance(joint, FixedJoint):
                continue
            
            if name not in joint_states:
                continue
            
            state = joint_states[name]
            joint_data = {'type': None}
            
            if isinstance(joint, RevoluteJoint):
                joint_data['type'] = 'revolute'
                joint_data['angle'] = float(state['q'])
            
            elif isinstance(joint, PrismaticJoint):
                joint_data['type'] = 'prismatic'
                joint_data['displacement'] = float(state['q'])
            
            elif isinstance(joint, SphericalJoint):
                joint_data['type'] = 'spherical'
                quat = np.array(state['quaternion'])
                
                # 转换为 XYZ 欧拉角
                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                # 构造旋转矩阵
                R_mat = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ])
                rot = R.from_matrix(R_mat)
                euler_rad = rot.as_euler('XYZ', degrees=False)
                
                joint_data['euler'] = [float(e) for e in euler_rad]
                joint_data['quaternion'] = [float(w), float(x), float(y), float(z)]
            
            frame_out['joints'][name] = joint_data
        
        frames_output.append(frame_out)
    
    output = {'frames': frames_output}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_solver(sys.argv[1])
    else:
        run_solver()