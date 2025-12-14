"""
数据交换功能实现
"""
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R, Slerp
from model.joint import JointNode, RevoluteJoint, PrismaticJoint, FixedJoint, SphericalJoint


def load_skeleton(json_path: str) -> Tuple[JointNode, Dict[str, JointNode]]:
    """
    从skeleton.json加载骨骼定义，构建场景图
    
    :param json_path: skeleton.json文件路径
    :return: (root节点, 关节名称到节点的映射字典)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    root_name = data['root_name']
    joints_data = data['joints']
    
    # 创建所有关节对象
    joint_map: Dict[str, JointNode] = {}
    
    for joint_data in joints_data:
        name = joint_data['name']
        joint_type = joint_data['type']
        offset = np.array(joint_data['offset'], dtype=np.float64)
        parent_name = joint_data.get('parent')
        
        # 根据类型创建关节
        if joint_type == 'fixed':
            quat = None
            if 'quaternion' in joint_data and joint_data['quaternion'] is not None:
                quat = np.array(joint_data['quaternion'], dtype=np.float64)
            joint = FixedJoint(name, offset, quat)
        elif joint_type == 'revolute':
            axis = np.array(joint_data['axis'], dtype=np.float64)
            limits = None
            if 'limits' in joint_data and joint_data['limits'] is not None:
                limits = tuple(joint_data['limits'])
            joint = RevoluteJoint(name, offset, axis, limits)
        elif joint_type == 'prismatic':
            axis = np.array(joint_data['axis'], dtype=np.float64)
            limits = None
            if 'limits' in joint_data and joint_data['limits'] is not None:
                limits = tuple(joint_data['limits'])
            joint = PrismaticJoint(name, offset, axis, limits)
        elif joint_type == 'spherical':
            limits = None
            if 'limits' in joint_data and joint_data['limits'] is not None:
                # 处理SphericalJoint的limits格式（针对XYZ欧拉角）
                limits_list = joint_data['limits']
                limits = []
                for limit_item in limits_list:
                    if limit_item is None:
                        limits.append(None)
                    else:
                        limits.append(tuple(limit_item))
            joint = SphericalJoint(name, offset, limits)
        else:
            raise ValueError(f"Unknown joint type: {joint_type}")
        
        joint_map[name] = joint
    
    # 建立父子关系
    for joint_data in joints_data:
        name = joint_data['name']
        parent_name = joint_data.get('parent')
        
        if parent_name is not None:
            if parent_name not in joint_map:
                raise ValueError(f"Parent '{parent_name}' not found for joint '{name}'")
            parent = joint_map[parent_name]
            parent.add_child(joint_map[name])
    
    # 找到根节点
    if root_name not in joint_map:
        raise ValueError(f"Root node '{root_name}' not found")
    root = joint_map[root_name]
    
    # 初始化关节状态（T-Pose）
    initialize_tpose(root)
    
    return root, joint_map


def initialize_tpose(root: JointNode):
    """
    初始化所有关节到T-Pose状态
    """
    def traverse(node: JointNode):
        if isinstance(node, RevoluteJoint):
            node.q = 0.0
        elif isinstance(node, PrismaticJoint):
            node.q = 0.0
        elif isinstance(node, SphericalJoint):
            node.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
        
        for child in node.children:
            traverse(child)
    
    traverse(root)
    root.update_global_transform()


def load_targets(json_path: str) -> List[Dict]:
    """
    从targets.json加载目标轨迹
    
    :param json_path: targets.json文件路径
    :return: 关键帧列表，每个元素为 {"frame": int, "pos": [x,y,z], "euler": [x,y,z]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    keyframes = []
    for item in data:
        keyframe = {
            'frame': int(item['frame']),
            'pos': np.array(item['pos'], dtype=np.float64),
            'euler': np.array(item['euler'], dtype=np.float64)  # 度
        }
        keyframes.append(keyframe)
    
    # 按帧号排序
    keyframes.sort(key=lambda kf: kf['frame'])
    
    return keyframes


def euler_to_transform(pos: np.ndarray, euler_deg: np.ndarray) -> np.ndarray:
    """
    将位置和欧拉角（度，XYZ顺序）转换为4x4变换矩阵
    
    :param pos: 位置 [x, y, z]（米）
    :param euler_deg: 欧拉角 [x, y, z]（度，XYZ顺序）
    :return: 4x4变换矩阵
    """
    transform = np.identity(4, dtype=np.float64)
    
    # 转换欧拉角：度 -> 弧度
    euler_rad = np.deg2rad(euler_deg)
    
    # 使用内旋XYZ顺序构建旋转矩阵
    # 内旋XYZ等价于外旋ZYX，矩阵乘法顺序：R = R_x @ R_y @ R_z
    rot = R.from_euler('XYZ', euler_rad, degrees=False)
    transform[:3, :3] = rot.as_matrix()
    
    # 设置位置
    transform[:3, 3] = pos
    
    return transform


def interpolate_targets(keyframes: List[Dict], frame: int) -> np.ndarray:
    """
    在关键帧之间进行线性插值，生成目标变换矩阵
    
    :param keyframes: 关键帧列表
    :param frame: 当前帧号
    :return: 4x4插值变换矩阵
    """
    # 找到当前帧所在的区间
    if frame <= keyframes[0]['frame']:
        kf = keyframes[0]
        return euler_to_transform(kf['pos'], kf['euler'])
    
    if frame >= keyframes[-1]['frame']:
        kf = keyframes[-1]
        return euler_to_transform(kf['pos'], kf['euler'])
    
    # 找到区间
    start_kf = keyframes[0]
    end_kf = keyframes[-1]
    
    for i in range(len(keyframes) - 1):
        if keyframes[i]['frame'] <= frame < keyframes[i+1]['frame']:
            start_kf = keyframes[i]
            end_kf = keyframes[i+1]
            break
    
    # 计算插值系数
    start_frame = start_kf['frame']
    end_frame = end_kf['frame']
    
    if end_frame == start_frame:
        alpha = 0.0
    else:
        alpha = (frame - start_frame) / (end_frame - start_frame)
    
    # 位置线性插值
    start_pos = start_kf['pos']
    end_pos = end_kf['pos']
    interp_pos = (1.0 - alpha) * start_pos + alpha * end_pos
    
    # 姿态球面线性插值（SLERP）
    start_euler_rad = np.deg2rad(start_kf['euler'])
    end_euler_rad = np.deg2rad(end_kf['euler'])
    
    start_rot = R.from_euler('XYZ', start_euler_rad, degrees=False)
    end_rot = R.from_euler('XYZ', end_euler_rad, degrees=False)
    
    # 使用SLERP插值
    from scipy.spatial.transform import Slerp
    slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))
    interp_rot = slerp(alpha)
    
    # 组装变换矩阵
    transform = np.identity(4, dtype=np.float64)
    transform[:3, :3] = interp_rot.as_matrix()
    transform[:3, 3] = interp_pos
    
    return transform


def interpolate_joint_states(joint_map: Dict[str, JointNode],
                             start_frame_data: Dict,
                             end_frame_data: Dict,
                             frame: int) -> Dict:
    """
    在关键帧之间对关节状态进行插值
    
    :param joint_map: 关节名称到节点的映射字典
    :param start_frame_data: 起始关键帧数据 {'frame': int, 'joint_states': dict}
    :param end_frame_data: 结束关键帧数据 {'frame': int, 'joint_states': dict}
    :param frame: 当前帧号
    :return: 插值后的帧数据 {'frame': int, 'joint_states': dict}
    """
    start_frame = start_frame_data['frame']
    end_frame = end_frame_data['frame']
    start_states = start_frame_data['joint_states']
    end_states = end_frame_data['joint_states']
    
    # 计算插值系数
    if end_frame == start_frame:
        alpha = 0.0
    else:
        alpha = (frame - start_frame) / (end_frame - start_frame)
    alpha = max(0.0, min(1.0, alpha))  # 限制在[0, 1]范围内
    
    # 创建插值后的帧数据
    interpolated_frame = {
        'frame': frame,
        'joint_states': {}
    }
    
    # 对每个关节进行插值
    for joint_name, joint in joint_map.items():
        if isinstance(joint, FixedJoint):
            continue
        
        if joint_name not in start_states or joint_name not in end_states:
            continue
        
        start_state = start_states[joint_name]
        end_state = end_states[joint_name]
        interpolated_state = {}
        
        if isinstance(joint, RevoluteJoint) or isinstance(joint, PrismaticJoint):
            # 旋转关节和移动关节：线性插值
            start_q = start_state.get('q', 0.0)
            end_q = end_state.get('q', 0.0)
            interpolated_state['q'] = (1.0 - alpha) * start_q + alpha * end_q
        
        elif isinstance(joint, SphericalJoint):
            # 球形关节：四元数SLERP插值
            start_quat = np.array(start_state.get('quaternion', [1.0, 0.0, 0.0, 0.0]))
            end_quat = np.array(end_state.get('quaternion', [1.0, 0.0, 0.0, 0.0]))
            
            # 归一化四元数
            start_quat = start_quat / np.linalg.norm(start_quat)
            end_quat = end_quat / np.linalg.norm(end_quat)
            
            # 使用scipy的Slerp进行球面线性插值
            # scipy使用[x, y, z, w]格式，我们需要转换为[w, x, y, z]格式
            start_quat_scipy = np.array([start_quat[1], start_quat[2], start_quat[3], start_quat[0]])
            end_quat_scipy = np.array([end_quat[1], end_quat[2], end_quat[3], end_quat[0]])
            
            start_rot = R.from_quat(start_quat_scipy)
            end_rot = R.from_quat(end_quat_scipy)
            
            slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))
            interp_rot = slerp(alpha)
            
            # 转换回[w, x, y, z]格式
            interp_quat_scipy = interp_rot.as_quat()  # [x, y, z, w]
            interp_quat = np.array([interp_quat_scipy[3], interp_quat_scipy[0], 
                                   interp_quat_scipy[1], interp_quat_scipy[2]])  # [w, x, y, z]
            
            interpolated_state['quaternion'] = interp_quat.tolist()
        
        interpolated_frame['joint_states'][joint_name] = interpolated_state
    
    return interpolated_frame


def export_animation(joint_map: Dict[str, JointNode], 
                    total_frames: int,
                    output_path: str):
    """
    导出动画数据到animation.json
    
    :param joint_map: 关节名称到节点的映射字典
    :param total_frames: 总帧数
    :param output_path: 输出文件路径
    """
    frames_data = []
    
    for frame in range(total_frames + 1):
        frame_data = {
            'frame': frame,
            'joints': {}
        }
        
        # 遍历所有关节，提取局部变量
        for name, joint in joint_map.items():
            if isinstance(joint, FixedJoint):
                continue  # FixedJoint不参与导出
            
            joint_data = {'type': None}
            
            if isinstance(joint, RevoluteJoint):
                joint_data['type'] = 'revolute'
                joint_data['angle'] = float(joint.q)
            
            elif isinstance(joint, PrismaticJoint):
                joint_data['type'] = 'prismatic'
                joint_data['displacement'] = float(joint.q)
            
            elif isinstance(joint, SphericalJoint):
                joint_data['type'] = 'spherical'
                
                # 将四元数转换为XYZ欧拉角
                q = joint.quaternion
                w, x, y, z = q[0], q[1], q[2], q[3]
                R_mat = np.array([
                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                ])
                rot = R.from_matrix(R_mat)
                euler_rad = rot.as_euler('XYZ', degrees=False)  # 固定使用XYZ顺序
                joint_data['euler'] = [float(e) for e in euler_rad]
                
                # 四元数（w, x, y, z格式）
                joint_data['quaternion'] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            
            frame_data['joints'][name] = joint_data
        
        frames_data.append(frame_data)
    
    # 写入JSON文件
    output = {'frames': frames_data}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
