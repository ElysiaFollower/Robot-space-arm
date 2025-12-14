"""
IK核心算法实现
符合 SPEC.md 规范
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Optional
from model.joint import JointNode, SphericalJoint, FixedJoint


def build_ik_chain(root: JointNode, effector: JointNode) -> List[JointNode]:
    """
    构建IK Chain：从指定的“链的根”节点(root)到末端执行器(effector)路径上的所有可控关节的有序列表。
    注意：root不一定是全局树的根节点，而是本IK链的起点，可以用来定义短链。effector不一定是全局树的末端，而是本IK链的终点。
    路径包含root节点和effector节点本身——如果是FixedJoint，会被自动跳过（不加入IK链，故不影响结果）。
    
    :param root: 这条IK链的根节点
    :param effector: 末端执行器节点
    :return: IK Chain列表，仅包含1-DoF节点（RevoluteJoint或PrismaticJoint），SphericalJoint会被展开为虚拟关节
    """
    # 路径查找：从effector开始向上搜索直到root
    path: List[JointNode] = []
    current = effector
    
    # 从effector开始，不断向上遍历parent，直到找到root
    while current is not None:
        path.append(current)
        if current == root:
            break
        current = current.parent
    
    # 检查是否找到了root
    if path[-1] != root:
        raise ValueError(f"Cannot find path from {root.name} to {effector.name}")
    
    # 反转路径，使其从root到effector的顺序
    path.reverse()
    
    # 更新全局变换
    root.update_global_transform()
    
    # 节点筛选和展开：使用多态方法append_to_ik_chain
    ik_chain: List[JointNode] = []
    
    for node in path:
        # 使用多态方法，让每个节点自己决定如何添加到IK链
        # - FixedJoint: 跳过（不添加）
        # - RevoluteJoint/PrismaticJoint: 直接添加
        # - SphericalJoint: 将虚拟关节逐个加入
        node.append_to_ik_chain(ik_chain)
    
    return ik_chain


def compute_jacobian(ik_chain: List[JointNode], end_effector_pos: np.ndarray) -> np.ndarray:
    """
    构建雅可比矩阵 J (6xN)
    
    :param ik_chain: IK Chain（仅包含1-DoF节点）
    :param end_effector_pos: 末端执行器当前的世界坐标位置（3x1向量）
    :return: 6xN 雅可比矩阵
    """
    # 计算总自由度
    num_dofs = sum(node.get_dof() for node in ik_chain)
    
    # 初始化雅可比矩阵
    jacobian = np.zeros((6, num_dofs), dtype=np.float64)
    
    # 遍历IK Chain，计算每一列
    col_idx = 0
    for node in ik_chain:
        dof = node.get_dof()
        if dof == 0:
            continue  # 不应出现在Chain中
        elif dof == 1:
            # 计算雅可比列向量
            jacobian[:, col_idx] = node.compute_jacobian_column(end_effector_pos)
            col_idx += 1
        else:
            raise ValueError(f"Unexpected DoF: {dof} for node {node.name}")
    
    return jacobian


def compute_error_vector(current_transform: np.ndarray, 
                        target_transform: np.ndarray) -> np.ndarray:
    """
    计算当前末端姿态和目标姿态之间的 6x1 误差向量 (delta_x)
    
    :param current_transform: 末端执行器当前的 4x4 全局变换矩阵
    :param target_transform: 目标 4x4 全局变换矩阵
    :return: 6x1 的误差向量 [delta_p (3x1), delta_r (3x1)]
    """
    # 位置误差 (delta_p) - (3x1 向量)
    current_pos = current_transform[:3, 3]
    target_pos = target_transform[:3, 3]
    delta_p = target_pos - current_pos
    
    # 姿态误差 (delta_r) - (3x1 向量)
    # R_error = R_target * R_current^(-1)
    current_rot_mat = current_transform[:3, :3]
    target_rot_mat = target_transform[:3, :3]
    R_error_mat = target_rot_mat @ current_rot_mat.T # 旋转矩阵转置即是逆，即R_current^(-1)
    R_error = R.from_matrix(R_error_mat)
    
    # 将误差旋转转换为轴-角向量 (Rotation Vector)
    # 轴-角向量的方向是旋转轴，模长是旋转角度（弧度）
    delta_r = R_error.as_rotvec()
    
    # 组合为 6x1 向量
    delta_x = np.concatenate([delta_p, delta_r])
    
    return delta_x

