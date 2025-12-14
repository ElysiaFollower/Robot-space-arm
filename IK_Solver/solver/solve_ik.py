"""
IK求解器实现
使用阻尼最小二乘法 (Damped Least Squares, DLS)
"""
import numpy as np
from typing import List, Optional
from model.joint import JointNode, SphericalJoint
from .ik_core import (
    build_ik_chain, 
    compute_jacobian, 
    compute_error_vector
)


def solve_ik(
    root: JointNode,
    effector: JointNode,
    target_transform: np.ndarray,
    ik_chain: Optional[List[JointNode]] = None,
    max_iterations: int = 100,
    position_tolerance: float = 1e-3,
    orientation_tolerance: float = 1e-1,
    damping: float = 0.0001,
    enable_line_search: bool = True,  # 默认启用线搜索
    line_search_alpha: float = 1.0,  
    line_search_alpha_min: float = 1e-2
) -> bool:
    """
    使用阻尼最小二乘法 (DLS) 求解IK

    :param root: 这条IK链的根节点
    :param effector: 这条IK链的终点节点
    :param target_transform: 目标变换矩阵（4x4），包含目标位置和姿态
    :param ik_chain: IK Chain（可选，如果为None则自动构建; 当自行提供的时候，要确保是合法的IK链，即从root到effector的顺序）
    :param max_iterations: 最大迭代次数，默认值100
    :param position_tolerance: 位置收敛容差（单位：米），默认值1e-3
    :param orientation_tolerance: 姿态收敛容差（单位：弧度），默认值1e-1
    :param damping: 阻尼系数λ，默认值0.0001
    :param enable_line_search: 是否启用线搜索，默认值True
    :param line_search_alpha: 线搜索初始步长，默认值1.0
    :param line_search_alpha_min: 线搜索最小步长，默认值1e-2； 若步长小于该值仍无法改进，则停止迭代
    :return: True表示成功收敛，False表示失败
    """
    # 如果未提供IK Chain，自动构建
    if ik_chain is None:
        ik_chain = build_ik_chain(root, effector)

    # 获取6x6单位矩阵，用于DLS
    identity_6x6 = np.identity(6, dtype=np.float64)

    # 保存每轮迭代开始时的最佳状态；线搜索启用时每轮迭代保证误差减小，故而实际上是保存当前解得的最优状态
    # 需要保存虚拟关节的q值和对应的SphericalJoint的quaternion
    best_states: List[float] = []
    best_spherical_quaternions = {}  # 映射：SphericalJoint对象 -> quaternion

    # FK更新：刷新全树变换
    root.update_global_transform()
    
    # 主迭代循环
    for iteration in range(max(max_iterations, 1)):
        # 在每次迭代开始时保存当前状态; 至少抵达一次
        for node in ik_chain:
            best_states = [node.q for node in ik_chain]
            # 如果虚拟关节属于SphericalJoint，保存SphericalJoint的quaternion
            if hasattr(node, 'spherical_parent') and node.spherical_parent is not None:
                spherical: SphericalJoint = node.spherical_parent
                best_spherical_quaternions[spherical] = spherical.quaternion.copy()

        # 计算误差 ΔX
        current_transform = effector.global_transform
        delta_x = compute_error_vector(current_transform, target_transform)
        current_error_norm = np.linalg.norm(delta_x)

        # 收敛检查
        pos_error_norm = np.linalg.norm(delta_x[:3])
        ori_error_norm = np.linalg.norm(delta_x[3:])
        if pos_error_norm < position_tolerance and ori_error_norm < orientation_tolerance:
            return True  # 成功收敛

        # 构建雅可比矩阵 J
        end_effector_pos = effector.global_transform[:3, 3]
        J = compute_jacobian(ik_chain, end_effector_pos)

        # 求解 Δq
        # 构建阻尼矩阵: A = J * J^T + λ * I
        A = J @ J.T + damping * identity_6x6

        # 求解线性方程组: A * β = ΔX
        # 使用更稳定的求解方法，即使矩阵仍然接近奇异也能处理
        try:
            # 尝试使用标准求解
            beta = np.linalg.solve(A, delta_x)
        except np.linalg.LinAlgError:
            # 矩阵奇异或接近奇异，使用最小二乘求解（更稳定）
            try:
                beta, residuals, rank, s = np.linalg.lstsq(A, delta_x, rcond=None)
            except:
                # 如果仍然失败，回滚到最佳状态(之前迭代得到的最优状态)并返回False
                for i, node in enumerate(ik_chain):
                    node.q = best_states[i]
                # 恢复SphericalJoint的quaternion
                for spherical, quat in best_spherical_quaternions.items():
                    spherical.quaternion = quat.copy()
                root.update_global_transform()
                return False

        # 计算增量: Δq = J.T * β
        delta_q = J.T @ beta

        # 线搜索（默认启用）； 寻找到最优步长 —— 实际上是不断缩小dx，dtheta，直到局部线性化的假设在允许误差的范围内成立
        alpha = line_search_alpha
        if enable_line_search:
            # 线搜索循环
            while True:
                # 回滚到搜索开始前的初始状态，即探测起点，即先前的最优状态
                for i, node in enumerate(ik_chain):
                    node.q = best_states[i]
                for spherical, quat in best_spherical_quaternions.items():
                    spherical.quaternion = quat.copy()

                # 应用测试增量（所有节点统一处理）
                for i, node in enumerate(ik_chain):
                    node.apply_delta(alpha * delta_q[i])

                root.update_global_transform()

                # 计算新误差
                new_transform = effector.global_transform
                new_delta_x = compute_error_vector(new_transform, target_transform)
                new_error_norm = np.linalg.norm(new_delta_x)

                # 检查是否接受
                if new_error_norm < current_error_norm:
                    break  # 误差减小，接受

                # 否则缩小步长
                alpha = alpha / 2.0
                if alpha < line_search_alpha_min:
                    # 线搜索失败：即使使用很小的步长也无法改进
                    # 在带阻尼的DLS方法中，这可能意味着目标无法到达或已接近最优
                    # 则回滚状态并退出
                    for i, node in enumerate(ik_chain):
                        node.q = best_states[i]
                    # 恢复SphericalJoint的quaternion
                    for spherical, quat in best_spherical_quaternions.items():
                        spherical.quaternion = quat.copy()
                    root.update_global_transform()
                    return False
        else:
            # 不使用线搜索，直接应用增量（所有节点统一处理）
            # 但在奇异情况下，delta_q可能很大，需要限制步长以避免抖动
            delta_q_norm = np.linalg.norm(delta_q)
            if delta_q_norm > 1.0:  # 如果增量过大，进行缩放
                delta_q = delta_q / delta_q_norm * 1.0  # 限制最大步长为1.0

            for i, node in enumerate(ik_chain):
                node.apply_delta(delta_q[i])

        # 更新FK（为下次迭代准备）
        root.update_global_transform()

    # 超过最大迭代次数, 停止迭代
    root.update_global_transform()
    return False