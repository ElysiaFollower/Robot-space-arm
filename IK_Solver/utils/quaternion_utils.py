"""
四元数工具函数
"""
import numpy as np
from typing import Union


def quaternion_to_rotation_matrix(quaternion: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    将四元数转换为旋转矩阵
    
    :param quaternion: 四元数，格式为 [w, x, y, z] 或 (w, x, y, z)
    :return: 3x3 旋转矩阵
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    
    if quaternion.shape != (4,):
        raise ValueError(f"Quaternion must be a 4-element array, got shape {quaternion.shape}")
    
    # 归一化四元数
    norm = np.linalg.norm(quaternion)
    if norm < 1e-10:
        raise ValueError(f"Quaternion norm too small: {norm}, cannot normalize")
    quaternion = quaternion / norm
    
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
        [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
        [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ], dtype=np.float64)
    
    return R

