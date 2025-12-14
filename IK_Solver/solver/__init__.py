"""
求解层 (Solver Layer)
纯数学计算，负责正向运动学更新、雅可比矩阵构建、线性方程组求解及变量更新
"""

from .ik_core import (
    build_ik_chain,
    compute_jacobian,
    compute_error_vector
)
from .solve_ik import solve_ik

__all__ = [
    'build_ik_chain',
    'compute_jacobian',
    'compute_error_vector',
    'solve_ik'
]

