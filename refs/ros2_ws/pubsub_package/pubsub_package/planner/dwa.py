# -*- coding: utf-8 -*-


import numpy as np
import math
import copy
import time


class DWA:
    """
    动态窗口算法类，包含使用动态窗口算法完成局部运动规划所需要用到的函数
    """

    def __init__(self):
        """
        初始化一些全局参数
        """
