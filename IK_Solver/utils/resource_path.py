"""
资源路径工具函数
处理PyInstaller打包后的资源文件路径
"""
import sys
import os


def resource_path(relative_path: str) -> str:
    """
    获取资源文件的绝对路径
    兼容开发环境和PyInstaller打包后的环境
    
    :param relative_path: 相对于项目根目录的路径（如 'data/skeleton.json'）
    :return: 资源文件的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller打包后的环境
        # sys._MEIPASS是临时解压目录，包含所有打包的文件
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        # 开发环境：使用项目根目录
        # 获取src/utils的父目录（项目根目录）
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    return os.path.join(base_path, relative_path)


def get_data_path(filename: str) -> str:
    """
    获取data目录下文件的路径
    
    :param filename: 文件名（如 'skeleton.json'）
    :return: 文件的绝对路径
    """
    return resource_path(os.path.join('data', filename))

