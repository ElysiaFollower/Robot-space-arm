"""
关节类层次结构实现
"""
import numpy as np
from abc import ABC, abstractmethod
from typing_extensions import override
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, List

from utils import quaternion_to_rotation_matrix


class JointNode(ABC):
    """
    所有关节类型的抽象基类，定义求解器接口。
    """
    
    def __init__(self, name: str, offset: np.ndarray):
        """
        初始化关节节点
        
        :param name: 关节名称
        :param offset: 相对父级的静态位移 (Vec3)
        """
        self.name = name
        self.parent: Optional['JointNode'] = None
        self.children: List['JointNode'] = []
        self.local_offset: np.ndarray = np.asarray(offset, dtype=np.float64) 
        self.global_transform: np.ndarray = np.identity(4, dtype=np.float64)
    
    def add_child(self, child: 'JointNode'):
        """
        添加子节点（多态方法）
        基本逻辑在基类中实现，特殊逻辑通过钩子方法处理
        """
        # 基本逻辑：设置父子关系
        child.parent = self
        self.children.append(child)
        
        # 调用钩子方法，让子类处理特殊逻辑
        self.on_child_added(child)
        child.on_parent_set(self)
    
    def on_child_added(self, child: 'JointNode'):
        """
        钩子方法：当添加子节点时调用
        子类可以重写此方法以处理特殊逻辑
        
        :param child: 被添加的子节点
        """
        pass
    
    def on_parent_set(self, parent: 'JointNode'):
        """
        钩子方法：当设置父节点时调用
        子类可以重写此方法以处理特殊逻辑
        
        :param parent: 新设置的父节点
        """
        pass
    
    @abstractmethod
    def get_local_matrix(self) -> np.ndarray:
        """
        根据当前内部变量计算局部变换矩阵。
        
        :return: 4x4 局部变换矩阵
        """
        pass
    
    @abstractmethod
    def compute_jacobian_column(self, end_effector_pos: np.ndarray) -> np.ndarray: #note: for spherical joint, this method won't be used, since it's unvisible for IK solver
        """
        计算并返回该关节对应的雅可比列向量 (6x1)。
        
        :param end_effector_pos: 末端执行器当前在世界坐标系中的位置 (3x1 向量)
        :return: 6x1 列向量，前3个元素为线速度贡献，后3个元素为角速度贡献
        """
        pass
    
    @abstractmethod
    def apply_delta(self, delta_q: float):
        """
        接收求解器增量，更新内部变量，执行约束检查。
        
        :param delta_q: 关节变量的增量（弧度或米）
        """
        pass
    
    @abstractmethod
    def get_dof(self) -> int:
        """
        返回自由度数量 (0 或 1 或 3)。
        
        :return: 自由度数量
        """
        pass
    
    @abstractmethod
    def append_to_ik_chain(self, ik_chain: List['JointNode']):
        """
        将节点添加到IK链列表的末尾
        
        :param ik_chain: IK链列表（引用传递，直接修改）
        """
        pass
    
    def update_global_transform(self):
        """
        递归更新此关节及其所有子关节的 global_transform。
        使用多态特性调用 get_local_matrix()。
        """
        # 计算局部变换矩阵
        local_transform = self.get_local_matrix()
        
        # 计算全局变换矩阵
        if self.parent is None:
            # 根节点
            self.global_transform = local_transform
        else:
            # global = parent_global @ local
            self.global_transform = self.parent.global_transform @ local_transform
        
        # 递归更新所有子节点
        for child in self.children:
            child.update_global_transform()
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"


class RevoluteJoint(JointNode):
    """
    旋转关节 - 绕固定轴旋转的铰链
    """
    
    def __init__(self, name: str, offset: np.ndarray, axis: np.ndarray, 
                 limits: Optional[Tuple[float, float]] = None):
        """
        初始化旋转关节
        
        :param name: 关节名称
        :param offset: 相对父级的静态位移 (Vec3)
        :param axis: 旋转轴（局部坐标系，不能为零向量。程序自动归一化）
        :param limits: 约束范围 [min, max]（弧度），None 表示无约束
        """
        super().__init__(name, offset)
        self.axis = np.asarray(axis, dtype=np.float64)
        # 归一化轴向量
        axis_norm = np.linalg.norm(self.axis)
        if axis_norm > 1e-6:
            self.axis = self.axis / axis_norm
        else:
            raise ValueError(f"Axis vector is not a unit vector and too small to be normalized: {self.axis}")
        self.q: float = 0.0  # 角度（弧度）
        self.limits: Optional[Tuple[float, float]] = limits
    
    def get_local_matrix(self) -> np.ndarray:
        """生成绕 axis 旋转 q 的矩阵"""
        local_transform = np.identity(4, dtype=np.float64)
        
        # 构建旋转矩阵：绕局部轴 axis 旋转 q 弧度
        # 先计算四元数，然后通过四元数得到R_mat
        self.axis = self.axis / np.linalg.norm(self.axis)
        axis_normalized = self.axis
        theta = self.q

        # 计算四元数 [w, x, y, z]
        half_theta = theta / 2.0
        w = np.cos(half_theta)
        xyz = axis_normalized * np.sin(half_theta)
        quat = np.array([w, xyz[0], xyz[1], xyz[2]])

        # 根据四元数计算旋转矩阵
        R_mat = quaternion_to_rotation_matrix(quat)
        
        local_transform[:3, :3] = R_mat
        local_transform[:3, 3] = self.local_offset
        
        return local_transform
    
    def compute_jacobian_column(self, end_effector_pos: np.ndarray) -> np.ndarray:
        """
        计算雅可比列向量: J_i = [z_i cross (p_end - p_i), z_i]^T
        所有变量必须处于世界坐标系下
        """
        # 从 global_transform 提取旋转轴在世界坐标系中的方向向量
        # global_transform[:3, :3] 的列向量是局部坐标轴在世界坐标系中的表示
        # 需要将局部 axis 转换到世界坐标系 —— 进行坐标基变换
        R_world = self.global_transform[:3, :3]
        z_i = R_world @ self.axis  # 世界坐标系中的旋转轴方向
        # 确保轴向量是单位向量（数值稳定性）
        z_i_norm = np.linalg.norm(z_i)
        if z_i_norm > 1e-6:
            z_i = z_i / z_i_norm
        else:
            raise ValueError(f"Axis vector is not a unit vector and too small to be normalized: {z_i}")
        
        # 从 global_transform 提取关节位置（世界坐标系）
        p_i = self.global_transform[:3, 3]
        
        # 计算线速度贡献: z_i cross (p_end - p_i)
        vector_to_end = end_effector_pos - p_i
        J_v = np.cross(z_i, vector_to_end)
        
        # 角速度贡献: z_i
        J_w = z_i
        
        # 组合为 6x1 列向量
        return np.concatenate([J_v, J_w])
    
    def apply_delta(self, delta_q: float):
        """更新角度，执行约束检查"""
        self.q += delta_q
        if self.limits is not None:
            min_val, max_val = self.limits
            self.q = np.clip(self.q, min_val, max_val)
    
    def get_dof(self) -> int:
        return 1
    
    def append_to_ik_chain(self, ik_chain: List['JointNode']):
        """RevoluteJoint直接添加到IK链"""
        ik_chain.append(self)


class PrismaticJoint(JointNode):
    """
    移动关节 - 沿固定轴滑动的滑块
    """
    
    def __init__(self, name: str, offset: np.ndarray, axis: np.ndarray,
                 limits: Optional[Tuple[float, float]] = None):
        """
        初始化移动关节
        
        :param name: 关节名称
        :param offset: 相对父级的静态位移 (Vec3)
        :param axis: 移动轴（局部坐标系，不能为零向量。程序自动归一化）
        :param limits: 约束范围 [min, max](米), None 表示无约束
        """
        super().__init__(name, offset)
        self.axis = np.asarray(axis, dtype=np.float64)
        # 归一化轴向量
        axis_norm = np.linalg.norm(self.axis)
        if axis_norm > 1e-6:
            self.axis = self.axis / axis_norm
        else:
            raise ValueError(f"Axis vector is not a unit vector and too small to be normalized: {self.axis}")
        self.q: float = 0.0  # 位移（米）
        self.limits: Optional[Tuple[float, float]] = limits
    
    def get_local_matrix(self) -> np.ndarray:
        """生成沿 axis 平移 q 的矩阵: T = [I | q * axis]"""
        local_transform = np.identity(4, dtype=np.float64)
        local_transform[:3, 3] = self.local_offset + self.q * self.axis
        return local_transform
    
    def compute_jacobian_column(self, end_effector_pos: np.ndarray) -> np.ndarray:
        """
        计算雅可比列向量: J_i = [z_i, 0]^T
        所有变量必须处于世界坐标系下
        """
        # 从 global_transform 提取移动轴在世界坐标系中的方向向量
        R_world = self.global_transform[:3, :3]
        z_i = R_world @ self.axis  # 世界坐标系中的移动轴方向
        # 确保轴向量是单位向量（数值稳定性）
        z_i_norm = np.linalg.norm(z_i)
        if z_i_norm > 1e-6:
            z_i = z_i / z_i_norm
        else:
            raise ValueError(f"Axis vector is not a unit vector and too small to be normalized: {z_i}")
        
        # 线速度贡献: z_i
        J_v = z_i
        
        # 角速度贡献: 0（移动关节不产生旋转）
        J_w = np.zeros(3)
        
        # 组合为 6x1 列向量
        return np.concatenate([J_v, J_w])
    
    def apply_delta(self, delta_q: float):
        """更新位移，执行约束检查"""
        self.q += delta_q
        if self.limits is not None:
            min_val, max_val = self.limits
            self.q = np.clip(self.q, min_val, max_val)
    
    def get_dof(self) -> int:
        return 1
    
    def append_to_ik_chain(self, ik_chain: List['JointNode']):
        """PrismaticJoint直接添加到IK链"""
        ik_chain.append(self)


class FixedJoint(JointNode):
    """
    固定关节 - 无变量的结构连接或末端执行器
    增加quaternion属性，表示固定关节的本地旋转姿态（四元数, [w, x, y, z]）

    用于表示固定的偏移(包括位置和姿态)
    """
    
    def __init__(self, name: str, offset: np.ndarray, quaternion: Optional[np.ndarray] = None):
        """
        初始化固定关节
        
        :param name: 关节名称
        :param offset: 相对父级的静态位移 (Vec3)
        :param quaternion: 固定关节的本地旋转（四元数，格式为[w, x, y, z]），若为None则默认为无旋转单位四元数
        """
        super().__init__(name, offset)
        if quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 单位四元数
        else:
            self.quaternion = np.array(quaternion, dtype=np.float64)
            norm = np.linalg.norm(self.quaternion)
            if norm > 1e-6:
                self.quaternion /= norm
            else:
                raise ValueError(f"Quaternion norm too small: {self.quaternion}")

    def get_local_matrix(self) -> np.ndarray:
        """
        返回本地变换矩阵：先旋转，再平移（平移为local_offset, 旋转用self.quaternion）
        """
        local_transform = np.identity(4, dtype=np.float64)
        # 四元数转旋转矩阵
        R = quaternion_to_rotation_matrix(self.quaternion)
        local_transform[:3, :3] = R
        local_transform[:3, 3] = self.local_offset
        return local_transform

    def compute_jacobian_column(self, end_effector_pos: np.ndarray) -> np.ndarray:
        """返回 6x1 零向量（固定关节不参与雅可比构建）"""
        import inspect, os
        frame_info = inspect.currentframe()
        code_info = inspect.getframeinfo(frame_info)
        filename = os.path.basename(code_info.filename)
        lineno = code_info.lineno
        print(f"{filename}:{lineno} WARNING： 固定关节不应该参与雅可比构建(请检查ik_chain构建过程是否正确)") #有问题，但返回零向量还能用，所以不报错
        return np.zeros(6)
    
    def apply_delta(self, delta_q: float):
        """无操作（固定关节无变量）"""
        pass

    def get_dof(self) -> int:
        return 0
    
    def append_to_ik_chain(self, ik_chain: List['JointNode']):
        """FixedJoint跳过，不添加到IK链"""
        pass


class SphericalJoint(JointNode):
    """
    球形关节（SphericalJoint）
    在IK链构建时，球形关节会被拆解为虚拟固定关节与三个虚拟转动关节（沿XYZ轴依次串联），前者同步球形关节的旋转姿态，后者表示当前姿态下以XYZ欧拉角顺序的增量旋转并参与IK计算
    IK链中只包含这些虚拟关节，球形关节本体不会直接参与IK计算。
    """
    
    def __init__(self, name: str, offset: np.ndarray,
                 limits: Optional[List[Optional[Tuple[float, float]]]] = None):
        """
        初始化球形关节
        
        :param name: 关节名称
        :param offset: 相对父级的静态位移 (Vec3)
        :param limits: 三个轴的约束范围(弧度, XYZ欧拉角顺序), 每个元素为 [min, max] 或 None
        """
        super().__init__(name, offset)
        
        # 四元数表示的旋转状态
        self.quaternion: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        
        self.limits: Optional[List[Optional[Tuple[float, float]]]] = limits
        
        # 初始化虚拟固定关节和虚拟关节链
        # 虚拟固定关节用于同步球形关节当前的旋转姿态
        self.virtual_fix: Optional[FixedJoint] = None
        # 三个虚拟旋转关节：v1, v2, v3
        self.virtual_chain: List[RevoluteJoint] = []
        self._initialize_virtual_chain() # 预创建以避免频繁创建和销毁虚拟关节
    
    def _initialize_virtual_chain(self):
        """
        初始化虚拟关节链
        - 虚拟固定关节(fix): parent=SphericalJoint.parent, offset=SphericalJoint.offset, quaternion=SphericalJoint.quaternion
        - 虚拟关节1(v1): parent=fix, offset=[0,0,0], axis=[1,0,0], q=0
        - 虚拟关节2(v2): parent=v1, offset=[0,0,0], axis=[0,1,0], q=0
        - 虚拟关节3(v3): parent=v2, offset=[0,0,0], axis=[0,0,1], q=0, children=SphericalJoint.children
        """
        # 清理旧的虚拟关节链（如果存在）
        if self.virtual_fix is not None:
            # 清理虚拟固定关节的children
            self.virtual_fix.children.clear()
        if self.virtual_chain:
            # 清理虚拟关节链
            for node in self.virtual_chain:
                node.children.clear()
            self.virtual_chain.clear()
        
        # 创建虚拟固定关节(fix)
        self.virtual_fix = FixedJoint(
            name=f"_{self.name}_fix",
            offset=self.local_offset.copy(),
            quaternion=self.quaternion.copy()
        )
        self.virtual_fix.parent = self.parent
        self.virtual_fix.spherical_parent = self # 必要的时候可以用来区分虚拟关节和真实关节，并链接到虚拟关节的原型球形关节
        
        # 创建三个虚拟旋转关节：v1, v2, v3
        # 轴固定为局部坐标系的XYZ轴：[1,0,0], [0,1,0], [0,0,1]
        axes = [
            np.array([1.0, 0.0, 0.0]),  # X轴
            np.array([0.0, 1.0, 0.0]),  # Y轴
            np.array([0.0, 0.0, 1.0]),  # Z轴
        ]
        
        for i in range(3):
            # 获取对应轴的约束
            limits = None
            if self.limits is not None and i < len(self.limits) and self.limits[i] is not None:
                limits = self.limits[i]
            
            # 创建虚拟旋转关节
            node = RevoluteJoint(
                name=f"_{self.name}_v{i+1}",
                offset=np.zeros(3),  # 所有虚拟关节的offset都是[0,0,0]
                axis=axes[i],  # 固定为局部坐标系的XYZ轴
                limits=limits
            )
            
            # 为虚拟关节添加对SphericalJoint的引用
            node.spherical_parent = self
            
            # 建立虚拟关节的串联关系
            if i == 0:
                # v1：parent = fix
                self.virtual_fix.add_child(node)
            else:
                # v2和v3：parent指向前一个虚拟关节
                self.virtual_chain[i-1].add_child(node)
            
            self.virtual_chain.append(node)
        
        # v3继承球形关节的children
        if len(self.virtual_chain) > 0:
            v3 = self.virtual_chain[2]
            v3.children = self.children.copy()  # 复制列表，避免直接引用 # 注意不能使用add_child()方法，因为会修改children的父
    
    def get_local_matrix(self) -> np.ndarray:
        """
        基于 quaternion 计算局部旋转矩阵
        """
        # 直接用当前四元数生成旋转矩阵
        R_sphere_local = quaternion_to_rotation_matrix(self.quaternion)
        
        local_transform = np.identity(4, dtype=np.float64)
        local_transform[:3, :3] = R_sphere_local
        local_transform[:3, 3] = self.local_offset
        
        return local_transform
    
    @override
    def update_global_transform(self):
        """
        更新SphericalJoint的global_transform 
        增添逻辑：同步实际关节和虚拟关节的状态：
        1. 检查增量：若v1.q, v2.q, v3.q均为0，跳过
        2. 应用旋转：将v1.q, v2.q, v3.q视为绕当前v1.axis, v2.axis, v3.axis的增量欧拉角
        3. 依照XYZ欧拉角顺序构建增量旋转
        4. 更新球形关节姿态：R_new = R_curr · R_delta
        5. 更新约束：更新虚拟关节的Limit（若不为None）—— 以v1.q为例，v1.q清零时，v1对应的Limit上下界分别减去v1.q，因为相应增量已经应用在当前姿态上了
        6. 同步姿态：将球形关节的四元数赋值给fix
        """
        # 状态同步：检查虚拟关节的增量并同步到球形关节
        v1, v2, v3 = self.virtual_chain[0], self.virtual_chain[1], self.virtual_chain[2]
        
        # 检查增量：若v1.q, v2.q, v3.q均为0，跳过
        if abs(v1.q) > 1e-10 or abs(v2.q) > 1e-10 or abs(v3.q) > 1e-10:
            # 应用旋转：将v1.q, v2.q, v3.q视为绕当前v1.axis, v2.axis, v3.axis的增量欧拉角
            # 依照XYZ欧拉角顺序构建增量旋转
            rot_delta = R.from_euler('XYZ', [v1.q, v2.q, v3.q], degrees=False)
            quat_delta = rot_delta.as_quat()  # [x, y, z, w]
            quat_delta = np.array([quat_delta[3], quat_delta[0], quat_delta[1], quat_delta[2]])  # [w, x, y, z]
            
            # 更新球形关节姿态：R_new = R_curr · R_delta (四元数乘法)
            q_old = self.quaternion
            w1, x1, y1, z1 = q_old[0], q_old[1], q_old[2], q_old[3]
            w2, x2, y2, z2 = quat_delta[0], quat_delta[1], quat_delta[2], quat_delta[3]
            
            # 四元数乘法：q_new = q_old * q_delta
            q_new = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
                w1*x2 + w2*x1 + y1*z2 - z1*y2,  # x
                w1*y2 + w2*y1 + z1*x2 - x1*z2,  # y
                w1*z2 + w2*z1 + x1*y2 - y1*x2   # z
            ])
            q_new = q_new / np.linalg.norm(q_new)
            self.quaternion = q_new

            # 更新约束：更新虚拟关节的Limit（若不为None）
            # 以v1.q为例，v1.q清零时，v1对应的Limit上下界分别减去v1.q
            q_v1, q_v2, q_v3 = v1.q, v2.q, v3.q
            if v1.limits is not None:
                min_val, max_val = v1.limits
                v1.limits = (min_val - q_v1, max_val - q_v1)
            if v2.limits is not None:
                min_val, max_val = v2.limits
                v2.limits = (min_val - q_v2, max_val - q_v2)
            if v3.limits is not None:
                min_val, max_val = v3.limits
                v3.limits = (min_val - q_v3, max_val - q_v3)

            # 清空增量
            v1.q = v2.q = v3.q = 0.0
            
            # 同步姿态：将球形关节的四元数赋值给fix
            self.virtual_fix.quaternion = self.quaternion.copy()
        
        local_transform = self.get_local_matrix()
        if self.parent is None:
            self.global_transform = local_transform
        else:
            self.global_transform = self.parent.global_transform @ local_transform
        
        # 更新虚拟固定关节的global_transform，就是SphericalJoint的global_transform
        self.virtual_fix.global_transform = self.global_transform
        
        # 递归更新虚拟关节链的global_transform
        # 虚拟关节链：fix -> v1 -> v2 -> v3
        # 由于虚拟关节的q都是0（已同步），并且平移offset都是0，所以它们的local_transform只是单位矩阵
        # 所以它们的global_transform都是SphericalJoint的global_transform
        for virtual_node in self.virtual_chain:
            virtual_node.global_transform = self.global_transform

        # 递归更新所有子节点
        for child in self.children:
            child.update_global_transform()
    
    def compute_jacobian_column(self, end_effector_pos: np.ndarray) -> np.ndarray:
        """
        不直接调用。
        SphericalJoint在IK Chain构建时会被展开为3个内部节点。
        """
        raise NotImplementedError("SphericalJoint should be expanded in IK Chain")
    
    def apply_delta(self, delta_q: float): 
        """
        不直接调用。
        SphericalJoint的增量通过虚拟节点处理。
        在任何需要使用到SphericalJoint位置/姿态的地方，都意味着场景图的刷新，也就意味着增量的同步。Apply相当于在那时进行一次同步
        """
        raise NotImplementedError("SphericalJoint delta should be applied to internal nodes")
    
    def get_dof(self) -> int:
        return 3
    
    def append_to_ik_chain(self, ik_chain: List['JointNode']):
        """SphericalJoint将虚拟关节逐个加入IK链"""
        if not self.virtual_chain:
            raise RuntimeError(f"SphericalJoint {self.name} has no virtual chain")
        
        # 按照虚拟关节的串联顺序，依次加入Chain
        for virtual_node in self.virtual_chain:
            ik_chain.append(virtual_node)
    
    @override
    def on_child_added(self, child: 'JointNode'):
        """
        当SphericalJoint添加子节点时，需要更新v3的children
        v3应该继承SphericalJoint的所有children
        """
        if self.virtual_chain and len(self.virtual_chain) >= 3:
            v3 = self.virtual_chain[2]
            if child not in v3.children:
                v3.children.append(child)
    
    @override
    def on_parent_set(self, parent: 'JointNode'):
        """
        当SphericalJoint设置父节点时，需要更新virtual_fix的parent
        virtual_fix的parent应该等于SphericalJoint的parent
        """
        if self.virtual_fix is not None:
            self.virtual_fix.parent = parent


if __name__ == "__main__":
    pass