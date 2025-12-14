# Write the path to the IK folder here
# For example, the IK folder is in your Documents folder
import sys
import getpass
import math
sys.path.append(f"D:/robotics/task3/my_IK")


import numpy as np
from my_IK import my_solver

####################################
### You Can Write Your Code Here ###
####################################
def linearTrajPlaning(time_total, t_current):
    global self
    global linear_flag
    # Initialize trajectory state
    if linear_flag:
        linear_flag = 0
        qin = np.array(self.qin, dtype=np.float64)
        tin = np.array(self.tin, dtype=np.float64)
        tout = np.array(self.tout, dtype=np.float64)
        cart_line = {
            "x_start": tin[0], "x_end": tout[0],
            "y_fix": tin[1], "z_fix": tin[2],
            "rx_fix": tin[3], "ry_fix": tin[4], "rz_fix": tin[5]
        }
        cart_x_diff = cart_line["x_end"] - cart_line["x_start"]
        
        x_vel = (self.tout[0] - self.tin[0]) / time_total
        
        t_trans = 0.5
        t_trans1_end = t_trans
        t_trans2_start = time_total - t_trans
        vin_cart = vout_cart = np.zeros(6)
        vin_cart[0] = vout_cart[0] = x_vel
        trans1_dis = trans2_dis = x_vel * t_trans
        constant_vel_time = time_total - 2 * t_trans

        v_avg = np.zeros(6)
        v_avg[0] = (cart_x_diff - trans1_dis - trans2_dis) / constant_vel_time
        
        joint_limits = np.array([
            [-10*np.pi/9, 10*np.pi/9],
            [-np.pi/2, np.pi/2],
            [-2*np.pi/3, 2*np.pi/3],
            [-5*np.pi/6, 5*np.pi/6],
            [-5*np.pi/6, 5*np.pi/6],
            [-np.pi, np.pi]
        ])
        linearTrajPlaning._state = {
            "t_trans": t_trans,
            "t_trans1_end": t_trans1_end, "t_trans2_start": t_trans2_start,
            "qin": qin, "current_q": qin.copy(),
            "cart_line": cart_line, "vin_cart": vin_cart, "v_avg": v_avg, "vout_cart": vout_cart,
            "x_vel": x_vel,
            "joint_limits": joint_limits, "last_t": 0.0, "J_last": calculate_jacobian(qin)
        }
    
    state = linearTrajPlaning._state
    t_current_clamped = min(t_current, time_total)
    dt = max(t_current_clamped - state["last_t"], 1e-6)
    state["last_t"] = t_current_clamped
    current_q = state["current_q"].copy()

    # Boundary handling
    if t_current_clamped <= 0:
        state["current_q"] = state["qin"]
        return state["qin"]
    if t_current_clamped >= time_total:
        delattr(linearTrajPlaning, "_state")
        return current_q

    # Calculate target Cartesian velocity
    target_cart_vel = np.zeros(6)
    t = t_current_clamped
    if t < state["t_trans1_end"]:
        t_norm = t / state["t_trans"]
        vel_factor = 10 * t_norm**3 - 15 * t_norm**4 + 6 * t_norm**5
        target_cart_vel = state["vin_cart"] + (state["v_avg"] - state["vin_cart"]) * vel_factor
    elif t < state["t_trans2_start"]:
        target_cart_vel = state["v_avg"].copy()
    else:
        t_norm = (time_total - t) / state["t_trans"]
        vel_factor = 10 * t_norm**3 - 15 * t_norm**4 + 6 * t_norm**5
        target_cart_vel = state["v_avg"] + (state["vout_cart"] - state["v_avg"]) * vel_factor

    # Calculate target Cartesian position
    target_cart_pos = np.zeros(6)
    cart_line = state["cart_line"]
    target_cart_pos[1:] = [cart_line["y_fix"], cart_line["z_fix"], 
                          cart_line["rx_fix"], cart_line["ry_fix"], cart_line["rz_fix"]]
    
    if t < state["t_trans1_end"]:
        t_val = t
        T = state["t_trans"]
        a = state["vin_cart"][0]  # Equivalent to state["x_vel"], keep original logic
        b = state["v_avg"][0]
        target_cart_pos[0] = cart_line["x_start"] + (a*t_val + 0.5*(b - a)*(2*T*t_val**3 - 3*t_val**4/T + t_val**5/T**2))
    elif t < state["t_trans2_start"]:
        trans1_total = 0.5 * (state["vin_cart"][0] + state["v_avg"][0]) * state["t_trans"]
        constant_vel_duration = t - state["t_trans1_end"]
        target_cart_pos[0] = cart_line["x_start"] + trans1_total + state["v_avg"][0] * constant_vel_duration
    else:
        trans1_total = 0.5 * (state["vin_cart"][0] + state["v_avg"][0]) * state["t_trans"]
        constant_vel_total = state["v_avg"][0] * (state["t_trans2_start"] - state["t_trans1_end"])
        t_remain = time_total - t
        T = state["t_trans"]
        a = state["v_avg"][0]
        b = state["vout_cart"][0]  # Equivalent to state["x_vel"], keep original logic
        trans2_progress = a*(state["t_trans"] - t_remain) + 0.5*(b - a)*(
            2*T*(state["t_trans"] - t_remain)**3 - 3*(state["t_trans"] - t_remain)**4/T + 
            (state["t_trans"] - t_remain)**5/T**2
        )
        target_cart_pos[0] = cart_line["x_start"] + trans1_total + constant_vel_total + trans2_progress

    # Calculate Jacobian matrix and joint velocities
    J_current = calculate_jacobian(current_q)
    J_current[:3, :] /= 1000.0
    state["J_last"] = J_current

    damping = 0.01
    J_T = J_current.T
    joint_vel = J_T @ np.linalg.inv(J_current @ J_T + damping**2 * np.eye(6)) @ target_cart_vel

    vel_limits = np.array([100, 100, 100, 100, 100, 100]) / 180 * np.pi
    joint_vel = np.clip(joint_vel, -vel_limits, vel_limits)

    # Update joint angles
    current_q += joint_vel * dt
    current_q[0] = math.remainder(current_q[0], 2 * np.pi)
    current_q[5] = math.remainder(current_q[5], 2 * np.pi)
    current_q = np.clip(current_q, state["joint_limits"][:, 0], state["joint_limits"][:, 1])

    state["current_q"] = current_q
    return current_q    
def compute_joint_velocities(joints):
    cartesian_velocity = np.array([0.34/3, 0, 0, 0, 0, 0], dtype=np.float64)  * 1000.0
    J = calculate_jacobian(joints)
    J_pinv = np.linalg.pinv(J)
    joint_velocity = J_pinv @ cartesian_velocity
    return -joint_velocity
    
def transform_local_to_global(pose, local_offset):
    """
    Transform local offset to global pose
    :param pose: [x, y, z, rx, ry, rz] Cartesian xyz + Euler angles
    :param local_offset: [dx, dy, dz] Local offset in end-effector frame
    :return: Global pose with offset applied
    """
    x, y, z, rx, ry, rz = pose
    dx, dy, dz = local_offset
    
    # Compute ZYX Euler angle rotation matrix
    # Note: ZYX corresponds to Yaw-Pitch-Roll
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)
    
    # Compute R = Rz * Ry * Rx
    R = np.array([
        [cos_ry*cos_rz, -cos_ry*sin_rz, sin_ry],
        [cos_rx*sin_rz + sin_rx*sin_ry*cos_rz, cos_rx*cos_rz - sin_rx*sin_ry*sin_rz, -sin_rx*cos_ry],
        [sin_rx*sin_rz - cos_rx*sin_ry*cos_rz, sin_rx*cos_rz + cos_rx*sin_ry*sin_rz, cos_rx*cos_ry]
    ])
    
    # Transform local offset to global frame
    global_offset = R @ np.array([dx, dy, dz])
    
    # Return updated pose
    return np.array([x + global_offset[0], y + global_offset[1], z + global_offset[2], rx, ry, rz])

def ik_suitable(pose_data):
    ik_solutions = my_solver(np.array(pose_data))
    
    if ik_solutions.size == 0:
        return np.zeros(6)
    
    solutions = ik_solutions.T
    return solutions[2]


def sysCall_init():
    sim = require('sim')
    # Initialize the simulation
    doSomeInit()    # must have    
    self.curobj=0
    self.objcnt=4
    self.Tperiod=25
    #------------------------------------------------------------------------
    # Using the codes, you can obtain the poses and positions of four blocks
    pointHandles = []
    suckPointPaths = [
    '/Platform1/Cuboid2/SuckPoint',
    '/Platform1/Prism2/SuckPoint',
    '/Platform1/Prism1/SuckPoint',
    '/Platform1/Cuboid1/SuckPoint'
    ]
    for path in suckPointPaths:
        pointHandles.append(sim.getObject(path))
    self.qobj=[]
    self.qobjabove=[]
    for i in range(4):
        position = sim.getObjectPosition(pointHandles[i], -1)
        orientation = sim.getObjectOrientation(pointHandles[i], -1)
        self.qobj.append(ik_suitable(position + orientation))
        self.qobjabove.append(ik_suitable(np.array(position + orientation)+np.array([0,0,0.05,0,0,0])))
    self.pttarg=[[-0.35, 0.0, 0.20, -3.141592653589793, 0, +1.5707963267948968],
    [-0.35, -0.05, 0.175, -2.356210045920425, 0, -3.1415924163059246],
    [-0.35, 0.05, 0.175, 2.356210045920425, 0, 0],
    [-0.35, 0.0, 0.25, -3.141592653589793, 0, +1.5707963267948968]]
    self.qtarg=[]
    for i in range(4):
        self.qtarg.append(ik_suitable(np.array(self.pttarg[i])))
    self.qtargdrop=[]
    for i in range(4):
        self.qtargdrop.append(ik_suitable(np.array(self.pttarg[i])+np.array([0,0,0.01,0,0,0])))
    self.qtargdrop[1][0] += math.pi * 2    
    self.qtargabove=[]
    for i in range(4):
        targdrop_pose = np.array(self.pttarg[i]) + np.array([0,0,0.06,0,0,0])
        targabove_pose = transform_local_to_global(targdrop_pose, [0, 0, -0.02])
        self.qtargabove.append(ik_suitable(targabove_pose))
    self.qtargabove[1][0] += math.pi * 2
    self.q0 = np.zeros(6) # initialize q0 with all zeros
    self.qmid = ik_suitable(np.array([0, 0.3, 0.30, -3.141592653589793, 0, 1.5707963267948966]))
    self.tin = np.array([0.17, 0.35, 0.20, -3.141592653589793, 0, -1.5707963267948968])
    self.tout = np.array([-0.17, 0.35, 0.20, -3.141592653589793, 0, -1.5707963267948968])
    self.qin=ik_suitable(self.tin)
    self.qout=ik_suitable(self.tout)
    self.vin = compute_joint_velocities(self.qin)
    self.vout = compute_joint_velocities(self.qout)
q_start = None
mark_start = 0
linear_flag = 1
def sysCall_actuation():
    global q_start, mark_start, linear_flag
    t_origin2above = 5
    t_above2obj = 1
    t_obj2in = 3
    t_in2out = 3
    t_out2above = 5
    t_above2mid = 5 #t_above2mid = t_origin2above
    t_grip = 0.5
    
    t = sim.getSimulationTime() - self.Tperiod * self.curobj

    if self.curobj >= self.objcnt:
        sim.pauseSimulation()
        return

    if t < t_origin2above + t_above2obj:
        if self.curobj == 0:
            q = quinticTraj_1_mid(self.q0, self.qobjabove[self.curobj], self.qobj[self.curobj], t, t_origin2above, t_above2obj)
        else :
            q = quinticTraj_3_mid(self.qtargdrop[self.curobj-1], self.qtargabove[self.curobj-1], self.qmid, self.qobjabove[self.curobj],self.qobj[self.curobj], t+t_above2mid+t_above2obj, t_above2obj, t_above2mid,t_above2mid,t_above2obj)
        state = False
    elif t < t_origin2above + t_above2obj + t_grip:
        q = self.qobj[self.curobj]
        state = True  
        # Grip #33333333333333333333
    elif t < t_origin2above + t_above2obj + t_grip + t_obj2in:
        q = quinticTraj_0_mid(self.qobj[self.curobj], self.qin, t-(t_origin2above + t_above2obj + t_grip), t_obj2in, None, self.vin)
        state = True
        # Grip #33333333333333333333
        linear_flag = 1
    elif t < t_origin2above + t_above2obj + t_grip + t_obj2in + t_in2out:
        # Linear traversal stage: t_linear from 0 to t_in2out
        t_linear = t - (t_origin2above + t_above2obj + t_grip + t_obj2in)
        # Record starting joint angles
        if mark_start == 0:
            q_start = getCurrentq()
        mark_start = 1
        q = linearTrajPlaning(t_in2out, t_linear)
        state = True
    elif t < t_origin2above + 2 * t_above2obj + t_grip + t_obj2in + t_in2out + t_out2above:
        linear_flag = 1
        if mark_start == 1:
            q_start = getCurrentq()
        mark_start = 0
        q = quinticTraj_1_mid(q_start,self.qtargabove[self.curobj], self.qtargdrop[self.curobj], t-(t_origin2above + t_above2obj + t_grip + t_obj2in + t_in2out - 0.07), t_out2above, t_above2obj, self.vout) 
        state = True
    elif t < t_origin2above + 2 * t_above2obj + 2 * t_grip + t_obj2in + t_in2out + t_out2above:
        # print("drop_begin:  ", self.curobj)
        # print("linear_flag",linear_flag)
        # q = self.qtargdrop[self.curobj]
        # if linear_flag == 0 and self.curobj == 1:
        #     linear_flag = 1
        #     q[0] += math.pi * 2
        #     print("trans:  ", self.curobj)
        # state = False
        # print("drop_end:  ", self.curobj)
        state = False
        q = self.qtargdrop[self.curobj]
    elif t < t_origin2above + 3 * t_above2obj + 2 * t_grip + t_obj2in + t_in2out + t_out2above + t_above2mid:
        if self.curobj == self.objcnt - 1:
            q = quinticTraj_1_mid(self.qtargdrop[self.curobj], self.qtargabove[self.curobj], self.q0, t-(t_origin2above + 2 * t_above2obj + 2 * t_grip + t_obj2in + t_in2out + t_out2above), t_above2obj, t_origin2above, self.vout)
        else:
            # if self.curobj == 1:
            #     print("="*5)
            #     print(linear_flag)
            #     print(self.qtargdrop[self.curobj])
            #     self.qtargdrop[self.curobj][0] = self.qtargdrop[self.curobj][0] + math.pi * 2
            #     print(self.qtargdrop[self.curobj])
            #     print("="*5)
            
            
            # if linear_flag == 1 and self.curobj == 1:
            #     linear_flag = 0
            #     print("="*5)
            #     print(self.qtargdrop[self.curobj])
            #     self.qtargdrop[self.curobj][0] = self.qtargdrop[self.curobj][0] + math.pi * 2
            #     print(self.qtargdrop[self.curobj])
            #     print("="*5)
            q = quinticTraj_3_mid(self.qtargdrop[self.curobj], self.qtargabove[self.curobj], self.qmid, self.qobjabove[self.curobj+1],self.qobj[self.curobj+1], t-(t_origin2above + 2 * t_above2obj + 2 * t_grip + t_obj2in + t_in2out + t_out2above), t_above2obj, t_above2mid,t_above2mid,t_above2obj)
        state = False
    else:
        self.curobj += 1
        if self.curobj == self.objcnt:
            q = self.q0
        else:
            q = self.qmid
        state = False

    runState = move(q, state)

    if not runState:
        sim.pauseSimulation()
        
def getCurrentq():
    return np.array([sim.getJointPosition(handle) for handle in self.jointHandles])

####################################################
### You Don't Have to Change the following Codes ###
####################################################
def doSomeInit():
    self.Joint_limits = np.array([[-200, -90, -120, -150, -150, -180],
                            [200, 90, 120, 150, 150, 180]]).transpose()/180*np.pi
    self.Vel_limits = np.array([100, 100, 100, 100, 100, 100])/180*np.pi
    self.Acc_limits = np.array([500, 500, 500, 500, 500, 500])/180*np.pi
    
    self.lastPos = np.zeros(6)
    self.lastVel = np.zeros(6)
    self.sensorVel = np.zeros(6)
    
    self.robotHandle = sim.getObject('/Robot')
    self.suctionHandle = sim.getObject('/Robot/SuctionCup')
    self.jointHandles = []
    for i in range(6):
        self.jointHandles.append(sim.getObject('/Robot/Joint' + str(i+1)))
    sim.writeCustomStringData(self.suctionHandle, 'activity', 'off')
    sim.writeCustomStringData(self.robotHandle, 'error', '0')
    
    self.dataPos = []
    self.dataVel = []
    self.dataAcc = []
    self.graphPos = sim.getObject('/Robot/DataPos')
    self.graphVel = sim.getObject('/Robot/DataVel')
    self.graphAcc = sim.getObject('/Robot/DataAcc')
    color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    for i in range(6):
        self.dataPos.append(sim.addGraphStream(self.graphPos, 'Joint'+str(i+1), 'deg', 0, color[i]))
        self.dataVel.append(sim.addGraphStream(self.graphVel, 'Joint'+str(i+1), 'deg/s', 0, color[i]))
        self.dataAcc.append(sim.addGraphStream(self.graphAcc, 'Joint'+str(i+1), 'deg/s2', 0, color[i]))

def sysCall_sensing():
    # put your sensing code here
    if sim.readCustomStringData(self.robotHandle,'error') == '1':
        return
    for i in range(6):
        pos = sim.getJointPosition(self.jointHandles[i])
        if i == 0:
            if pos < -160/180*np.pi:
                pos += 2*np.pi
        vel = sim.getJointVelocity(self.jointHandles[i])
        acc = (vel - self.sensorVel[i])/sim.getSimulationTimeStep()
        if pos < self.Joint_limits[i, 0] or pos > self.Joint_limits[i, 1]:
            print("Error: Joint" + str(i+1) + " Position Out of Range!")
            sim.writeCustomStringData(self.robotHandle, 'error', '1')
            return
        
        if abs(vel) > self.Vel_limits[i]:
            print("Error: Joint" + str(i+1) + " Velocity Out of Range!")
            sim.writeCustomStringData(self.robotHandle, 'error', '1')
            return
        
        if abs(acc) > self.Acc_limits[i]:
            print("Error: Joint" + str(i+1) + " Acceleration Out of Range!")
            sim.writeCustomStringData(self.robotHandle, 'error', '1')
            return
        
        sim.setGraphStreamValue(self.graphPos, self.dataPos[i], pos*180/np.pi)
        sim.setGraphStreamValue(self.graphVel, self.dataVel[i], vel*180/np.pi)
        sim.setGraphStreamValue(self.graphAcc, self.dataAcc[i], acc*180/np.pi)
        self.sensorVel[i] = vel

def sysCall_cleanup():
    # do some clean-up here
    sim.writeCustomStringData(self.suctionHandle, 'activity', 'off')
    sim.writeCustomStringData(self.robotHandle, 'error', '0')

def move(q, state):
    if sim.readCustomStringData(self.robotHandle,'error') == '1':
        return
    # print("q",q)
    for i in range(6):
        if q[i] < self.Joint_limits[i, 0] or q[i] > self.Joint_limits[i, 1]:
            print("move(): Joint" + str(i+1) + " Position Out of Range!")
            return False
        if abs(q[i] - self.lastPos[i])/sim.getSimulationTimeStep() > self.Vel_limits[i]:
            print("move(): Joint" + str(i+1) + " Velocity Out of Range!")
            return False
        if abs(self.lastVel[i] - (q[i] - self.lastPos[i]))/sim.getSimulationTimeStep() > self.Acc_limits[i]:
            print("move(): Joint" + str(i+1) + " Acceleration Out of Range!")
            return False
    # print(q[i])
                
    self.lastPos = q
    self.lastVel = q - self.lastPos
    
    for i in range(6):
        sim.setJointTargetPosition(self.jointHandles[i], q[i])
        
    if state:
        sim.writeCustomStringData(self.suctionHandle, 'activity', 'on')
    else:
        sim.writeCustomStringData(self.suctionHandle, 'activity', 'off')
    
    return True

def quinticTraj_0_mid(start, end, t, time, v_start=None, v_end=None, a_start=None, a_end=None):
    """Quintic trajectory with no intermediate points - single segment"""
    if v_start is None:
        v_start = np.zeros_like(start)
    if v_end is None:
        v_end = np.zeros_like(end)
    if a_start is None:
        a_start = np.zeros_like(start)
    if a_end is None:
        a_end = np.zeros_like(end)

    if t < time:
        x = np.zeros_like(start)
        for i in range(len(start)):
            target_angle = _find_shortest_path(start[i], end[i])
            a0, a1, a2, a3, a4, a5 = _calculate_quintic_coefficients(
                start[i], target_angle, v_start[i], v_end[i], a_start[i], a_end[i], time
            )
            x[i] = a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t
    else:
        x = np.array(end)
    
    return x

def quinticTraj_1_mid(start, mid, end, t, time_1, time_2, v_start=None, v_end=None, a_start=None, a_end=None):
    """Quintic trajectory with 1 intermediate point - two segments"""
    if v_start is None:
        v_start = np.zeros_like(start)
    if v_end is None:
        v_end = np.zeros_like(end)
    if a_start is None:
        a_start = np.zeros_like(start)
    if a_end is None:
        a_end = np.zeros_like(end)

    # Process angles for continuity
    processed_mid = _process_angles_global(start, mid)
    processed_end = _process_angles_global(processed_mid, end)
    
    # Solve coefficients
    coeffs = _solve_global_quintic_1_mid(start, processed_mid, processed_end, 
                                       v_start, v_end, a_start, a_end, 
                                       time_1, time_2)

    if t < time_1:
        # First segment
        x = _evaluate_quintic(coeffs[0], t)
    elif t < time_1 + time_2:
        # Second segment
        x = _evaluate_quintic(coeffs[1], t - time_1)
    else:
        x = np.array(end)
    
    return x

def quinticTraj_2_mid(start, mid_1, mid_2, end, t, time_1, time_2, time_3, v_start=None, v_end=None, a_start=None, a_end=None):
    """Quintic trajectory with 2 intermediate points - three segments"""
    if v_start is None:
        v_start = np.zeros_like(start)
    if v_end is None:
        v_end = np.zeros_like(end)
    if a_start is None:
        a_start = np.zeros_like(start)
    if a_end is None:
        a_end = np.zeros_like(end)

    # Process angles
    processed_mid1 = _process_angles_global(start, mid_1)
    processed_mid2 = _process_angles_global(processed_mid1, mid_2)
    processed_end = _process_angles_global(processed_mid2, end)
    
    # Solve coefficients
    coeffs = _solve_global_quintic_2_mid(start, processed_mid1, processed_mid2, processed_end,
                                       v_start, v_end, a_start, a_end, 
                                       time_1, time_2, time_3)

    if t < time_1:
        # First segment
        x = _evaluate_quintic(coeffs[0], t)
    elif t < time_1 + time_2:
        # Second segment
        x = _evaluate_quintic(coeffs[1], t - time_1)
    elif t < time_1 + time_2 + time_3:
        # Third segment
        x = _evaluate_quintic(coeffs[2], t - time_1 - time_2)
    else:
        x = np.array(end)
    
    return x

def quinticTraj_3_mid(start, mid_1, mid_2, mid_3, end, t, time_1, time_2, time_3, time_4, v_start=None, v_end=None, a_start=None, a_end=None):
    """Quintic trajectory with 3 intermediate points - four segments"""
    if v_start is None:
        v_start = np.zeros_like(start)
    if v_end is None:
        v_end = np.zeros_like(end)
    if a_start is None:
        a_start = np.zeros_like(start)
    if a_end is None:
        a_end = np.zeros_like(end)

    # Process angles
    processed_mid1 = _process_angles_global(start, mid_1)
    processed_mid2 = _process_angles_global(processed_mid1, mid_2)
    processed_mid3 = _process_angles_global(processed_mid2, mid_3)
    processed_end = _process_angles_global(processed_mid3, end)
    
    # Solve coefficients
    coeffs = _solve_global_quintic_3_mid(start, processed_mid1, processed_mid2, processed_mid3, processed_end,
                                       v_start, v_end, a_start, a_end, 
                                       time_1, time_2, time_3, time_4)

    if t < time_1:
        # First segment
        x = _evaluate_quintic(coeffs[0], t)
    elif t < time_1 + time_2:
        # Second segment
        x = _evaluate_quintic(coeffs[1], t - time_1)
    elif t < time_1 + time_2 + time_3:
        # Third segment
        x = _evaluate_quintic(coeffs[2], t - time_1 - time_2)
    elif t < time_1 + time_2 + time_3 + time_4:
        # Fourth segment
        x = _evaluate_quintic(coeffs[3], t - time_1 - time_2 - time_3)
    else:
        x = np.array(end)
    
    return x

def _solve_global_quintic_1_mid(start, mid, end, v_start, v_end, a_start, a_end, time_1, time_2):
    """Solve global quintic coefficients for 1 intermediate point"""
    n_joints = len(start)
    coeffs = [None] * 2  # 2 segments
    
    for i in range(n_joints):
        # Build constraint matrix (12 equations)
        A = np.zeros((12, 12))
        b = np.zeros(12)
        
        # Start point constraints q v a
        A[0, 0:6] = [1, 0, 0, 0, 0, 0]  # position
        A[1, 0:6] = [0, 1, 0, 0, 0, 0]  # velocity
        A[2, 0:6] = [0, 0, 2, 0, 0, 0]  # acceleration
        b[0] = start[i]
        b[1] = v_start[i]
        b[2] = a_start[i]
        
        # Mid point (t=time_1)
        t = time_1
        A[3, 0:6] = [1, t, t**2, t**3, t**4, t**5]  # position
        A[4, 0:6] = [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]  # velocity
        A[5, 0:6] = [0, 0, 2, 6*t, 12*t**2, 20*t**3]  # acceleration
        b[3] = mid[i]
        
        # Second segment start (t=0)
        A[6, 6:12] = [1, 0, 0, 0, 0, 0]  # position
        A[4, 6:12] = [0, -1, 0, 0, 0, 0]  # velocity (continuity)
        A[5, 6:12] = [0, 0, -2, 0, 0, 0]  # acceleration (continuity)
        A[7, 6:12] = [0, 1, 0, 0, 0, 0]  # velocity
        A[8, 6:12] = [0, 0, 2, 0, 0, 0]  # acceleration
        b[6] = mid[i]
        b[7] = 0.5 * ((mid[i]- start[i])/time_1 +  (end[i]-mid[i])/time_2)  # mid velocity
        b[8] = ((end[i] - mid[i])/time_2 - (mid[i] - start[i])/time_1) / (time_1 + time_2) * 2  
        
        # End point (t=time_2)
        t = time_2
        A[9, 6:12] = [1, t, t**2, t**3, t**4, t**5]  # position
        A[10, 6:12] = [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]  # velocity
        A[11, 6:12] = [0, 0, 2, 6*t, 12*t**2, 20*t**3]  # acceleration
        b[9] = end[i]
        b[10] = v_end[i]
        b[11] = a_end[i]
        
        # Solve linear system
        
        # Solve
        x = np.linalg.solve(A, b)
        
        if i == 0:
            coeffs[0] = np.zeros((n_joints, 6))
            coeffs[1] = np.zeros((n_joints, 6))
        
        coeffs[0][i] = x[0:6]
        coeffs[1][i] = x[6:12]
    
    return coeffs

def _solve_global_quintic_2_mid(start, mid1, mid2, end, v_start, v_end, a_start, a_end, time_1, time_2, time_3):
    """Solve global quintic coefficients for 2 intermediate points"""
    n_joints = len(start)
    coeffs = [None] * 3  # 3 segments
    
    for i in range(n_joints):
        # Build constraint matrix (18 equations)
        A = np.zeros((18, 18))
        b = np.zeros(18)
        
        # First segment
        A[0, 0:6] = [1, 0, 0, 0, 0, 0]  # t=0 position
        A[1, 0:6] = [0, 1, 0, 0, 0, 0]  # t=0 velocity
        A[2, 0:6] = [0, 0, 2, 0, 0, 0]  # t=0 acceleration
        b[0] = start[i]
        b[1] = v_start[i]
        b[2] = a_start[i]
        
        t = time_1
        A[3, 0:6] = [1, t, t**2, t**3, t**4, t**5]  # t=time_1 position
        A[4, 0:6] = [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]  # velocity
        A[5, 0:6] = [0, 0, 2, 6*t, 12*t**2, 20*t**3]  # acceleration
        b[3] = mid1[i]
        b[4] = 0.5 * ( (mid1[i]-start[i])/time_1 +  (mid2[i]-mid1[i])/time_2)  # mid1 velocity
        b[5] = ((mid2[i] - mid1[i])/time_2 - (mid1[i] - start[i])/time_1) / (time_1 + time_2) * 2

        # Second segment
        A[6, 6:12] = [1, 0, 0, 0, 0, 0]  # t=0 position
        A[7, 6:12] = [0, 1, 0, 0, 0, 0]  # velocity
        A[8, 6:12] = [0, 0, 2, 0, 0, 0]  # acceleration
        b[6] = mid1[i]
        b[7] = b[4]# mid1 velocity
        b[8] = b[5]
        
        t = time_2
        A[9, 6:12] = [1, t, t**2, t**3, t**4, t**5]  # t=time_2 position
        A[10, 6:12] = [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]  # velocity
        A[11, 6:12] = [0, 0, 2, 6*t, 12*t**2, 20*t**3]  # acceleration
        b[9] = mid2[i]
        b[10] = 0.5 * ( (end[i]-mid2[i])/time_3 +  (mid2[i]-mid1[i])/time_2)  # mid2 velocity
        b[11] = ((end[i] - mid2[i])/time_3 - (mid2[i] - mid1[i])/time_2) / (time_2 + time_3) * 2
        
        # Third segment
        A[12, 12:18] = [1, 0, 0, 0, 0, 0]  # t=0 position
        A[13, 12:18] = [0, 1, 0, 0, 0, 0]  # velocity
        A[14, 12:18] = [0, 0, 2, 0, 0, 0]  # acceleration
        b[12] = mid2[i]
        b[13] = b[10]
        b[14] = b[11]
        
        t = time_3
        A[15, 12:18] = [1, t, t**2, t**3, t**4, t**5]  # t=time_3 position
        A[16, 12:18] = [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]  # velocity
        A[17, 12:18] = [0, 0, 2, 6*t, 12*t**2, 20*t**3]  # acceleration
        b[15] = end[i]
        b[16] = v_end[i]
        b[17] = a_end[i]
        
        # Solve
        x = np.linalg.solve(A, b)
        
        if i == 0:
            coeffs[0] = np.zeros((n_joints, 6))
            coeffs[1] = np.zeros((n_joints, 6))
            coeffs[2] = np.zeros((n_joints, 6))
        
        coeffs[0][i] = x[0:6]
        coeffs[1][i] = x[6:12]
        coeffs[2][i] = x[12:18]
    
    return coeffs

def _solve_global_quintic_3_mid(start, mid1, mid2, mid3, end, v_start, v_end, a_start, a_end, time_1, time_2, time_3, time_4):
    """Solve global quintic coefficients for 3 intermediate points"""
    n_joints = len(start)
    coeffs = [None] * 4  # 4 segments
    
    for i in range(n_joints):
        # Build constraint matrix (24 equations)
        A = np.zeros((24, 24))
        b = np.zeros(24)
        
        # First segment
        A[0:3, 0:6] = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]]
        b[0] = start[i]; b[1] = v_start[i]; b[2] = a_start[i]
        
        t = time_1
        A[3:6, 0:6] = [[1, t, t**2, t**3, t**4, t**5], 
                        [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
                        [0, 0, 2, 6*t, 12*t**2, 20*t**3]]
        b[3] = mid1[i]
        b[4] = 0.5 * ( (mid1[i]-start[i])/time_1 +  (mid2[i]-mid1[i])/time_2)  # mid1 velocity
        b[5] = ((mid2[i] - mid1[i])/time_2 - (mid1[i] - start[i])/time_1) / (time_1 + time_2) * 2
        
        # Second segment
        A[6:9, 6:12] = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]]
        b[6] = mid1[i]
        b[7] = b[4]
        b[8] = b[5]
        
        t = time_2
        A[9:12, 6:12] = [[1, t, t**2, t**3, t**4, t**5],
                         [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
                         [0, 0, 2, 6*t, 12*t**2, 20*t**3]]
        b[9] = mid2[i]
        b[10] = 0.5 * ( (mid3[i]-mid2[i])/time_3 +  (mid2[i]-mid1[i])/time_2)  # mid2 velocity
        b[11] = ((mid3[i] - mid2[i])/time_3 - (mid2[i] - mid1[i])/time_2) / (time_2 + time_3) * 2
        
        # Third segment
        A[12:15, 12:18] = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]]
        b[12] = mid2[i]
        b[13] = 0.5 * ( (mid3[i]-mid2[i])/time_3 +  (mid2[i]-mid1[i])/time_2)  # mid2 velocity
        b[14] = b[11]
        
        t = time_3
        A[15:18, 12:18] = [[1, t, t**2, t**3, t**4, t**5],
                           [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
                           [0, 0, 2, 6*t, 12*t**2, 20*t**3]]
        b[15] = mid3[i]
        b[16] = 0.5 * ( (end[i]-mid3[i])/time_4 +  (mid3[i]-mid2[i])/time_3)  # mid3 velocity
        b[17] = ((end[i] - mid3[i])/time_4 - (mid3[i] - mid2[i])/time_3) / (time_3 + time_4) * 2
        
        # Fourth segment
        A[18:21, 18:24] = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0]]
        b[18] = mid3[i]
        b[19] = 0.5 * ( (end[i]-mid3[i])/time_4 + (mid3[i]-mid2[i])/time_3)  # mid3 velocity
        b[20] = b[17]
        
        t = time_4
        A[21:24, 18:24] = [[1, t, t**2, t**3, t**4, t**5],
                           [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
                           [0, 0, 2, 6*t, 12*t**2, 20*t**3]]
        b[21] = end[i]; b[22] = v_end[i]; b[23] = a_end[i]
        
        # Solve
        x = np.linalg.solve(A, b)
        
        if i == 0:
            coeffs[0] = np.zeros((n_joints, 6))
            coeffs[1] = np.zeros((n_joints, 6))
            coeffs[2] = np.zeros((n_joints, 6))
            coeffs[3] = np.zeros((n_joints, 6))
        
        coeffs[0][i] = x[0:6]
        coeffs[1][i] = x[6:12]
        coeffs[2][i] = x[12:18]
        coeffs[3][i] = x[18:24]
    
    return coeffs

def _evaluate_quintic(coeffs, t):
    """Evaluate quintic polynomial at time t"""
    n_joints = coeffs.shape[0]
    x = np.zeros(n_joints)
    for i in range(n_joints):
        a0, a1, a2, a3, a4, a5 = coeffs[i]
        x[i] = a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t
    return x

def _process_angles_global(current, target):
    # 190->90  
    processed = np.array(target, dtype=float)
    for i in range(0,5):
        diff = target[i] - current[i]
        # For all joints, first apply basic shortest path correction (original logic retained)
        if diff > math.pi:
            processed[i] = target[i] - 2*math.pi
        elif diff < -math.pi:
            processed[i] = target[i] + 2*math.pi
        
        # For joint 6 (index 5), add "reverse prediction" with 10° margin (π/18 radians)
        if i == 5:  # Only process joint 6
            # If corrected angle is close to 180° (π), force switch to reverse direction
            if processed[i] > math.pi - math.pi/18:  # Exceeds 170°
                processed[i] -= 2*math.pi  # Switch to -190° → actually equivalent to 170°, but reverse motion direction
            elif processed[i] < -math.pi + math.pi/18:  # Below -170°
                processed[i] += 2*math.pi  # Switch to 190° → actually equivalent to -170°, but reverse motion direction
    return processed

def _find_shortest_path(current, target):
#     """Find shortest path for angle"""
     diff = target - current
#     while diff > math.pi:
#         diff -= 2 * math.pi
#     while diff < -math.pi:
#         diff += 2 * math.pi
     return current + diff

def _calculate_quintic_coefficients(p0, p1, v0, v1, a0, a1, T):
    """Calculate quintic polynomial coefficients"""
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    
    a0_coeff = p0
    a1_coeff = v0
    a2_coeff = a0 / 2.0
    a3_coeff = (20*(p1 - p0) - (8*v1 + 12*v0)*T - (3*a0 - a1)*T2) / (2*T3)
    a4_coeff = (30*(p0 - p1) + (14*v1 + 16*v0)*T + (3*a0 - 2*a1)*T2) / (2*T4)
    a5_coeff = (12*(p1 - p0) - (6*v1 + 6*v0)*T - (a1 - a0)*T2) / (2*T5)
    
    return a0_coeff, a1_coeff, a2_coeff, a3_coeff, a4_coeff, a5_coeff

def calculate_jacobian(joints):
    # Calculate Jacobian matrix
    t1, t2, t3, t4, t5, _ = joints
    sin_t1, cos_t1 = math.sin(t1), math.cos(t1)
    sin_t2, cos_t2 = math.sin(t2), math.cos(t2)
    sin_t3, cos_t3 = math.sin(t3), math.cos(t3)
    sin_t4, cos_t4 = math.sin(t4), math.cos(t4)
    sin_t5, cos_t5 = math.sin(t5), math.cos(t5)
    s = t2 + t3 + t4
    sin_s, cos_s = math.sin(s), math.cos(s)
    t2t3 = t2 + t3
    sin_t2t3, cos_t2t3 = math.sin(t2t3), math.cos(t2t3)
    c171_2, c77, c185, c170, c23 = 85.5, 77.0, 185.0, 170.0, 23.0
    
    J = np.zeros((6, 6), dtype=np.float64)
    
    J[0,0] = c77*sin_t1*sin_t2*sin_t3*sin_t4 - c171_2*cos_t1*sin_t5 - c185*sin_t1*sin_t2 - c170*cos_t2*sin_t1*sin_t3 - c170*cos_t3*sin_t1*sin_t2 - c77*cos_t2*cos_t3*sin_t1*sin_t4 - c77*cos_t2*cos_t4*sin_t1*sin_t3 - c77*cos_t3*cos_t4*sin_t1*sin_t2 - c23*cos_t1 - c171_2*cos_t2*cos_t3*cos_t4*cos_t5*sin_t1 + c171_2*cos_t2*cos_t5*sin_t1*sin_t3*sin_t4 + c171_2*cos_t3*cos_t5*sin_t1*sin_t2*sin_t4 + c171_2*cos_t4*cos_t5*sin_t1*sin_t2*sin_t3
    J[0,1] = cos_t1*(c185*cos_t2 + c170*cos_t2t3 + c77*cos_s - c171_2*sin_s*cos_t5)
    J[0,2] = cos_t1*(c170*cos_t2t3 + c77*cos_s - c171_2*sin_s*cos_t5)
    J[0,3] = cos_t1*(c77*cos_s - c171_2*sin_s*cos_t5)
    J[0,4] = -c171_2*(cos_t1*sin_t5*cos_s + cos_t5*sin_t1)
    J[0,5] = 0.0
    
    J[1,0] = c185*cos_t1*sin_t2 - c23*sin_t1 - c171_2*sin_t1*sin_t5 + c170*cos_t1*cos_t2*sin_t3 + c170*cos_t1*cos_t3*sin_t2 + c77*cos_t1*cos_t2*cos_t3*sin_t4 + c77*cos_t1*cos_t2*cos_t4*sin_t3 + c77*cos_t1*cos_t3*cos_t4*sin_t2 - c77*cos_t1*sin_t2*sin_t3*sin_t4 + c171_2*cos_t1*cos_t2*cos_t3*cos_t4*cos_t5 - c171_2*cos_t1*cos_t2*cos_t5*sin_t3*sin_t4 - c171_2*cos_t1*cos_t3*cos_t5*sin_t2*sin_t4 - c171_2*cos_t1*cos_t4*cos_t5*sin_t2*sin_t3
    J[1,1] = sin_t1*(c185*cos_t2 + c170*cos_t2t3 + c77*cos_s - c171_2*sin_s*cos_t5)
    J[1,2] = sin_t1*(c170*cos_t2t3 + c77*cos_s - c171_2*sin_s*cos_t5)
    J[1,3] = sin_t1*(c77*cos_s - c171_2*sin_s*cos_t5)
    J[1,4] = c171_2*(cos_t1*cos_t5 - sin_t1*sin_t5*cos_s)
    J[1,5] = 0.0
    
    J[2,0] = 0.0
    J[2,1] = -(c185*sin_t2 + c170*sin_t2t3 + c77*sin_s + c171_2*cos_s*cos_t5)
    J[2,2] = -(c170*sin_t2t3 + c77*sin_s + c171_2*cos_s*cos_t5)
    J[2,3] = -(c77*sin_s + c171_2*cos_s*cos_t5)
    J[2,4] = c171_2*sin_s*sin_t5
    J[2,5] = 0.0
    
    J[3,0] = 0.0
    J[3,1] = -sin_t1
    J[3,2] = -sin_t1
    J[3,3] = -sin_t1
    J[3,4] = (math.sin(t2 - t1 + s) + math.sin(t1 + s))/2
    J[3,5] = cos_t1*cos_t5*cos_s - sin_t1*sin_t5
    
    J[4,0] = 0.0
    J[4,1] = cos_t1
    J[4,2] = cos_t1
    J[4,3] = cos_t1
    J[4,4] = (math.cos(t2 - t1 + s) - math.cos(t1 + s))/2
    J[4,5] = cos_t1*sin_t5 + sin_t1*cos_t5*cos_s
    
    J[5,0] = 1.0
    J[5,1] = 0.0
    J[5,2] = 0.0
    J[5,3] = 0.0
    J[5,4] = cos_s
    J[5,5] = -sin_s*cos_t5
    
    return J