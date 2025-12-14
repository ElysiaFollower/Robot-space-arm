# Before running this example, make sure you have installed the following package.
# pip install coppeliasim-zmqremoteapi-client numpy
# You can find more information about ZeroMQ remote API 
# in the file path <Coppeliasim_install_path>\programming\zmqRemoteApi
# or on https://github.com/CoppeliaRobotics/zmqRemoteApi
#
# You can get more API about coppleliasim on https://coppeliarobotics.com/helpFiles/en/apiFunctions.htm

import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import utils

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
response = client.call('sim.setStepping', [True])
print('Response from sim.setStepping:', response)
client.setStepping(True)
sim.startSimulation()

# Get object handle
l_joint1 = sim.getObject('./L_Joint1')
l_joint2 = sim.getObject('./L_Joint2')
l_joint3 = sim.getObject('./L_Joint3')
joint4 = sim.getObject('./Joint4')
r_joint1 = sim.getObject('./R_Joint1')
r_joint2 = sim.getObject('./R_Joint2')
r_joint3 = sim.getObject('./R_Joint3')

l_base = sim.getObject('./L_Base')
l_link1 = sim.getObject('./L_Link1')
l_link2 = sim.getObject('./L_Link2')
l_link3 = sim.getObject('./L_Link3')
r_base = sim.getObject('./R_Base')
r_link1 = sim.getObject('./R_Link1')
r_link2 = sim.getObject('./R_Link2')
r_link3 = sim.getObject('./R_Link3')

joints = [l_joint1, l_joint2, l_joint3, joint4, r_joint1, r_joint2, r_joint3]
links = [l_base, l_link1, l_link2, l_link3, r_link1, r_link2, r_link3, r_base]



# =================================
# main control logic begin
# =================================
while (t := sim.getSimulationTime()) < 3:
    sim.setJointPosition(joints[0], 2 * np.pi * t / 3)
    sim.setJointPosition(l_joint2, 2 * np.pi * t / 3)

    message = f'Simulation time: {t:.2f} s'
    print(message)
    sim.addLog(sim.verbosity_scriptinfos, message)
    if t >= 1.00 and t <= 1.05:
        utils.switch_base(sim, joints, links, base="R")
    client.step()  # triggers next simulation step
    time.sleep(0.01)
# =================================
# main control logic end
# =================================

# Stop simulation
sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

print('Program ended')