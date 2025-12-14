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

while (t := sim.getSimulationTime()) < 3:
    sim.setJointPosition(l_joint1, 2 * np.pi * t / 3)

    message = f'Simulation time: {t:.2f} s'
    print(message)
    sim.addLog(sim.verbosity_scriptinfos, message)
    client.step()  # triggers next simulation step
    time.sleep(0.01)

# Stop simulation
sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

print('Program ended')