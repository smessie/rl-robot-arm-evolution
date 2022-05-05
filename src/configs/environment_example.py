import psutil

# Copy this file to env.py and then adjust all settings to your wishes, in that env.py file.
PATH_TO_UNITY_EXECUTABLE = '../build/simenv.x86_64'
PATH_TO_ROBOT_URDF = "environment/robot.urdf"
MORPHEVO_USE_GRAPHICS = False
RL_USE_GRAPHICS_TRAINING = False
RL_USE_GRAPHICS_TESTING = True
NUM_CORES = psutil.cpu_count()
MODULES_MAY_ROTATE = True
MODULES_MAY_TILT = True
