import os
import ctypes

dll_dir = "C:/ProgramData/miniconda3/envs/ai/Lib/site-packages/mujoco/plugin/sdf_plugin.dll"
dll_dir2 = "C:/ProgramData/miniconda3/envs/ai/Lib/site-packages/mujoco/mujoco.dll"
print(os.path.exists(dll_dir))
print(os.path.exists(dll_dir2))

try:
    ctypes.CDLL(dll_dir2)
except:
    print("Finished 1")

try:
    ctypes.CDLL(dll_dir)
except:
    print("Finished 2")