"""freq_family: run a script on the GPU with TF32 off in cuDNN and in matmul.

b4_probes.sh uses it to test if TF32 is the cause of the difference between
the B4 score of the GPU and of the CPU.

Usage: python3 tf32_off.py <script> [flags of the script]
"""
import runpy
import sys

import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
