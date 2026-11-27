"""freq_family: train the heads of a wave with TF32 on in matmul and in cuDNN.

The user, 2026-11-23: "well we need if faster." TF32 rounds the inputs of each
matrix product to 10 bits and keeps fp32 sums, on the tensor cores. Heads of
500 steps scored within 0.02 % of the fp32 heads (GM-Relative MASE, 11
configs) at 1.4 to 1.8 times the speed. The mirror of tf32_off.py: it runs
train_forecasting_heads_shared.py of the code folder (the first entry of
PYTHONPATH, as follow_multi.sh sets it) with the same flags.

Usage, in the environment of a follower:
    FF_TRAINER=<this file> bash follow_multi.sh
"""
import os
import runpy
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
code = os.environ["PYTHONPATH"].split(os.pathsep)[0]
script = os.path.join(code, "experiments/2026-04-13_gift-eval/scripts/train_forecasting_heads_shared.py")
sys.path.insert(0, os.path.dirname(script))
sys.argv = [script] + sys.argv[1:]
print("train_heads_tf32: TF32 on in matmul and cuDNN", flush=True)
runpy.run_path(script, run_name="__main__")
