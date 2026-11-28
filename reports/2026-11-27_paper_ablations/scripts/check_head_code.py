"""Compare the backbone that two score-code folders build for one checkpoint (#482).

The rows 6 and 8 of #482 score with a patched copy of the score code of #425
(make_head_code.sh). This check builds the frozen backbone of a checkpoint
with the head trainer (backbone_config, load_frozen_backbone) and with the
GIFT-Eval script (load_models) of each folder, with the flags of a head job of
follow_multi.sh. It prints whether the two folders give the same config and
the same weights. Each folder runs in its own process: both hold a package
`src`.

Usage:
    python3 check_head_code.py --old <code> --new <code> --ckpt <backbone .pth>
"""
import argparse
import json
import os
import subprocess
import sys

SHAPE = ("--t-raw 4096 --n-channels 1 --d-model 320 --n-heads 8 --num-layers 6 "
         "--encoder-type gru --rev-norm-kind ewma --rev-norm-span 128").split()
CHILD = r'''
import hashlib, importlib.util, json, sys, torch
from unittest.mock import patch
code, ckpt, shape = sys.argv[1], sys.argv[2], sys.argv[3:]
gift = code + "/experiments/2026-04-13_gift-eval/scripts/"

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

head = load(gift + "train_forecasting_head.py", "head")
args = head.parse_args(["--backbone-path", ckpt, "--device", "cpu",
                        "--freq-emb-dim", "3", "--seasonality-emb-dim", "3", *shape])
sd = torch.load(ckpt, map_location="cpu", weights_only=True)
config, _ = head.backbone_config(args, sd)
head_model = head.load_frozen_backbone(args, config, sd, "cpu")
ev = load(gift + "eval_gift_eval_official.py", "ev")
with patch.object(sys, "argv", ["eval", "--backbone-path", ckpt, "--zero-head",
                                "--strategy", "R", "--device", "cpu", *shape]):
    eargs = ev.parse_args()
eval_model, _ = ev.load_models(eargs, torch.device("cpu"))
print("RESULT " + json.dumps({
    "head_config": {k: str(v) for k, v in sorted(config.items())},
    "head_weights": digest(head_model),
    "eval_config": {k: str(v) for k, v in sorted(ev.BACKBONE_CONFIG.items())},
    "eval_weights": digest(eval_model)}))
'''


def build(code, ckpt):
    """The configs and the weight digests that the folder ``code`` builds."""
    env = dict(os.environ, PYTHONPATH=code, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", CHILD, code, ckpt, *SHAPE],
                       capture_output=True, text=True, env=env, cwd=code, timeout=1800)
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
    assert r.returncode == 0 and line, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(line[-1][len("RESULT "):])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--ckpt", required=True)
    a = p.parse_args()
    old, new = build(a.old, a.ckpt), build(a.new, a.ckpt)
    for part in ("head_config", "head_weights", "eval_config", "eval_weights"):
        same = old[part] == new[part]
        print(f"{part}: {'SAME' if same else 'DIFFERENT'}")
        if not same and part.endswith("config"):
            for key in sorted(set(old[part]) | set(new[part])):
                if old[part].get(key) != new[part].get(key):
                    print(f"  {key}: {old[part].get(key)} -> {new[part].get(key)}")
    print(f"encoder_type {new['head_config']['encoder_type']}, "
          f"depthwise_conv {new['head_config']['depthwise_conv']}")
    return 0 if old == new else 1


if __name__ == "__main__":
    sys.exit(main())
