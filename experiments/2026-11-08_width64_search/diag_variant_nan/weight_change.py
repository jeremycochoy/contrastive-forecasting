"""Per-tensor weight change between two saves of a run: ||B - A|| / ||A|| per 1k steps, and ||B|| / ||A||.

Usage: python3 weight_change.py  (prints the variant 10k -> NaN saves, red and 10-L-ALIGN 10k -> 15k)
"""
import os
import torch

H = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64")
P = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere"
PAIRS = {  # label: (save A, save B, steps between them)
    "variant pass 2": (f"{P}_align10pw2_rollfix/ckpt/{P}_align10pw2_rollfix_10k.pth",
                       f"{P}_align10pw2_rollfix/ckpt/{P}_align10pw2_rollfix_r2_EMERGENCY_13756.pth", 3756),
    "variant pass 1": (f"{P}_align10pw2_rollfix/ckpt/{P}_align10pw2_rollfix_10k.pth",
                       f"{P}_align10pw2_rollfix/ckpt/{P}_align10pw2_rollfix_EMERGENCY_12325.pth", 2325),
    "red": (f"{P}_sigf_align10pw2_rollfix/ckpt/{P}_sigf_align10pw2_rollfix_10k.pth",
            f"{P}_sigf_align10pw2_rollfix/ckpt/{P}_sigf_align10pw2_rollfix_15k.pth", 5000),
    "10-L-ALIGN": (f"{P}_align10_rollfix/ckpt/{P}_align10_rollfix_10k.pth",
                   f"{P}_align10_rollfix/ckpt/{P}_align10_rollfix_15k.pth", 5000),
}


def load(path):
    return {k: v.float() for k, v in torch.load(f"{H}/{path}", map_location="cpu", weights_only=False).items()
            if torch.is_tensor(v) and v.is_floating_point()}


def changes(label):
    a_path, b_path, steps = PAIRS[label]
    a, b = load(a_path), load(b_path)
    bad = [k for k, v in b.items() if not torch.isfinite(v).all()]
    out = {}
    for k in a:
        if k in b and a[k].norm() > 0:
            out[k] = ((b[k] - a[k]).norm() / a[k].norm() * 1000 / steps, b[k].norm() / a[k].norm(), b[k].norm())
    return out, bad


def main():
    res = {label: changes(label) for label in PAIRS}
    for label, (_, bad) in res.items():
        print(f"{label}: non-finite tensors in save B: {len(bad)} {bad[:5]}")
    var = res["variant pass 2"][0]
    top = sorted(var, key=lambda k: -var[k][0])[:22]
    print("tensor | relative change per 1k steps (variant pass 2, pass 1, red, 10-L-ALIGN) | norm ratio B/A (same order) | norm B (variant pass 2)")
    for k in top:
        rel = " ".join(f"{res[l][0][k][0]:.3f}" if k in res[l][0] else "  -  " for l in PAIRS)
        ratio = " ".join(f"{res[l][0][k][1]:.3f}" if k in res[l][0] else "  -  " for l in PAIRS)
        print(f"{k:60s} | {rel} | {ratio} | {var[k][2]:.2f}")


if __name__ == "__main__":
    main()
