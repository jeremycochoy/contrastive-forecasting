#!/usr/bin/env python3
"""#328 disentanglement scoreboard — GM + paired-bootstrap Δ vs base for every
arm at best AND last (full-97). GM/bootstrap logic copied verbatim from analyze.py."""
import math, os, random

CF = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
BASE_RES = f"{CF}/2026-05-29_forked_6Lf_b1024/results"
TRIP_RES = f"{CF}/2026-06-03_crossfade_triplet/results"
BASE_TAG = "xshh_allt_forked2_qk_aon_b1024"
HEADS = ["2L", "6L"]
ARMS = [  # display, tag
    ("L3",                       "allt08_L3_qk_aon_b1024"),
    ("L6+nobn",                  "allt08_nobn_qk_aon_b1024"),
    ("L3+nobn",                  "allt08_L3_nobn_qk_aon_b1024"),
    ("L3+nobn+triplet",          "allt08_xftrip_nobn_enc3_qk_aon_b1024"),
    ("L3+nobn+triplet @25k",     "allt08_xftrip_nobn_enc3_qk_aon_b1024_to25000"),
    ("L6+nobn+triplet",          "allt08_xftrip_nobn_enc6_qk_aon_b1024"),
    ("base+triplet",             "allt08_xftrip_bn_enc6_qk_aon_b1024"),
]

def relatives(sum_txt):
    out = {}
    if not os.path.exists(sum_txt): return out
    for line in open(sum_txt):
        p = line.split()
        if len(p) == 4 and "/" in p[0]:
            try: out[p[0]] = float(p[3])
            except ValueError: pass
    return out

def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None

def paired_delta_ci(da, db, n=2000, seed=0):
    common = sorted(set(da) & set(db))
    if len(common) < 2: return (None, None, None, 0)
    a = [da[c] for c in common]; b = [db[c] for c in common]
    delta = gm(b) - gm(a); rng = random.Random(seed); ds = []
    for _ in range(n):
        idx = [rng.randrange(len(common)) for _ in common]
        ds.append(gm([b[i] for i in idx]) - gm([a[i] for i in idx]))
    ds.sort()
    return (delta, ds[int(0.05*n)], ds[int(0.95*n)], len(common))

def rels(res, tag, head): return relatives(f"{res}/gift_eval_full_{tag}_{head}/summary.txt")

def verdict(lo, hi):
    if lo is None: return "?"
    if hi < 0: return "BETTER (reliable)"
    if lo > 0: return "worse (reliable)"
    return "ns (CI straddles 0)"

for head in HEADS:
    base = rels(BASE_RES, BASE_TAG, head)
    print(f"\n===== {head} head =====  base GM = {gm(list(base.values())):.4f}  (n_configs={len(base)})")
    print(f"{'arm':<28}{'ckpt':<6}{'GM':>8}{'Δ vs base':>11}{'  90% CI':>20}   verdict")
    for name, tag in ARMS:
        for ck, t in [("best", tag), ("last", tag + "_last")]:
            r = rels(TRIP_RES, t, head)
            if not r:
                print(f"{name:<28}{ck:<6}{'  running/NA':>8}"); continue
            g = gm(list(r.values()))
            d, lo, hi, npair = paired_delta_ci(base, r)
            ci = f"({lo:+.3f},{hi:+.3f})" if lo is not None else "(NA)"
            print(f"{name:<28}{ck:<6}{g:>8.4f}{d:>+11.4f}{ci:>20}   {verdict(lo,hi)}")
