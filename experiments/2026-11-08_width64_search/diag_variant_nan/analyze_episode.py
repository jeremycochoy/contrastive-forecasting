"""Print the replay's records of each step in a range: the losses, the 6 L_align terms, the largest gradient and update
groups, the input batch, and (in the probe windows) the max |output| and max |gradient| of the key modules.

Usage: python3 analyze_episode.py <first step> <last step>
"""
import os
import csv
import json
import sys

D = os.environ.get("HP64_ROOT", "/home/jupyter/cf_runs/hp64") + "/diag_variant_nan"
NAME = "g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10pw2_rollfix"
KEY_FWD = ["rev_norm", "encoder", "transformer.encoder_layers.0", "transformer.fcst_down_proj", "transformer.layers.0",
           "transformer.layers.2", "transformer.fcst_up_proj", "teacher_input_to_latent", "teacher_encoder_layers.0"]
KEY_BWD = ["encoder", "transformer.encoder_layers.0", "transformer.fcst_down_proj", "transformer.layers.0",
           "transformer.layers.2", "transformer.fcst_up_proj"]


def by_tag(records):
    out = {}
    for tag, value in records:
        base = tag.replace(" (again, in backward)", "")
        out[base] = max(out.get(base, 0.0), value) if value == value else float("nan")
    return out


def main(first, last):
    losses = {int(r["step"]): r for r in csv.DictReader(open(f"{D}/ckpt/{NAME}_losses.csv"))}
    steps = {int(r["step"]): r for r in csv.DictReader(open(f"{D}/steps.csv"))}
    probe = {}
    for line in open(f"{D}/window_probe.jsonl"):
        p = json.loads(line)
        if first <= p["step"] <= last:
            probe[p["step"]] = (by_tag(p["fwd"]), by_tag(p["bwd"]))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    print("step  loss  d1    d2    d4    d8    d16   d32  | sig_e sig_h u_bt_e | grad  top grad groups | x_max  ratio flat")
    for s in range(first, last + 1):
        if s not in steps:
            continue
        r, l = steps[s], losses.get(s, {})
        groups = sorted(((k[5:], f(r, k)) for k in r if k.startswith("grad:")), key=lambda t: -t[1])[:3]
        print(f"{s} {f(r, 'loss'):6.2f} {f(l, 'l_align'):.3f} " + " ".join(f"{f(r, f'align_d{d}'):.3f}"
              for d in (2, 4, 8, 16, 32))
              + f" | {f(l, 'sigreg_e'):.3f} {f(l, 'sigreg_h'):.3f} {f(l, 'u_batchtime_e'):.3f} | "
              + f"{f(r, 'grad_total'):6.2f} " + ", ".join(f"{g.replace('transformer.', 't.')} {v:.2f}"
                                                         for g, v in groups)
              + f" | {f(r, 'x_max_abs'):.3g} {f(r, 'x_row_maxratio_max'):.3g} {f(r, 'x_rows_flat'):.0f}")
        if s in probe:
            fwd, bwd = probe[s]
            print("      fwd max: " + ", ".join(f"{t.replace('transformer.', 't.')} {fwd.get(t, float('nan')):.3g}"
                                             for t in KEY_FWD))
            print("      grad max: " + ", ".join(f"{t.replace('transformer.', 't.')} {bwd.get(t, float('nan')):.3g}"
                                              for t in KEY_BWD))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
