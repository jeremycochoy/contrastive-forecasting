"""Run train.py with read-only probes, to see the dynamics before a NaN (the variant 'red without SIGReg-F').

Usage (the run's environment, CF_NAN_DEBUG=1 and the train.py arguments of the run):
  CF_NAN_DEBUG=1 CF_NAN_DEBUG_DUMP=<dump.pt> DIAG_OUT=<folder> DIAG_WINDOWS='10440-10620,11880-12060' \
    python3 -u diag_replay.py <train.py> <train.py arguments>

Every step, <folder>/steps.csv gets: the loss, L_align at the distances 2, 4, 8, 16, 32 (the CSV of train.py logs
only distance 1), the gradient norm, the update norm and the weight norm of each module group, and statistics of
the input batch. In the windows, <folder>/window_probe.jsonl gets the max |output| and the max |gradient| of every
module call (the probe of CF_NAN_DEBUG, in call order) and <folder>/window_rows.jsonl the rows with the largest
L_align at distance 1. Each probe only reads tensors: the training numbers stay the same (compare the losses CSV of
the replay with the run's).
"""
import csv
import json
import os
import runpy
import sys

import torch

import src.loss as loss_mod
import src.nan_debug as nd

OUT = os.environ["DIAG_OUT"]
WINDOWS = [tuple(int(v) for v in w.split("-")) for w in os.environ.get("DIAG_WINDOWS", "").split(",") if w]
DISTANCES = (2, 4, 8, 16, 32)  # the copies after distance 1, in the order of the loss
ALIGN_WEIGHT = 10.0
TOP_ROWS = 8
S = {"row": None, "model": None, "groups": None, "params": None, "writer": None, "files": []}


def in_window(step):
    return any(a <= step <= b for a, b in WINDOWS)


def group_of(name):
    """The module group of a parameter: the attention, FFN and norms of each layer apart."""
    p = name.split(".")
    if p[0] == "transformer" and len(p) > 3 and p[1] in ("layers", "encoder_layers"):
        return ".".join(p[:4])
    return ".".join(p[:3]) if p[0] == "transformer" else ".".join(p[:2])


def group_norms(tensors):
    """{group: norm} over the tensors of each group, with one device sync."""
    names = S["groups"]
    by_group = {g: [] for g in names}
    for n, t in tensors:
        if t is not None:
            by_group[group_of(n)].append(t.detach().float().pow(2).sum())
    zero = torch.zeros((), device=next(S["model"].parameters()).device)
    total = torch.stack([torch.stack(v).sum() if v else zero for v in by_group.values()])
    values = total.sqrt().cpu().tolist()
    return dict(zip(names, values)), float(total.sum().sqrt())


# --- the probes ---------------------------------------------------------------------------------------------

orig_reduce = loss_mod.reduce_depth_terms


def reduce_depth_terms(terms, reduce):
    row = S["row"]
    if (row is not None and "align" not in row and len(terms) == 1 + len(DISTANCES)
            and torch.is_grad_enabled() and terms[1].requires_grad):
        row["align"] = [t.detach().float() for t in terms[1:]]
    return orig_reduce(terms, reduce)


loss_mod.reduce_depth_terms = reduce_depth_terms
orig_cll = loss_mod.contrastive_latent_loss


def contrastive_latent_loss(latents, *args, **kwargs):
    """In the windows, the outer call: L_align at distance 1 per row (read only)."""
    row = S["row"]
    if (row is not None and row.get("window") and "rows" not in row and kwargs.get("depth_index") in (None, 0)
            and kwargs.get("rollout_latents") and kwargs.get("teacher_original_latent") is not None):
        with torch.no_grad():
            f, teacher = latents[0], kwargs["teacher_original_latent"]
            cos = torch.nn.functional.cosine_similarity(f[:, :-1].float(), teacher[:, 1:].float(), dim=-1)
            per_pair = 2.0 - 2.0 * cos  # [B, T-1, C]
            pad = kwargs.get("pad_patches")
            keep = (torch.ones_like(per_pair, dtype=torch.bool) if pad is None else ~pad[:, :-1])
            per_row = (per_pair * keep).sum(dim=(1, 2)) / keep.sum(dim=(1, 2)).clamp(min=1)
            row["rows"] = per_row.cpu()
    return orig_cll(latents, *args, **kwargs)


loss_mod.contrastive_latent_loss = contrastive_latent_loss

ND = nd.NanDebug
orig = {k: getattr(ND, k) for k in ("__init__", "start_step", "note_batch", "loss_is_bad", "grads_are_bad",
                                    "weights_are_bad")}


def nd_init(self, model, optimizer, dump_path, inject=None):
    orig["__init__"](self, model, optimizer, dump_path, inject)
    S["model"] = model
    S["params"] = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    S["groups"] = sorted({group_of(n) for n, _ in S["params"]})
    os.makedirs(OUT, exist_ok=True)
    f = open(os.path.join(OUT, "steps.csv"), "a", newline="")
    cols = (["step", "loss", "lr"] + [f"align_d{d}" for d in DISTANCES] + ["grad_total"]
            + [f"grad:{g}" for g in S["groups"]] + [f"update:{g}" for g in S["groups"]]
            + [f"weight:{g}" for g in S["groups"]]
            + ["x_max_abs", "x_row_std_min", "x_row_std_max", "x_row_maxratio_max", "x_rows_flat"])
    S["writer"] = csv.DictWriter(f, fieldnames=cols)
    if f.tell() == 0:
        S["writer"].writeheader()
    S["files"] = [f, open(os.path.join(OUT, "window_probe.jsonl"), "a"),
                  open(os.path.join(OUT, "window_rows.jsonl"), "a")]
    print(f"[diag] probes on: {len(S['params'])} parameters in {len(S['groups'])} groups, windows {WINDOWS}",
          flush=True)


def write_row():
    row = S["row"]
    if row is None:
        return
    S["row"] = None
    out = {"step": row["step"], "loss": row.get("loss"), "lr": row.get("lr")}
    if "align" in row:
        values = torch.stack(row["align"]).cpu().tolist()
        out.update({f"align_d{d}": v / ALIGN_WEIGHT for d, v in zip(DISTANCES, values)})
    out.update(row.get("csv", {}))
    S["writer"].writerow(out)
    if row.get("window"):
        S["files"][1].write(json.dumps({"step": row["step"], "fwd": row.get("fwd", []),
                                        "bwd": row.get("bwd", [])}) + "\n")
        if "rows" in row:
            per_row = row["rows"]
            top = per_row.argsort(descending=True)[:TOP_ROWS].tolist()
            batch = row.get("batch_rows", {})
            S["files"][2].write(json.dumps({
                "step": row["step"], "align_d1_mean": float(per_row.mean()),
                "top": [{"row": i, "align_d1": float(per_row[i]),
                         **{k: (v[i] if isinstance(v, list) else v) for k, v in batch.items()}} for i in top]})
                + "\n")
    if row["step"] % 50 == 0 or row.get("window"):
        for f in S["files"]:
            f.flush()


def nd_start_step(self, step):
    write_row()
    S["row"] = {"step": step, "window": in_window(step), "csv": {},
                "lr": self.optimizer.param_groups[0]["lr"]}
    orig["start_step"](self, step)
    if not S["row"]["window"]:
        nd.PROBE.active = False  # the module probe only in the windows


def nd_note_batch(self, **items):
    orig["note_batch"](self, **items)
    x = items.get("values")
    row = S["row"]
    if row is None or x is None or "x_max_abs" in row["csv"]:
        return
    with torch.no_grad():
        flat = x.detach().float().reshape(x.shape[0], -1)
        amax, std = flat.abs().amax(dim=1), flat.std(dim=1)
        ratio = amax / std.clamp(min=1e-12)
        stats = torch.stack([amax.max(), std.min(), std.max(), ratio.max(), (std < 1e-6).sum().float()]).cpu()
        row["csv"].update(dict(zip(["x_max_abs", "x_row_std_min", "x_row_std_max", "x_row_maxratio_max",
                                    "x_rows_flat"], stats.tolist())))
        if row["window"]:
            batch = {"x_max_abs": amax.cpu().tolist(), "x_std": std.cpu().tolist()}
            for key in ("freq_ids", "seasonality_ids"):
                if torch.is_tensor(items.get(key)):
                    batch[key] = items[key].detach().reshape(-1).cpu().tolist()[:x.shape[0]]
            if items.get("row_kind") is not None:
                batch["row_kind"] = list(items["row_kind"])
            if torch.is_tensor(items.get("sign_flipped")):
                batch["sign_flipped"] = items["sign_flipped"].detach().reshape(-1).cpu().int().tolist()
            row["batch_rows"] = batch


def nd_loss_is_bad(self, loss):
    if S["row"] is not None:
        S["row"]["loss"] = loss
    bad = orig["loss_is_bad"](self, loss)
    if bad:
        write_row()
        flush_all()
    return bad


def nd_grads_are_bad(self, loss):
    row = S["row"]
    if row is not None:
        norms, total = group_norms([(n, p.grad) for n, p in S["params"]])
        row["csv"]["grad_total"] = total
        row["csv"].update({f"grad:{g}": v for g, v in norms.items()})
        if row["window"]:
            row["fwd"] = [[t, v] for t, v in nd._maxima(nd.PROBE.forward)]
            row["bwd"] = [[t, v] for t, v in nd._maxima(nd.PROBE.backward)]
    bad = orig["grads_are_bad"](self, loss)
    if bad:
        write_row()
        flush_all()
    return bad


def nd_weights_are_bad(self, loss, grad_norm=None):
    bad = orig["weights_are_bad"](self, loss, grad_norm)
    row = S["row"]
    if row is not None and self.before is not None:
        upd, _ = group_norms([(n, p.detach().float() - self.before[n].float()) for n, p in S["params"]])
        w, _ = group_norms([(n, p) for n, p in S["params"]])
        row["csv"].update({f"update:{g}": v for g, v in upd.items()})
        row["csv"].update({f"weight:{g}": v for g, v in w.items()})
    if bad:
        write_row()
        flush_all()
    return bad


def flush_all():
    for f in S["files"]:
        f.flush()


ND.__init__, ND.start_step, ND.note_batch = nd_init, nd_start_step, nd_note_batch
ND.loss_is_bad, ND.grads_are_bad, ND.weights_are_bad = nd_loss_is_bad, nd_grads_are_bad, nd_weights_are_bad

if __name__ == "__main__":
    if os.environ.get("CF_NAN_DEBUG") != "1":
        raise SystemExit("diag_replay.py needs CF_NAN_DEBUG=1: its probes ride on the NaN debug mode")
    train_py = sys.argv[1]
    sys.argv = [train_py] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(train_py)))  # as a plain run of train.py
    try:
        runpy.run_path(train_py, run_name="__main__")
    finally:
        write_row()
        flush_all()
