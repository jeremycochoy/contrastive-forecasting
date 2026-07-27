#!/usr/bin/env python3
"""Per-family logit magnitudes + gradient shares for #374 (arms 1/3/4).

Loads backbone checkpoints, pushes two fixed batches through the
five-negative-family decomposition shared by
`cosine_similarity_batch_split_pred_rep` and
`cosine_similarity_batch_full_hh_negs_xshh_allt` (src/loss.py), and
reports per-tensor mean / max plus the per-anchor gradient share
exp((tensor_i - log_denom).mean()) at tau=0.10.

Batches (both fixed by --seed):
  * mixed    — first --batch-size windows of the training HF stream
               (jeremycochoy/gift-pretrain-full-4096 / small_v1), offset
               by seed %% 100000 rows.
  * periodic — solar/H + electricity/H series from the local GIFT-Eval
               catalog (the tasks training was benchmarked against),
               sliced into non-overlapping T=4096 windows. The pretrain
               dataset's source_id column cannot isolate solar /
               electricity, hence the catalog fallback.

Measurement-time simplifications (read the numbers accordingly):
  * NO MoCo swap: the cross-batch f<->h keys come from the STUDENT's
    h_{t+1} even for the --moco-negatives arms (3, 4). We characterize
    the objective landscape as the student sees it; the EMA teacher is
    not run at measurement time.
  * log_pos likewise uses the student's h_{t+1} (training used the EMA
    teacher positive).
  * model.eval(): encoder dropkey (0.70 during training), dropout, and
    train-time augmentations (sign flip, mixup, synth/crossfade rows)
    are all off.
  * log_neg_xx (cross-channel h<->h) is identically -inf at C=1; its
    rows carry "n/a" but it still enters the denominators as a no-op,
    exactly as in training.

Denominators: the pooled shape reports every family's share against the
single per-anchor LSE of all five families; the split shape reports the
f-anchored families against LSE(zy, cross_batch) (`denom=pred`) and the
h-anchored families against LSE(xx, hh_all, xs_allt) (`denom=rep`).
log_pos rows carry an empty share.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import src.dataloader as dataloader_mod
from src.dataloader import HFStreamingLoader, _forward_fill_nan
from src.freq_embedding import SOURCE_ID_TO_LABELS
from src.loss import cosine_similarity_from_normalized
from src.models import ConfigurableModel

SPLIT_SHAPE = "cosine_similarity_batch_split_pred_rep"
POOLED_SHAPE = "cosine_similarity_batch_full_hh_negs_xshh_allt"
PRED_FAMILIES = ("log_neg_zy", "log_neg_cross_batch")
REP_FAMILIES = ("log_neg_xx", "log_neg_hh_all", "log_neg_xs_allt")
NEG_FAMILIES = ("log_neg_xx", "log_neg_zy", "log_neg_hh_all",
                "log_neg_cross_batch", "log_neg_xs_allt")
CSV_HEADER = ["arm_name", "ckpt_path", "loss_shape", "batch_type",
              "denom", "tensor", "mean", "max", "share"]

# Architecture shared by arms 1/3/4 — mirrors train.py's model_config for
# the train_backbone_split_pred_rep.sh / vast_moco_arm3_launch.sh /
# elisa_moco_arm4_launch.sh command line (all three are identical here).
MODEL_CONFIG = dict(
    C=1, H=384, W=16,
    encoder_type="gru", num_layers=6, nhead=6, ffn_mult=4.0,
    activation="gelu", depthwise_conv=3, deprecated_depthwise_conv=0,
    dropout=0.1,
    freq_emb_dim=3, seasonality_emb_dim=3,
    rev_norm_kind="ewma", rev_norm_span=128, patch_stats_kind="none",
    learnable_tau=False, tau_init=0.10,
    num_encoder_layers=3,
    encoder_dropkey=0.70, encoder_dropkey_share_heads=True,
    encoder_dropkey_share_layers=True,
    residual_dtype="fp32", attn_dtype="fp16", ffn_dtype="fp16",
    conv_dtype="fp16", patch_emb_dtype="fp32",
    forecaster_kind="transformer", cpc_k_steps=12,
    cpc_infonce=True,                       # cpc_w1 lives in the ckpt
    qk_norm=True, attn_out_norm=True,
    ema_embedding=True, ema_encoder=True,   # teacher_* keys in the ckpt
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Backbone .pth state_dict files.")
    p.add_argument("--arm-name", nargs="+", required=True,
                   help="Row label per checkpoint (parallel to --ckpts).")
    p.add_argument("--loss-shape", nargs="+", required=True,
                   choices=[SPLIT_SHAPE, POOLED_SHAPE],
                   help="Loss shape per checkpoint (parallel to --ckpts).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--out-csv", default=os.path.join(
        REPO_ROOT, "experiments", "2026-07-10_split_pred_rep", "results",
        "gradient_share_measurement.csv"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--t-raw", type=int, default=4096)
    p.add_argument("--hf-repo", default="jeremycochoy/gift-pretrain-full-4096")
    p.add_argument("--hf-path", default="small_v1")
    p.add_argument("--gift-eval-dir", default=os.environ.get(
        "GIFT_EVAL", "/home/jupyter/workspaces/gift-eval-data"))
    p.add_argument("--xs-chunk", type=int, default=16,
                   help="Source-batch chunk for the xs_allt LSE (memory "
                        "knob only; logsumexp is associative so the "
                        "result is exact and chunk-size independent).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build both batches, forward the FIRST checkpoint "
                        "only, print tensor shapes + first CSV row.")
    return p.parse_args()


def validate_args(args):
    n = len(args.ckpts)
    if len(args.arm_name) != n or len(args.loss_shape) != n:
        raise SystemExit(
            f"--ckpts ({n}), --arm-name ({len(args.arm_name)}) and "
            f"--loss-shape ({len(args.loss_shape)}) must be parallel lists.")
    missing = [c for c in args.ckpts if not os.path.isfile(c)]
    if missing:
        raise SystemExit("Missing checkpoint(s):\n  " + "\n  ".join(missing))


def set_determinism(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_hf_token():
    path = os.path.join(REPO_ROOT, "experiments", "hf_token.txt")
    if not os.path.isfile(path):
        return
    tok = open(path).read().strip()
    if tok:
        os.environ.setdefault("HF_TOKEN", tok)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)


def source_ids_to_labels(sids):
    """Training-time (freq_id, seasonality_id) lookup, as the mixed
    loaders do it. gift rows map to (0, 0)."""
    max_sid = max(SOURCE_ID_TO_LABELS)
    to_freq = torch.zeros(max_sid + 1, dtype=torch.long)
    to_seas = torch.zeros(max_sid + 1, dtype=torch.long)
    for sid, (fid, eid) in SOURCE_ID_TO_LABELS.items():
        to_freq[sid], to_seas[sid] = fid, eid
    safe = sids.clamp(min=0, max=max_sid)
    return to_freq[safe], to_seas[safe]


def build_mixed_batch(args):
    """First --batch-size HF windows after a seed-derived row offset."""
    dataloader_mod.T_RAW = args.t_raw  # loader crops module-level T_RAW
    loader = HFStreamingLoader(
        repo_id=args.hf_repo, batch_size=args.batch_size, C=1,
        path_in_repo=args.hf_path, skip_rows=args.seed % 100_000,
        emit_source_ids=True)
    it = iter(loader)
    x, sids = next(it)
    it.close()
    freq_ids, seas_ids = source_ids_to_labels(sids)
    return x, freq_ids, seas_ids


def catalog_windows(path, n_windows, t_raw, rng):
    """n non-overlapping t_raw windows from one GIFT-Eval arrow dataset,
    drawn by rng from all (series, offset) candidates."""
    from datasets import load_from_disk
    series = []
    for t in load_from_disk(path)["target"]:
        a = np.asarray(t, dtype=np.float32)
        series.extend(list(a) if a.ndim == 2 else [a])
    cands = [(i, s) for i, arr in enumerate(series)
             for s in range(0, len(arr) - t_raw + 1, t_raw)]
    out = []
    for j in rng.permutation(len(cands)):
        i, s = cands[j]
        window = series[i][s:s + t_raw].copy()
        if _forward_fill_nan(window):
            out.append(window)
        if len(out) == n_windows:
            return np.stack(out)
    raise SystemExit(
        f"Only {len(out)}/{n_windows} valid T={t_raw} windows in {path}; "
        f"cannot build the periodic-only batch. Lower --batch-size or "
        f"check the dataset.")


def build_periodic_batch(args):
    """solar/H + electricity/H windows (half each) with (0, 0) labels —
    the same labels every gift-source HF row carried in training."""
    dirs = [os.path.join(args.gift_eval_dir, d)
            for d in ("solar/H", "electricity/H")]
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(
            "Cannot build the periodic-only batch — GIFT-Eval catalog "
            "dirs not found:\n  " + "\n  ".join(missing) +
            "\nPass --gift-eval-dir or set $GIFT_EVAL (the dir used by "
            "eval_gift_eval_official.py). NOT falling back to the mixed "
            "batch.")
    rng = np.random.default_rng(args.seed)
    n_solar = args.batch_size // 2
    wins = np.concatenate([
        catalog_windows(dirs[0], n_solar, args.t_raw, rng),
        catalog_windows(dirs[1], args.batch_size - n_solar, args.t_raw, rng)])
    x = torch.from_numpy(wins).unsqueeze(-1)  # [B, T_raw, 1]
    zeros = torch.zeros(x.shape[0], dtype=torch.long)
    return x, zeros, zeros


@torch.no_grad()
def forward_latents(model, batch, device):
    """rev_norm -> prepare_encoder_input -> transformer, exactly as the
    trainer's forward_step (no channel mixing on this path)."""
    x, freq_ids, seas_ids = (t.to(device) for t in batch)
    xn = model.rev_norm(x, mode="norm") if model.rev_norm is not None else x
    xr = model.prepare_encoder_input(
        xn, freq_ids=freq_ids, seasonality_ids=seas_ids)
    f_flat, o_flat = model.transformer(xr)
    B, T = x.shape[0], x.shape[1] // model.W
    f_lat = f_flat.reshape(B, model.C, T, model.H).permute(0, 2, 1, 3)
    o_lat = o_flat.reshape(B, model.C, T, model.H).permute(0, 2, 1, 3)
    return f_lat.float(), o_lat.float()


@torch.no_grad()
def xs_allt_lse(hx_norm, orig_norm, tau, chunk):
    """LSE_{b' != b, all l} cos(h_{b,t}, h_{b',l})/tau -> [B, T-1, C],
    chunked over the source batch (exact; see --xs-chunk)."""
    B = hx_norm.shape[0]
    anchor = hx_norm.permute(0, 2, 1, 3).contiguous()   # [B, C, T-1, H]
    src = orig_norm.permute(0, 2, 1, 3).contiguous()    # [B, C, T,   H]
    b_all = torch.arange(B, device=anchor.device)
    run = None
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        gram = torch.matmul(anchor.unsqueeze(1),
                            src[s:e].permute(0, 1, 3, 2).unsqueeze(0)) / tau
        same = (b_all.view(B, 1) == b_all[s:e].view(1, e - s))
        gram = gram.masked_fill(same.view(B, e - s, 1, 1, 1), float("-inf"))
        lse = torch.logsumexp(torch.logsumexp(gram, dim=4), dim=1)
        run = lse if run is None else torch.logsumexp(
            torch.stack([run, lse]), dim=0)
    return run.permute(0, 2, 1)


@torch.no_grad()
def family_tensors(f_lat, o_lat, tau, chunk):
    """The positive + five negative families, each [B, T-1, C], mirroring
    src/loss.py's split_pred_rep / xshh_allt branches (student-only)."""
    neg_inf = float("-inf")
    B, T, C, _ = f_lat.shape
    orig_norm = F.normalize(o_lat, p=2, dim=-1)
    fore_norm = F.normalize(f_lat, p=2, dim=-1)
    hy_hat, hz_hat = fore_norm[:, :-1], fore_norm[:, 1:]
    hx, hy = orig_norm[:, :-1], orig_norm[:, 1:]
    out = {"log_pos": cosine_similarity_from_normalized(hy, hy_hat) / tau}

    sims_xx = cosine_similarity_from_normalized(
        hx.unsqueeze(3), hx.unsqueeze(2))
    mask_mat = ~torch.eye(C, dtype=torch.bool, device=hx.device)
    out["log_neg_xx"] = torch.logsumexp(
        (sims_xx / tau).masked_fill(~mask_mat.view(1, 1, C, C), neg_inf),
        dim=2)

    sims_zy = cosine_similarity_from_normalized(
        hz_hat.unsqueeze(3), hy_hat.unsqueeze(2))
    out["log_neg_zy"] = torch.logsumexp(sims_zy / tau, dim=2)

    t_idx = torch.arange(T - 1, device=hx.device).view(T - 1, 1)
    l_idx = torch.arange(T, device=hx.device).view(1, T)
    sims_hh = torch.matmul(hx.permute(0, 2, 1, 3), orig_norm.permute(0, 2, 3, 1))
    drop_self = (l_idx == t_idx).view(1, 1, T - 1, T)
    out["log_neg_hh_all"] = torch.logsumexp(
        (sims_hh / tau).masked_fill(drop_self, neg_inf), dim=3
    ).permute(0, 2, 1)

    # Cross-batch f<->h with STUDENT keys — no MoCo/teacher swap here.
    mask_batch = ~torch.eye(B, dtype=torch.bool, device=hx.device)
    sims_cb = torch.matmul(
        hy_hat.permute(1, 2, 0, 3),
        hy.permute(1, 2, 0, 3).transpose(-2, -1)
    ).permute(2, 3, 0, 1)
    out["log_neg_cross_batch"] = torch.logsumexp(
        (sims_cb / tau).masked_fill(~mask_batch.view(B, B, 1, 1), neg_inf),
        dim=1)

    out["log_neg_xs_allt"] = xs_allt_lse(hx, orig_norm, tau, chunk)
    return out


def family_shares(fams, names):
    """share_i = exp((tensor_i - per-anchor LSE of `names`).mean())."""
    per_anchor = torch.logsumexp(
        torch.stack([fams[n] for n in names]), dim=0)
    return {n: torch.exp((fams[n] - per_anchor).mean()).item()
            for n in names}


def rows_for_checkpoint(arm, ckpt, shape, batch_type, fams):
    def row(denom, name, share):
        t = fams[name]
        if name == "log_neg_xx" and t.shape[-1] == 1:  # -inf at C=1
            return [arm, ckpt, shape, batch_type, denom, name,
                    "n/a", "n/a", "n/a"]
        return [arm, ckpt, shape, batch_type, denom, name,
                f"{t.mean().item():.6g}", f"{t.max().item():.6g}", share]

    groups = ([("pooled", NEG_FAMILIES)] if shape == POOLED_SHAPE
              else [("pred", PRED_FAMILIES), ("rep", REP_FAMILIES)])
    rows = [row(groups[0][0], "log_pos", "")]
    for denom, names in groups:
        shares = family_shares(fams, names)
        rows += [row(denom, n, f"{shares[n]:.6g}") for n in names]
    return rows


def main():
    args = parse_args()
    validate_args(args)
    set_determinism(args.seed)
    load_hf_token()
    device = torch.device(args.device)
    print("[note] MoCo teacher swap NOT applied at measurement time: all "
          "tensors (incl. log_pos and cross-batch keys) use the student.")

    batches = {"mixed": build_mixed_batch(args),
               "periodic": build_periodic_batch(args)}
    for name, (x, _, _) in batches.items():
        print(f"[batch] {name}: x={tuple(x.shape)}")

    model = ConfigurableModel(**MODEL_CONFIG).to(device).eval()
    triples = list(zip(args.arm_name, args.ckpts, args.loss_shape))
    if args.dry_run:
        triples = triples[:1]

    all_rows = []
    for arm, ckpt, shape in triples:
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True))
        for batch_type, batch in batches.items():
            t0 = time.time()
            fams = family_tensors(*forward_latents(model, batch, device),
                                  args.tau, args.xs_chunk)
            if args.dry_run and batch_type == "mixed":
                for n, t in fams.items():
                    print(f"[dry-run] {n}: shape={tuple(t.shape)}")
            all_rows += rows_for_checkpoint(arm, ckpt, shape, batch_type, fams)
            print(f"[done] {arm} x {batch_type} "
                  f"({os.path.basename(ckpt)}) in {time.time() - t0:.1f}s")

    if args.dry_run:
        print("[dry-run] CSV header:", ",".join(CSV_HEADER))
        print("[dry-run] first row: ", ",".join(map(str, all_rows[0])))
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)
    print(f"[csv] {len(all_rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
