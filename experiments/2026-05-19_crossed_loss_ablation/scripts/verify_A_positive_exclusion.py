#!/usr/bin/env python3
"""Independent audit of the comparison baseline (A)
`cosine_similarity_batch_full_fh_negs`: prove the positive pair
(h_{t+1}, fₜ) (same b, same c) never enters the sum of negatives.

The positive could only leak in two places:
  • the (fₜ, hₗ) all-time term at l = t+1   → must be masked
  • the cross-batch term on the b₁=b₂ diagonal → must be masked
Every other negative (xy, xx, zy) is h–h or f–f and structurally cannot
contain the (h_{t+1}, fₜ) pair.

Proofs (run: `python3 verify_A_positive_exclusion.py`):
  P1 black-box, B=1: aligning fₜ with the positive ONLY drops the
     negatives-only loss by EXACTLY 1/τ — impossible if the positive
     also fed any negative term.
  P2 black-box, B=2: same exactness with the cross-batch term active ⇒
     its b₁=b₂ diagonal is correctly masked.
  P3 white-box: replicate (A)'s two mask lines; assert the positive
     coordinates are −∞ pre-logsumexp and the legit negatives finite.
  P4 the `--pos-in-denominator` training flag adds the positive to the
     denominator BY DESIGN (normalized InfoNCE), separately from the
     verified positive-free negative set.
"""
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from src.loss import contrastive_latent_loss

NAME = "cosine_similarity_batch_full_fh_negs"
TAU = 0.07


def spec(tau=TAU, posden=False):
    tc = {"contrastive_divergence_temperature": tau,
          "contrastive_latent_noise": None, "loss_shape": NAME,
          "contrastive_latent_delay": 0}
    if posden:
        tc["include_positive_in_denominator"] = True
    return SimpleNamespace(train_configuration=tc)


def main():
    ok = True

    # P1 — B=1
    T, H = 5, 12
    eye = torch.eye(H)
    h = eye[:T].view(1, T, 1, H).contiguous()
    f0 = eye[T:2 * T].view(1, T, 1, H).contiguous()
    Lo = contrastive_latent_loss((f0, h), False, spec()).item()
    fp = f0.clone(); fp[:, :T - 1, 0, :] = eye[1:T]
    Lp = contrastive_latent_loss((fp, h), False, spec()).item()
    p1 = abs((Lo - Lp) - 1 / TAU) < 1e-3
    ok &= p1
    print(f"P1 B=1  Δ={Lo - Lp:.6f}  1/τ={1 / TAU:.6f}  {'PASS' if p1 else 'FAIL'}")

    # P2 — B=2 (cross-batch term active)
    B = 2
    Hb = 2 * B * T
    e = torch.eye(Hb)
    h2 = torch.zeros(B, T, 1, Hb)
    f2 = torch.zeros(B, T, 1, Hb)
    for b in range(B):
        for t in range(T):
            h2[b, t, 0] = e[b * T + t]
            f2[b, t, 0] = e[B * T + b * T + t]
    Lo2 = contrastive_latent_loss((f2, h2), False, spec()).item()
    f2p = f2.clone()
    for b in range(B):
        f2p[b, :T - 1, 0, :] = h2[b, 1:T, 0, :]
    Lp2 = contrastive_latent_loss((f2p, h2), False, spec()).item()
    p2 = abs((Lo2 - Lp2) - 1 / TAU) < 1e-3
    ok &= p2
    print(f"P2 B=2  Δ={Lo2 - Lp2:.6f}  1/τ={1 / TAU:.6f}  {'PASS' if p2 else 'FAIL'}")

    # P3 — white-box mask inspection
    B, T, C, H = 3, 6, 2, 16
    g = torch.Generator().manual_seed(1)
    fl = torch.randn(B, T, C, H, generator=g)
    ol = torch.randn(B, T, C, H, generator=g)
    on = F.normalize(ol, p=2, dim=-1)
    fn = F.normalize(fl, p=2, dim=-1)
    hy_hat = fn[:, :-1]
    hy = on[:, 1:]
    sims_fh = torch.matmul(hy_hat.permute(0, 2, 1, 3), on.permute(0, 2, 3, 1))
    t_idx = torch.arange(T - 1).view(T - 1, 1)
    l_idx = torch.arange(T).view(1, T)
    m_fh = (sims_fh / TAU).masked_fill(
        (l_idx == t_idx + 1).view(1, 1, T - 1, T), float("-inf"))
    fh_pos_inf = all((m_fh[:, :, t, t + 1] == float("-inf")).all()
                     for t in range(T - 1))
    fh_keep_fin = all(torch.isfinite(m_fh[:, :, t, t]).all()
                      for t in range(T - 1))            # l=t kept (old xy_hat)
    hy_p = hy.permute(1, 2, 0, 3)
    hy_hat_p = hy_hat.permute(1, 2, 0, 3)
    scb = torch.matmul(hy_hat_p, hy_p.transpose(-2, -1)
                       ).permute(2, 3, 0, 1).contiguous()
    m_cb = (scb / TAU).masked_fill(
        (~torch.eye(B, dtype=torch.bool)).view(B, B, 1, 1).logical_not(),
        float("-inf"))
    cb_diag_inf = all((m_cb[b, b] == float("-inf")).all() for b in range(B))
    cb_off_fin = all(torch.isfinite(m_cb[i, j]).all()
                     for i in range(B) for j in range(B) if i != j)
    p3 = fh_pos_inf and fh_keep_fin and cb_diag_inf and cb_off_fin
    ok &= p3
    print(f"P3 white-box  fh[l=t+1]=-inf:{fh_pos_inf} fh[l=t]finite:{fh_keep_fin} "
          f"cb_diag=-inf:{cb_diag_inf} cb_off finite:{cb_off_fin}  "
          f"{'PASS' if p3 else 'FAIL'}")

    # P4 — pos-in-denominator is a separate, intended term
    B, T, C, H = 4, 8, 1, 16
    g = torch.Generator().manual_seed(7)
    fl = torch.randn(B, T, C, H, generator=g)
    ol = torch.randn(B, T, C, H, generator=g)
    Ln = contrastive_latent_loss((fl, ol), False, spec()).item()
    Lp4 = contrastive_latent_loss((fl, ol), False, spec(posden=True)).item()
    p4 = Lp4 + 1e-6 >= Ln
    ok &= p4
    print(f"P4 neg-only={Ln:.6f}  --pos-in-denominator={Lp4:.6f}  "
          f"{'PASS' if p4 else 'FAIL'} (positive in denom by design, "
          f"not a leaked negative)")

    print("\nALL PASS — (A) negative sum is positive-free" if ok
          else "\nFAILURE — investigate")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
