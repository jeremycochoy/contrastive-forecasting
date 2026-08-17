import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def cosine_similarity_from_normalized(a, b):
    return (a * b).sum(dim=-1)


# Logsumexp-form variants that compute `log_pos` / `log_neg_total` and so
# support the normalized-InfoNCE objective (positive-in-denominator), the
# BYOL alignment add-on (`align_loss_weight`), and the floor subtraction
# (`subtract_contrastive_floor`). Any other loss_shape requesting one of
# those raises rather than silently returning an unintended value.
_NORMALIZED_FORM_SHAPES = (
    'cosine_similarity_batch',
    'cosine_similarity_batch_no_time_neg',
    'cosine_similarity_batch_square',
    'cosine_similarity_batch_full_fh_negs',
    'cosine_similarity_batch_full_hh_negs',
    'cosine_similarity_batch_full_ff_negs',
    'cosine_similarity_batch_full_fh_hh_negs',
    'cosine_similarity_batch_full_hh_ff_negs',
    'cosine_similarity_batch_full_fh_hh_ff_negs',
    'cosine_similarity_batch_full_hh_negs_xbfree',
    'cosine_similarity_batch_full_hh_negs_xshh',
    'cosine_similarity_batch_full_hh_negs_xshh_allt',
)


# --- Rollout depth (#373) --------------------------------------------------
#
# `train_rollout_depth = k` duplicates every loss term that ties the
# forecaster output f to the encoder latent h. Copy j ties f^(j)_t to
# h_{t+1+j}, where f^(j) is the forecaster applied to its own output j more
# times (`src.forecasting_head.rollout_forecaster_latents`). k = 0 is the
# historical objective, byte for byte.
#
# `train_rollout_reduce` (#401) says how the k + 1 copies combine:
#   'sum'  (default) adds them, which is #373's objective. A k = 3 run is the
#          baseline plus three added terms, so the f-side carries 4x its k = 0
#          weight against the terms that carry no f.
#   'mean' divides them by k + 1, so the f-side holds its k = 0 weight at
#          every depth. The depth then changes what the model trains on, and
#          not how much the f-side outweighs the rest.
# At k = 0 there is one copy and the two reductions agree exactly. The mean
# runs over the f-bearing copies ONLY: the terms that carry no f enter once,
# at their configured weight, under both reductions.
#
# Every copy runs the SAME code path as depth 0, on views where every h index
# has shifted by j — so f^(j) lands wherever f^(0) lands, in the same role,
# and no depth-j term can quietly keep a depth-0 tensor. The h-anchored
# negative families (xy, xx, hh_all, xshh, xs_allt) shift with the depth too,
# rather than being computed once and reused unshifted: both variants give
# the h side k+1× weight inside the pooled denominators, and shifting keeps a
# depth-j copy a literal copy of the depth-0 objective. Terms that carry no f
# at all — L_rep, L_rep_moco, and `mse`'s − mse(h_t, h_{t+1}) half — are
# computed once and added once (`depth_index` in `contrastive_latent_loss`).

def rollout_depth_views(forecasted_latent, original_latent, depth,
                        teacher_original_latent=None):
    """Depth-`j` views of the latents, for an unchanged loss body.

    `forecasted_latent` is f^(j). The window keeps f^(j)_0..f^(j)_{T-1-j}
    against h_j..h_{T-1}, so the body reads hy_hat = f^(j)_t, hy = h_{t+1+j},
    hx = h_{t+j} and hz_hat = f^(j)_{t+1}, over anchors t = 0..T-2-j.

    Returns (f_view, h_view, teacher_view | None), each with T − j steps.
    """
    steps = original_latent.shape[1] - depth
    teacher_view = (None if teacher_original_latent is None
                    else teacher_original_latent[:, depth:])
    return (forecasted_latent[:, :steps], original_latent[:, depth:],
            teacher_view)


def resolve_rollout_depth(depth_override, config_depth, rollout_latents,
                          n_steps):
    """k for the rollout-depth objective, with its consistency checks.

    The function argument `depth_override` wins over the training-config key
    `config_depth` — the precedence every other run-level knob here uses, and
    what lets the `loss_tau_ref` diagnostic stay a depth-0 reference on a
    k > 0 run. The caller passes f^(1)..f^(k) explicitly, so a count that
    disagrees with k would train at a different depth than the run asked for:
    raise instead.
    """
    depth = int(config_depth or 0) if depth_override is None \
        else int(depth_override)
    given = 0 if rollout_latents is None else len(rollout_latents)
    if depth != given:
        raise ValueError(
            f"train_rollout_depth={depth} but {given} rollout latent(s) were "
            "passed; the caller supplies f^(1)..f^(k) (see "
            "src.forecasting_head.rollout_forecaster_latents). Pass "
            "train_rollout_depth=0 to force the depth-0 objective.")
    if depth and depth + 2 > n_steps:
        raise ValueError(
            f"train_rollout_depth={depth} needs at least {depth + 2} time "
            f"steps to keep one anchor; got {n_steps}.")
    return depth


ROLLOUT_REDUCTIONS = ('sum', 'mean')


def resolve_rollout_reduce(reduce_override, config_reduce):
    """How the k + 1 rollout-depth copies combine (#401).

    `'sum'` (default) is #373's objective. `'mean'` divides the copies by
    k + 1, so the f-side holds its k = 0 weight at every depth. The function
    argument wins over the training-config key `train_rollout_reduce`, the
    precedence every other run-level knob here uses.

    An unknown word raises: a run that asked for a reduction this code does
    not have would otherwise train on the sum and say nothing.
    """
    reduce = config_reduce if reduce_override is None else reduce_override
    if reduce is None:
        return 'sum'
    if reduce not in ROLLOUT_REDUCTIONS:
        raise ValueError(
            f"train_rollout_reduce={reduce!r} is not a reduction over the "
            f"k + 1 rollout-depth copies; use one of {ROLLOUT_REDUCTIONS}.")
    return reduce


def reduce_depth_terms(terms, reduce):
    """Combine the k + 1 copies of an f-bearing term (#401).

    `terms[0]` is the depth-0 copy. One copy returns itself, so k = 0 is
    unchanged under either reduction.
    """
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    if reduce == 'sum' or len(terms) == 1:
        return total
    return total / len(terms)


def infonce_floor(tau, n_negatives):
    """Theoretical minimum of the normalized-InfoNCE loss (positive in the
    denominator): the value at perfect alignment (cos(f, h⁺) = 1) with
    maximally-spread negatives (cos⁻ ≈ 0)::

        floor = log(1 + N · e^(−1/τ))            # a CONSTANT given (τ, N)

    Subtracting it only RE-BASES the curve so that ~0 means "at the
    uniformity floor". It is **gradient-neutral** (a constant has zero
    gradient; a monotonic shift leaves argmin / EMA / NaN-checks
    unchanged). It is a function of τ AND the negative count N (not τ
    alone), and it is a *lower* bound (assumes cos⁻ ≡ 0, cos⁺ ≡ 1), so a
    real run's contrastive loss approaches it from above — the re-based
    curve settles slightly above 0, not exactly at it. `float(tau)`
    detaches a learnable-τ tensor, keeping the subtraction grad-neutral.
    """
    return math.log1p(float(n_negatives) * math.exp(-1.0 / float(tau)))


def _split_pred_rep_floors(tau, B, T, C, tau_rep=None):
    """(f_pred, f_rep) — the two constants to re-base L_pred + L_rep by.

    L_pred has one positive against B batch-pooled f-anchored negatives per
    (t,c):
        N_pred = B · (C + (B − 1))
                          ─── adj-f (LSE over C channels of f_{t+1})
                              ─── cross-batch fh (LSE over B−1 batches)
    Its floor is the InfoNCE floor at that count:
        f_pred = log(1 + N_pred · e^(−1/τ))

    L_rep has NO positive; its value at cos ≡ 0 is simply log(#terms in the
    pooled logsumexp), i.e. log(N_rep) where:
        N_rep = B · ((C − 1) + (T − 1) + (B − 1) · T)
                       ─── xx (LSE over C−1 other channels)
                                ─── hh_all (LSE over T−1 other times, same series)
                                            ─── xs_allt (LSE over (B−1)·T other series×times)

    The pure-LSE (moco_rep_keys off) form of L_rep is log(N_rep) at cos≡0
    for any temperature — the τ divisor cancels between the exponents and
    the resulting logsumexp — so `tau_rep` does not enter that expression.
    `tau_rep` is accepted for symmetry with the split branch's `tau` /
    `tau_rep` wiring (#379); it's a no-op here. The moco_rep_keys shape,
    which does depend on τ_rep via `log(1 + N_rep · e^(−1/τ_rep))`, is not
    currently routed through `_split_pred_rep_floors` — no bimoco arm sets
    `--subtract-contrastive-floor` in #374/#379.
    """
    del tau_rep  # placeholder for the moco_rep_keys shape (see docstring).
    n_pred = B * (C + (B - 1))
    n_rep  = B * ((C - 1) + (T - 1) + (B - 1) * T)
    return (infonce_floor(tau, n_pred), math.log(n_rep))


def _effective_negative_count(loss_shape, B, T, C):
    """Exact count of negative cosine terms aggregated into `log_neg_total`
    for a logsumexp variant — used only by `infonce_floor`. `log_neg_total`
    pools the per-anchor negatives over the batch (logsumexp dim=0), so
    N = B · (per-anchor term count). Per-family per-anchor counts (general
    C): xy = C, xx = C−1 (diag-masked), zy = C, xy_hat = C (base) / C−1
    (no_time_neg, diag-masked), each all-time crossed term (fh/hh/ff) = T−1,
    each cross-batch edge = B−1. Returns None for unsupported shapes."""
    xb = B - 1                     # one cross-batch edge
    nt = T - 1                     # one all-time crossed term
    base3 = C + (C - 1) + C        # xy + xx + zy
    if loss_shape == 'cosine_similarity_batch':
        per = base3 + C + xb                       # + xy_hat + cross_fe
    elif loss_shape == 'cosine_similarity_batch_no_time_neg':
        per = (C - 1) + (C - 1) + xb               # xx + xy_hat(diag) + cross_fe
    elif loss_shape == 'cosine_similarity_batch_square':
        per = base3 + C + 3 * xb                    # + xy_hat + cross_fe/ff/hh
    elif loss_shape == 'cosine_similarity_batch_full_hh_negs_xbfree':
        per = base3 + nt + 2 * xb                   # + hh_all + cross_ff/hh (no fe)
    elif loss_shape == 'cosine_similarity_batch_full_hh_negs_xshh':
        # (B) minus xy (−C), plus the cross-series same-step h↔h edge (+xb):
        #   xx + zy + hh_all(T-1) + cross_fe(B-1) + cross_xshh(B-1)
        per = (C - 1) + C + nt + 2 * xb
    elif loss_shape == 'cosine_similarity_batch_full_hh_negs_xshh_allt':
        # As _xshh but the cross-series h↔h edge ranges over ALL l: (B-1)·T:
        #   xx + zy + hh_all(T-1) + cross_fe(B-1) + xs_allt((B-1)·T)
        per = (C - 1) + C + nt + xb + xb * T
    else:
        n_alltime = {
            'cosine_similarity_batch_full_fh_negs': 1,
            'cosine_similarity_batch_full_hh_negs': 1,
            'cosine_similarity_batch_full_ff_negs': 1,
            'cosine_similarity_batch_full_fh_hh_negs': 2,
            'cosine_similarity_batch_full_hh_ff_negs': 2,
            'cosine_similarity_batch_full_fh_hh_ff_negs': 3,
        }.get(loss_shape)
        if n_alltime is None:
            return None
        per = base3 + n_alltime * nt + xb           # + all-time term(s) + cross_fe
    return B * per

def _full_hh_negs_terms(hy_hat_norm, hx_norm, hy_norm, hz_hat_norm, orig_norm,
                        B, T, C, tau):
    """Exact replication of β's `cosine_similarity_batch_full_hh_negs`
    numerator + denominator (want_hh variant), returning
    ``(log_pos, log_neg_total)`` so a caller can reuse β's NEGATIVES unchanged.

    Mirrors the inline branch in `contrastive_latent_loss` term-for-term — xy,
    xx, zy, the encoder all-time `hh`, and cross-batch — including the
    batch-pooling of `log_neg_total` (logsumexp over the batch axis). Verified
    numerically equal to that branch by scripts/test_cpc.py (#316 review).
    log_pos: [B, T-1, C]; log_neg_total: [1, T-1, C].
    """
    neg_inf = float('-inf')
    log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
    sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
    log_neg_xy = torch.logsumexp(sims_xy / tau, dim=2)
    sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
    mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
    mask_mat = mask_mat.view(1, 1, C, C)
    log_neg_xx = torch.logsumexp((sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2)
    sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
    log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)
    t_idx = torch.arange(T - 1, device=orig_norm.device).view(T - 1, 1)
    l_idx = torch.arange(T, device=orig_norm.device).view(1, T)
    sims_hh = torch.matmul(hx_norm.permute(0, 2, 1, 3), orig_norm.permute(0, 2, 3, 1))  # [B,C,T-1,T]
    mask_hh = (l_idx == t_idx).view(1, 1, T - 1, T)
    log_neg_hh = torch.logsumexp(
        (sims_hh / tau).masked_fill(mask_hh, neg_inf), dim=3).permute(0, 2, 1)
    hy_p = hy_norm.permute(1, 2, 0, 3)
    hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)
    sims_cb = torch.matmul(
        hy_hat_p, hy_p.transpose(-2, -1)).permute(2, 3, 0, 1).contiguous()  # [B,B,T-1,C]
    mask_b = ~torch.eye(B, dtype=torch.bool, device=sims_cb.device)
    mask_b = mask_b.view(B, B, 1, 1)
    log_neg_cb = torch.logsumexp((sims_cb / tau).masked_fill(~mask_b, neg_inf), dim=1)
    negatives = torch.stack(
        [log_neg_xy, log_neg_xx, log_neg_zy, log_neg_hh, log_neg_cb], dim=0)
    log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
    log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)  # [1,T-1,C]
    return log_pos, log_neg_total


def cpc_multistep_loss(forecasted_multi, original_latent, tau,
                       include_positive_in_denominator=True):
    """Multi-step positive on β's negatives — β's
    `cosine_similarity_batch_full_hh_negs` with ONLY the positive changed (#316).

    β contrasts the single 1-step pair (f_t, h_{t+1}) against its negative pool
    (xy, xx, zy, encoder all-time hh, cross-batch — batch-pooled into
    `log_neg_total`). Here the forecaster is K linear heads, f^{(k)}_t = W_k h_t,
    so we keep β's negative pool **exactly** — computed once from the k = 1 head
    (the analogue of β's f_t) — and only the positive changes: for each horizon
    k the InfoNCE positive is cos(f^{(k)}_t, h_{t+k}) against that same pool, and
    the per-k losses are averaged. At k = 1 this is byte-for-byte β's loss.

    Args:
        forecasted_multi: [B, T, C, K, H] — head-k output at each (b, t, c).
        original_latent:  [B, T, C, H]    — encoder latents h (targets).
        tau: scalar or 0-d tensor temperature.
        include_positive_in_denominator: normalized InfoNCE (loss ≥ 0) when
            True (β's --pos-in-denominator); negatives-only form when False.

    Returns: scalar loss (mean over k of the per-step InfoNCE on β's negatives).
    """
    B, T, C, K, H = forecasted_multi.shape
    f_norm = F.normalize(forecasted_multi, p=2, dim=-1)   # [B,T,C,K,H]
    orig_norm = F.normalize(original_latent, p=2, dim=-1)  # [B,T,C,H]
    f1_norm = f_norm[:, :, :, 0, :]                        # k=1 head = β's f_t

    # β's negatives, computed once from the k=1 head (identical to β).
    log_pos_1, log_neg_total = _full_hh_negs_terms(
        f1_norm[:, :-1], orig_norm[:, :-1], orig_norm[:, 1:],
        f1_norm[:, 1:], orig_norm, B, T, C, tau)           # log_neg_total: [1,T-1,C]

    per_k = []
    for k in range(1, K + 1):
        Tk = T - k
        if Tk <= 0:
            break
        if k == 1:
            log_pos = log_pos_1                             # [B,T-1,C] (β's positive)
        else:
            anchor = f_norm[:, :Tk, :, k - 1, :]           # f^{(k)}_t [B,Tk,C,H]
            pos_tgt = orig_norm[:, k:, :, :]               # h_{t+k}   [B,Tk,C,H]
            log_pos = (anchor * pos_tgt).sum(-1) / tau     # [B,Tk,C]
        lnt = log_neg_total[:, :Tk, :]                      # β's negatives at anchors 0..Tk-1
        if include_positive_in_denominator:
            log_denom = torch.logsumexp(
                torch.stack([log_pos, lnt.expand_as(log_pos)], dim=0), dim=0)
            per_k.append((log_denom - log_pos).mean())
        else:
            per_k.append((lnt.expand_as(log_pos) - log_pos).mean())
    return torch.stack(per_k).mean()


def cpc_multistep_cpcnegs_loss(forecasted_multi, original_latent, tau,
                               include_positive_in_denominator=True):
    """CPC multi-step with CPC-CANONICAL negatives (#316 study arm #3) — the
    ORIGINAL loss that trained the linear+CPC-negs k=12 backbones, kept so
    that family's k=1 baseline is matched. For each k the InfoNCE positive is
    cos(f^{(k)}_t, h_{t+k}); negatives are encoder latents h from (a) other
    sequences at the matched target time (cross-batch) and (b) all other times
    in the same sequence (the target l = t+k masked). NOT β's negatives — this
    is the confounded control family, per-anchor (not batch-pooled). Averaged
    over k.
    """
    B, T, C, K, H = forecasted_multi.shape
    neg_inf = float('-inf')
    f_norm = F.normalize(forecasted_multi, p=2, dim=-1)
    h_norm = F.normalize(original_latent, p=2, dim=-1)
    per_k = []
    for k in range(1, K + 1):
        Tk = T - k
        if Tk <= 0:
            break
        anchor = f_norm[:, :Tk, :, k - 1, :]
        pos_tgt = h_norm[:, k:, :, :]
        log_pos = (anchor * pos_tgt).sum(-1) / tau
        a_p = anchor.permute(1, 2, 0, 3)
        p_p = pos_tgt.permute(1, 2, 0, 3)
        sims_cb = torch.matmul(
            a_p, p_p.transpose(-2, -1)).permute(2, 3, 0, 1)
        mask_b = ~torch.eye(B, dtype=torch.bool, device=sims_cb.device)
        mask_b = mask_b.view(B, B, 1, 1)
        log_neg_cb = torch.logsumexp(
            (sims_cb / tau).masked_fill(~mask_b, neg_inf), dim=1)
        a_q = anchor.permute(0, 2, 1, 3)
        h_q = h_norm.permute(0, 2, 3, 1)
        sims_t = torch.matmul(a_q, h_q)
        t_idx = torch.arange(Tk, device=sims_t.device).view(Tk, 1)
        l_idx = torch.arange(T, device=sims_t.device).view(1, T)
        drop = (l_idx == (t_idx + k)).view(1, 1, Tk, T)
        log_neg_t = torch.logsumexp(
            (sims_t / tau).masked_fill(drop, neg_inf), dim=3).permute(0, 2, 1)
        log_neg_total = torch.logsumexp(
            torch.stack([log_neg_cb, log_neg_t], dim=0), dim=0)
        if include_positive_in_denominator:
            log_denom = torch.logsumexp(
                torch.stack([log_pos, log_neg_total], dim=0), dim=0)
            per_k.append((log_denom - log_pos).mean())
        else:
            per_k.append((log_neg_total - log_pos).mean())
    return torch.stack(per_k).mean()


def cpc_infonce_aux_loss(forecasted_latent, original_latent, w1,
                         cross_batch_chunk=256, rollout_latents=None,
                         depth_reduce='sum'):
    """CPC InfoNCE auxiliary term (van den Oord et al. 2018, Eq. 4; k=1), #344.

    Predict the next-step encoder embedding ``e_{t+1}`` from the
    autoregressive context ``h_t`` (the forecaster latent) through a
    learnable LOG-BILINEAR score (Eq. 3) ``f(e_j, h_t) = exp(e_j^T W_1 h_t)``
    over a candidate set ``C = {e_{t+1}} ∪ {negatives}``::

        L = − log( exp(e_{t+1}^T W_1 h_t) / Σ_{e_j ∈ C} exp(e_j^T W_1 h_t) )

    The positive ``e_{t+1}`` is itself one of the denominator terms (plain
    softmax / cross-entropy over candidates), so ``L ≥ 0``. Both numerator
    and denominator score the SAME projected context ``W_1 h_t`` against an
    encoder embedding ``e_j = enc(x_j)``.

    Roles, matched to the existing 1-step positive (f_t ↔ h_{t+1}):
        ``h_t = forecasted_latent[:, t]`` — the AR context ``g_ar(e_{≤t})``
            (the paper's ``c_t``), a tensor distinct from ``e_t``;
        ``e_t = original_latent[:, t]``   — the per-step encoder embedding
            (the paper's ``z_t``), L2-normalised here.

    NO stop-gradient (paper-exact: encoder and AR model trained jointly,
    gradient flows through both ``h_t`` and the targets ``e_j``). NO
    temperature divisor: the unbounded bilinear ``W_1`` carries the scale,
    so the term's theoretical minimum is already 0 (floor-subtraction would
    be a no-op for it; the existing contrastive term keeps its own τ/floor).

    Negatives are drawn from the empirical proposal ``p(x_{t+1})`` — the same
    CPC-canonical set as :func:`cpc_multistep_cpcnegs_loss` (per-anchor, NOT
    batch-pooled):
        (a) cross-batch: ``e_{b', t+1}`` for every other sequence ``b' ≠ b``
            at the matched target step ``t+1``;
        (b) cross-time: ``e_{b, l}`` for every other step ``l ≠ t+1`` in the
            same sequence.
    The cross-batch Gram ``[B, B, T−1]`` (~1 GB at B=1024, T=256) is
    GRADIENT-CHECKPOINTED in chunks over the source batch ``b'``: logsumexp
    is associative, so the chunked result is exact and chunk-size
    independent (chunk size only trades memory for kernel launches), and the
    Gram is recomputed in backward one chunk at a time — capping peak memory
    at a single chunk, exactly as the ``xshh_allt`` term does.

    Args:
        forecasted_latent: ``[B, T, C, H]`` — AR context ``h`` (forecaster).
        original_latent:   ``[B, T, C, H]`` — encoder embeddings ``e``.
        w1: ``nn.Linear(H, H, bias=False)`` applied to ``h_t`` (the bilinear
            ``W_1``).
        cross_batch_chunk: source-batch chunk for the checkpointed
            cross-batch logsumexp (env override ``CPC_CB_CHUNK``).
        rollout_latents: f^(1)..f^(k) (#373). ``f`` sits in the numerator AND
            in every denominator term (through ``W_1 f_t``), so each depth
            rebuilds the whole term from ``W_1 f^(j)_t`` against
            ``e_{t+1+j}``.
        depth_reduce: how the k + 1 copies combine (#401) — ``'sum'``
            (default, #373's objective) or ``'mean'``. The term carries no
            h-only half, so the mean is the plain mean of the copies.

    Returns: scalar loss (mean over B, C, t).
    """
    if forecasted_latent.dim() != 4:
        raise ValueError(
            "cpc_infonce_aux_loss expects a 4-D [B,T,C,H] forecaster latent "
            f"(got {forecasted_latent.dim()}-D); the CPC InfoNCE auxiliary (#344) "
            "is not defined for the cpc_multistep forecaster stack. Drop "
            "--cpc-infonce-weight or use the transformer forecaster.")
    B, T, C, H = forecasted_latent.shape
    if T < 2:
        return forecasted_latent.new_zeros(())
    neg_inf = float('-inf')
    e = F.normalize(original_latent, p=2, dim=-1)        # [B,T,C,H] unit embeddings e_j
    q = w1(forecasted_latent)                            # [B,T,C,H] W_1 h_t (raw scale)

    q_a = q[:, :-1]                                      # anchors h_t,  t=0..T-2 [B,T-1,C,H]
    e_pos = e[:, 1:]                                     # target e_{t+1}         [B,T-1,C,H]

    # Positive log-score  e_{t+1}^T W_1 h_t  (no τ; W_1 carries the scale).
    log_pos = (q_a * e_pos).sum(-1)                      # [B,T-1,C]

    # (b) cross-time negatives: e_{b,l} for all l != t+1, same sequence.
    #     sims_t[b,c,t,l] = e_{b,c,l}^T W_1 h_{b,c,t}.
    sims_t = torch.matmul(q_a.permute(0, 2, 1, 3),       # [B,C,T-1,H]
                          e.permute(0, 2, 3, 1))         # [B,C,H,T]  -> [B,C,T-1,T]
    t_idx = torch.arange(T - 1, device=e.device).view(T - 1, 1)
    l_idx = torch.arange(T, device=e.device).view(1, T)
    drop = (l_idx == (t_idx + 1)).view(1, 1, T - 1, T)   # mask the positive l=t+1
    log_neg_time = torch.logsumexp(
        sims_t.masked_fill(drop, neg_inf), dim=3).permute(0, 2, 1)  # [B,T-1,C]

    # (a) cross-batch negatives: e_{b',t+1} for b' != b at the matched step,
    #     chunked + checkpointed over the source batch b' (peak = one chunk).
    CH = int(os.environ.get('CPC_CB_CHUNK', str(cross_batch_chunk)))
    q_ap = q_a.permute(1, 2, 0, 3).contiguous()          # [T-1,C,B,H] anchors
    e_pp = e_pos.permute(1, 2, 0, 3).contiguous()        # [T-1,C,B,H] targets at t+1
    b_all = torch.arange(B, device=e.device)

    def _cb_chunk_lse(anc, tgt_chunk, same_mask):
        # anc [T-1,C,B,H], tgt_chunk [T-1,C,ch,H], same_mask [1,1,B,ch] bool.
        gram = torch.matmul(anc, tgt_chunk.transpose(-2, -1))       # [T-1,C,B,ch]
        gram = gram.masked_fill(same_mask, neg_inf)
        return torch.logsumexp(gram, dim=3)                         # [T-1,C,B]

    run = None
    for s in range(0, B, CH):
        ee = min(s + CH, B)
        same = (b_all.view(B, 1) == b_all[s:ee].view(1, ee - s)).view(1, 1, B, ee - s)
        chunk_lse = checkpoint(_cb_chunk_lse, q_ap, e_pp[:, :, s:ee], same,
                               use_reentrant=False)
        run = chunk_lse if run is None else torch.logsumexp(
            torch.stack([run, chunk_lse], dim=0), dim=0)
    log_neg_batch = run.permute(2, 0, 1)                 # [B,T-1,C]

    # Normalized InfoNCE: positive in the denominator (loss >= 0). Per-anchor
    # candidate set C = {e_{t+1}} ∪ {cross-time} ∪ {cross-batch}.
    log_neg_total = torch.logsumexp(
        torch.stack([log_neg_time, log_neg_batch], dim=0), dim=0)   # [B,T-1,C]
    log_denom = torch.logsumexp(
        torch.stack([log_pos, log_neg_total], dim=0), dim=0)
    copies = [(log_denom - log_pos).mean()]
    for depth, f_depth in enumerate(rollout_latents or (), start=1):
        f_view, e_view, _ = rollout_depth_views(
            f_depth, original_latent, depth)
        copies.append(cpc_infonce_aux_loss(
            f_view, e_view, w1, cross_batch_chunk=cross_batch_chunk))
    return reduce_depth_terms(
        copies, resolve_rollout_reduce(depth_reduce, None))


def cpc_infonce_all_loss(forecasted_latent, original_latent, w1, source_chunk=8,
                         marginal_only=True, rollout_latents=None,
                         depth_reduce='sum'):
    """CPC InfoNCE with the FULL marginal-proposal negative set (#348).

    The paper-faithful CPC InfoNCE (van den Oord et al. 2018, Eq. 4): predict
    ``z_{t+1}`` from the AR context ``c_t`` via the log-bilinear ``z_j^T W_1 c_t``
    (no τ, no stop-grad). The denominator ``Σ_{x_j ∈ X} f_k(x_j, c_t)`` sums over
    the whole candidate set ``X`` and the positive ``z_{t+1}`` is one of its terms
    — there is **no masking**.

    ``marginal_only=True`` (default): ``X = {positive z_{b,t+1}} ∪ {every OTHER
        sequence's embeddings z_{b',l}, b'≠b, all l}``. The negatives are
        independent of the context ``c_t`` — genuine marginal ``p(z_{t+1})``
        draws — so van den Oord's Theorem 1 holds: the unique optimum is
        ``f_k ∝ p(z|c)/p(z)`` and the loss attains the bound
        ``I(z_{t+1}; c_t) >= log N − L_N``. This is the strict,
        information-theoretically grounded form.
    ``marginal_only=False``: ``X`` additionally includes the SAME sequence's
        other steps (the literal full batch×time grid). Those are temporally
        correlated with ``c_t`` (not marginal draws), so the MI bound becomes
        approximate; kept as a maximal-negatives ablation. This full-grid set is
        a strict superset of the narrow :func:`cpc_infonce_aux_loss` set, so for
        identical inputs ``marginal_only=False`` >= the aux loss.

    The cross-sequence-all-time negatives form a ``[B, B, T-1, T]`` object; it
    is streamed over the SOURCE batch ``b'`` in chunks, each chunk
    gradient-checkpointed. logsumexp is associative, so the chunked result is
    exact and chunk-size independent (chunk size only trades memory for kernel
    launches). Env override: ``CPC_ALL_CHUNK``.

    Args mirror :func:`cpc_infonce_aux_loss`, ``rollout_latents`` (#373) and
    ``depth_reduce`` (#401) included: each depth rebuilds the whole term from
    ``W_1 c^(j)_t`` against ``z_{t+1+j}``, and the k + 1 copies combine by the
    sum (default) or by the mean. Returns a scalar (mean over B, C, t).
    """
    if forecasted_latent.dim() != 4:
        raise ValueError(
            "cpc_infonce_all_loss expects a 4-D [B,T,C,H] forecaster latent "
            f"(got {forecasted_latent.dim()}-D); the CPC InfoNCE term (#344/#348) "
            "is not defined for the cpc_multistep forecaster stack.")
    B, T, C, H = forecasted_latent.shape
    if T < 2:
        return forecasted_latent.new_zeros(())
    neg_inf = float('-inf')
    e = F.normalize(original_latent, p=2, dim=-1)        # [B,T,C,H] unit embeddings z_j
    q = w1(forecasted_latent)                            # [B,T,C,H] W_1 c_t (raw scale)
    q_a = q[:, :-1]                                      # contexts c_t  [B,T-1,C,H]
    e_pos = e[:, 1:]                                     # positive z_{t+1}  [B,T-1,C,H]
    log_pos = (q_a * e_pos).sum(-1)                      # numerator score  [B,T-1,C]

    # Denominator Σ_{x_j ∈ X} f_k(x_j, c_t) is over the ENTIRE batch×time grid of
    # candidates X (Eq. 4) — the positive z_{t+1} is itself one of these terms,
    # so there is NO masking. We sum it in two non-overlapping partitions to chunk
    # the cross-sequence part: (i) the anchor's OWN sequence over all l, (ii) every
    # OTHER sequence over all l. Their logsumexp is the full denominator.

    # Cross-sequence candidates: every OTHER sequence b' != b, ALL l — the
    # context-independent (marginal) draws. The [B,B,T-1,T] Gram is streamed over
    # the source batch b' in gradient-checkpointed chunks (logsumexp is
    # associative ⇒ exact and chunk-size independent; env CPC_ALL_CHUNK). The
    # b'=b slice is dropped here (excluded entirely under marginal_only, or
    # re-added via the same-sequence term below otherwise).
    CH = int(os.environ.get('CPC_ALL_CHUNK', str(source_chunk)))
    anchor = q_a.permute(0, 2, 1, 3).contiguous()        # [B,C,T-1,H]  (W_1 c_t)
    src = e.permute(0, 2, 1, 3).contiguous()             # [B,C,T,H]    (z_{b',l}, all l)
    b_all = torch.arange(B, device=e.device)

    def _cross_chunk(anc, src_chunk, same_mask):
        # anc [B,C,T-1,H], src_chunk [ch,C,T,H], same_mask [B,ch,1,1,1] bool.
        gram = torch.matmul(anc.unsqueeze(1),                          # [B,1, C,T-1,H]
                            src_chunk.permute(0, 1, 3, 2).unsqueeze(0))  # [1,ch,C,H,  T]
        gram = gram.masked_fill(same_mask, neg_inf)                    # drop b'=b
        return torch.logsumexp(torch.logsumexp(gram, dim=4), dim=1)    # [B,C,T-1]

    run = None
    for s in range(0, B, CH):
        ee = min(s + CH, B)
        same = (b_all.view(B, 1) == b_all[s:ee].view(1, ee - s)).view(B, ee - s, 1, 1, 1)
        chunk_lse = checkpoint(_cross_chunk, anchor, src[s:ee], same, use_reentrant=False)
        run = chunk_lse if run is None else torch.logsumexp(
            torch.stack([run, chunk_lse], dim=0), dim=0)
    log_den_cross = run.permute(0, 2, 1)                 # [B,C,T-1] -> [B,T-1,C]

    # Denominator Σ_{x_j ∈ X} f_k(x_j, c_t) (Eq. 4) — the positive z_{t+1} is
    # always one of its terms. marginal_only=True (default, strict van den Oord):
    # X = {positive} ∪ {all OTHER-sequence embeddings}; the negatives are i.i.d.
    # marginal draws (independent of c_t), so Theorem 1 holds EXACTLY — the unique
    # optimum is the density ratio p(z|c)/p(z), maximizing the I >= log N − L
    # bound. marginal_only=False: X also includes the SAME sequence's other steps
    # (the literal full batch×time grid); those are correlated with c_t (not
    # marginal draws), so the MI bound is only approximate.
    if marginal_only:
        log_denom = torch.logsumexp(
            torch.stack([log_pos, log_den_cross], dim=0), dim=0)      # {pos} ∪ {cross}
    else:
        sims_t = torch.matmul(q_a.permute(0, 2, 1, 3), e.permute(0, 2, 3, 1))  # [B,C,T-1,T]
        log_den_self = torch.logsumexp(sims_t, dim=3).permute(0, 2, 1)         # same-seq all l
        log_denom = torch.logsumexp(
            torch.stack([log_den_self, log_den_cross], dim=0), dim=0)
    copies = [(log_denom - log_pos).mean()]
    for depth, f_depth in enumerate(rollout_latents or (), start=1):
        f_view, e_view, _ = rollout_depth_views(
            f_depth, original_latent, depth)
        copies.append(cpc_infonce_all_loss(
            f_view, e_view, w1, source_chunk=source_chunk,
            marginal_only=marginal_only))
    return reduce_depth_terms(
        copies, resolve_rollout_reduce(depth_reduce, None))


def sigreg_loss(z, sigma2=None, M=1024, T_knots=17, W=None,
                post_normalize=False, n_chunk=8192, projections=None):
    """LeJEPA spherical SIGReg statistic (Balestriero et al. 2025, #355).

    Push the pooled marginal of the latent ``z`` toward ``Unif(S^{K-1})`` —
    a principled, isotropic anti-collapse term that regularises the
    *whole-batch* marginal distribution to fill the unit hypersphere
    instead of concentrating on a low-dimensional subspace. Composes with
    contrastive / EMA-target / CPC objectives — no pairwise relation,
    stop-grad, predictor head, or queue.

    Statistic — average the Epps–Pulley 1-D test over ``M`` random
    unit-direction projections::

        SIGReg(Z) = (1/M) · Σ_{m=1..M}  EP({ a_m^T z_i }_{i=1..N})

        EP(s) = ∫_{ω ∈ [−W, W]}  | φ̂(ω) − exp(−½ σ² ω²) |²  dω
        φ̂(ω) = (1/N) · Σ_i [ cos(ω s_i) + i sin(ω s_i) ]

    ``a_m ∼ Unif(S^{K-1})`` are RESAMPLED every call (fresh randomness is
    part of the regulariser, not a cached basis — no buffers, the term
    contributes nothing to ``state_dict``). The integral is the
    trapezoidal-rule sum on ``T_knots`` knots in ``[−W, W]``.

    For ``K`` at the d_model scale (#355 issue: 384), projections of
    ``Unif(S^{K-1})`` onto any unit direction are statistically
    indistinguishable from ``N(0, 1/K)`` (excess kurtosis ``−6/(K+2)``
    ≈ −0.016 at K=384), so the standard SIGReg target ``σ² = 1/K`` is
    the principled choice — no Beta-target needed.

    Args:
        z: tensor with feature axis ``K = z.shape[-1]``; every other axis
            is flattened into the sample axis ``N``.
        sigma2: target projected variance. ``None`` → ``1/K`` (auto-scale
            with the latent dim, matching the K-d-sphere marginal).
        M: number of random 1-D projection directions per call. Ignored
            when ``projections`` is supplied.
        T_knots: number of trapezoidal-rule knots in ``[−W, W]``.
        W: integration bound. ``None`` → ``6/√K`` (~6σ — the integrand
            has decayed near the endpoints).
        post_normalize: when True, ``z`` is L2-normalised on its last
            axis before projection (LeJEPA's post-sphere-projection
            placement; the strict ``σ² = 1/K`` regime). When False
            (default), the statistic is on the raw pre-normalisation
            vectors — pushes the encoder to LAND on the sphere with a
            uniform marginal, leaving the downstream L2-normalise as a
            near-identity.
        n_chunk: chunk size along the sample axis ``N`` for the
            ``cos/sin`` integrand evaluation — peaks at ``M·n_chunk·
            T_knots`` floats per chunk. logsumexp-free averaging over
            chunks is an exact sum split (the test pins this), so
            ``n_chunk`` is a memory/throughput tradeoff only.
        projections: optional ``[M, K]`` tensor of unit-direction
            projections to use INSTEAD of resampling. Tests use this to
            make the call deterministic; the training path never passes
            it (fresh ``a_m`` every step is the contract).

    Returns: scalar tensor (the mean EP statistic across the M directions).
    """
    K = z.shape[-1]
    if sigma2 is None:
        sigma2 = 1.0 / K
    if W is None:
        W = 6.0 / math.sqrt(K)
    if post_normalize:
        z = F.normalize(z, p=2, dim=-1)
    z_flat = z.reshape(-1, K)                            # [N, K]
    N = z_flat.shape[0]

    if projections is None:
        a = torch.randn(M, K, device=z.device, dtype=z.dtype)
        a = F.normalize(a, p=2, dim=-1)                  # [M, K]
    else:
        if projections.shape[-1] != K:
            raise ValueError(
                f"projections has last dim {projections.shape[-1]}, "
                f"expected K={K}.")
        a = projections.to(device=z.device, dtype=z.dtype)
        M = a.shape[0]

    s = a @ z_flat.t()                                    # [M, N]

    omega = torch.linspace(-W, W, T_knots, device=z.device, dtype=z.dtype)

    # φ̂(ω) over the sample axis — chunked along N to cap the [M,n_chunk,
    # T_knots] intermediate. cos/sin/sum is associative, so the chunked
    # mean equals the unchunked one (pinned in tests). Each chunk's body
    # (ws / cos / sin / sum) is wrapped in torch.utils.checkpoint so
    # backward recomputes the per-chunk intermediates instead of keeping
    # them in the graph — without this, n_chunk only trades chunk count
    # for per-chunk size and the M=1024 / K=384 path OOMs on a 24 GB card
    # (#356 review P0).
    def _chunk_phi(s_chunk, omega):
        ws = s_chunk.unsqueeze(-1) * omega                # [M, c, T_knots]
        return torch.cos(ws).sum(dim=1), torch.sin(ws).sum(dim=1)
    phi_re = z.new_zeros(M, T_knots)
    phi_im = z.new_zeros(M, T_knots)
    for start in range(0, N, n_chunk):
        end = min(start + n_chunk, N)
        chunk_re, chunk_im = checkpoint(
            _chunk_phi, s[:, start:end], omega, use_reentrant=False)
        phi_re = phi_re + chunk_re
        phi_im = phi_im + chunk_im
    phi_re = phi_re / N
    phi_im = phi_im / N

    # Target characteristic function — N(0, σ²) is purely real.
    target = torch.exp(-0.5 * sigma2 * omega.pow(2))      # [T_knots]
    integrand = (phi_re - target).pow(2) + phi_im.pow(2)  # [M, T_knots]

    # Trapezoidal rule on [−W, W]: step h, endpoints weighted h/2.
    h = 2.0 * W / (T_knots - 1)
    trap_w = z.new_full((T_knots,), h)
    trap_w[0] = trap_w[-1] = 0.5 * h

    ep_per_dir = (integrand * trap_w).sum(dim=-1)         # [M]
    return ep_per_dir.mean()


def align_loss(forecasted_latent, original_latent, weight=1.0,
               target_latent=None, rollout_latents=None,
               depth_reduce='sum'):
    """BYOL/SimSiam alignment term, standalone (#344 follow-up arm).

    ``L_align = weight · (2 − 2·cos(f_t, sg(h_{t+1}))).mean()`` — pull the
    forecaster output ``f_t`` toward the *next* encoder embedding ``h_{t+1}``,
    with the encoder target stop-gradded (gradient flows only through the
    forecaster). This is byte-for-byte the alignment add-on inside
    ``contrastive_latent_loss`` (the ``align_loss_weight`` path), exposed as a
    standalone so a run can train on it WITHOUT the main contrastive loss —
    to test whether CPC + a separate forecaster loss beats the contrastive
    objective. ``2 − 2·cos = ‖f̂ − ĥ‖²`` ∈ [0, 4], min 0 at perfect alignment.

    ``target_latent`` (#388) replaces ``original_latent`` on the target side
    only. Pass the EMA teacher's ``h`` to get the BYOL form the term was
    designed for: a slowly-moving target the student cannot chase within a
    step. Left at None the target is the student's own ``sg(h_{t+1})``, which
    is what #382's `align` arm trained on.

    ``rollout_latents`` (#373): f^(1)..f^(k). The term is f-bearing, so every
    depth builds a complete copy ``2 − 2·cos(f^(j)_t, sg(target_{t+1+j}))``.
    None (default) ⇒ the depth-0 term alone. ``depth_reduce`` (#401) combines
    the k + 1 copies: ``'sum'`` (default, #373's objective) or ``'mean'``,
    which holds the term at its k = 0 weight at every depth.

    forecasted_latent, original_latent, target_latent: ``[B, T, C, H]``.
    Returns scalar.
    """
    target = original_latent if target_latent is None else target_latent
    fore_norm = F.normalize(forecasted_latent, p=2, dim=-1)
    targ_norm = F.normalize(target, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1, :, :]              # f_t
    hy_norm = targ_norm[:, 1:, :, :].detach()          # sg(target_{t+1})
    cos_align = cosine_similarity_from_normalized(hy_hat_norm, hy_norm)
    copies = [weight * (2.0 - 2.0 * cos_align).mean()]
    for depth, f_depth in enumerate(rollout_latents or (), start=1):
        f_view, o_view, target_view = rollout_depth_views(
            f_depth, original_latent, depth, target_latent)
        copies.append(align_loss(f_view, o_view, weight,
                                 target_latent=target_view))
    return reduce_depth_terms(
        copies, resolve_rollout_reduce(depth_reduce, None))


def align_moco_loss(original_latent, teacher_original_latent, tau=0.10, weight=1.0):
    """MoCo-style InfoNCE with student encoder anchor + EMA-teacher keys.

    Positive:  cos(h_{b,t}, h^T_{b,t}) / τ
    Negatives: cos(h_{b,t}, h^T_{b',t}) / τ for all b' ≠ b (cross-batch, same t)

    Loss = LSE_{denom = {positive} ∪ {negatives}}(cos / τ) − log_pos, mean over
    anchors (b, t, c). Same-batch keys only (no cross-time), matching the
    `log_neg_cross_batch` denominator's structure — but here the anchor is
    the student's h_{b,t} (not the forecaster's f_{b,t}) and the keys are
    the teacher's h^T_{b',t} (not the student's h_{b',t}). Purpose is to
    prevent the student encoder from drifting too fast away from the EMA
    teacher: the query is student-side (gradient flows through h), the keys
    are teacher-side (gradient blocked by the EMA update path).

    original_latent, teacher_original_latent: ``[B, T, C, H]``. Returns scalar.
    """
    q = F.normalize(original_latent, p=2, dim=-1)                # [B, T, C, H]
    k = F.normalize(teacher_original_latent, p=2, dim=-1)        # [B, T, C, H]
    # sims[t, c, b1, b2] = cos(q[b1, t, c], k[b2, t, c]) / τ.
    q_p = q.permute(1, 2, 0, 3)                                   # [T, C, B, H]
    k_p = k.permute(1, 2, 0, 3)                                   # [T, C, B, H]
    sims = torch.matmul(q_p, k_p.transpose(-2, -1)) / tau         # [T, C, B, B]
    log_pos = torch.diagonal(sims, dim1=-2, dim2=-1).permute(2, 0, 1)   # [B, T, C]
    log_denom = torch.logsumexp(sims, dim=-1).permute(2, 0, 1)          # [B, T, C]
    return weight * (log_denom - log_pos).mean()


# --- All-time cross-series Gram speedups (#327) ----------------------------
#
# The all-time cross-series negative in `cosine_similarity_batch_full_hh_negs_
# xshh_allt` is an O((B·T)²·H) reduction whose full [B, B, T-1, T] Gram is
# never materialised — it is streamed over source chunks (`XSHH_ALLT_CHUNK`)
# and reduced on the fly by two nested logsumexp. At batch 2048 it dominates
# step time (#327). Two independent, default-OFF speedups, each toggled by its
# own env flag (both bit-identical to the default `checkpoint` path):
#
#   XSHH_ALLT_FUSED=1  — replace the per-chunk `torch.utils.checkpoint` with a
#       hand-written autograd.Function. Forward streams the chunks and keeps
#       only the final logsumexp `m`; backward recomputes each chunk's Gram,
#       forms the softmax weights w = exp(score − m) and accumulates the
#       anchor/source grads with one matmul each — same FLOPs as checkpoint
#       (the score recompute is the FlashAttention constraint) but without the
#       generic-autograd graph overhead or the many small chunk kernels.
#
#   XSHH_ALLT_SHARD=1  — under torchrun (world_size ≥ 2) shard the SOURCE
#       batch b' across ranks: each rank reduces its source slice with the
#       fused kernel, then an associative cross-rank logsumexp combines the
#       per-rank partials into the global value. Anchors stay global (the
#       existing DifferentiableAllGather already gives every rank all B
#       latents), so each rank does ~1/world_size of the (B·T)² matmul. The
#       backward all-reduces the source-slice grads into the global
#       anchor/source grads, so it composes with DifferentiableAllGather /
#       average_gradients exactly like the unsharded loss. Within each rank's
#       slice it uses the fused kernel, so it composes with XSHH_ALLT_FUSED.
#
# Both default OFF ⇒ the loss is byte-for-byte the checkpoint path.

def _xs_allt_chunked_lse(anchor, src, anchor_ids, src_ids, tau, chunk):
    """Streamed logsumexp of the all-time cross-series scores (no grad — used
    inside autograd.Function.forward).

      anchor [B, C, T-1, H]   anchors h_t
      src    [S, C, T,   H]   sources h_l (a contiguous slice of the B series)
      anchor_ids [B]          global batch id of each anchor
      src_ids    [S]          global batch id of each source (for self-masking)

    Returns `run` [B, C, T-1] = LSE over {source, l} of cos(h_t, h_l)/τ, with
    self-pairs (anchor id == source id) excluded. logsumexp is associative, so
    the result is exact and independent of `chunk`.
    """
    neg_inf = float('-inf')
    B = anchor.shape[0]
    S = src.shape[0]
    run = None
    for s in range(0, S, chunk):
        e = min(s + chunk, S)
        same = (anchor_ids.view(B, 1) == src_ids[s:e].view(1, e - s)).view(
            B, e - s, 1, 1, 1)
        gram = torch.matmul(
            anchor.unsqueeze(1),                              # [B, 1,  C, T-1, H]
            src[s:e].permute(0, 1, 3, 2).unsqueeze(0),        # [1, ch, C, H,   T]
        ) / tau                                               # [B, ch, C, T-1, T]
        gram = gram.masked_fill(same, neg_inf)
        chunk_lse = torch.logsumexp(torch.logsumexp(gram, dim=4), dim=1)  # [B,C,T-1]
        run = chunk_lse if run is None else torch.logsumexp(
            torch.stack([run, chunk_lse], dim=0), dim=0)
    if run is None:                                            # empty source slice
        run = anchor.new_full((B, anchor.shape[1], anchor.shape[2]), neg_inf)
    return run


def _xs_allt_chunked_grads(anchor, src, anchor_ids, src_ids, run, grad_out,
                           tau, chunk, need_tau):
    """Analytic backward of `_xs_allt_chunked_lse`. `run` is the logsumexp used
    as the softmax normaliser m (the global value for the single-device path,
    the local partial for the sharded path — folding the cross-rank weight into
    `grad_out` makes the local-m softmax reproduce the global one exactly).

    Returns (grad_anchor [B,C,T-1,H], grad_src [S,C,T,H], grad_tau or None).
    """
    neg_inf = float('-inf')
    B = anchor.shape[0]
    S = src.shape[0]
    grad_anchor = torch.zeros_like(anchor)
    grad_src = torch.zeros_like(src)
    grad_tau = None
    go = grad_out.unsqueeze(1).unsqueeze(-1)                   # [B, 1, C, T-1, 1]
    m = run.unsqueeze(1).unsqueeze(-1)                         # [B, 1, C, T-1, 1]
    for s in range(0, S, chunk):
        e = min(s + chunk, S)
        same = (anchor_ids.view(B, 1) == src_ids[s:e].view(1, e - s)).view(
            B, e - s, 1, 1, 1)
        score = torch.matmul(
            anchor.unsqueeze(1),
            src[s:e].permute(0, 1, 3, 2).unsqueeze(0),
        ) / tau                                               # [B, ch, C, T-1, T]
        # Softmax weight w = exp(score − m); self-pairs contribute exactly 0
        # (masked after the exp so an all-masked row's −inf−(−inf) NaN is
        # zeroed, not propagated).
        w = torch.exp(score.masked_fill(same, neg_inf) - m).masked_fill(same, 0.0)
        gw = w * go                                           # [B, ch, C, T-1, T]
        grad_anchor += torch.einsum('bjctl,jclh->bcth', gw, src[s:e]) / tau
        grad_src[s:e] = torch.einsum('bjctl,bcth->jclh', gw, anchor) / tau
        if need_tau:
            # ∂score/∂τ = −score/τ; `score` is finite (pre-mask) and gw is 0 at
            # masked positions, so the product is NaN-free.
            term = (gw * (-score / tau)).sum()
            grad_tau = term if grad_tau is None else grad_tau + term
    return grad_anchor, grad_src, grad_tau


class _XsAlltFusedLSE(torch.autograd.Function):
    """Single-device fused all-time cross-series logsumexp (XSHH_ALLT_FUSED=1).
    Drop-in for the per-chunk checkpoint loop: identical streamed forward, but
    a hand-written backward instead of an autograd graph over the chunks."""

    @staticmethod
    def forward(ctx, anchor, src, tau, chunk):
        ids = torch.arange(anchor.shape[0], device=anchor.device)
        with torch.no_grad():
            run = _xs_allt_chunked_lse(anchor, src, ids, ids, tau, chunk)
        ctx.tau_is_tensor = isinstance(tau, torch.Tensor)
        if ctx.tau_is_tensor:
            ctx.save_for_backward(anchor, src, run, tau)
        else:
            ctx.save_for_backward(anchor, src, run)
            ctx.tau = tau
        ctx.chunk = chunk
        return run

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.tau_is_tensor:
            anchor, src, run, tau = ctx.saved_tensors
        else:
            anchor, src, run = ctx.saved_tensors
            tau = ctx.tau
        ids = torch.arange(anchor.shape[0], device=anchor.device)
        need_tau = ctx.needs_input_grad[2]
        grad_anchor, grad_src, grad_tau = _xs_allt_chunked_grads(
            anchor, src, ids, ids, run, grad_out, tau, ctx.chunk, need_tau)
        return grad_anchor, grad_src, (grad_tau if need_tau else None), None


class _XsAlltShardedLSE(torch.autograd.Function):
    """Source-sharded all-time cross-series logsumexp (XSHH_ALLT_SHARD=1).
    Each rank reduces its slice of the source batch b' with the fused kernel,
    then an associative cross-rank logsumexp combines the per-rank partials.
    Backward all-reduces the source-slice grads into the global anchor/source
    grads, so it composes with DifferentiableAllGather / average_gradients
    exactly like the unsharded loss."""

    @staticmethod
    def forward(ctx, anchor, src, tau, chunk):
        B = anchor.shape[0]
        rank, world = dist.get_rank(), dist.get_world_size()
        start, end = rank * B // world, (rank + 1) * B // world
        ids = torch.arange(B, device=anchor.device)
        with torch.no_grad():
            partial = _xs_allt_chunked_lse(
                anchor, src[start:end], ids, ids[start:end], tau, chunk)
            gathered = [torch.empty_like(partial) for _ in range(world)]
            dist.all_gather(gathered, partial.contiguous())
            global_lse = torch.logsumexp(torch.stack(gathered, dim=0), dim=0)
        ctx.tau_is_tensor = isinstance(tau, torch.Tensor)
        if ctx.tau_is_tensor:
            ctx.save_for_backward(anchor, src, partial, global_lse, tau)
        else:
            ctx.save_for_backward(anchor, src, partial, global_lse)
            ctx.tau = tau
        ctx.chunk = chunk
        ctx.start, ctx.end = start, end
        return global_lse

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.tau_is_tensor:
            anchor, src, partial, global_lse, tau = ctx.saved_tensors
        else:
            anchor, src, partial, global_lse = ctx.saved_tensors
            tau = ctx.tau
        B = anchor.shape[0]
        start, end, chunk = ctx.start, ctx.end, ctx.chunk
        ids = torch.arange(B, device=anchor.device)
        need_tau = ctx.needs_input_grad[2]
        # ∂L/∂partial = grad_out · exp(partial − global_lse). Folding this α
        # into the local softmax (normaliser = partial) reproduces the global
        # weight exp(score − global_lse) exactly, so the per-rank source-slice
        # grads sum (all-reduce) to the full gradient.
        gp = grad_out * torch.exp(partial - global_lse)
        grad_anchor, grad_src_slice, grad_tau = _xs_allt_chunked_grads(
            anchor, src[start:end], ids, ids[start:end], partial, gp,
            tau, chunk, need_tau)
        grad_src = torch.zeros_like(src)
        grad_src[start:end] = grad_src_slice
        grad_anchor = grad_anchor.contiguous()
        grad_src = grad_src.contiguous()
        dist.all_reduce(grad_anchor, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_src, op=dist.ReduceOp.SUM)
        if need_tau:
            if grad_tau is None:
                grad_tau = anchor.new_zeros(())
            grad_tau = grad_tau.contiguous()
            dist.all_reduce(grad_tau, op=dist.ReduceOp.SUM)
        return grad_anchor, grad_src, (grad_tau if need_tau else None), None


def contrastive_latent_loss(predicted_position, validation, spec,
                            get_history=False, tau_override=None,
                            include_positive_in_denominator=False,
                            align_loss_weight=None,
                            align_target=None,
                            subtract_contrastive_floor=None,
                            moco_negatives=None,
                            moco_rep_keys=None,
                            teacher_original_latent=None,
                            rollout_latents=None,
                            train_rollout_depth=None,
                            train_rollout_reduce=None,
                            depth_index=0,
                            f_terms_only=False):
    """Compute the contrastive divergence loss.

    Args:
        predicted_position: tuple of (forecasted_latent, original_latent).
        validation: True during validation (skips training-only paths).
        spec: SimpleNamespace with `train_configuration` dict.
        get_history: if True, returns intermediate (kept for compat).
        tau_override: optional tensor or float overriding the dict's
            `contrastive_divergence_temperature`. Used by the
            learnable-τ trainer to pass a 0-d tensor that gets gradient.
        include_positive_in_denominator: opt-in, default False. When
            False the loss is byte-for-byte the historical training
            objective — for the logsumexp variants that is the
            negatives-only form ``(log_neg_total - log_pos).mean()``
            ≈ ``-log(e^pos / Σ_neg e^neg)``, which is unbounded above
            and goes NEGATIVE once positives separate from negatives.
            When True, the positive is added to BOTH numerator and
            denominator, giving a proper normalized InfoNCE
            ``(logsumexp([log_pos, log_neg_total]) - log_pos).mean()``
            = ``-log(e^pos / (e^pos + Σ_neg e^neg))`` which is always
            ≥ 0. May also be requested per-run via the training config
            key ``train_configuration['include_positive_in_denominator']``
            (the two are OR-ed; the function arg is what the diagnostic
            `loss_tau_ref` column passes, the config key is the knob a
            training run sets — e.g. the ``--pos-in-denominator`` CLI
            flag). The default stays False on both, so every
            past/running experiment's objective is byte-for-byte
            unchanged. Implemented for the logsumexp-form variants
            (`cosine_similarity_batch`,
            `cosine_similarity_batch_no_time_neg`,
            `cosine_similarity_batch_square`,
            `cosine_similarity_batch_full_fh_negs`,
            `cosine_similarity_batch_full_hh_negs`,
            `cosine_similarity_batch_full_ff_negs`,
            `cosine_similarity_batch_full_fh_hh_negs`,
            `cosine_similarity_batch_full_hh_ff_negs`,
            `cosine_similarity_batch_full_fh_hh_ff_negs`,
            `cosine_similarity_batch_full_hh_negs_xbfree`); requesting it
            with any other `loss_shape` raises NotImplementedError rather
            than silently returning an unintended value.

    The temperature τ acts as a divisor on cosine similarities. When
    `tau_override` is a tensor, gradient flows through the loss back to
    the caller's parameter (CLIP-style learnable temperature, #28).

    `align_loss_weight` (λ; default None → config key `align_loss_weight`,
    default 0.0 = off) adds a BYOL/SimSiam alignment term
    λ·(2 − 2·cos(f_t, sg(h_{t+1}))) on top of the contrastive loss — a
    non-saturating positive: its per-cosine gradient is a constant −2,
    independent of the negatives, vs the InfoNCE positive's −(1−p₊)/τ which
    fades once the negatives separate (p₊→1). `subtract_contrastive_floor`
    (default None → config key, default False) re-bases the loss by the
    constant `infonce_floor(τ, N)` (gradient-neutral; logged-value only).
    `align_loss_weight` applies to ANY `loss_shape` (it needs only the
    positive pair); `subtract_contrastive_floor` is restricted to
    `_NORMALIZED_FORM_SHAPES` with positive-in-denominator (it is the floor
    of THAT objective). An explicit function arg overrides the config key
    (the `loss_tau_ref` diagnostic passes 0/False to stay a pure
    contrastive reference). See #309.

    `align_target` (#390; default None → config key `align_target`, default
    `'student'`) picks whose h_{t+1} the alignment term above pulls toward:
    `'student'` is the encoder's own sg(h_{t+1}) — what #379 trained on —
    and `'teacher'` is the EMA teacher's h_{t+1} from
    `teacher_original_latent`, the BYOL form the term was designed for
    (slowly-moving target the student cannot chase within a step).
    `'teacher'` without a teacher latent raises: falling back to the
    student silently is the defect this argument removes.

    Config key ``stopgrad_positive_h`` (default False; the
    ``--stopgrad-positive-h`` CLI flag): SimSiam/BYOL-style target
    stop-grad on the InfoNCE positive — detach the encoder side h_{t+1}
    of sim(h_{t+1}, f_{t+1}) in the positive term (numerator and, with
    pos-in-denominator, denominator; negatives keep gradient on h).
    Forward value unchanged; xshh_allt loss shape only (raises otherwise).

    ``teacher_original_latent`` (#353): when provided, an EMA-teacher copy
    of the encoder output of the same input (shape matches
    ``original_latent``). The teacher's h_{t+1} replaces the student's
    as the main-contrastive POSITIVE (numerator and, with pos-in-denom,
    denominator) — a slowly-moving target with implicit stop-grad
    (the teacher carries no autograd graph). Negatives stay on the
    student. Mutually exclusive with ``stopgrad_positive_h`` (the
    teacher is the stop-grad). Only implemented for the xshh_allt
    loss shape (raises otherwise).

    Config keys ``pred_loss_weight`` / ``rep_loss_weight`` (#382; both
    default 1.0 → historical objective unchanged): per-term scalar
    weights on L_pred and L_rep in the ``..._split_pred_rep`` shape.
    Setting one to 0.0 isolates the other term for the loss-term-
    isolation ablation. No-op for every other loss shape.

    Rollout depth (#373; config key ``train_rollout_depth``, default 0 →
    historical objective unchanged). ``rollout_latents`` carries
    f^(1)..f^(k), the forecaster applied to its own output (see
    `src.forecasting_head.rollout_forecaster_latents`); ``k`` copies of
    every f-bearing term are then built beside the depth-0 one, copy ``j``
    tying f^(j)_t to h_{t+1+j}. ``train_rollout_depth`` is the function-arg
    override of the config key, with the usual precedence.
    ``depth_index`` says which copy THIS call computes: the depth loop below
    passes j > 0 with already-shifted views (see
    :func:`rollout_depth_views`), and a copy skips the terms that carry no
    ``f`` — L_rep, L_rep_moco, and ``mse``'s − mse(h_t, h_{t+1}) half — which
    the depth-0 call already counted once.

    Rollout reduction (#401; config key ``train_rollout_reduce``, default
    ``'sum'`` → #373's objective unchanged). ``'mean'`` divides the k + 1
    copies by k + 1, so the f-side holds its k = 0 weight at every depth.
    The mean covers the f-bearing copies only: the terms that carry no ``f``
    enter once at their configured weight under both reductions.
    ``train_rollout_reduce`` is the function-arg override of the config key.

    ``f_terms_only``: return the f-bearing part of THIS call alone, the part
    a depth copy carries. It is what the mean divides — the depth-0 call
    returns the f-side and the h-side added together, and the two have to be
    told apart before one of them can be divided.
    """
    forecasted_latent, original_latent = predicted_position
    train_config = spec.train_configuration

    # Rollout depth (#373). Resolved before the dispatch so a shape that
    # cannot compose the forecaster on its own output fails loud below
    # instead of ignoring the flag.
    depth = resolve_rollout_depth(
        train_rollout_depth, train_config.get('train_rollout_depth'),
        rollout_latents, original_latent.shape[1])
    reduce = resolve_rollout_reduce(
        train_rollout_reduce, train_config.get('train_rollout_reduce'))

    # True when this call must carry the f-bearing terms and nothing else:
    # every rollout-depth copy (depth_index > 0), and the depth-0 f-side the
    # `mean` reduction subtracts before it divides (#401).
    f_only = bool(depth_index) or bool(f_terms_only)

    # CPC multi-step (#316): the forecaster latent is a [B,T,C,K,H] stack of
    # K linear-head predictions, so the 4-D unpack/positives below do not
    # apply. Dispatch to the dedicated multi-step InfoNCE and return early.
    # align_loss_weight / subtract_contrastive_floor are not defined for this
    # variant (no single positive pair / closed-form floor) and are ignored.
    if train_config.get('loss_shape') in ('cpc_multistep', 'cpc_multistep_cpcnegs'):
        # #373: the only family that cannot compose. Its forecaster is K
        # parallel linear heads, f^(k)_t = W_k h_t, each predicting its own
        # horizon straight from h_t — there is no single operator to apply to
        # its own output. Raise rather than ignore the flag.
        if depth > 0:
            raise NotImplementedError(
                "train_rollout_depth is not defined for loss_shape="
                f"{train_config.get('loss_shape')!r}: its forecaster is K "
                "parallel linear heads (f^(k)_t = W_k h_t), so there is no "
                "single operator to apply to its own output. Use the "
                "transformer forecaster, or drop --train-rollout-depth.")
        # stopgrad_positive_h is implemented only in the xshh_allt branch;
        # the CPC variants return before the tail guard below, so fail loud
        # here rather than silently training without the stop-grad.
        if bool(train_config.get('stopgrad_positive_h', False)):
            raise NotImplementedError(
                "stopgrad_positive_h is only implemented for loss_shape="
                "'cosine_similarity_batch_full_hh_negs_xshh_allt'; got "
                f"{train_config.get('loss_shape')!r}.")
        if tau_override is not None:
            tau = tau_override
        else:
            tau = train_config.get('contrastive_divergence_temperature', 1.0)
        pos_in_denom = include_positive_in_denominator or bool(
            train_config.get('include_positive_in_denominator', False))
        _cpc_fn = (cpc_multistep_cpcnegs_loss
                   if train_config.get('loss_shape') == 'cpc_multistep_cpcnegs'
                   else cpc_multistep_loss)
        loss = _cpc_fn(
            forecasted_latent, original_latent, tau,
            include_positive_in_denominator=pos_in_denom)
        if get_history:
            return loss, (forecasted_latent, original_latent)
        return loss

    B, T, C, H = forecasted_latent.shape
    if tau_override is not None:
        tau = tau_override
    else:
        tau = train_config.get('contrastive_divergence_temperature', 1.0)

    # #379 — separate τ for the L_rep term of split shapes. When
    # `contrastive_divergence_temperature_rep` is absent or None (default),
    # `tau_rep` aliases `tau`, so the objective for the 6-base arms of #374
    # / #379 and every pre-#379 run is byte-for-byte unchanged. When set
    # (e.g. via the training script's --tau-rep flag), the split_pred_rep
    # and rep_only branches use it for the h-anchored family LSE terms
    # (log_neg_xx, log_neg_hh_all, log_neg_xs_allt, and log_pos_h when
    # moco_rep_keys is on); L_pred stays on `tau`. tau_override still wins
    # for both (it's the learnable-τ / loss_tau_ref pipeline).
    if tau_override is not None:
        tau_rep = tau_override
    else:
        _tr = train_config.get('contrastive_divergence_temperature_rep')
        tau_rep = tau if _tr is None else _tr

    # Normalized-InfoNCE (positive in BOTH numerator and denominator) is
    # opt-in via EITHER the function arg (diagnostic loss_tau_ref) OR the
    # training-config key (a run-level knob, e.g. the --pos-in-denominator
    # CLI flag). Default False on both ⇒ historical objective unchanged.
    pos_in_denom = include_positive_in_denominator or bool(
        train_config.get('include_positive_in_denominator', False)
    )

    # SimSiam/BYOL-style target stop-grad on the InfoNCE POSITIVE (#336
    # follow-up): detach the ENCODER side h_{t+1} of the positive cosine
    # sim(h_{t+1}, f_{t+1}) everywhere that term appears — the numerator
    # and, under pos_in_denom, the denominator (both read `log_pos`, so
    # detaching inside `log_pos` covers both). Every NEGATIVE keeps its
    # gradient on h. Detach cuts only the backward edge, so the forward
    # loss value is unchanged. Default False ⇒ byte-for-byte historical
    # objective; only implemented for the xshh_allt shape (guarded below).
    sg_pos = bool(train_config.get('stopgrad_positive_h', False))

    # BYOL/JEPA EMA-teacher positive (#353): when a teacher latent is supplied,
    # the encoder side of the main-contrastive positive comes from the slowly-
    # moving teacher (h^T_{t+1}) instead of the student. The teacher tensor
    # already carries no grad (computed under no_grad in the model), so this
    # is the stop-grad-on-target version of `sg_pos` with a smoothed target.
    # Negatives keep the student's `hy_norm`. Mutually exclusive with sg_pos:
    # the trainer rejects both flags together (cleaner intent).
    if teacher_original_latent is not None and sg_pos:
        raise ValueError(
            "teacher_original_latent and stopgrad_positive_h are mutually "
            "exclusive: the teacher path IS the target stop-grad.")

    noise_sigma = train_config.get('contrastive_latent_noise')
    if noise_sigma is not None and not validation:
        forecasted_latent = forecasted_latent + torch.randn_like(forecasted_latent) * noise_sigma

    hy_hat = forecasted_latent[:, :-1, :, :]
    hx = original_latent[:, :-1, :, :]
    hy = original_latent[:, 1:, :, :]

    orig_norm = F.normalize(original_latent, p=2, dim=-1)
    fore_norm = F.normalize(forecasted_latent, p=2, dim=-1)
    hy_hat_norm = fore_norm[:, :-1, :, :]
    hz_hat_norm = fore_norm[:, 1:, :, :]
    hx_norm = orig_norm[:, :-1, :, :]
    hy_norm = orig_norm[:, 1:, :, :]

    # Teacher-side normalized positive (only used in the xshh_allt branch).
    if teacher_original_latent is not None:
        teacher_orig_norm = F.normalize(teacher_original_latent, p=2, dim=-1)
        hy_teacher_norm = teacher_orig_norm[:, 1:, :, :]
    else:
        teacher_orig_norm = None
        hy_teacher_norm = None

    if train_config.get('loss_shape') == 'cosine_similarity_old':
        positives = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat
        loss = -torch.log(positives / negatives).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity':
        positives = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat
        # print(positives.shape, negatives.shape)
        # In the new version, all positives together, all negatives together, cross batch.
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch':
        # Numerically stable logsumexp form — equivalence + fp16/fp32 small-τ
        # stability pinned in tests/test_loss_stability.py.
        neg_inf = float('-inf')
        log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        log_neg_xy = torch.logsumexp(sims_xy / tau, dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_xy_hat = torch.logsumexp(sims_xy_hat / tau, dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

        # Cross-batch negatives: compare across batch dimension (not just C dimension).
        # Matmul form below is equivalent to the broadcast form but avoids
        # materialising the [B, B, T-1, C, H] intermediate (~25 GB at B=256,
        # H=384, T-1=255 fp32). For unit-normalised vectors, cos(u, v) = u · v.
        # Index convention: sims[b1, b2, t, c] = cos(hy[b2, t, c], hy_hat[b1, t, c]).
        hy_p = hy_norm.permute(1, 2, 0, 3)              # [T-1, C, B, H]
        hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)      # [T-1, C, B, H]
        sims_cross_batch = torch.matmul(
            hy_hat_p, hy_p.transpose(-2, -1)            # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()              # [B, B, T-1, C]

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        log_neg_cross_batch = torch.logsumexp(
            (sims_cross_batch / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )

        negatives = torch.stack(
            [log_neg_xy, log_neg_xx, log_neg_zy, log_neg_xy_hat, log_neg_cross_batch],
            dim=0,
        )
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            # Normalized InfoNCE (loss_tau_ref diagnostic OR the
            # --pos-in-denominator training knob): add the positive to
            # the denominator → loss = -log(e^pos / (e^pos + Σ_neg e^neg))
            # ≥ 0 always. Broadcast log_neg_total (keepdim batch=1)
            # against the per-anchor log_pos before logsumexp.
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_full_fh_negs':
        # Same as `cosine_similarity_batch`, except the single l = t
        # forecaster–encoder negative (`log_neg_xy_hat` = cos(h_t, f_t)) is
        # REPLACED by the full set of (f_t, h_l) negatives over EVERY time
        # position l, excluding only the positive target l = t+1. Anchor
        # f_t is the forecaster at index t (t = 0..T-2); h_l is the encoder
        # at index l (l = 0..T-1), same (b, c). The l = t slice equals the
        # old same-channel xy_hat term (identical for C = 1, the training
        # config); l ∈ {0..t-1, t+2..T-1} are the genuinely new negatives.
        # All other terms (xy, xx, zy, cross-batch) are byte-for-byte
        # unchanged from `cosine_similarity_batch`. Numerically stable
        # logsumexp form — same shape contract and cross-batch pooling.
        neg_inf = float('-inf')
        log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        log_neg_xy = torch.logsumexp(sims_xy / tau, dim=2)

        # REPLACES log_neg_xy_hat: full (f_t, h_l) negatives for all l != t+1.
        # sims_fh[b, c, t, l] = cos(f_t^{b,c}, h_l^{b,c}). One batched matmul
        # (no Python loop, no [B,T-1,C,T,H] broadcast intermediate); for
        # unit-normalised vectors cos(u, v) = u · v. Kept in [B,C,T-1,T] so
        # logsumexp reduces the contiguous last (l) axis and only the small
        # [B,T-1,C] result is permuted — avoids a full-size transpose-copy.
        sims_fh = torch.matmul(
            hy_hat_norm.permute(0, 2, 1, 3),            # [B, C, T-1, H]  (f_t)
            orig_norm.permute(0, 2, 3, 1),              # [B, C, H, T]    (h_l)
        )                                               # [B, C, T-1, T]
        # Mask only the positive target l = t+1 for each anchor t (0..T-2);
        # every other l (incl. l = t) stays an active negative.
        t_idx = torch.arange(T - 1, device=sims_fh.device).view(T - 1, 1)
        l_idx = torch.arange(T, device=sims_fh.device).view(1, T)
        pos_mask = (l_idx == t_idx + 1).view(1, 1, T - 1, T)
        log_neg_fh_all = torch.logsumexp(
            (sims_fh / tau).masked_fill(pos_mask, neg_inf), dim=3
        ).permute(0, 2, 1)                              # [B, T-1, C]

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

        # Cross-batch negatives: compare across batch dimension (not just C dimension).
        # Matmul form below is equivalent to the broadcast form but avoids
        # materialising the [B, B, T-1, C, H] intermediate (~25 GB at B=256,
        # H=384, T-1=255 fp32). For unit-normalised vectors, cos(u, v) = u · v.
        # Index convention: sims[b1, b2, t, c] = cos(hy[b2, t, c], hy_hat[b1, t, c]).
        hy_p = hy_norm.permute(1, 2, 0, 3)              # [T-1, C, B, H]
        hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)      # [T-1, C, B, H]
        sims_cross_batch = torch.matmul(
            hy_hat_p, hy_p.transpose(-2, -1)            # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()              # [B, B, T-1, C]

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        log_neg_cross_batch = torch.logsumexp(
            (sims_cross_batch / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )

        negatives = torch.stack(
            [log_neg_xy, log_neg_xx, log_neg_zy, log_neg_fh_all, log_neg_cross_batch],
            dim=0,
        )
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            # Normalized InfoNCE (loss_tau_ref diagnostic OR the
            # --pos-in-denominator training knob) — see the
            # cosine_similarity_batch branch for the rationale.
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') in (
        'cosine_similarity_batch_full_hh_negs',
        'cosine_similarity_batch_full_ff_negs',
        'cosine_similarity_batch_full_fh_hh_negs',
        'cosine_similarity_batch_full_hh_ff_negs',
        'cosine_similarity_batch_full_fh_hh_ff_negs',
    ):
        # Sibling crossed-negative variants of
        # `cosine_similarity_batch_full_fh_negs` (#303, extended #307).
        # Built by the IDENTICAL structural transform of
        # `cosine_similarity_batch`: the single l = t forecaster–encoder
        # negative (`log_neg_xy_hat` = cos(h_t, f_t)) is REPLACED by an
        # all-time-position crossed term, same (b, c), differing only in
        # which modality pair and which l is masked:
        #   full_hh  (B): (h_t, h_l) ∀ l ≠ t      encoder–encoder, mask self
        #   full_ff  (C): (f_t, f_l) ∀ l ≠ t      forecaster–forecaster, mask self
        #   full_fh_hh (A)+(B): BOTH the full_fh_negs (f_t, h_l) ∀ l ≠ t+1
        #                       term AND the (B) (h_t, h_l) ∀ l ≠ t term.
        #   full_hh_ff (B)+(C) [#307]: BOTH (B) and (C) all-time terms —
        #                       within-branch only on the all-time axis,
        #                       NO all-time f–h.
        #   full_fh_hh_ff (A)+(B)+(C) [#307]: all three all-time crossed
        #                       terms together.
        # The cross-batch negative is the standard f–h `log_neg_cross_batch`
        # for ALL of these (byte-for-byte the #303 arms' cross-batch — only
        # the all-time axis changes here; cf. the separate
        # `cosine_similarity_batch_full_hh_negs_xbfree` arm, which is the
        # only one that also alters the cross-batch axis).
        # These are the time-crossed (l ≠ t / l ≠ t+1, same b,c) siblings
        # of (A); distinct from the batch-crossed h×h / f×f already in
        # `cosine_similarity_batch_square` (b ≠ b', fixed t). Anchor index
        # t = 0..T-2 (the [B,T-1,C] contract); l ranges over all T encoder
        # / forecaster positions. (h_t, h_t) and (f_t, f_t) are cos = 1
        # self-pairs (masked); for (A) the masked l = t+1 is the positive
        # target. All other terms (xy, xx, zy, cross-batch) are
        # byte-for-byte identical to `cosine_similarity_batch`.
        # Numerically stable logsumexp form; --pos-in-denominator
        # supported via the shared tail below.
        which = train_config.get('loss_shape')
        want_fh = which in (
            'cosine_similarity_batch_full_fh_hh_negs',
            'cosine_similarity_batch_full_fh_hh_ff_negs',
        )
        want_hh = which in (
            'cosine_similarity_batch_full_hh_negs',
            'cosine_similarity_batch_full_fh_hh_negs',
            'cosine_similarity_batch_full_hh_ff_negs',
            'cosine_similarity_batch_full_fh_hh_ff_negs',
        )
        want_ff = which in (
            'cosine_similarity_batch_full_ff_negs',
            'cosine_similarity_batch_full_hh_ff_negs',
            'cosine_similarity_batch_full_fh_hh_ff_negs',
        )
        neg_inf = float('-inf')
        log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        log_neg_xy = torch.logsumexp(sims_xy / tau, dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

        # Shared all-time crossed-negative builder. anchor_t is the anchor
        # at t = 0..T-2 ([B,T-1,C,H]); src_l is the full l = 0..T-1 stack
        # ([B,T,C,H]). One batched matmul (no Python loop, no
        # [B,T-1,C,T,H] broadcast); for unit-normalised vectors
        # cos(u, v) = u · v. Kept in [B,C,T-1,T] so logsumexp reduces the
        # contiguous last (l) axis; only the small [B,T-1,C] result is
        # permuted (avoids a full-size transpose-copy). `drop_l` selects
        # which l is masked per anchor t: t+1 (the positive, for the
        # f–h term) or t (the cos=1 self-pair, for h–h / f–f).
        t_idx = torch.arange(T - 1, device=orig_norm.device).view(T - 1, 1)
        l_idx = torch.arange(T, device=orig_norm.device).view(1, T)

        def _full_time_neg(anchor_t_norm, src_l_norm, drop_l):
            sims = torch.matmul(
                anchor_t_norm.permute(0, 2, 1, 3),   # [B, C, T-1, H]
                src_l_norm.permute(0, 2, 3, 1),       # [B, C, H, T]
            )                                          # [B, C, T-1, T]
            mask = (l_idx == drop_l).view(1, 1, T - 1, T)
            return torch.logsumexp(
                (sims / tau).masked_fill(mask, neg_inf), dim=3
            ).permute(0, 2, 1)                         # [B, T-1, C]

        neg_terms = [log_neg_xy, log_neg_xx, log_neg_zy]
        if want_fh:
            # (A) term: (f_t, h_l) ∀ l ≠ t+1 — drop the positive target.
            neg_terms.append(
                _full_time_neg(hy_hat_norm, orig_norm, t_idx + 1)
            )
        if want_hh:
            # (B) term: (h_t, h_l) ∀ l ≠ t — drop the cos=1 self-pair.
            neg_terms.append(
                _full_time_neg(hx_norm, orig_norm, t_idx)
            )
        if want_ff:
            # (C) term: (f_t, f_l) ∀ l ≠ t — drop the cos=1 self-pair.
            neg_terms.append(
                _full_time_neg(hy_hat_norm, fore_norm, t_idx)
            )

        # Cross-batch negatives: byte-for-byte the
        # `cosine_similarity_batch_full_fh_negs` matmul form. Index
        # convention: sims[b1, b2, t, c] = cos(hy[b2,t,c], hy_hat[b1,t,c]).
        hy_p = hy_norm.permute(1, 2, 0, 3)              # [T-1, C, B, H]
        hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)      # [T-1, C, B, H]
        sims_cross_batch = torch.matmul(
            hy_hat_p, hy_p.transpose(-2, -1)            # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()              # [B, B, T-1, C]
        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)
        log_neg_cross_batch = torch.logsumexp(
            (sims_cross_batch / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )
        neg_terms.append(log_neg_cross_batch)

        negatives = torch.stack(neg_terms, dim=0)
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            # Normalized InfoNCE (loss_tau_ref diagnostic OR the
            # --pos-in-denominator training knob) — see the
            # cosine_similarity_batch branch for the rationale.
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_full_hh_negs_xbfree':
        # (B), cross-branch-(f↔h)-negative-free (#307). Arm (B)'s all-time
        # encoder–encoder transform — the single l = t forecaster–encoder
        # negative (`log_neg_xy_hat` = cos(h_t, f_t)) REPLACED by the
        # all-time (h_t, h_l) ∀ l ≠ t term (identical builder/mask to
        # `cosine_similarity_batch_full_hh_negs`) — AND the cross-batch
        # axis rebuilt the `cosine_similarity_batch_square` way: the f↔h
        # cross-batch term (`log_neg_cross_fe`, the ONLY cross-batch term
        # in `cosine_similarity_batch` / the #303 arms) is DROPPED and the
        # two within-branch square edges are kept instead —
        #   cross_ff: f_b ↔ f_b'        at same t, b ≠ b'
        #   cross_hh: h_{b,t+1} ↔ h_{b',t+1} at same t, b ≠ b'
        # NET: NO f↔h NEGATIVE anywhere (neither all-time nor cross-batch);
        # the f↔h *positive* (log_pos = cos(h_{t+1}, f_t)) is retained.
        # xy (cos(h_t,h_{t+1}) cross-channel = h–h), xx (h–h cross-channel
        # same-time), zy (f_{t+1}↔f_t cross-channel = f–f) are byte-for-byte
        # identical to `cosine_similarity_batch` and are all within-branch
        # (no f↔h). square's cross_ff/cross_hh are written here in the
        # matmul form, NOT square's [B,B,T-1,C,H] broadcast (that is ~10 TB
        # at the of-record T=4096,B=128); this is the memory-safe exact
        # equivalent (logsumexp over the other-batch axis of a symmetric
        # f·f / h·h Gram, diagonal masked), pinned in tests. Numerically
        # stable logsumexp; --pos-in-denominator supported via the shared
        # tail below.
        neg_inf = float('-inf')
        log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        log_neg_xy = torch.logsumexp(sims_xy / tau, dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

        # (B) all-time term: (h_t, h_l) ∀ l ≠ t. anchor h_t (t=0..T-2),
        # src h_l (l=0..T-1), drop the cos≡1 self-pair l = t. Byte-for-byte
        # the `cosine_similarity_batch_full_hh_negs` builder/mask.
        t_idx = torch.arange(T - 1, device=orig_norm.device).view(T - 1, 1)
        l_idx = torch.arange(T, device=orig_norm.device).view(1, T)
        sims_hh_all = torch.matmul(
            hx_norm.permute(0, 2, 1, 3),     # [B, C, T-1, H]  (h_t)
            orig_norm.permute(0, 2, 3, 1),   # [B, C, H, T]    (h_l)
        )                                     # [B, C, T-1, T]
        drop_self = (l_idx == t_idx).view(1, 1, T - 1, T)
        log_neg_hh_all = torch.logsumexp(
            (sims_hh_all / tau).masked_fill(drop_self, neg_inf), dim=3
        ).permute(0, 2, 1)                    # [B, T-1, C]

        # Within-branch cross-batch edges (square's log_neg_cross_ff /
        # log_neg_cross_hh), matmul form. sims is the symmetric f·f / h·h
        # Gram over the batch axis (per t, c); diagonal b1=b2 masked;
        # logsumexp over the other-batch axis — exactly square's
        # broadcast + logsumexp(dim=1). The f↔h `log_neg_cross_fe` is
        # intentionally ABSENT (the experiment's defining change).
        mask_batch = ~torch.eye(B, dtype=torch.bool, device=orig_norm.device)
        mask_batch = mask_batch.view(B, B, 1, 1)
        hy_p = hy_norm.permute(1, 2, 0, 3)          # [T-1, C, B, H]  h_{t+1}
        hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)  # [T-1, C, B, H]  f_t
        sims_cb_ff = torch.matmul(
            hy_hat_p, hy_hat_p.transpose(-2, -1)    # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()          # [B, B, T-1, C]
        log_neg_cross_ff = torch.logsumexp(
            (sims_cb_ff / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )
        sims_cb_hh = torch.matmul(
            hy_p, hy_p.transpose(-2, -1)            # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()          # [B, B, T-1, C]
        log_neg_cross_hh = torch.logsumexp(
            (sims_cb_hh / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )

        negatives = torch.stack(
            [log_neg_xy, log_neg_xx, log_neg_zy, log_neg_hh_all,
             log_neg_cross_ff, log_neg_cross_hh],
            dim=0,
        )
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            # Normalized InfoNCE (loss_tau_ref diagnostic OR the
            # --pos-in-denominator training knob) — see the
            # cosine_similarity_batch branch for the rationale.
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_full_hh_negs_xshh':
        # #318 — β (`cosine_similarity_batch_full_hh_negs`) + cross-SERIES,
        # same-STEP encoder–encoder negatives at EVERY step ("deny the
        # positional shortcut"). Two clean edits on top of (B)/β:
        #   (1) ADD `log_neg_xshh`: cos(h_{b,t}, h_{b',t}) ∀ b' ≠ b — the
        #       cross-series, same-step h↔h repulsion. Anchored at h_t
        #       (`hx_norm`, t = 0..T-2), so it acts at every anchor step l = t,
        #       NOT only the target step t+1 (the one step the
        #       `cosine_similarity_batch_square` cross_hh edge touched). At a
        #       fixed step the only structure DIFFERENT series share is the
        #       content-free positional code; repelling it moves distinctness
        #       onto (series-specific, forecastable) content.
        #   (2) REMOVE `log_neg_xy`: cos(h_t, h_{t+1}) (adjacent encoder). For
        #       the C = 1 training config it is BYTE-FOR-BYTE the l = t+1 slice
        #       of (B)'s all-time (h_t, h_l) ∀ l ≠ t term — a duplicate — so
        #       dropping it de-duplicates rather than weakening the objective.
        # Everything else (`log_pos`, xx, zy, the all-time hh term, and the
        # cross-batch f↔h `log_neg_cross_batch`) is byte-for-byte (B). The new
        # edge uses the memory-safe matmul Gram form over the batch axis (NOT a
        # [B,B,T-1,C,H] broadcast); diagonal b = b' (cos ≡ 1 self-pair) masked.
        # Numerically stable logsumexp; --pos-in-denominator supported via the
        # shared tail below. See #318.
        neg_inf = float('-inf')
        log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau

        # xx (cross-channel same-time h↔h; −inf / no-op at C=1) and zy
        # (f_{t+1}↔f_t cross-channel) — byte-for-byte (B). `log_neg_xy`
        # (adjacent h↔h) is intentionally ABSENT (edit 2).
        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

        # (B) all-time term: (h_t, h_l) ∀ l ≠ t. Byte-for-byte the
        # `cosine_similarity_batch_full_hh_negs` builder/mask (l = t self-pair
        # dropped); the l = t+1 slice it contains is what edit (2) removed the
        # duplicate of.
        t_idx = torch.arange(T - 1, device=orig_norm.device).view(T - 1, 1)
        l_idx = torch.arange(T, device=orig_norm.device).view(1, T)
        sims_hh_all = torch.matmul(
            hx_norm.permute(0, 2, 1, 3),     # [B, C, T-1, H]  (h_t)
            orig_norm.permute(0, 2, 3, 1),   # [B, C, H, T]    (h_l)
        )                                     # [B, C, T-1, T]
        drop_self = (l_idx == t_idx).view(1, 1, T - 1, T)
        log_neg_hh_all = torch.logsumexp(
            (sims_hh_all / tau).masked_fill(drop_self, neg_inf), dim=3
        ).permute(0, 2, 1)                    # [B, T-1, C]

        # Cross-batch f↔h negative — byte-for-byte (B)'s `log_neg_cross_batch`.
        # sims[b1,b2,t,c] = cos(hy[b2,t,c], hy_hat[b1,t,c]) = cos(f_{b1,t}, h_{b2,t+1}).
        mask_batch = ~torch.eye(B, dtype=torch.bool, device=orig_norm.device)
        mask_batch = mask_batch.view(B, B, 1, 1)
        hy_p = hy_norm.permute(1, 2, 0, 3)          # [T-1, C, B, H]  h_{t+1}
        hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)  # [T-1, C, B, H]  f_t
        sims_cross_batch = torch.matmul(
            hy_hat_p, hy_p.transpose(-2, -1)        # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()          # [B, B, T-1, C]
        log_neg_cross_batch = torch.logsumexp(
            (sims_cross_batch / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )

        # NEW (edit 1): cross-series, same-step h↔h — cos(h_{b,t}, h_{b',t})
        # ∀ b' ≠ b, anchored at h_t (so it acts at every step l = t). Same
        # memory-safe batch-Gram form as cross_batch, but h_t↔h_t (SAME step)
        # across the batch; diagonal b = b' (cos ≡ 1) masked.
        hx_p = hx_norm.permute(1, 2, 0, 3)          # [T-1, C, B, H]  h_t
        sims_xshh = torch.matmul(
            hx_p, hx_p.transpose(-2, -1)            # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()          # [B, B, T-1, C]
        log_neg_xshh = torch.logsumexp(
            (sims_xshh / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )

        negatives = torch.stack(
            [log_neg_xx, log_neg_zy, log_neg_hh_all,
             log_neg_cross_batch, log_neg_xshh],
            dim=0,
        )
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            # Normalized InfoNCE (loss_tau_ref diagnostic OR the
            # --pos-in-denominator training knob) — see the
            # cosine_similarity_batch branch for the rationale.
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_full_hh_negs_xshh_allt':
        # #318 ablation (the l=t-vs-all-l fork) — the ALL-TIME sibling of
        # `..._xshh`. Identical to it (β, drop adjacent log_neg_xy, keep
        # everything else byte-for-β) EXCEPT the cross-series h↔h edge ranges
        # over EVERY source step l, not only the same step:
        #     log_neg_xs_allt = LSE_{b'≠b, ∀ l} cos(h_{b,t}, h_{b',l}) / τ
        # i.e. the cross-SERIES analog of (B)'s within-series all-time term
        # (which repels h_t from h_l ∀ l within the SAME series). `..._xshh`
        # is the l = t slice of this; this arm is the strict superset. It
        # tests whether the BROAD cross-series repulsion beats the targeted
        # same-step one, or instead over-repels genuinely shared structure
        # (e.g. same-frequency seasonal phase at different absolute steps).
        #
        # The full edge is a [B, B, T-1, T] object (~17 GB at B=256, T=256),
        # so it is computed with a logsumexp that is CHUNKED over the source
        # batch b' — logsumexp is associative, so the chunked result is exact
        # and chunk-size-independent (chunk size only trades memory for kernel
        # launches). The anchor's OWN series (b'=b) is excluded entirely (its
        # within-series h↔h is the kept all-time `log_neg_hh_all`). Numerically
        # stable logsumexp; --pos-in-denominator supported via the shared tail.
        neg_inf = float('-inf')
        # stopgrad_positive_h: cut the encoder-side backward edge of the
        # positive only. `hy_norm` keeps gradient in every negative term
        # below (cross-batch f↔h, and h↔h via hx_norm/orig_norm).
        # EMA teacher (#353): when a teacher positive is supplied, it
        # replaces the student `hy_norm` ONLY inside `log_pos` (negatives
        # below still consume the student's `hy_norm`).
        if hy_teacher_norm is not None:
            hy_pos = hy_teacher_norm  # already no-grad (computed under no_grad)
        elif sg_pos:
            hy_pos = hy_norm.detach()
        else:
            hy_pos = hy_norm
        log_pos = cosine_similarity_from_normalized(hy_pos, hy_hat_norm) / tau

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

        # (B) all-time within-series term: (h_t, h_l) ∀ l ≠ t. Byte-for-β.
        t_idx = torch.arange(T - 1, device=orig_norm.device).view(T - 1, 1)
        l_idx = torch.arange(T, device=orig_norm.device).view(1, T)
        sims_hh_all = torch.matmul(
            hx_norm.permute(0, 2, 1, 3),     # [B, C, T-1, H]  (h_t)
            orig_norm.permute(0, 2, 3, 1),   # [B, C, H, T]    (h_l)
        )                                     # [B, C, T-1, T]
        drop_self = (l_idx == t_idx).view(1, 1, T - 1, T)
        log_neg_hh_all = torch.logsumexp(
            (sims_hh_all / tau).masked_fill(drop_self, neg_inf), dim=3
        ).permute(0, 2, 1)                    # [B, T-1, C]

        # Cross-batch f↔h negative — byte-for-β's `log_neg_cross_batch`.
        # MoCo-negatives (#374 arm 4, config key `moco_negatives`): when on
        # AND an EMA teacher is available, the keys h'_{t+1} come from the
        # teacher instead of the student — same swap as the split shape
        # below. The three h↔h families (xx, hh_all, xs_allt) stay pure
        # student on both sides.
        mask_batch = ~torch.eye(B, dtype=torch.bool, device=orig_norm.device)
        mask_batch = mask_batch.view(B, B, 1, 1)
        moco_negs = (
            moco_negatives if moco_negatives is not None
            else bool(train_config.get('moco_negatives', False)))
        if moco_negs and hy_teacher_norm is not None:
            hy_p = hy_teacher_norm.permute(1, 2, 0, 3)  # [T-1, C, B, H]  h^T_{t+1}
        else:
            hy_p = hy_norm.permute(1, 2, 0, 3)          # [T-1, C, B, H]  h_{t+1}
        hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)  # [T-1, C, B, H]  f_t
        sims_cross_batch = torch.matmul(
            hy_hat_p, hy_p.transpose(-2, -1)        # [T-1, C, B, B]
        ).permute(2, 3, 0, 1).contiguous()          # [B, B, T-1, C]
        log_neg_cross_batch = torch.logsumexp(
            (sims_cross_batch / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )

        # NEW: all-time cross-series h↔h — LSE_{b'≠b, ∀l} cos(h_{b,t}, h_{b',l})/τ.
        # The full [B, B, T-1, T] Gram is ~17 GB at B=256, T=256 and — crucially
        # — autograd would retain EVERY chunk's Gram for backward, so plain
        # chunking saves forward but not backward memory. So each chunk is
        # GRADIENT-CHECKPOINTED: its Gram is recomputed in backward one chunk at
        # a time, capping peak memory at a single chunk. logsumexp is
        # associative, so the result is exact and independent of the chunk size
        # (XSHH_ALLT_CHUNK; smaller fits a more crowded GPU at more kernel
        # launches). The anchor's own series (b'=b) is excluded entirely.
        anchor = hx_norm.permute(0, 2, 1, 3).contiguous()    # [B, C, T-1, H]  (h_t)
        src = orig_norm.permute(0, 2, 1, 3).contiguous()      # [B, C, T,   H]  (h_l, ∀l)
        b_all = torch.arange(B, device=orig_norm.device)
        CH = int(os.environ.get('XSHH_ALLT_CHUNK', '8'))
        # Optional speedups (#327), both default OFF ⇒ the checkpoint loop in
        # the `else` below runs byte-for-byte unchanged. See the module-level
        # note on _XsAlltFusedLSE / _XsAlltShardedLSE.
        fused = os.environ.get('XSHH_ALLT_FUSED', '0') == '1'
        shard = os.environ.get('XSHH_ALLT_SHARD', '0') == '1' and (
            dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1)

        if shard:
            run = _XsAlltShardedLSE.apply(anchor, src, tau, CH)
        elif fused:
            run = _XsAlltFusedLSE.apply(anchor, src, tau, CH)
        else:
            def _chunk_lse(anc, src_chunk, same_mask):
                # anc [B,C,T-1,H], src_chunk [ch,C,T,H], same_mask [B,ch,1,1,1] bool.
                gram = torch.matmul(
                    anc.unsqueeze(1),                             # [B, 1, C, T-1, H]
                    src_chunk.permute(0, 1, 3, 2).unsqueeze(0),   # [1, ch, C, H, T]
                ) / tau                                           # [B, ch, C, T-1, T]
                gram = gram.masked_fill(same_mask, neg_inf)
                return torch.logsumexp(torch.logsumexp(gram, dim=4), dim=1)  # [B, C, T-1]

            run = None                                            # [B, C, T-1] running LSE
            for s in range(0, B, CH):
                e = min(s + CH, B)
                same = (b_all.view(B, 1) == b_all[s:e].view(1, e - s)).view(B, e - s, 1, 1, 1)
                chunk_lse = checkpoint(_chunk_lse, anchor, src[s:e], same, use_reentrant=False)
                run = chunk_lse if run is None else torch.logsumexp(
                    torch.stack([run, chunk_lse], dim=0), dim=0)
        log_neg_xs_allt = run.permute(0, 2, 1)                # [B, T-1, C]

        negatives = torch.stack(
            [log_neg_xx, log_neg_zy, log_neg_hh_all,
             log_neg_cross_batch, log_neg_xs_allt],
            dim=0,
        )
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_split_pred_rep':
        # #374 split. Same negative families as `..._xshh_allt`, but the
        # single pooled denominator is broken into TWO independent terms:
        #
        #   L_pred = normalized InfoNCE — positive cos(f_t, h^T_{t+1})/τ
        #            over denominator = f-anchored families only
        #            (cross-batch f↔h'_{t+1} + adjacent f↔f).
        #   L_rep  = pooled LSE of h-anchored families, NO positive
        #            (cross-channel xx + within-series all-time hh
        #             + cross-series all-time xs).
        #   L      = L_pred + L_rep.
        #
        # Same τ in both, same teacher-side/stopgrad positive, same batch
        # pooling inside each term. `include_positive_in_denominator` is
        # a no-op for this shape (L_pred is normalized-InfoNCE by
        # construction). `subtract_contrastive_floor` IS supported and
        # handled below in `contrastive_latent_loss` — it subtracts the
        # per-term constants `f_pred + f_rep` (see `_split_pred_rep_floors`)
        # and is gradient-neutral.
        # Per-term scalar weights (#382). Default 1.0 keeps the historical
        # objective byte-for-byte; setting one side to 0.0 isolates the
        # other term for the loss-term-isolation ablation. Read early so we
        # can short-circuit the Gram compute for a disabled side entirely
        # (see the `w_pred != 0.0` / `w_rep != 0.0` guards below).
        w_pred = float(train_config.get('pred_loss_weight', 1.0))
        w_rep = float(train_config.get('rep_loss_weight', 1.0))
        # L_rep holds no forecaster output, so a rollout-depth copy (#373)
        # must not duplicate it: the depth-0 call already added it once, at
        # its configured weight.
        if f_only:
            w_rep = 0.0
        zero_scalar = torch.zeros(
            (), device=hy_hat_norm.device, dtype=hy_hat_norm.dtype,
            requires_grad=False)

        neg_inf = float('-inf')

        # L_pred: normalized InfoNCE — positive cos(f_t, h^T_{t+1})/τ over
        # denominator = f-anchored families only (adjacent f↔f + cross-batch
        # f↔h'_{t+1}). Every operation below is L_pred-only; when the arm
        # sets `--pred-loss-weight 0.0` we skip the entire block (no positive
        # gather, no cross-batch matmul) — reviewer round-2 Gap 3.
        if w_pred != 0.0:
            if hy_teacher_norm is not None:
                hy_pos = hy_teacher_norm
            elif sg_pos:
                hy_pos = hy_norm.detach()
            else:
                hy_pos = hy_norm
            log_pos = cosine_similarity_from_normalized(hy_pos, hy_hat_norm) / tau

            # f-anchored family: adjacent f_{t+1}↔f_t (zy), cross-channel.
            sims_zy = cosine_similarity_from_normalized(
                hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
            log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

            # f-anchored family: cross-batch f_t ↔ h'_{t+1}.
            # MoCo-negatives (#374 arm 3, config key `moco_negatives`): when
            # on AND an EMA teacher is available, replace the student
            # h'_{t+1} with the teacher's h^T_{t+1} — the "keys" in the
            # cross-batch f↔h term then come from the same slowly-moving
            # encoder as the positive, so positive and negatives share one
            # space (MoCo-style).
            mask_batch = ~torch.eye(B, dtype=torch.bool, device=orig_norm.device)
            mask_batch = mask_batch.view(B, B, 1, 1)
            moco_negs = (
                moco_negatives if moco_negatives is not None
                else bool(train_config.get('moco_negatives', False)))
            if moco_negs and hy_teacher_norm is not None:
                hy_p = hy_teacher_norm.permute(1, 2, 0, 3)
            else:
                hy_p = hy_norm.permute(1, 2, 0, 3)
            hy_hat_p = hy_hat_norm.permute(1, 2, 0, 3)
            sims_cross_batch = torch.matmul(
                hy_hat_p, hy_p.transpose(-2, -1)
            ).permute(2, 3, 0, 1).contiguous()
            log_neg_cross_batch = torch.logsumexp(
                (sims_cross_batch / tau).masked_fill(~mask_batch, neg_inf), dim=1)

            negs_pred = torch.stack([log_neg_zy, log_neg_cross_batch], dim=0)
            log_neg_per_anchor_pred = torch.logsumexp(negs_pred, dim=0)
            log_neg_total_pred = torch.logsumexp(
                log_neg_per_anchor_pred, dim=0, keepdim=True)
            log_denom_pred = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total_pred.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss_pred = (log_denom_pred - log_pos).mean()
        else:
            loss_pred = zero_scalar

        # L_rep: pooled LSE of h-anchored families (cross-channel xx +
        # within-series all-time hh + cross-series all-time xs). MoCo-keys
        # (`moco_rep_keys`) replaces the student KEY side with the EMA
        # teacher — turns L_rep into a normalized-InfoNCE with the
        # student↔teacher same-batch same-time pair as its positive.
        # `--rep-loss-weight 0.0` skips the entire block (no Gram matmul,
        # no xs_allt chunked accumulator).
        if w_rep != 0.0:
            # MoCo-keys on L_rep (#374 arm bimoco, config key
            # `moco_rep_keys`): when on AND an EMA teacher is available,
            # replace the STUDENT h on the KEY side of the three h-anchored
            # families with the teacher's h^T. Anchors (hx_norm) stay
            # student-side; keys become teacher-side. Mirrors
            # `moco_negatives` (which does the same for the cross-batch f↔h
            # family on the L_pred side).
            moco_rep = (
                moco_rep_keys if moco_rep_keys is not None
                else bool(train_config.get('moco_rep_keys', False)))
            if moco_rep and teacher_orig_norm is not None:
                hx_key = teacher_orig_norm[:, :-1, :, :]           # teacher h at t
                orig_key = teacher_orig_norm                       # teacher h at all times
            else:
                hx_key = hx_norm
                orig_key = orig_norm

            # h-anchored family 1: cross-channel same-time h↔h (xx).
            sims_xx = cosine_similarity_from_normalized(
                hx_norm.unsqueeze(3), hx_key.unsqueeze(2))
            mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
            mask_mat = mask_mat.view(1, 1, C, C)
            log_neg_xx = torch.logsumexp(
                (sims_xx / tau_rep).masked_fill(~mask_mat, neg_inf), dim=2)

            # h-anchored family 2: within-series all-time h↔h (hh_all).
            t_idx = torch.arange(T - 1, device=orig_norm.device).view(T - 1, 1)
            l_idx = torch.arange(T, device=orig_norm.device).view(1, T)
            sims_hh_all = torch.matmul(
                hx_norm.permute(0, 2, 1, 3),
                orig_key.permute(0, 2, 3, 1),
            )
            drop_self = (l_idx == t_idx).view(1, 1, T - 1, T)
            log_neg_hh_all = torch.logsumexp(
                (sims_hh_all / tau_rep).masked_fill(drop_self, neg_inf), dim=3
            ).permute(0, 2, 1)

            # h-anchored family 3: cross-series all-time h↔h' (xs_allt),
            # gradient-checkpointed via the same chunked path as
            # `..._xshh_allt`. With moco_rep_keys on, the KEY side uses
            # teacher_orig_norm (see `orig_key` selection above); the anchor
            # stays student-side.
            anchor = hx_norm.permute(0, 2, 1, 3).contiguous()
            src = orig_key.permute(0, 2, 1, 3).contiguous()
            b_all = torch.arange(B, device=orig_norm.device)
            CH = int(os.environ.get('XSHH_ALLT_CHUNK', '8'))
            fused = os.environ.get('XSHH_ALLT_FUSED', '0') == '1'
            shard = os.environ.get('XSHH_ALLT_SHARD', '0') == '1' and (
                dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1)
            if shard:
                run = _XsAlltShardedLSE.apply(anchor, src, tau_rep, CH)
            elif fused:
                run = _XsAlltFusedLSE.apply(anchor, src, tau_rep, CH)
            else:
                def _chunk_lse(anc, src_chunk, same_mask):
                    gram = torch.matmul(
                        anc.unsqueeze(1),
                        src_chunk.permute(0, 1, 3, 2).unsqueeze(0),
                    ) / tau_rep
                    gram = gram.masked_fill(same_mask, neg_inf)
                    return torch.logsumexp(torch.logsumexp(gram, dim=4), dim=1)
                run = None
                for s in range(0, B, CH):
                    e = min(s + CH, B)
                    same = (b_all.view(B, 1) == b_all[s:e].view(1, e - s)
                            ).view(B, e - s, 1, 1, 1)
                    chunk_lse = checkpoint(
                        _chunk_lse, anchor, src[s:e], same, use_reentrant=False)
                    run = chunk_lse if run is None else torch.logsumexp(
                        torch.stack([run, chunk_lse], dim=0), dim=0)
            log_neg_xs_allt = run.permute(0, 2, 1)

            # Without moco_rep_keys: student anchor + student keys everywhere.
            # The same-batch same-time <h_{b,t}, h_{b,t}> pair is trivially 1
            # (self-cosine); we omit it (adds a constant to the loss,
            # gradient-neutral). With moco_rep_keys: keys are teacher-side.
            # The same-batch same-time pair becomes the non-trivial
            # student↔teacher <h_{b,t}, h^T_{b,t}>, so the loss becomes a
            # normalized InfoNCE with that positive in the denominator
            # (same form as L_pred, symmetric on the h side).
            negs_rep = torch.stack(
                [log_neg_xx, log_neg_hh_all, log_neg_xs_allt], dim=0)
            log_neg_per_anchor_rep = torch.logsumexp(negs_rep, dim=0)
            log_neg_total_rep = torch.logsumexp(
                log_neg_per_anchor_rep, dim=0, keepdim=True)
            if moco_rep and teacher_orig_norm is not None:
                hx_teacher_norm = teacher_orig_norm[:, :-1, :, :]
                log_pos_h = cosine_similarity_from_normalized(
                    hx_norm, hx_teacher_norm) / tau_rep                  # [B, T-1, C]
                log_denom_rep = torch.logsumexp(
                    torch.stack(
                        [log_pos_h, log_neg_total_rep.expand_as(log_pos_h)],
                        dim=0,
                    ),
                    dim=0,
                )
                loss_rep = (log_denom_rep - log_pos_h).mean()
            else:
                loss_rep = log_neg_total_rep.mean()
        else:
            loss_rep = zero_scalar

        loss = w_pred * loss_pred + w_rep * loss_rep

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_rep_only' \
            and f_only:
        # #373: the whole shape is h-anchored (L_rep only), so a rollout-depth
        # copy has nothing to substitute and adds exactly zero. The f-bearing
        # `align_loss_weight` add-on this arm pairs L_rep with IS duplicated,
        # in the shared tail below.
        loss = hy_hat_norm.new_zeros(())

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_rep_only':
        # #374 follow-up arm: drop L_pred from the split shape and pair
        # `L_rep` (h-anchored logsumexp) with `align_loss` added by the
        # training script via --align-loss-weight. Same h-anchored families
        # as `..._split_pred_rep`'s L_rep branch.
        #
        # Without moco_rep_keys: keys are student-side; same-batch same-time
        # pair is trivially <h,h>=1 so no positive is added (constant drop).
        # With moco_rep_keys: keys are teacher-side; the same-batch same-time
        # pair becomes non-trivial <h_student, h_teacher> and enters the
        # normalized-InfoNCE positive-in-denominator form (identical to the
        # split shape's L_rep_moco).
        #
        # #379: the entire loss IS L_rep here, so every τ divisor is
        # tau_rep (which aliases tau when --tau-rep is unset).
        neg_inf = float('-inf')
        moco_rep = (
            moco_rep_keys if moco_rep_keys is not None
            else bool(train_config.get('moco_rep_keys', False)))
        if moco_rep and teacher_original_latent is not None:
            teacher_orig_norm_local = F.normalize(teacher_original_latent, p=2, dim=-1)
            hx_key = teacher_orig_norm_local[:, :-1, :, :]
            orig_key = teacher_orig_norm_local
        else:
            teacher_orig_norm_local = None
            hx_key = hx_norm
            orig_key = orig_norm
        # h-anchored family 1: cross-channel same-time h↔h (xx).
        sims_xx = cosine_similarity_from_normalized(
            hx_norm.unsqueeze(3), hx_key.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau_rep).masked_fill(~mask_mat, neg_inf), dim=2)
        # h-anchored family 2: within-series all-time h↔h (hh_all).
        t_idx = torch.arange(T - 1, device=orig_norm.device).view(T - 1, 1)
        l_idx = torch.arange(T, device=orig_norm.device).view(1, T)
        sims_hh_all = torch.matmul(
            hx_norm.permute(0, 2, 1, 3),
            orig_key.permute(0, 2, 3, 1),
        )
        drop_self = (l_idx == t_idx).view(1, 1, T - 1, T)
        log_neg_hh_all = torch.logsumexp(
            (sims_hh_all / tau_rep).masked_fill(drop_self, neg_inf), dim=3
        ).permute(0, 2, 1)
        # h-anchored family 3: cross-series all-time h↔h′ (xs_allt),
        # gradient-checkpointed via the same chunked path as `..._xshh_allt`.
        anchor = hx_norm.permute(0, 2, 1, 3).contiguous()
        src = orig_key.permute(0, 2, 1, 3).contiguous()
        b_all = torch.arange(B, device=orig_norm.device)
        CH = int(os.environ.get('XSHH_ALLT_CHUNK', '8'))
        fused = os.environ.get('XSHH_ALLT_FUSED', '0') == '1'
        shard = os.environ.get('XSHH_ALLT_SHARD', '0') == '1' and (
            dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1)
        if shard:
            run = _XsAlltShardedLSE.apply(anchor, src, tau_rep, CH)
        elif fused:
            run = _XsAlltFusedLSE.apply(anchor, src, tau_rep, CH)
        else:
            def _chunk_lse(anc, src_chunk, same_mask):
                gram = torch.matmul(
                    anc.unsqueeze(1),
                    src_chunk.permute(0, 1, 3, 2).unsqueeze(0),
                ) / tau_rep
                gram = gram.masked_fill(same_mask, neg_inf)
                return torch.logsumexp(torch.logsumexp(gram, dim=4), dim=1)
            run = None
            for s in range(0, B, CH):
                e = min(s + CH, B)
                same = (b_all.view(B, 1) == b_all[s:e].view(1, e - s)
                        ).view(B, e - s, 1, 1, 1)
                chunk_lse = checkpoint(
                    _chunk_lse, anchor, src[s:e], same, use_reentrant=False)
                run = chunk_lse if run is None else torch.logsumexp(
                    torch.stack([run, chunk_lse], dim=0), dim=0)
        log_neg_xs_allt = run.permute(0, 2, 1)
        # L_rep: pooled LSE of h-anchored families.
        # See split branch for the moco_rep_keys positive-in-denominator form.
        negs_rep = torch.stack(
            [log_neg_xx, log_neg_hh_all, log_neg_xs_allt], dim=0)
        log_neg_per_anchor_rep = torch.logsumexp(negs_rep, dim=0)
        log_neg_total_rep = torch.logsumexp(
            log_neg_per_anchor_rep, dim=0, keepdim=True)
        if moco_rep and teacher_orig_norm_local is not None:
            hx_teacher_norm = teacher_orig_norm_local[:, :-1, :, :]
            log_pos_h = cosine_similarity_from_normalized(
                hx_norm, hx_teacher_norm) / tau_rep
            log_denom_rep = torch.logsumexp(
                torch.stack(
                    [log_pos_h, log_neg_total_rep.expand_as(log_pos_h)],
                    dim=0,
                ),
                dim=0,
            )
            loss = (log_denom_rep - log_pos_h).mean()
        else:
            loss = log_neg_total_rep.mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_add_f_cross_negs':
        # NON-cumulative variant: identical to `cosine_similarity_batch` except
        # `negatives` includes an extra f-side cross-(b,c) term at fixed t —
        # i.e. cos(f_{b,c,t}, f_{b',c',t}) for all (b,c)≠(b',c'). Numerator
        # (positives) is unchanged: just exp(cos(hy_norm, hy_hat_norm) / tau).
        # Tests Exp 4's f-cross-bc term on its own, without inheriting Exp 3's
        # (h_t, f_t) positive (which is a degenerate shortcut for our arch).
        positives = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

        # Cross-batch negatives (h-side): compare across batch dimension
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
        neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

        # NEW (Exp 4 standalone): f-side cross-(b,c) negatives at same time t.
        # Reshape f at non-final positions to [T-1, B*C, H], compute pairwise
        # similarity with a single matmul, mask the diagonal, sum over the
        # second B*C dim, reshape back to [B, T-1, C]. Code ported as-is from
        # the cumulative variant `cosine_similarity_batch_add_pos_htft_add_f_cross_negs`
        # (PR #181) — same f-cross-bc term, but added to the cosine_similarity_batch
        # baseline rather than to Exp 3's predecessor.
        f_perm = hy_hat_norm.permute(1, 0, 2, 3).reshape(T - 1, B * C, H)
        sims_ff = torch.matmul(f_perm, f_perm.transpose(-1, -2))                          # [T-1, B*C, B*C]
        mask_bc = ~torch.eye(B * C, dtype=torch.bool, device=sims_ff.device)
        mask_bc = mask_bc.view(1, B * C, B * C)
        neg_f_cross_bc_flat = torch.exp(sims_ff / tau).masked_fill(~mask_bc, 0).sum(dim=2)  # [T-1, B*C]
        neg_f_cross_bc = neg_f_cross_bc_flat.reshape(T - 1, B, C).permute(1, 0, 2)          # [B, T-1, C]

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch + neg_f_cross_bc
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_add_pos_htft':
        # Same as cosine_similarity_batch, but adds (h_t, f_t) — same-channel,
        # same-time encoder-vs-forecaster — as an *additional* positive pair on
        # top of the existing (h_{t+1}, f_t) positive. Multi-positive InfoNCE:
        # sum the positive exponentials in the numerator. Negatives are
        # unchanged: neg_xy_hat is cross-channel only, so adding the
        # same-channel positive does not double-count.
        pos_h_t1 = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )
        pos_h_t = torch.exp(
            cosine_similarity_from_normalized(hx_norm, hy_hat_norm) / tau
        )
        positives = pos_h_t1 + pos_h_t

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

        # Cross-batch negatives: compare across batch dimension
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
        neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_add_pos_htft_add_f_cross_negs':
        # Cumulative on top of `cosine_similarity_batch_add_pos_htft` (Exp 3):
        # keeps the same multi-positive numerator (pos_h_t1 + pos_h_t) and the
        # same h-side negatives, then ADDS f-side cross-(b,c) negatives at the
        # same time t — i.e. cos(f_{b,c,t}, f_{b',c',t}) for all (b,c)≠(b',c').
        # That single term covers cross-batch same-channel + cross-channel
        # same-batch + cross-both at fixed t in one efficient B*C × B*C matmul.
        pos_h_t1 = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )
        pos_h_t = torch.exp(
            cosine_similarity_from_normalized(hx_norm, hy_hat_norm) / tau
        )
        positives = pos_h_t1 + pos_h_t

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

        # Cross-batch negatives (h-side): compare across batch dimension
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
        neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

        # NEW (Exp 4): f-side cross-(b,c) negatives at same time t.
        # Reshape f at non-final positions to [T-1, B*C, H], compute pairwise
        # similarity with a single matmul, mask the diagonal, sum over the
        # second B*C dim, reshape back to [B, T-1, C].
        f_perm = hy_hat_norm.permute(1, 0, 2, 3).reshape(T - 1, B * C, H)
        sims_ff = torch.matmul(f_perm, f_perm.transpose(-1, -2))                          # [T-1, B*C, B*C]
        mask_bc = ~torch.eye(B * C, dtype=torch.bool, device=sims_ff.device)
        mask_bc = mask_bc.view(1, B * C, B * C)
        neg_f_cross_bc_flat = torch.exp(sims_ff / tau).masked_fill(~mask_bc, 0).sum(dim=2)  # [T-1, B*C]
        neg_f_cross_bc = neg_f_cross_bc_flat.reshape(T - 1, B, C).permute(1, 0, 2)          # [B, T-1, C]

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch + neg_f_cross_bc
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_add_neg_htft':
        # Corrected Exp 3: identical to `cosine_similarity_batch` except the
        # `negatives` term includes an explicit per-(b,t,c) (h_t, f_t)
        # same-channel same-time NEGATIVE — i.e. cos(hx[b,t,c], hy_hat[b,t,c]).
        # Numerator (positives) is unchanged: just exp(cos(hy_norm, hy_hat_norm)/tau)
        # = (h_{t+1}, f_t).
        #
        # Why a NEGATIVE (not a positive as in `cosine_similarity_batch_add_pos_htft`,
        # PR #179): f_t is the forecaster output at position t, which predicts
        # h_{t+1}. h_t is the encoder output at the SAME position. f_t already
        # has e_t in its causal context, so pulling (h_t, f_t) together (the
        # original Exp 3 PR #179) created a degenerate f_t ≈ h_t shortcut —
        # the forecaster can satisfy the positive trivially without learning
        # to predict the future. The corrected formulation pushes (h_t, f_t)
        # APART, forcing the forecaster to differ from the present encoder
        # state and actually predict h_{t+1}.
        #
        # Double-count note: the existing `neg_xy_hat` is built as
        #   sum_{c1} exp(cos(hx[b,t,c1], hy_hat[b,t,c2])/tau)   (no c1≠c2 mask)
        # so for any C≥1 it ALREADY contains the same-channel (h_t, f_t) term
        # (c1=c2). Adding `neg_h_t_f_t` here ON TOP gives the same-channel
        # slice 2× weight in the denominator while the cross-channel slice
        # stays 1× — i.e. the new term doubles the (h_t, f_t) repulsion
        # signal rather than introducing it from scratch. We chose this
        # (option (b) in the design doc) over subtracting the same-channel
        # diagonal from `neg_xy_hat` first (option (a)) because option (a)
        # would be a net no-op vs `cosine_similarity_batch` baseline — the
        # explicit negative term needs to add genuine extra weight to test
        # the corrected hypothesis. See PR body for the full analysis.
        positives = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

        # Cross-batch negatives (h-side): compare across batch dimension
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
        neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

        # NEW (corrected Exp 3): explicit per-(b,t,c) same-channel (h_t, f_t)
        # negative. Shape [B, T-1, C], aligned with all other negative terms.
        neg_h_t_f_t = torch.exp(
            cosine_similarity_from_normalized(hx_norm, hy_hat_norm) / tau
        )

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch + neg_h_t_f_t
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_add_skip_f_negs':
        # NON-cumulative variant: identical to `cosine_similarity_batch` except
        # `negatives` includes an extra "skip-step" forecaster term —
        # cos(f_{b,c,t}, f_{b,c,t+2}) — i.e. f_t vs f_{t+2} same-(b, c). For
        # C=1 the existing `neg_zy` already covers cos(f_{t+1, c1}, f_{t, c2})
        # for (c1=c2), which is f_t vs f_{t+1} same-channel. f_t vs f_{t+2}
        # is a genuinely novel skip-step pair not in any other negative term.
        # This is the Exp 5 reformulation of the user's "f_t vs f_{t+1}"
        # original spec — testing whether a longer skip step adds discriminative
        # signal beyond the existing adjacent-step terms.
        positives = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        neg_xy = torch.exp(sims_xy / tau).sum(dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_xy_hat = torch.exp(sims_xy_hat / tau).sum(dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        neg_zy = torch.exp(sims_zy / tau).sum(dim=2)

        # Cross-batch negatives (h-side): compare across batch dimension
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
        neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

        # NEW (Exp 5): f_t vs f_{t+2} skip-step forecaster negatives,
        # same-(b, c). Valid pairs cover t = 0..T-3, so the skip-step
        # term only has T-2 positions. We pad the last position with 0
        # so the time dim aligns with the T-1 shape of the other neg
        # terms (the padded position contributes 0 to negatives.sum at
        # t = T-2, leaving that timestep's loss unaffected by this term).
        # For T<3 there are no valid skip pairs — keep the term zero.
        if T >= 3:
            f_t_pre = fore_norm[:, :T - 2, :, :]    # f_t for t=0..T-3,    [B, T-2, C, H]
            f_t_post = fore_norm[:, 2:T, :, :]       # f_{t+2} for t=0..T-3, [B, T-2, C, H]
            sims_skip = (f_t_pre * f_t_post).sum(dim=-1)         # [B, T-2, C]
            neg_skip_f_unpadded = torch.exp(sims_skip / tau)     # [B, T-2, C]
            # Pad time dim from T-2 to T-1 with zeros at the end. Tensor is
            # [B, T-2, C]; F.pad uses last-dim-first ordering, so to add 1
            # zero at the END of dim=1 (T axis) we pass (0, 0, 0, 1):
            # (left_C=0, right_C=0, left_T=0, right_T=1). Result: [B, T-1, C].
            neg_skip_f = F.pad(neg_skip_f_unpadded, (0, 0, 0, 1))
        else:
            neg_skip_f = torch.zeros_like(neg_zy)

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch + neg_skip_f
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_no_time_neg':
        # Same as cosine_similarity_batch but without cross-time negatives (t <-> t+1).
        # Keeps SAME-TIME cross-channel negatives (both h×h and h×h_hat) and
        # cross-batch negatives.
        # Useful for ARMA experiments where consecutive time slices are nearly identical.
        # Numerically stable logsumexp form — equivalence + fp16/fp32 small-τ stability
        # pinned in tests/test_loss_stability.py.
        neg_inf = float('-inf')
        log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau

        # Cross-channel negatives, h × h, same-time (encoder spread)
        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        # Cross-channel negatives, h × h_hat, same-time (forecaster spread).
        # Diagonal (c1 == c2) excluded: that is the same-time same-channel
        # FP comparison, which is cross-time-adjacent for autocorrelated
        # data and was the reason the `_no_time_neg` variant exists.
        sims_xy_hat = cosine_similarity_from_normalized(
            hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2)
        )
        log_neg_xy_hat = torch.logsumexp(
            (sims_xy_hat / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        # Cross-batch negatives: compare across batch dimension
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        log_neg_cross_batch = torch.logsumexp(
            (sims_cross_batch / tau).masked_fill(~mask_batch, neg_inf), dim=1
        )

        negatives = torch.stack(
            [log_neg_xx, log_neg_xy_hat, log_neg_cross_batch], dim=0)
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            # Normalized InfoNCE (loss_tau_ref diagnostic OR the
            # --pos-in-denominator training knob) — see the
            # cosine_similarity_batch branch for the rationale.
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') == 'cosine_similarity_batch_square':
        # Extends `cosine_similarity_batch` with the two missing clean batch-axis
        # edges of the (batch × time) square of prediction pairs:
        #   neg_cross_batch_forecast:   f_b vs f_b' at same t  (b ≠ b')
        #   neg_cross_batch_embedding:  h_{b,t+1} vs h_{b',t+1} at same t (b ≠ b')
        # The existing cross-batch diagonal term (h_{b',t+1} ↔ f_{b,t}) is kept.
        # Numerically stable logsumexp form — equivalence + fp16/fp32 small-τ
        # stability pinned in tests/test_loss_stability.py.
        neg_inf = float('-inf')
        log_pos = cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau

        sims_xy = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_norm.unsqueeze(2))
        log_neg_xy = torch.logsumexp(sims_xy / tau, dim=2)

        sims_xy_hat = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_xy_hat = torch.logsumexp(sims_xy_hat / tau, dim=2)

        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        log_neg_xx = torch.logsumexp(
            (sims_xx / tau).masked_fill(~mask_mat, neg_inf), dim=2
        )

        sims_zy = cosine_similarity_from_normalized(hz_hat_norm.unsqueeze(3), hy_hat_norm.unsqueeze(2))
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)

        # Existing inner diagonal: h_{b',t+1} ↔ f_{b,t}  [B,B,T-1,C] → [B,T-1,C]
        hy_norm_exp     = hy_norm.unsqueeze(0)      # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]
        sims_cross = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)
        mask_b = ~torch.eye(B, dtype=torch.bool, device=sims_cross.device).view(B, B, 1, 1)
        log_neg_cross_fe = torch.logsumexp(
            (sims_cross / tau).masked_fill(~mask_b, neg_inf), dim=1
        )

        # NEW: f_b vs f_b' at same t (b ≠ b')  [B,B,T-1,C] → [B,T-1,C]
        f_anchor = hy_hat_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        f_other  = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]
        sims_ff  = cosine_similarity_from_normalized(f_anchor, f_other)
        log_neg_cross_ff = torch.logsumexp(
            (sims_ff / tau).masked_fill(~mask_b, neg_inf), dim=1
        )

        # NEW: h_{b,t+1} vs h_{b',t+1} at same t (b ≠ b')  [B,B,T-1,C] → [B,T-1,C]
        h_anchor = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        h_other  = hy_norm.unsqueeze(1)  # [B, 1, T-1, C, H]
        sims_hh  = cosine_similarity_from_normalized(h_anchor, h_other)
        log_neg_cross_hh = torch.logsumexp(
            (sims_hh / tau).masked_fill(~mask_b, neg_inf), dim=1
        )

        negatives = torch.stack(
            [log_neg_xy, log_neg_xx, log_neg_zy, log_neg_xy_hat,
             log_neg_cross_fe, log_neg_cross_ff, log_neg_cross_hh],
            dim=0,
        )
        log_neg_per_anchor = torch.logsumexp(negatives, dim=0)
        log_neg_total = torch.logsumexp(log_neg_per_anchor, dim=0, keepdim=True)
        if pos_in_denom:
            # Normalized InfoNCE (loss_tau_ref diagnostic OR the
            # --pos-in-denominator training knob) — see the
            # cosine_similarity_batch branch for the rationale.
            log_denom = torch.logsumexp(
                torch.stack(
                    [log_pos, log_neg_total.expand_as(log_pos)], dim=0
                ),
                dim=0,
            )
            loss = (log_denom - log_pos).mean()
        else:
            loss = (log_neg_total - log_pos).mean()

    elif train_config.get('loss_shape') == 'mse':
        # Only the first term carries f. − mse(h_t, h_{t+1}) is h-only, so a
        # rollout-depth copy (#373) leaves it to the depth-0 call.
        loss = F.mse_loss(hy, hy_hat)
        if not f_only:
            loss = loss - F.mse_loss(hx, hy)
    else:
        shape = train_config.get('loss_shape')
        raise Exception(f"Loss shape {shape} not implemented")

    # Guard: positive-in-denominator (whether requested via the function
    # arg OR the training-config key) is only meaningful for the
    # logsumexp-form variants that compute `log_pos`/`log_neg_total` above
    # and set `loss` via that path. Any other `loss_shape` reaching here
    # with it set means the normalized form was NOT applied — fail loud
    # rather than silently returning the wrong objective. The default
    # (False on both) path is byte-for-byte unchanged for every variant,
    # so this never affects historical training losses.
    if pos_in_denom and train_config.get('loss_shape') not in _NORMALIZED_FORM_SHAPES:
        raise NotImplementedError(
            "include_positive_in_denominator (function arg or "
            "train_configuration key) is only implemented for loss_shape "
            "in {cosine_similarity_batch, cosine_similarity_batch_no_time_neg, "
            "cosine_similarity_batch_square, "
            "cosine_similarity_batch_full_fh_negs, "
            "cosine_similarity_batch_full_hh_negs, "
            "cosine_similarity_batch_full_ff_negs, "
            "cosine_similarity_batch_full_fh_hh_negs, "
            "cosine_similarity_batch_full_hh_ff_negs, "
            "cosine_similarity_batch_full_fh_hh_ff_negs, "
            "cosine_similarity_batch_full_hh_negs_xbfree, "
            "cosine_similarity_batch_full_hh_negs_xshh, "
            "cosine_similarity_batch_full_hh_negs_xshh_allt}; got "
            f"{train_config.get('loss_shape')!r}."
        )

    # Guard: stopgrad_positive_h is applied inside the xshh_allt / split
    # branches only. Any other `loss_shape` reaching here with it set
    # would have silently trained WITHOUT the stop-grad — fail loud instead.
    _SG_POS_SHAPES = (
        'cosine_similarity_batch_full_hh_negs_xshh_allt',
        'cosine_similarity_batch_split_pred_rep',
        'cosine_similarity_batch_rep_only',
    )
    if sg_pos and train_config.get('loss_shape') not in _SG_POS_SHAPES:
        raise NotImplementedError(
            "stopgrad_positive_h is only implemented for loss_shape in "
            f"{_SG_POS_SHAPES}; got {train_config.get('loss_shape')!r}.")

    # Same guard for the EMA-teacher positive (#353): only the xshh_allt /
    # split branches read it; any other shape reaching here with a teacher
    # latent would have silently trained on the student positive — fail loud.
    if hy_teacher_norm is not None and \
            train_config.get('loss_shape') not in _SG_POS_SHAPES:
        raise NotImplementedError(
            "teacher_original_latent is only implemented for loss_shape in "
            f"{_SG_POS_SHAPES}; got {train_config.get('loss_shape')!r}.")

    # Guard: moco_negatives (#374 arms 3+4) is only wired into the split /
    # xshh_allt branches; any other shape reaching here with it set would
    # have silently trained WITHOUT sending negatives through the teacher —
    # fail loud. Also requires the EMA-teacher path to be active (no teacher
    # → the flag has nothing to route through and is silently ignored
    # otherwise). The function-arg override (moco_negatives=False from the
    # loss_tau_ref diagnostic) takes precedence over the config key.
    _MOCO_NEG_SHAPES = (
        'cosine_similarity_batch_split_pred_rep',
        'cosine_similarity_batch_full_hh_negs_xshh_allt',
    )
    _moco_effective = (
        moco_negatives if moco_negatives is not None
        else bool(train_config.get('moco_negatives', False)))
    if _moco_effective:
        if train_config.get('loss_shape') not in _MOCO_NEG_SHAPES:
            raise NotImplementedError(
                "moco_negatives is only implemented for loss_shape in "
                f"{_MOCO_NEG_SHAPES}; got "
                f"{train_config.get('loss_shape')!r}.")
        if hy_teacher_norm is None:
            raise ValueError(
                "moco_negatives requires an EMA teacher (pass "
                "teacher_original_latent, i.e. --ema-embedding / "
                "--ema-encoder at training time).")

    # Guard: moco_rep_keys (#374 arm bimoco) is only wired into the split
    # branch's L_rep families; any other shape reaching here with it set
    # would have silently trained on student-side keys — fail loud. Also
    # requires the EMA-teacher path to be active. The function-arg override
    # (moco_rep_keys=False from the loss_tau_ref diagnostic) takes precedence
    # over the config key, so the diagnostic can force it off even when the
    # run has it on.
    _moco_rep_effective = (
        moco_rep_keys if moco_rep_keys is not None
        else bool(train_config.get('moco_rep_keys', False)))
    _MOCO_REP_SHAPES = (
        'cosine_similarity_batch_split_pred_rep',
        'cosine_similarity_batch_rep_only',
    )
    if _moco_rep_effective:
        if train_config.get('loss_shape') not in _MOCO_REP_SHAPES:
            raise NotImplementedError(
                "moco_rep_keys is only implemented for loss_shape in "
                f"{_MOCO_REP_SHAPES}; got "
                f"{train_config.get('loss_shape')!r}.")
        if teacher_original_latent is None:
            raise ValueError(
                "moco_rep_keys requires an EMA teacher (pass "
                "teacher_original_latent, i.e. --ema-embedding / "
                "--ema-encoder at training time).")

    # Optional add-ons (#309), both default OFF ⇒ the objective is
    # byte-for-byte unchanged for every existing run/test. Resolve from the
    # explicit function arg (an override — the loss_tau_ref diagnostic
    # passes 0/False to keep that reference a pure contrastive value) else
    # the training-config key (the run-level knob a CLI flag sets).
    align_w = float(
        align_loss_weight if align_loss_weight is not None
        else train_config.get('align_loss_weight', 0.0))
    sub_floor = bool(
        subtract_contrastive_floor if subtract_contrastive_floor is not None
        else train_config.get('subtract_contrastive_floor', False))

    if align_w != 0.0:
        # Whose h_{t+1} the term pulls toward (#390). Same arg-over-config-
        # key precedence; default 'student' keeps #379's runs reproducible.
        # Resolved inside this branch so a run with the term off (the
        # `loss_tau_ref` diagnostic passes weight 0 and no teacher) is
        # unaffected by the knob.
        align_tgt = (
            align_target if align_target is not None
            else train_config.get('align_target', 'student'))
        if align_tgt not in ('student', 'teacher'):
            raise ValueError(
                "align_target must be 'student' or 'teacher'; got "
                f"{align_tgt!r}.")
        if align_tgt == 'teacher' and hy_teacher_norm is None:
            raise ValueError(
                "align_target='teacher' requires an EMA teacher (pass "
                "teacher_original_latent, i.e. --ema-embedding / "
                "--ema-encoder at training time). Falling back to the "
                "student target is the #382 defect this argument removes.")

        # BYOL/SimSiam alignment term: L_align = (2 − 2·cos(f_t,
        # sg(h_{t+1}))).mean(), added to the loss with weight λ. Applies to
        # ANY loss_shape — it needs only the positive pair (hy_hat_norm,
        # hy_norm), which is computed above for every variant.
        # Same positive pair as `log_pos`, but stop-grad on the encoder
        # target (gradient flows only through the forecaster f_t). Unlike
        # the InfoNCE positive — whose per-cosine gradient −(1−p₊)/τ fades
        # once the NEGATIVES separate (p₊→1), even while cos⁺ < 1 —
        # L_align's per-cosine gradient is a constant −2, independent of the
        # negatives (in embedding space both pick up the same sinθ factor,
        # which cancels in their ratio; the tangent magnitude is 2·sinθ,
        # → 0 only as the positive itself aligns). So L_align keeps pulling
        # whenever cos⁺ < 1 — it fades as the positive aligns, not when the
        # contrastive task is "won". The form 2 − 2·cos = ‖f̂ − ĥ‖²
        # is already in [0, 4] with minimum 0 at cos = 1 — the `2` is the
        # built-in constant, so L_align needs no extra offset to be ≥ 0 /
        # min-0. With --subtract-contrastive-floor on, the total loss
        # (L_c − floor) + λ·L_align then has theoretical minimum 0.
        # With align_target='teacher' (#390) the target is the EMA
        # teacher's h^T_{t+1} instead: same term, slowly-moving target.
        align_ref = (hy_teacher_norm if align_tgt == 'teacher' else hy_norm)
        cos_align = cosine_similarity_from_normalized(
            hy_hat_norm, align_ref.detach())
        loss = loss + align_w * (2.0 - 2.0 * cos_align).mean()

    if sub_floor:
        # Re-base the loss by the (constant) uniformity floor so the logged
        # curve reads ~0 at that floor. Gradient-neutral.
        if train_config.get('loss_shape') == \
                'cosine_similarity_batch_split_pred_rep':
            # Split shape has TWO independent terms; subtract each side's
            # own floor. L_pred is normalized-InfoNCE by construction (its
            # own floor is the InfoNCE floor at its negative count);
            # L_rep has no positive, so its floor is log(#terms). A
            # rollout-depth copy (#373) carries L_pred only, so it re-bases
            # by that side's floor alone.
            f_pred, f_rep = _split_pred_rep_floors(tau, B, T, C, tau_rep=tau_rep)
            loss = loss - float(f_pred)
            if not f_only:
                loss = loss - float(f_rep)
        else:
            # Pooled shapes: floor requires the normalized form (positive
            # in the denominator).
            if not pos_in_denom:
                raise NotImplementedError(
                    "subtract_contrastive_floor requires the normalized-"
                    "InfoNCE objective (include_positive_in_denominator): "
                    "log(1 + N·e^(−1/τ)) is the floor of THAT form, not of "
                    "the default negatives-only loss.")
            n_neg = _effective_negative_count(
                train_config.get('loss_shape'), B, T, C)
            if n_neg is None:
                raise NotImplementedError(
                    "subtract_contrastive_floor is not implemented for "
                    f"loss_shape={train_config.get('loss_shape')!r}.")
            loss = loss - infonce_floor(tau, n_neg)

    # Rollout depth (#373): one complete copy of every f-bearing term per
    # depth j = 1..k. Each copy re-enters this same function — same shape
    # branch, same modifiers, same add-ons — on views where f is f^(j) and
    # every h index has shifted by j, so the copy cannot silently keep a
    # depth-0 tensor. `depth_index=j` drops the terms that carry no f, which
    # the call above already counted once.
    depth_copies = []
    for j, f_depth in enumerate(rollout_latents or (), start=1):
        f_view, h_view, teacher_view = rollout_depth_views(
            f_depth, original_latent, j, teacher_original_latent)
        depth_copies.append(contrastive_latent_loss(
            (f_view, h_view), validation, spec,
            tau_override=tau_override,
            include_positive_in_denominator=include_positive_in_denominator,
            align_loss_weight=align_loss_weight,
            align_target=align_target,
            subtract_contrastive_floor=subtract_contrastive_floor,
            moco_negatives=moco_negatives,
            moco_rep_keys=moco_rep_keys,
            teacher_original_latent=teacher_view,
            train_rollout_depth=0,
            depth_index=j))

    if depth_copies and reduce == 'sum':
        # #373's objective, byte for byte: the copies are added to a `loss`
        # that already holds both sides.
        loss = reduce_depth_terms([loss] + depth_copies, 'sum')
    elif depth_copies:
        # #401 `mean`. The division covers the f-bearing copies only, so the
        # f-side of depth 0 has to be told apart from the h-side it is added
        # to. One more pass over this call's own views, with the h-only terms
        # dropped, reads it. That pass is what the mean costs; on the
        # `rep_only` cell it is the L_align add-on alone, because the shape
        # body returns zeros for an f-only call.
        #
        # `loss - f_side_depth0` is the h-side to within one rounding of the
        # f-side: the two passes run the same deterministic code on the same
        # tensors, so they return the same number, and (H + F) - F carries an
        # error of order eps·|F|. That is 1e-7 relative in fp32, against a
        # loss the run reports to four decimals.
        if noise_sigma is not None and not validation:
            # The pass below would draw its own noise, so `loss - f_side`
            # would not be the h-side. No trainer sets this key; refuse
            # rather than return a total nobody can reproduce.
            raise NotImplementedError(
                "train_rollout_reduce='mean' is not defined with "
                "contrastive_latent_noise: the f-side probe would redraw the "
                "noise, so the h-side could not be held out of the division.")
        f_side_depth0 = contrastive_latent_loss(
            (forecasted_latent, original_latent), validation, spec,
            tau_override=tau_override,
            include_positive_in_denominator=include_positive_in_denominator,
            align_loss_weight=align_loss_weight,
            align_target=align_target,
            subtract_contrastive_floor=subtract_contrastive_floor,
            moco_negatives=moco_negatives,
            moco_rep_keys=moco_rep_keys,
            teacher_original_latent=teacher_original_latent,
            train_rollout_depth=0,
            f_terms_only=True)
        h_side = loss - f_side_depth0
        loss = h_side + reduce_depth_terms(
            [f_side_depth0] + depth_copies, 'mean')

    if get_history:
        return loss, (forecasted_latent, original_latent)
    return loss
