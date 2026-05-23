import math

import torch
import torch.nn.functional as F

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
)


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


def contrastive_latent_loss(predicted_position, validation, spec,
                            get_history=False, tau_override=None,
                            include_positive_in_denominator=False,
                            align_loss_weight=None,
                            subtract_contrastive_floor=None):
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
    """
    forecasted_latent, original_latent = predicted_position
    train_config = spec.train_configuration

    # CPC multi-step (#316): the forecaster latent is a [B,T,C,K,H] stack of
    # K linear-head predictions, so the 4-D unpack/positives below do not
    # apply. Dispatch to the dedicated multi-step InfoNCE and return early.
    # align_loss_weight / subtract_contrastive_floor are not defined for this
    # variant (no single positive pair / closed-form floor) and are ignored.
    if train_config.get('loss_shape') == 'cpc_multistep':
        if tau_override is not None:
            tau = tau_override
        else:
            tau = train_config.get('contrastive_divergence_temperature', 1.0)
        pos_in_denom = include_positive_in_denominator or bool(
            train_config.get('include_positive_in_denominator', False))
        loss = cpc_multistep_loss(
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

    # Normalized-InfoNCE (positive in BOTH numerator and denominator) is
    # opt-in via EITHER the function arg (diagnostic loss_tau_ref) OR the
    # training-config key (a run-level knob, e.g. the --pos-in-denominator
    # CLI flag). Default False on both ⇒ historical objective unchanged.
    pos_in_denom = include_positive_in_denominator or bool(
        train_config.get('include_positive_in_denominator', False)
    )

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
        loss = F.mse_loss(hy, hy_hat) - F.mse_loss(hx, hy)
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
            "cosine_similarity_batch_full_hh_negs_xbfree}; got "
            f"{train_config.get('loss_shape')!r}."
        )

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
        cos_align = cosine_similarity_from_normalized(
            hy_hat_norm, hy_norm.detach())
        loss = loss + align_w * (2.0 - 2.0 * cos_align).mean()

    if sub_floor:
        # Re-base the loss by the (constant) normalized-InfoNCE floor so the
        # logged curve reads ~0 at the uniformity floor. Gradient-neutral.
        if not pos_in_denom:
            raise NotImplementedError(
                "subtract_contrastive_floor requires the normalized-InfoNCE "
                "objective (include_positive_in_denominator): "
                "log(1 + N·e^(−1/τ)) is the floor of THAT form, not of the "
                "default negatives-only loss.")
        n_neg = _effective_negative_count(
            train_config.get('loss_shape'), B, T, C)
        if n_neg is None:
            raise NotImplementedError(
                "subtract_contrastive_floor is not implemented for "
                f"loss_shape={train_config.get('loss_shape')!r}.")
        loss = loss - infonce_floor(tau, n_neg)

    if get_history:
        return loss, (forecasted_latent, original_latent)
    return loss
