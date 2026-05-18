import torch
import torch.nn.functional as F

def cosine_similarity_from_normalized(a, b):
    return (a * b).sum(dim=-1)

def contrastive_latent_loss(predicted_position, validation, spec,
                            get_history=False, tau_override=None,
                            include_positive_in_denominator=False):
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
            `cosine_similarity_batch_full_fh_negs`); requesting it with
            any other `loss_shape` raises NotImplementedError rather
            than silently returning an unintended value.

    The temperature τ acts as a divisor on cosine similarities. When
    `tau_override` is a tensor, gradient flows through the loss back to
    the caller's parameter (CLIP-style learnable temperature, #28).
    """
    forecasted_latent, original_latent = predicted_position
    B, T, C, H = forecasted_latent.shape
    train_config = spec.train_configuration
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
    if pos_in_denom and train_config.get('loss_shape') not in (
        'cosine_similarity_batch',
        'cosine_similarity_batch_no_time_neg',
        'cosine_similarity_batch_square',
        'cosine_similarity_batch_full_fh_negs',
    ):
        raise NotImplementedError(
            "include_positive_in_denominator (function arg or "
            "train_configuration key) is only implemented for loss_shape "
            "in {cosine_similarity_batch, cosine_similarity_batch_no_time_neg, "
            "cosine_similarity_batch_square, "
            "cosine_similarity_batch_full_fh_negs}; got "
            f"{train_config.get('loss_shape')!r}."
        )

    if get_history:
        return loss, (forecasted_latent, original_latent)
    return loss
