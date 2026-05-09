import torch
import torch.nn.functional as F

def cosine_similarity_from_normalized(a, b):
    return (a * b).sum(dim=-1)

def contrastive_latent_loss(predicted_position, validation, spec,
                            get_history=False, tau_override=None):
    """Compute the contrastive divergence loss.

    Args:
        predicted_position: tuple of (forecasted_latent, original_latent).
        validation: True during validation (skips training-only paths).
        spec: SimpleNamespace with `train_configuration` dict.
        get_history: if True, returns intermediate (kept for compat).
        tau_override: optional tensor or float overriding the dict's
            `contrastive_divergence_temperature`. Used by the
            learnable-τ trainer to pass a 0-d tensor that gets gradient.

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
        
        # Cross-batch negatives: compare across batch dimension (not just C dimension)
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]
        
        # Compute similarities across batch dimension: [B, B, T-1, C]
        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)
        
        # Create mask to exclude same batch element (diagonal)
        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)
        
        # Exponential and mask: [B, B, T-1, C]
        neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
        
        # Sum across second batch dimension: [B, T-1, C]
        neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch
        # In the new version, all positives together, all negatives together, cross batch.
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

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
        # Only keeps cross-channel and cross-batch negatives.
        # Useful for ARMA experiments where consecutive time slices are nearly identical.
        positives = torch.exp(
            cosine_similarity_from_normalized(hy_norm, hy_hat_norm) / tau
        )

        # Cross-channel negatives (same time step, different channels)
        sims_xx = cosine_similarity_from_normalized(hx_norm.unsqueeze(3), hx_norm.unsqueeze(2))
        mask_mat = ~torch.eye(C, dtype=torch.bool, device=sims_xx.device)
        mask_mat = mask_mat.view(1, 1, C, C)
        neg_xx = torch.exp(sims_xx / tau).masked_fill(~mask_mat, 0).sum(dim=2)

        # Cross-batch negatives: compare across batch dimension
        hy_norm_exp = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
        hy_hat_norm_exp = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

        sims_cross_batch = cosine_similarity_from_normalized(hy_norm_exp, hy_hat_norm_exp)

        mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
        mask_batch = mask_batch.view(B, B, 1, 1)

        neg_cross_batch_exp = torch.exp(sims_cross_batch / tau).masked_fill(~mask_batch, 0)
        neg_cross_batch = neg_cross_batch_exp.sum(dim=1)

        negatives = neg_xx + neg_cross_batch
        loss = -torch.log(positives / negatives.sum(dim=0, keepdim=True)).mean()

    elif train_config.get('loss_shape') == 'mse':
        loss = F.mse_loss(hy, hy_hat) - F.mse_loss(hx, hy)
    else:
        shape = train_config.get('loss_shape')
        raise Exception(f"Loss shape {shape} not implemented")

    if get_history:
        return loss, (forecasted_latent, original_latent)
    return loss
