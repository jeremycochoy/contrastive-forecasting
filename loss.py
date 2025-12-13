import torch
import torch.nn.functional as F

def cosine_similarity_from_normalized(a, b):
    return (a * b).sum(dim=-1)

def contrastive_latent_loss(predicted_position, validation, spec, get_history=False):
    forecasted_latent, original_latent = predicted_position
    B, T, C, H = forecasted_latent.shape
    train_config = spec.train_configuration
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

    elif train_config.get('loss_shape') == 'mse':
        loss = F.mse_loss(hy, hy_hat) - F.mse_loss(hx, hy)
    else:
        shape = train_config.get('loss_shape')
        raise Exception(f"Loss shape {shape} not implemented")

    if get_history:
        return loss, (forecasted_latent, original_latent)
    return loss
