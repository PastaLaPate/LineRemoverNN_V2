import torch
import torch.nn.functional as F


def ssim_loss(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """1 - SSIM, so minimizing this maximizes structural similarity."""
    C1, C2 = 0.01**2, 0.03**2
    channel = pred.shape[1]

    kernel = _gaussian_kernel(window_size, 1.5, channel).to(pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=channel)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = (
        F.conv2d(pred * pred, kernel, padding=pad, groups=channel) - mu1_sq
    ).clamp(min=0)
    sigma2_sq = (
        F.conv2d(target * target, kernel, padding=pad, groups=channel) - mu2_sq
    ).clamp(min=0)
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return 1 - ssim_map.mean()


def _gaussian_kernel(size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, size, size).contiguous()


def criterion(pred, logits, blank, ruled):
    # The target is the difference (the lines themselves)
    target_mask = (blank - ruled).clamp(0, 1)

    # 1. Mask Loss (Use BCE with Logits for numerical stability)
    # This replaces the L1 loss on the mask for better gradient flow
    mask_loss = F.binary_cross_entropy_with_logits(logits, target_mask)

    # 2. Reconstruction Loss (SSIM)
    # pred is now bounded, so SSIM won't explode
    ssim = ssim_loss(pred, blank)

    # 3. Optional: Add a small L1 penalty on the final image
    # to ensure colors stay accurate
    l1_reconstruction = F.l1_loss(pred, blank)

    return mask_loss * 0.4 + ssim * 0.4 + l1_reconstruction * 0.2
