"""
DiffPure - Adversarial Purification via Diffusion
Paper: Nie et al., "Diffusion Models for Adversarial Purification," ICML 2022.
arXiv: https://arxiv.org/abs/2205.07460

Adapted for SD img2img: uses VAE encode-decode as the purification bottleneck.
"""
import torch
import torch.nn.functional as F


def diffpure_defense(image_tensor: torch.Tensor, vae) -> torch.Tensor:
    """
    DiffPure defense via VAE encode-decode purification.

    Algorithm:
        1. Pre-smoothing: blend with Gaussian blur (sigma=2.0, weight=0.3)
        2. VAE encode: 512x512x3 -> 64x64x4 (48x compression)
        3. VAE decode: reconstruct from purified latent
        4. Post-smoothing: light blur (sigma=0.5)

    The 48x compression bottleneck destroys high-frequency adversarial
    perturbations while preserving semantic content.

    Limitation: Blind reconstruction - treats ALL values equally.

    Args:
        image_tensor: Adversarial input [1,3,H,W] in [0,1]
        vae: SD VAE model

    Returns:
        Purified image tensor [1,3,H,W]
    """
    device = image_tensor.device
    with torch.no_grad():
        # Pre-smoothing
        ks = 9
        coords = torch.arange(ks, dtype=image_tensor.dtype, device=device) - ks // 2
        g1d = torch.exp(-0.5 * (coords / 2.0) ** 2); g1d = g1d / g1d.sum()
        g2d = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        blurred = F.conv2d(image_tensor, g2d, padding=ks // 2, groups=3)
        x_blend = (0.7 * image_tensor + 0.3 * blurred).clamp(0, 1)

        # VAE encode-decode
        z = vae.encode(x_blend * 2 - 1).latent_dist.mean
        z = z * vae.config.scaling_factor / vae.config.scaling_factor
        x_purified = (vae.decode(z).sample / 2 + 0.5).clamp(0, 1)

        # Post-smoothing
        ks2 = 3
        c2 = torch.arange(ks2, dtype=x_purified.dtype, device=device) - ks2 // 2
        g2 = torch.exp(-0.5 * (c2 / 0.5) ** 2); g2 = g2 / g2.sum()
        g2k = (g2[:, None] * g2[None, :]).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        x_purified = F.conv2d(x_purified, g2k, padding=ks2 // 2, groups=3).clamp(0, 1)

    return x_purified
