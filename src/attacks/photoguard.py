"""
PhotoGuard Encoder Attack
Paper: Salman et al., "Raising the Cost of Malicious AI-Powered Image Editing," ICML 2023.
arXiv: https://arxiv.org/abs/2302.06588
"""
import torch
import torch.nn.functional as F


def photoguard_attack(image_tensor: torch.Tensor, vae, eps: float = 32/255,
                      alpha: float = 2/255, steps: int = 100) -> torch.Tensor:
    """
    PhotoGuard encoder attack - pushes latent toward uninformative gray target.

    Algorithm:
        1. Target: z_target = VAE_encode(gray_image)
        2. For t = 1..T:
           a. x_adv = clamp(x + delta, 0, 1)
           b. L = MSE(VAE_encode(x_adv), z_target)  (minimize to target)
           c. delta -= alpha * sign(grad L)
           d. delta = clamp(delta, -eps, eps)

    Args:
        image_tensor: Input [1,3,H,W]
        vae: SD VAE
        eps: L-inf budget (default: 32/255)
        alpha: Step size
        steps: Iterations (default: 100)
    """
    dtype = image_tensor.dtype
    gray = torch.ones_like(image_tensor) * 0.5
    with torch.no_grad():
        z_target = vae.encode(gray * 2 - 1).latent_dist.mean

    delta = torch.zeros_like(image_tensor, dtype=dtype)
    for _ in range(steps):
        x_adv = (image_tensor + delta).clamp(0, 1).detach().requires_grad_(True)
        loss = F.mse_loss(vae.encode(x_adv * 2 - 1).latent_dist.mean, z_target)
        loss.backward()
        with torch.no_grad():
            delta = (delta - alpha * x_adv.grad.sign()).clamp(-eps, eps)

    return (image_tensor + delta).clamp(0, 1).detach()
