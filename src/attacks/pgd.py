"""
PGD - Projected Gradient Descent
Paper: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR 2018.
arXiv: https://arxiv.org/abs/1706.06083
"""
import torch
import torch.nn.functional as F


def pgd_attack(image_tensor: torch.Tensor, vae, eps: float = 16/255,
               alpha: float = 2/255, steps: int = 20) -> torch.Tensor:
    """
    PGD attack - iterative FGSM with projection onto epsilon-ball.

    Algorithm:
        1. Initialize delta ~ Uniform(-eps, eps)
        2. For t = 1..T:
           a. x_adv = clamp(x + delta, 0, 1)
           b. z_adv = VAE_encode(x_adv)
           c. L = -MSE(z_adv, z_clean)
           d. delta = delta - alpha * sign(grad L)
           e. delta = clamp(delta, -eps, eps)
        3. Return clamp(x + delta, 0, 1)

    Args:
        image_tensor: Input [1,3,H,W] in [0,1]
        vae: SD VAE model
        eps: L-inf budget (default: 16/255)
        alpha: Step size (default: 2/255)
        steps: Number of iterations (default: 20)

    Returns:
        Adversarial image tensor
    """
    dtype = image_tensor.dtype
    with torch.no_grad():
        z_clean = vae.encode(image_tensor * 2 - 1).latent_dist.mean

    delta = torch.empty_like(image_tensor).uniform_(-eps, eps).to(dtype=dtype)

    for _ in range(steps):
        x_adv = (image_tensor + delta).clamp(0, 1).detach().requires_grad_(True)
        z_adv = vae.encode(x_adv * 2 - 1).latent_dist.mean
        loss = -F.mse_loss(z_adv, z_clean)
        loss.backward()
        with torch.no_grad():
            delta = (delta - alpha * x_adv.grad.sign()).clamp(-eps, eps)

    return (image_tensor + delta).clamp(0, 1).detach()
