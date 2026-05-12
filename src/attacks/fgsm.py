"""
FGSM - Fast Gradient Sign Method
Paper: Goodfellow et al., "Explaining and Harnessing Adversarial Examples," ICLR 2015.
arXiv: https://arxiv.org/abs/1412.6572
"""
import torch
import torch.nn.functional as F


def fgsm_attack(image_tensor: torch.Tensor, vae, eps: float = 16/255) -> torch.Tensor:
    """
    FGSM attack targeting the VAE encoder of Stable Diffusion.

    Algorithm:
        1. Compute clean latent: z_clean = VAE_encode(x)
        2. Forward with gradients: z_adv = VAE_encode(x_adv)
        3. Loss L = -MSE(z_adv, z_clean)  (maximize latent distortion)
        4. Perturbation: delta = eps * sign(grad_x L)
        5. x_adv = clamp(x - delta, 0, 1)

    Args:
        image_tensor: Input image tensor [1,3,H,W] in [0,1], on CUDA
        vae: SD VAE model
        eps: L-inf perturbation budget (default: 16/255)

    Returns:
        Adversarial image tensor [1,3,H,W] in [0,1]
    """
    dtype = image_tensor.dtype
    x = image_tensor.clone().detach().to(dtype=dtype).requires_grad_(True)

    with torch.no_grad():
        z_clean = vae.encode(image_tensor * 2 - 1).latent_dist.mean

    with torch.cuda.amp.autocast(enabled=True):
        z_adv = vae.encode(x * 2 - 1).latent_dist.mean
        loss = -F.mse_loss(z_adv.float(), z_clean.float())

    loss.backward()
    x_adv = (image_tensor - eps * x.grad.data.sign()).clamp(0, 1)
    return x_adv.detach()
