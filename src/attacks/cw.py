"""
C&W L2 Attack
Paper: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks," IEEE S&P 2017.
arXiv: https://arxiv.org/abs/1608.04644
"""
import torch
import torch.nn.functional as F


def cw_attack(pipe, image_tensor: torch.Tensor, lr: float = 0.02,
              max_iter: int = 300, initial_c: float = 10.0,
              binary_search_steps: int = 5, max_l2: float = 25.0,
              target_distance: float = 0.5) -> torch.Tensor:
    """
    C&W L2 attack using tanh reparameterization and binary search.

    Two-phase approach:
        Phase 1: Maximize latent distortion (high c, 200 iters)
        Phase 2: Binary search to minimize L2 while maintaining attack

    Args:
        pipe: SD pipeline (uses pipe.vae)
        image_tensor: Input [1,3,H,W] in [0,1]
        lr: Adam learning rate
        max_iter: Phase 2 iterations per binary search step
        max_l2: L2 norm soft cap
    """
    vae = pipe.vae
    dtype = image_tensor.dtype
    x = image_tensor.clone().detach().to(dtype=torch.float32)

    with torch.no_grad():
        z_clean = vae.encode(x.to(dtype=dtype) * 2 - 1).latent_dist.mean.float()

    w_init = torch.atanh(2 * x.clamp(1e-6, 1 - 1e-6) - 1).clone().detach()

    # Phase 1: Pure attack
    w = w_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(200):
        opt.zero_grad()
        x_adv = (torch.tanh(w) + 1) / 2
        z_adv = vae.encode(x_adv.to(dtype=dtype) * 2 - 1).latent_dist.mean.float()
        loss = torch.sum((x_adv - x) ** 2) + 1000 * (-F.mse_loss(z_adv, z_clean))
        loss.backward(); opt.step()
        with torch.no_grad():
            d = (torch.tanh(w) + 1) / 2 - x
            if d.norm().item() > max_l2:
                w.data = torch.atanh(2 * (x + d * max_l2 / d.norm().item()).clamp(1e-6, 1-1e-6) - 1)

    best_adv = ((torch.tanh(w) + 1) / 2).clone().detach()
    with torch.no_grad():
        p1_ld = F.mse_loss(vae.encode(best_adv.to(dtype=dtype)*2-1).latent_dist.mean.float(), z_clean).item()
    eff_th = max(p1_ld * 0.5, target_distance)
    best_l2 = (best_adv - x).norm().item()

    # Phase 2: Binary search
    c_lo, c_hi = 1.0, 1000.0
    for _ in range(binary_search_steps):
        c = (c_lo + c_hi) / 2
        w = w_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        b_l2, b_adv = float('inf'), None
        for _ in range(max_iter):
            opt.zero_grad()
            x_adv = (torch.tanh(w) + 1) / 2
            z_adv = vae.encode(x_adv.to(dtype=dtype)*2-1).latent_dist.mean.float()
            l2 = torch.sum((x_adv - x) ** 2)
            ld = F.mse_loss(z_adv, z_clean)
            (l2 + c * (-ld + torch.clamp(eff_th - ld, min=0) * 10)).backward()
            opt.step()
            with torch.no_grad():
                d = (torch.tanh(w)+1)/2 - x
                if d.norm().item() > max_l2:
                    w.data = torch.atanh(2*(x+d*max_l2/d.norm().item()).clamp(1e-6,1-1e-6)-1)
                if ld.item() > eff_th and l2.item() < b_l2:
                    b_l2 = l2.item(); b_adv = ((torch.tanh(w)+1)/2).clone().detach()
        if b_adv is not None:
            c_hi = c
            if (b_adv-x).norm().item() < best_l2:
                best_l2 = (b_adv-x).norm().item(); best_adv = b_adv
        else:
            c_lo = c

    return best_adv.detach().to(dtype=dtype)
