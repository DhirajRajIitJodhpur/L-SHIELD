"""Image quality metrics: MSE, PSNR, SSIM, LPIPS, ASR."""
import torch
import torch.nn.functional as F
import numpy as np


def compute_mse(t1: torch.Tensor, t2: torch.Tensor) -> float:
    return F.mse_loss(t1, t2).item()

def compute_psnr(t1: torch.Tensor, t2: torch.Tensor) -> float:
    mse = compute_mse(t1, t2)
    return 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)

def compute_ssim(t1: torch.Tensor, t2: torch.Tensor, window_size: int = 11) -> float:
    """SSIM - Wang et al., IEEE TIP 2004."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2
    mu1 = F.avg_pool2d(t1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(t2, window_size, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    sig1_sq = F.avg_pool2d(t1**2, window_size, stride=1, padding=pad) - mu1_sq
    sig2_sq = F.avg_pool2d(t2**2, window_size, stride=1, padding=pad) - mu2_sq
    sig12 = F.avg_pool2d(t1*t2, window_size, stride=1, padding=pad) - mu12
    ssim_map = ((2*mu12+C1)*(2*sig12+C2)) / ((mu1_sq+mu2_sq+C1)*(sig1_sq+sig2_sq+C2))
    return ssim_map.mean().item()

def compute_asr(clean_out: torch.Tensor, attacked_out: torch.Tensor,
                threshold: float = 0.7) -> tuple:
    """Attack Success Rate: 1 if SSIM < threshold."""
    ssim = compute_ssim(clean_out, attacked_out)
    return (1.0 if ssim < threshold else 0.0), ssim

def compute_all_metrics(t1: torch.Tensor, t2: torch.Tensor, lpips_fn) -> dict:
    return {"MSE": compute_mse(t1,t2), "PSNR": compute_psnr(t1,t2),
            "SSIM": compute_ssim(t1,t2), "LPIPS": lpips_fn(t1,t2)}


class LPIPSMetric:
    """LPIPS - Zhang et al., CVPR 2018. arXiv:1801.03924"""
    def __init__(self):
        self._fn = None
    def load(self, device="cuda"):
        try:
            import lpips
            self._fn = lpips.LPIPS(net="alex").to(device).eval()
        except Exception:
            pass
    def __call__(self, t1, t2):
        if self._fn:
            with torch.no_grad(): return self._fn(t1*2-1, t2*2-1).item()
        return compute_mse(t1, t2) * 10
