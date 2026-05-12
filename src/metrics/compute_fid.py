"""FID - Frechet Inception Distance. Heusel et al., NeurIPS 2017."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def compute_fid(dir1: str, dir2: str) -> float:
    """Compute FID between two directories of images.
    Uses clean-fid library if available, else manual Inception v3."""
    try:
        from cleanfid import fid as cleanfid
        return cleanfid.compute_fid(str(dir1), str(dir2))
    except Exception:
        return _manual_fid(dir1, dir2)


def _manual_fid(dir1: str, dir2: str) -> float:
    from torchvision.models import inception_v3
    from torchvision import transforms
    from scipy.linalg import sqrtm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    tf = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(),
                             transforms.Normalize([0.5]*3, [0.5]*3)])

    def feats(d):
        fs = []
        for p in sorted(list(Path(d).glob("*.png")) + list(Path(d).glob("*.jpg"))):
            try:
                with torch.no_grad():
                    fs.append(model(tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)).cpu().numpy().flatten())
            except: continue
        return np.array(fs) if fs else np.zeros((1, 2048))

    f1, f2 = feats(dir1), feats(dir2)
    if len(f1) < 2 or len(f2) < 2: return float('inf')
    mu1, s1 = f1.mean(0), np.cov(f1, rowvar=False)
    mu2, s2 = f2.mean(0), np.cov(f2, rowvar=False)
    d = mu1 - mu2
    cm = sqrtm(s1 @ s2)
    if np.iscomplexobj(cm): cm = cm.real
    return float(d @ d + np.trace(s1 + s2 - 2 * cm))
