"""Chart generation utilities."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CB, CG = '#2C5F7C', '#2E8B57'
ATK = ['FGSM', 'PGD', 'PhotoGuard', 'C&W']


def generate_defense_charts(avg: dict, recovery: dict, fid_scores: dict, output_dir: str):
    """Generate all defense comparison charts."""
    od = Path(output_dir); od.mkdir(parents=True, exist_ok=True)
    x = np.arange(4)

    # SSIM
    dp = [avg.get(f"{a} + DiffPure Defense",{}).get("SSIM",0) for a in ATK]
    ls = [avg.get(f"{a} + LSSP Defense",{}).get("SSIM",0) for a in ATK]
    fig,ax = plt.subplots(figsize=(10,6))
    ax.bar(x-0.15,dp,0.3,label='DiffPure',color=CB,edgecolor='black')
    ax.bar(x+0.15,ls,0.3,label='LSSP (Ours)',color=CG,edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(ATK); ax.set_ylabel('SSIM')
    ax.set_title('Defense SSIM',fontweight='bold'); ax.legend(); ax.grid(axis='y',alpha=0.3)
    plt.tight_layout(); plt.savefig(od/'defense_ssim_comparison.png',dpi=150); plt.close()

    # FID
    if fid_scores:
        fa = [fid_scores.get(f"{a}_attack",0) for a in ATK]
        fd = [fid_scores.get(f"{a}_diffpure",0) for a in ATK]
        fl = [fid_scores.get(f"{a}_lssp",0) for a in ATK]
        fig,ax = plt.subplots(figsize=(10,6)); w=0.25
        ax.bar(x-w,fa,w,label='Attack',color='#F44336',edgecolor='black')
        ax.bar(x,fd,w,label='DiffPure',color=CB,edgecolor='black')
        ax.bar(x+w,fl,w,label='LSSP',color=CG,edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(ATK); ax.set_ylabel('FID')
        ax.set_title('FID Comparison',fontweight='bold'); ax.legend(); ax.grid(axis='y',alpha=0.3)
        plt.tight_layout(); plt.savefig(od/'fid_comparison.png',dpi=150); plt.close()
