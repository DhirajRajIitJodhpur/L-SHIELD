# Adversarial Attacks and Defenses on Image-to-Image Stable Diffusion Model

**Author:** Dhiraj Raj (M24CSA009) | **Supervisor:** Dr. Palash Das | **IIT Jodhpur**

## Overview

This project evaluates adversarial robustness of Stable Diffusion v1.5 img2img pipeline. We implement 4 attacks and propose **LSSP (Latent Space Statistical Projection)**, a novel defense extending DiffPure.

## Key Results

| Dataset | Defense | Avg Recovery | Avg FID |
|---------|---------|-------------|---------|
| COCO-50 | DiffPure | 9.4% | 221.1 |
| COCO-50 | **LSSP (Ours)** | **14.4%** | **182.7** |
| ImageNet-50 | DiffPure | 8.1% | 203.9 |
| ImageNet-50 | **LSSP (Ours)** | **14.4%** | **166.1** |

LSSP consistently outperforms DiffPure across both datasets and all metrics.

## Quick Start

```bash
pip install -r requirements.txt
python run_both_experiments.py --device cuda
```

See [execute.md](execute.md) for detailed instructions.

## Project Structure

```
src/attacks/       -- 4 attack implementations (FGSM, PGD, PhotoGuard, C&W)
src/defenses/      -- 2 defense implementations (DiffPure, LSSP)
src/metrics/       -- Metrics (MSE, PSNR, SSIM, LPIPS, ASR, FID)
results/coco_50/   -- COCO experiment results
results/imagenet_50/ -- ImageNet experiment results
results/cross_dataset/ -- Cross-dataset comparison
```

## Novel Contribution: LSSP

LSSP extends DiffPure by adding statistical projection between VAE encode and decode:
1. Pre-compute per-position mean and std from clean images
2. At runtime: encode, detect outliers (>3 sigma), clip to boundary, decode
3. Result: 98.5% clean content preserved, only 1-2% outliers corrected

## References

- [1] Goodfellow et al., FGSM, ICLR 2015
- [2] Madry et al., PGD, ICLR 2018
- [3] Salman et al., PhotoGuard, ICML 2023
- [4] Carlini & Wagner, C&W, IEEE S&P 2017
- [5] Nie et al., DiffPure, ICML 2022
- [6] Rombach et al., Stable Diffusion, CVPR 2022
- [7] Heusel et al., FID, NeurIPS 2017
