# How to Execute the Experiments

**Author:** Dhiraj Raj (M24CSA009) | **Supervisor:** Dr. Palash Das | **IIT Jodhpur**

## Prerequisites

- NVIDIA GPU with 24GB+ VRAM (tested on RTX A6000 48GB)
- Python 3.10+, CUDA 12.1+

## Setup

```bash
git clone <repo_url>
cd mtp_dhiraj_50coco_50imgnet
pip install -r requirements.txt
```

## Run Experiments

### Both experiments (recommended, ~12 hours)
```bash
python run_both_experiments.py --device cuda
```

### Individual experiments (~6 hours each)
```bash
python run_coco_experiment.py --num_images 50 --device cuda
python run_imagenet_experiment.py --num_images 50 --device cuda
```

### Quick test (10 images, ~1 hour)
```bash
python run_both_experiments.py --device cuda --num_images 10
```

### Remote GPU with nohup
```bash
nohup python run_both_experiments.py --device cuda > experiment.log 2>&1 &
```

## Output Structure

```
results/
  coco_50/metrics/     -- CSV results
  coco_50/charts/      -- PNG charts
  imagenet_50/metrics/
  imagenet_50/charts/
  cross_dataset/       -- Cross-dataset comparison
```

## Metrics

| Metric | Measures | Better |
|--------|----------|--------|
| MSE | Pixel error | Lower |
| PSNR | Signal quality (dB) | Higher |
| SSIM | Structural similarity | Higher |
| LPIPS | Perceptual distance | Lower |
| ASR | Attack success (SSIM<0.7) | Lower |
| FID | Distribution similarity | Lower |

## Expected Results

| Defense | COCO Recovery | ImageNet Recovery | COCO FID | ImageNet FID |
|---------|--------------|------------------|----------|-------------|
| DiffPure | 9.4% | 8.1% | 221.1 | 203.9 |
| **LSSP** | **14.4%** | **14.4%** | **182.7** | **166.1** |
