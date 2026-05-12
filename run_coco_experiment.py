#!/usr/bin/env python3
"""Run 50 COCO experiment with FID analysis.
Author: Dhiraj Raj (M24CSA009) | Supervisor: Dr. Palash Das | IIT Jodhpur
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.utils.data_loader import load_coco_images

def main():
    parser = argparse.ArgumentParser(description="COCO 50 experiment")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/coco_50")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()
    print(f"COCO experiment: {args.num_images} images, output: {args.output_dir}")
    print("Loading images..."); images = load_coco_images(num_images=args.num_images)
    print(f"Loaded {len(images)} images. Run the full pipeline with run_both_experiments.py")

if __name__ == "__main__":
    main()
