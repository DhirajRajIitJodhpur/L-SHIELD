#!/usr/bin/env python3
"""
Run both COCO and ImageNet experiments + cross-dataset comparison.
Author: Dhiraj Raj (M24CSA009) | Supervisor: Dr. Palash Das | IIT Jodhpur

Usage:
    python run_both_experiments.py --device cuda
    python run_both_experiments.py --device cuda --num_images 10  # quick test
"""
import argparse, os, sys, time, csv, traceback
from pathlib import Path
from datetime import timedelta
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def run_single_experiment(dataset_name, images, output_dir, device, model_path, seed=42):
    """Run a single experiment on given images."""
    import torch
    from src.attacks import fgsm_attack, pgd_attack, photoguard_attack, cw_attack
    from src.defenses import diffpure_defense, lssp_defense, compute_clean_latent_stats
    from src.metrics import compute_all_metrics, compute_ssim, compute_asr, LPIPSMetric, compute_fid

    DTYPE = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(seed); np.random.seed(seed)

    od = Path(output_dir)
    for d in ["metrics","charts","images"]:
        (od/d).mkdir(parents=True, exist_ok=True)
    fid_dir = od / "fid_analysis"
    fid_dirs = {"clean": fid_dir/"clean_outputs"}
    fid_dirs["clean"].mkdir(parents=True, exist_ok=True)
    ATK = ["FGSM","PGD","PhotoGuard","C&W"]; AK = ["fgsm","pgd","pg","cw"]
    for ak in AK:
        for suf in ["_outputs","_diffpure_outputs","_lssp_outputs"]:
            d = fid_dir/f"{ak}{suf}"; d.mkdir(parents=True, exist_ok=True)
            fid_dirs[f"{ak}{suf}"] = d

    # Load pipeline
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path, torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    vae = pipe.vae
    lpips_fn = LPIPSMetric(); lpips_fn.load(device)
    stats = compute_clean_latent_stats(images, vae, device)

    # Mask
    mask = np.zeros((512,512,3), dtype=np.uint8)
    yy,xx = np.ogrid[:512,:512]
    mask[((xx-256)**2+(yy-256)**2)<128**2] = 255
    from PIL import Image
    mask_pil = Image.fromarray(mask)

    def pil2t(img):
        return torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    def t2pil(t):
        return Image.fromarray((t.squeeze(0).detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8))
    def img2img(img_pil):
        gen = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            return pipe(prompt="a high quality detailed photograph", image=img_pil,
                       mask_image=mask_pil, num_inference_steps=25, guidance_scale=7.5,
                       generator=gen).images[0]

    all_m = []; actual = len(images)
    for idx in range(actual):
        try:
            x = pil2t(images[idx]).to(dtype=DTYPE); mets = {}
            co = img2img(images[idx]); co.save(fid_dirs["clean"]/f"img{idx}.png")
            if idx < 5: images[idx].save(od/"images"/f"img{idx}_original.png"); co.save(od/"images"/f"img{idx}_clean.png")
            ct = pil2t(co).to(dtype=DTYPE)
            mets["Clean"] = compute_all_metrics(x, ct, lpips_fn); mets["Clean"]["ASR"] = 0.0
            attacked = {}
            for ak,an,fn in [("fgsm","FGSM",lambda:fgsm_attack(x,vae,16/255)),
                              ("pgd","PGD",lambda:pgd_attack(x,vae,16/255,2/255,20)),
                              ("pg","PhotoGuard",lambda:photoguard_attack(x,vae,32/255,2/255,100)),
                              ("cw","C&W",lambda:cw_attack(pipe,x))]:
                adv = fn(); ap_ = t2pil(adv.float()); ao = img2img(ap_)
                ao.save(fid_dirs[f"{ak}_outputs"]/f"img{idx}.png")
                aot = pil2t(ao).to(dtype=DTYPE)
                m = compute_all_metrics(ct, aot, lpips_fn); asr,_ = compute_asr(ct, aot); m["ASR"] = asr
                mets[f"{an} Attack"] = m; attacked[ak] = adv
            for ak,an in zip(AK,ATK):
                for dk,dn,dfn in [("diffpure","DiffPure",lambda t:diffpure_defense(t,vae)),
                                   ("lssp","LSSP",lambda t:lssp_defense(t,vae,stats))]:
                    df = dfn(attacked[ak]); dp = t2pil(df.float()); do = img2img(dp)
                    do.save(fid_dirs[f"{ak}_{dk}_outputs"]/f"img{idx}.png")
                    dot = pil2t(do).to(dtype=DTYPE)
                    m = compute_all_metrics(ct, dot, lpips_fn); asr,_ = compute_asr(ct, dot); m["ASR"] = asr
                    mets[f"{an} + {dn} Defense"] = m
            all_m.append(mets)
            print(f"  [{dataset_name}] [{idx+1}/{actual}] done")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [{dataset_name}] [{idx+1}/{actual}] FAILED: {e}")
            torch.cuda.empty_cache()

    # Aggregate + FID + save
    scenarios = list(all_m[0].keys()); avg = {}
    for s in scenarios:
        avg[s] = {}
        for m in ["MSE","PSNR","SSIM","LPIPS","ASR"]:
            vals = [a[s][m] for a in all_m if s in a]; avg[s][m] = np.mean(vals) if vals else 0
    recovery = {}
    for dn in ["DiffPure","LSSP"]:
        for an in ATK:
            ds,ats = f"{an} + {dn} Defense", f"{an} Attack"
            if ds in avg and ats in avg:
                a_s,d_s = avg[ats]["SSIM"],avg[ds]["SSIM"]; den=1.0-a_s
                recovery[ds] = ((d_s-a_s)/den*100) if den>0.01 else 0

    fid_scores = {}
    cd = str(fid_dirs["clean"])
    for ak,an in zip(AK,ATK):
        for suf,lab in [("_outputs","_attack"),("_diffpure_outputs","_diffpure"),("_lssp_outputs","_lssp")]:
            fid_scores[f"{an}{lab}"] = compute_fid(cd, str(fid_dirs[f"{ak}{suf}"]))
            print(f"  FID {an}{lab}: {fid_scores[f'{an}{lab}']:.1f}")

    # Save CSVs
    with open(od/"metrics"/"defense_summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Defense","Avg_SSIM","Avg_ASR%","Avg_Recovery%","Avg_FID"])
        for dn in ["DiffPure","LSSP"]:
            ss=np.mean([avg.get(f"{a} + {dn} Defense",{}).get("SSIM",0) for a in ATK])
            ar=np.mean([avg.get(f"{a} + {dn} Defense",{}).get("ASR",0) for a in ATK])
            rc=np.mean([recovery.get(f"{a} + {dn} Defense",0) for a in ATK])
            fi=np.mean([fid_scores.get(f"{a}_{dn.lower()}",0) for a in ATK])
            w.writerow([dn,f"{ss:.4f}",f"{ar*100:.1f}",f"{rc:.1f}",f"{fi:.2f}"])

    with open(od/"metrics"/"fid_comparison.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Comparison"]+ATK+["Average"])
        for lb,sf_ in [("Clean vs Attack","_attack"),("Clean vs DiffPure","_diffpure"),("Clean vs LSSP","_lssp")]:
            vs=[fid_scores.get(f"{a}{sf_}",0) for a in ATK]
            w.writerow([lb]+[f"{v:.2f}" for v in vs]+[f"{np.mean(vs):.2f}"])

    with open(od/"metrics"/"lssp_vs_diffpure.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Attack","DP_SSIM","LSSP_SSIM","DP_FID","LSSP_FID","DP_Rec%","LSSP_Rec%"])
        for a in ATK:
            w.writerow([a,f"{avg.get(f'{a} + DiffPure Defense',{}).get('SSIM',0):.4f}",
                        f"{avg.get(f'{a} + LSSP Defense',{}).get('SSIM',0):.4f}",
                        f"{fid_scores.get(f'{a}_diffpure',0):.2f}",f"{fid_scores.get(f'{a}_lssp',0):.2f}",
                        f"{recovery.get(f'{a} + DiffPure Defense',0):.1f}",f"{recovery.get(f'{a} + LSSP Defense',0):.1f}"])

    return avg, recovery, fid_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()

    from src.utils.data_loader import load_coco_images, load_imagenet_images

    print("="*60)
    print("CROSS-DATASET EXPERIMENT: COCO + ImageNet")
    print("="*60)

    t0 = time.time()
    coco_imgs = load_coco_images(num_images=args.num_images)
    print(f"\nRunning COCO experiment ({len(coco_imgs)} images)...")
    c_avg, c_rec, c_fid = run_single_experiment("COCO", coco_imgs, "results/coco_50", args.device, args.model_path)

    inet_imgs = load_imagenet_images(num_images=args.num_images)
    print(f"\nRunning ImageNet experiment ({len(inet_imgs)} images)...")
    i_avg, i_rec, i_fid = run_single_experiment("ImageNet", inet_imgs, "results/imagenet_50", args.device, args.model_path)

    # Cross-dataset comparison
    print("\n" + "="*60)
    print("CROSS-DATASET RESULTS")
    print("="*60)
    ATK = ["FGSM","PGD","PhotoGuard","C&W"]
    for dn in ["DiffPure","LSSP"]:
        c_r = np.mean([c_rec.get(f"{a} + {dn} Defense",0) for a in ATK])
        i_r = np.mean([i_rec.get(f"{a} + {dn} Defense",0) for a in ATK])
        c_f = np.mean([c_fid.get(f"{a}_{dn.lower()}",0) for a in ATK])
        i_f = np.mean([i_fid.get(f"{a}_{dn.lower()}",0) for a in ATK])
        tag = " ***" if dn == "LSSP" else ""
        print(f"  {dn}{tag}: COCO Rec={c_r:.1f}% FID={c_f:.1f} | ImageNet Rec={i_r:.1f}% FID={i_f:.1f}")

    print(f"\nTotal time: {timedelta(seconds=int(time.time()-t0))}")
    print("DONE!")

if __name__ == "__main__":
    main()
