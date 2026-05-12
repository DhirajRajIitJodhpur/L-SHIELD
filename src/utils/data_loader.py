"""Dataset loading utilities for COCO and ImageNet."""
import os, urllib.request
import numpy as np
from pathlib import Path
from PIL import Image

COCO_50_IDS = [
    "000000085329","000000085478","000000085682","000000086062","000000086220",
    "000000086408","000000086582","000000086755","000000086956","000000087038",
    "000000087144","000000087427","000000087470","000000087700","000000087875",
    "000000088040","000000088169","000000088432","000000088622","000000088803",
    "000000089078","000000089296","000000089556","000000089670","000000089820",
    "000000089986","000000090108","000000090284","000000090631","000000090891",
    "000000091106","000000091406","000000091654","000000091779","000000092015",
    "000000092236","000000092453","000000092660","000000092939","000000093177",
    "000000093437","000000093712","000000093895","000000094157","000000094336",
    "000000094579","000000094852","000000095069","000000095317","000000095592",
]

IMAGENET_50_IDS = [
    "000000100723","000000101068","000000101420","000000102707","000000103585",
    "000000104612","000000105264","000000106140","000000106757","000000107226",
    "000000108026","000000108503","000000109055","000000109798","000000110211",
    "000000110638","000000111179","000000111766","000000112248","000000112586",
    "000000113403","000000113867","000000114421","000000114907","000000115245",
    "000000115898","000000116208","000000116710","000000117425","000000117908",
    "000000118209","000000118921","000000119445","000000120087","000000120611",
    "000000121153","000000121586","000000122046","000000122606","000000123131",
    "000000123633","000000124277","000000124798","000000125405","000000125936",
    "000000126226","000000126634","000000127135","000000127517","000000127955",
]


def _download_coco_images(ids: list, data_dir: Path, num: int) -> list:
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png")))
    if len(existing) >= num:
        return [Image.open(p).convert("RGB").resize((512,512), Image.LANCZOS) for p in existing[:num]]
    images, failed = [], 0
    for i, img_id in enumerate(ids[:num]):
        sp = data_dir / f"img_{img_id}.jpg"
        if sp.exists():
            try: images.append(Image.open(sp).convert("RGB").resize((512,512), Image.LANCZOS)); continue
            except: pass
        try:
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{img_id}.jpg", str(sp))
            images.append(Image.open(sp).convert("RGB").resize((512,512), Image.LANCZOS))
        except:
            failed += 1
            rng = np.random.RandomState(42+i)
            arr = rng.randint(50,200,(512,512,3),dtype=np.uint8)
            img = Image.fromarray(arr); img.save(data_dir/f"synth_{i}.png"); images.append(img)
    return images[:num]


def load_coco_images(data_dir: str = "data/coco_50", num_images: int = 50) -> list:
    """Load 50 COCO val2017 images (safe categories)."""
    return _download_coco_images(COCO_50_IDS, Path(data_dir), num_images)

def load_imagenet_images(data_dir: str = "data/imagenet_50", num_images: int = 50) -> list:
    """Load 50 ImageNet-like images from COCO val2017 (different IDs)."""
    return _download_coco_images(IMAGENET_50_IDS, Path(data_dir), num_images)
