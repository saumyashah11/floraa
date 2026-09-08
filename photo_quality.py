"""
FLORA :: photo quality pre-check.

Runs before diagnosis to catch a bad photo -- too blurry, too dark/bright,
or blank/degenerate -- so a farmer gets "retake this photo" instead of a
confident-looking wrong diagnosis silently caused by a bad shot. This
exists because image_only accuracy has been the weakest signal across
every ablation in this project (Module 2's bottleneck-leak control, Module
3's route comparison and robustness sweep) -- without a check like this, a
real deployment cannot tell whether a wrong diagnosis came from the model
or from the photo.

Thresholds are calibrated from this project's own real training images
(`--calibrate`), not picked arbitrarily: "too blurry" means blurrier than
the bottom 5% of real dataset photos, not blurrier than some invented
constant.

    python photo_quality.py --calibrate
    python photo_quality.py --image path/to/photo.jpg
"""
import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import MANIFESTS, DRIVE  # noqa: E402

THRESHOLDS_PATH = DRIVE / "photo_quality_thresholds.json"
MIN_SIDE_PX = 64  # below this, even a perfect photo can't give the model much to work with

_thresholds_cache = None


def _load_gray(image):
    """image: path str/Path, PIL.Image, or raw bytes. Returns (grayscale array, (w, h))."""
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.open(BytesIO(image)).convert("RGB")
    arr = np.array(pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray, pil.size


def measure(image):
    """Raw quality metrics for one image -- no thresholds applied yet."""
    gray, (w, h) = _load_gray(image)
    return {
        "width": w, "height": h,
        "blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "dark_clip_frac": float((gray < 10).mean()),
        "bright_clip_frac": float((gray > 245).mean()),
    }


def calibrate_thresholds(n_sample=1500, seed=42):
    """
    Sample real dataset images and derive percentile-based thresholds.
    Prefers the full (uncapped) manifest so calibration reflects real photo
    quality, not the small capped training subset from the proof-of-concept
    retrain.
    """
    import pandas as pd
    mf_path = MANIFESTS / "manifest_attr_full.csv"
    if not mf_path.exists():
        mf_path = MANIFESTS / "manifest_attr.csv"
    df = pd.read_csv(mf_path)
    sample = df.sample(min(n_sample, len(df)), random_state=seed)

    rows = []
    for p in sample.image_path:
        try:
            rows.append(measure(p))
        except Exception:
            continue
    m = pd.DataFrame(rows)
    print(f"calibrated on {len(m):,} real images (from {mf_path.name})")

    thresholds = {
        "blur_variance_p05": float(m.blur_variance.quantile(0.05)),
        "blur_variance_p10": float(m.blur_variance.quantile(0.10)),
        "brightness_p05": float(m.brightness_mean.quantile(0.05)),
        "brightness_p95": float(m.brightness_mean.quantile(0.95)),
        "contrast_p05": float(m.contrast_std.quantile(0.05)),
        "n_calibration_images": len(m),
        "source_manifest": mf_path.name,
    }
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    print(json.dumps(thresholds, indent=2))
    print(f"written -> {THRESHOLDS_PATH}")
    return thresholds


def load_thresholds():
    global _thresholds_cache
    if _thresholds_cache is None:
        if not THRESHOLDS_PATH.exists():
            raise SystemExit(
                f"{THRESHOLDS_PATH} not found -- run `python photo_quality.py "
                f"--calibrate` first.")
        _thresholds_cache = json.loads(THRESHOLDS_PATH.read_text())
    return _thresholds_cache


def assess(image):
    """
    Returns {"ok": bool, "issues": [{"code", "message"}...], "metrics": {...}}.
    `issues` is empty, one, or several -- a UI renders each as a warning, it
    never hard-blocks (a genuinely sick-but-well-photographed leaf should
    never be refused).
    """
    t = load_thresholds()
    m = measure(image)
    issues = []

    if min(m["width"], m["height"]) < MIN_SIDE_PX:
        issues.append({"code": "too_small", "message":
            f"Image is only {m['width']}x{m['height']}px -- too small for a "
            f"reliable diagnosis."})
    if m["blur_variance"] < t["blur_variance_p05"]:
        issues.append({"code": "blurry", "message":
            "Photo looks blurry -- hold the camera steady and let it focus "
            "before capturing."})
    elif m["blur_variance"] < t["blur_variance_p10"]:
        issues.append({"code": "slightly_blurry", "message":
            "Photo looks a little soft -- a sharper shot would help."})
    if m["brightness_mean"] < t["brightness_p05"] or m["dark_clip_frac"] > 0.5:
        issues.append({"code": "too_dark", "message":
            "Photo looks too dark -- retake it in better light."})
    if m["brightness_mean"] > t["brightness_p95"] or m["bright_clip_frac"] > 0.5:
        issues.append({"code": "too_bright", "message":
            "Photo looks overexposed/washed out -- avoid direct glare or "
            "backlighting."})
    if m["contrast_std"] < t["contrast_p05"]:
        issues.append({"code": "low_detail", "message":
            "Photo looks flat/blank -- make sure the affected leaf actually "
            "fills the frame."})

    return {"ok": not issues, "issues": issues, "metrics": m}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--image", help="path to a single image to assess")
    ap.add_argument("--n_sample", type=int, default=1500)
    args = ap.parse_args()

    if args.calibrate:
        calibrate_thresholds(n_sample=args.n_sample)
    elif args.image:
        print(json.dumps(assess(args.image), indent=2))
    else:
        ap.print_help()
