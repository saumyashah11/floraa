"""
FLORA :: script 01
Stage A (--stage discover) : download archives, walk them, emit the list of raw
                             folder labels plus a machine-proposed mapping.
Stage B (--stage build)    : read your reviewed label_map.csv, hash every image,
                             drop duplicates, and write the final manifest.

    !python 01_download_and_manifest.py --stage discover
    # ... review MANIFESTS/label_map.csv by hand ...
    !python 01_download_and_manifest.py --stage build
"""
import argparse, os, re, shutil, sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (RAW, MANIFESTS, KAGGLE_SETS, MANUAL_SETS, IMG_EXT,
                    SEED, VAL_FRAC, TEST_FRAC, ZSL_UNSEEN, DRIVE)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --------------------------------------------------------------- download
def setup_kaggle():
    """Expects kaggle.json placed at DRIVE/'kaggle.json'."""
    src = DRIVE / "kaggle.json"
    if not src.exists():
        raise SystemExit(
            f"Put your kaggle.json at {src}. Get it from Kaggle > Settings > "
            "Create New Token.")
    dst = Path.home() / ".kaggle"
    dst.mkdir(exist_ok=True)
    shutil.copy(src, dst / "kaggle.json")
    os.chmod(dst / "kaggle.json", 0o600)


def download_all():
    try:
        setup_kaggle()
        import kaggle
    except (Exception, SystemExit) as e:
        print(f"[WARN] Kaggle download unavailable ({e}). "
              "Continuing with manual sets already on disk only.")
    else:
        for slug, folder in KAGGLE_SETS.items():
            out = RAW / folder
            if out.exists() and any(out.rglob("*")):
                print(f"[skip] {folder} already present")
                continue
            out.mkdir(parents=True, exist_ok=True)
            print(f"[get ] {slug} -> {out}")
            try:
                kaggle.api.dataset_download_files(slug, path=str(out), unzip=True, quiet=False)
            except Exception as e:
                # One dataset being gated/renamed/rate-limited must not abort
                # discovery for every dataset queued after it.
                print(f"[FAIL] {slug} did not download ({e}). Skipping it; "
                      "rerun --stage discover later to retry just this one.")

    for folder in MANUAL_SETS:
        p = RAW / folder
        if not p.exists() or not any(p.rglob("*")):
            print(f"[WARN] {p} is empty. Download the Mendeley set by hand from\n"
                  f"       https://data.mendeley.com/datasets/2vzp6ss7vg/1\n"
                  f"       unzip it, and place the folders under {p}")


# --------------------------------------------------------------- discovery
def walk_images():
    rows = []
    roots = [RAW / f for f in list(KAGGLE_SETS.values()) + MANUAL_SETS]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in IMG_EXT or not p.is_file():
                continue
            rel = p.relative_to(root)
            rows.append({
                "image_path": str(p),
                "source": root.name,
                "raw_label": rel.parent.name,
                "raw_path": str(rel.parent),
            })
    return pd.DataFrame(rows)


DISEASE_PATTERNS = [
    # Compound / crop-specific terms first -- each must be checked before the
    # generic single-word patterns below or it gets swallowed by them (e.g.
    # "Cedar_apple_rust" would otherwise match the generic "rust" pattern and
    # propose "apple_rust", which is not a class in class_priors.csv).
    (r"cedar[\s_-]?apple[\s_-]?rust",             "cedar_apple_rust"),
    (r"common[\s_-]?rust",                        "common_rust"),
    (r"northern[\s_-]?leaf[\s_-]?blight",         "northern_leaf_blight"),
    (r"gray[\s_-]?leaf[\s_-]?spot|grey[\s_-]?leaf[\s_-]?spot|cercospora", "gray_leaf_spot"),
    (r"target[\s_-]?spot",                        "target_spot"),
    (r"septoria",                                 "septoria_leaf_spot"),
    (r"leaf[\s_-]?mold|leaf[\s_-]?mould",         "leaf_mold"),
    (r"spider[\s_-]?mite",                        "spider_mite_damage"),
    (r"yellow[\s_-]?leaf[\s_-]?curl",             "yellow_leaf_curl_virus"),
    (r"mosaic",                                   "mosaic_virus"),
    (r"bacterial[\s_-]?spot",                     "bacterial_spot"),
    (r"scab",                                     "scab"),
    (r"black[\s_-]?rot",                          "black_rot"),
    (r"esca|black[\s_-]?measles",                 "esca"),
    (r"isariopsis|leaf[\s_-]?blight",             "leaf_blight"),
    (r"scorch",                                   "leaf_scorch"),
    (r"early[\s_-]?blight",                       "early_blight"),
    (r"late[\s_-]?blight",                        "late_blight"),
    (r"nutrient|deficien|chloros",                 "nutrient_deficiency"),
    # Original flower-corpus patterns, generic fallbacks.
    (r"hea+lthy|fresh|normal|good",               "healthy"),
    (r"black[\s_-]?spot|blackspot",              "black_spot"),
    (r"powder",                                  "powdery_mildew"),
    (r"downy",                                   "downy_mildew"),
    (r"rust",                                    "rust"),
    (r"anthrac",                                 "anthracnose"),
    (r"botrytis|grey[\s_-]?mou?ld|gray[\s_-]?mou?ld|blossom[\s_-]?blight", "botrytis_flower_blight"),
    (r"blight",                                  "blight"),
    (r"leaf[\s_-]?spot|alternaria",               "leaf_spot"),
    (r"virus",                                    "mosaic_virus"),
    (r"mite|insect|pest|aphid",                  "pest_damage"),
]
SPECIES_PATTERNS = [
    # Hibiscus's own binomial name, Hibiscus rosa-sinensis, contains "rosa" --
    # must be checked before the rose pattern or every hibiscus folder in the
    # Mendeley set silently mislabels as rose.
    (r"hibiscus|china[\s_-]?rose|jaba", "hibiscus"),
    (r"rose|rosa|gulab", "rose"),
    (r"marigold|tagetes|genda", "marigold"),
    (r"chrysanth", "chrysanthemum"),
    (r"jasmin|mogra", "jasmine"),
    # Crop expansion set (matches folder names in vipoooool/new-plant-diseases-dataset
    # and spMohanty/plantvillage-dataset, e.g. "Tomato___Early_blight").
    (r"tomato", "tomato"),
    (r"potato", "potato"),
    (r"pepper|bell[\s_-]?pepper|capsicum", "pepper"),
    (r"apple", "apple"),
    (r"cherry", "cherry"),
    (r"peach", "peach"),
    (r"grape", "grape"),
    (r"strawberry", "strawberry"),
    (r"squash", "squash"),
    (r"corn|maize", "maize"),
]


def propose(raw_label: str, source: str, raw_path: str) -> str:
    """Heuristic first guess. YOU MUST REVIEW THIS BY HAND."""
    hay = f"{raw_path} {raw_label} {source}".lower()
    species = next((s for pat, s in SPECIES_PATTERNS if re.search(pat, hay)), None)
    disease = next((d for pat, d in DISEASE_PATTERNS if re.search(pat, hay)), None)
    if disease is None:
        return "UNMAPPED"
    if species is None:
        return f"generic_{disease}" if disease != "healthy" else "generic_healthy"
    return f"{species}_{disease}"


def stage_discover():
    download_all()
    df = walk_images()
    if df.empty:
        raise SystemExit("No images found. Check that the archives unzipped into RAW.")
    print(f"found {len(df):,} images across {df.source.nunique()} sources")

    keys = (df[["source", "raw_label", "raw_path"]]
            .value_counts().reset_index(name="n_images"))
    keys["proposed_canonical"] = [
        propose(r.raw_label, r.source, r.raw_path) for r in keys.itertuples()]
    keys["canonical_label"] = keys["proposed_canonical"]
    keys["keep"] = 1
    out = MANIFESTS / "label_map.csv"
    keys.to_csv(out, index=False)

    df.to_csv(MANIFESTS / "_raw_index.csv", index=False)
    print(f"\nwrote {out}")
    print(keys[["source", "raw_label", "n_images", "proposed_canonical"]].to_string(index=False))
    unmapped = keys[keys.proposed_canonical == "UNMAPPED"].n_images.sum()
    print(f"\n>>> {unmapped:,} images are UNMAPPED. Open label_map.csv, fill the "
          f"'canonical_label' column, set keep=0 for anything you want to drop, "
          f"then run --stage build.")


# --------------------------------------------------------------- build
def phash_of(path):
    import imagehash
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception:
        return None


def stage_build():
    lm_path = MANIFESTS / "label_map.csv"
    if not lm_path.exists():
        raise SystemExit("Run --stage discover first, then review label_map.csv.")
    lm = pd.read_csv(lm_path)
    lm = lm[lm.keep == 1]
    if (lm.canonical_label == "UNMAPPED").any():
        raise SystemExit("label_map.csv still contains UNMAPPED rows. Fix or set keep=0.")

    df = pd.read_csv(MANIFESTS / "_raw_index.csv")
    df = df.merge(lm[["source", "raw_label", "raw_path", "canonical_label"]],
                  on=["source", "raw_label", "raw_path"], how="inner")
    print(f"{len(df):,} images survive the label map")

    tqdm.pandas(desc="perceptual hash")
    df["phash"] = df.image_path.progress_map(phash_of)
    df = df[df.phash.notna()].copy()

    before = len(df)
    df = df.drop_duplicates(subset="phash", keep="first")
    print(f"exact-duplicate removal: {before:,} -> {len(df):,}")

    # near-duplicate bucket: images sharing a 32-bit prefix stay in the same split
    df["dup_bucket"] = df.phash.str[:8]

    # ---- split, grouped by bucket, stratified on canonical_label
    import numpy as np
    rng = np.random.default_rng(SEED)
    df["split"] = "train"
    for label, grp in df.groupby("canonical_label"):
        buckets = grp.dup_bucket.unique()
        rng.shuffle(buckets)
        n = len(buckets)
        n_test = max(1, int(round(n * TEST_FRAC)))
        n_val = max(1, int(round(n * VAL_FRAC)))
        test_b = set(buckets[:n_test])
        val_b = set(buckets[n_test:n_test + n_val])
        df.loc[df.dup_bucket.isin(test_b) & (df.canonical_label == label), "split"] = "test"
        df.loc[df.dup_bucket.isin(val_b) & (df.canonical_label == label), "split"] = "val"

    # ---- zero-shot classes leave the training pool entirely
    df["zsl_unseen"] = df.canonical_label.isin(ZSL_UNSEEN).astype(int)
    df.loc[df.zsl_unseen == 1, "split"] = "zsl_test"

    out = MANIFESTS / "manifest.csv"
    df.to_csv(out, index=False)

    print(f"\nwrote {out}  ({len(df):,} rows)")
    print(pd.crosstab(df.canonical_label, df.split).to_string())
    print("\nzero-shot held out:", ZSL_UNSEEN)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["discover", "build"], required=True)
    a = ap.parse_args()
    stage_discover() if a.stage == "discover" else stage_build()
