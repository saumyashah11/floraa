"""
FLORA :: script 06 -- run the trained module 1 network over EVERY split,
including the zero-shot classes it has never seen, and dump the vectors that
module 2 trains on.

Script 05 exported the test split only. Module 2 needs train, val, test and
zsl_test, so this runs the full manifest once.

    !python 06_extract_embeddings.py --tag m1_swint_joint
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MANIFESTS, CKPTS, RESULTS, NUM_WORKERS, BATCH_SIZE
from flora_model import FloraDataset, FloraAttributeNet, CAT_HEADS, CAT_VALUES


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.load(CKPTS / f"{a.tag}_best.pt", map_location="cpu")
    classes = s["classes"]
    model = FloraAttributeNet(s["backbone"], n_disease=len(classes),
                              mode=s["mode"], pretrained=False,
                              img_size=s.get("img_size", 224))
    model.load_state_dict(s["model"]); model.to(dev).eval()

    df = pd.read_csv(MANIFESTS / "manifest_attr.csv").reset_index(drop=True)
    ds = FloraDataset(df, classes, train=False, size=s.get("img_size", 224))
    dl = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    feats, attrs, cats = [], [], {h: [] for h in CAT_HEADS}
    with torch.no_grad():
        for x, _ in tqdm(dl, desc="extract"):
            with torch.cuda.amp.autocast():
                o = model(x.to(dev, non_blocking=True), return_feat=True)
            feats.append(o["feat"].float().cpu().numpy())
            attrs.append(o["attr_vec"].float().cpu().numpy())
            for h in CAT_HEADS:
                cats[h].append(o[h].argmax(1).cpu().numpy())

    feats = np.concatenate(feats); attrs = np.concatenate(attrs)
    out = RESULTS / f"{a.tag}_full_export"; out.mkdir(exist_ok=True)
    np.save(out / "features.npy", feats.astype(np.float32))
    np.save(out / "attr_logits.npy", attrs.astype(np.float32))

    meta = df[["image_path", "phash", "canonical_label", "species", "split",
               "zsl_unseen", "severity", "coverage", "is_healthy"]].copy()
    for h in CAT_HEADS:
        meta[f"pred_{h}"] = [CAT_VALUES[h][i] for i in np.concatenate(cats[h])]
    meta.to_csv(out / "meta.csv", index=False)
    (out / "seen_classes.json").write_text(json.dumps(classes, indent=2))

    print(f"\nfeatures {feats.shape}  attr {attrs.shape}")
    print(meta.groupby(["split", "zsl_unseen"]).size().to_string())
    print(f"written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1_swint_joint")
    main(ap.parse_args())
