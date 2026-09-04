"""
FLORA :: script 05 -- test-set report, counterfactual explanations, and the
handover artefacts module 2 consumes.

    !python 05_evaluate_and_export.py --tag m1_swint_joint
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import MANIFESTS, CKPTS, RESULTS, NUM_WORKERS, BATCH_SIZE
from flora_model import (FloraDataset, FloraAttributeNet, CAT_HEADS, CAT_VALUES,
                         severity_expectation, N_SEVERITY)


def load(tag, dev):
    s = torch.load(CKPTS / f"{tag}_best.pt", map_location="cpu")
    m = FloraAttributeNet(s["backbone"], n_disease=len(s["classes"]),
                          mode=s["mode"], pretrained=False,
                          img_size=s.get("img_size", 224))
    m.load_state_dict(s["model"]); m.to(dev).eval()
    return m, s["classes"], s.get("img_size", 224)


@torch.no_grad()
def run(model, dl, dev):
    feats, attrs, dpred, dtrue, sev = [], [], [], [], []
    cats = {h: [] for h in CAT_HEADS}
    for x, y in tqdm(dl, desc="infer"):
        with torch.cuda.amp.autocast():
            o = model(x.to(dev), return_feat=True)
        feats.append(o["feat"].float().cpu().numpy())
        attrs.append(o["attr_vec"].float().cpu().numpy())
        for h in CAT_HEADS:
            cats[h].append(o[h].argmax(1).cpu().numpy())
        sev.append(severity_expectation(o["severity"].float()).cpu().numpy())
        key = "disease_bottleneck" if "disease_bottleneck" in o else "disease_direct"
        dpred.append(o[key].argmax(1).cpu().numpy())
        dtrue.append(y["disease"].numpy())
    return (np.concatenate(feats), np.concatenate(attrs),
            {h: np.concatenate(v) for h, v in cats.items()},
            np.concatenate(sev), np.concatenate(dpred), np.concatenate(dtrue))


def counterfactuals(model, attrs, classes, n=25):
    """
    The bottleneck's payoff. Perturb ONE descriptor at a time and record which
    flips the diagnosis. Gives you sentences of the form: 'had the margin read
    diffuse rather than sharply defined, the call would move to X'.
    """
    dev = next(model.parameters()).device
    head = model.bottleneck_head
    spans, i = {}, 0
    for h in CAT_HEADS:
        spans[h] = (i, i + len(CAT_VALUES[h])); i += len(CAT_VALUES[h])
    spans["severity"] = (i, i + N_SEVERITY)

    rows = []
    for idx in range(min(n, len(attrs))):
        v = torch.tensor(attrs[idx: idx + 1], device=dev)
        base = int(head(v).argmax(1))
        for h, (s, e) in spans.items():
            for k in range(e - s):
                w = v.clone()
                w[0, s:e] = -6.0
                w[0, s + k] = 6.0
                new = int(head(w).argmax(1))
                if new != base:
                    val = (CAT_VALUES[h][k] if h in CAT_VALUES else f"severity={k}")
                    rows.append({"sample": idx, "descriptor": h,
                                 "counterfactual_value": val,
                                 "from": classes[base], "to": classes[new]})
    return pd.DataFrame(rows)


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, size = load(a.tag, dev)

    df = pd.read_csv(MANIFESTS / "manifest_attr.csv")
    te = df[(df.split == "test") & (df.zsl_unseen == 0)]
    ds = FloraDataset(te, classes, train=False, size=size)
    dl = DataLoader(ds, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS)

    feats, attrs, cats, sev, dpred, dtrue = run(model, dl, dev)
    keep = dtrue != -100

    rep = classification_report(dtrue[keep], dpred[keep],
                                labels=list(range(len(classes))),
                                target_names=classes, zero_division=0)
    print(rep)
    (RESULTS / f"{a.tag}_test_report.txt").write_text(rep)
    np.savetxt(RESULTS / f"{a.tag}_confusion.csv",
               confusion_matrix(dtrue[keep], dpred[keep],
                                labels=list(range(len(classes)))),
               fmt="%d", delimiter=",")

    sev_mae = float(np.abs(sev - te.severity.values[:len(sev)]).mean())
    print(f"\nseverity MAE on test: {sev_mae:.3f}")

    # ---- handover to module 2
    out = RESULTS / f"{a.tag}_export"
    out.mkdir(exist_ok=True)
    np.save(out / "features.npy", feats)
    np.save(out / "attr_logits.npy", attrs)
    meta = te.iloc[:len(feats)][["image_path", "phash", "canonical_label",
                                 "species", "split"]].copy()
    for h in CAT_HEADS:
        meta[f"pred_{h}"] = [CAT_VALUES[h][i] for i in cats[h]]
    meta["pred_severity"] = np.round(sev, 2)
    meta["pred_disease"] = [classes[i] for i in dpred]
    meta.to_csv(out / "meta.csv", index=False)
    (out / "classes.json").write_text(json.dumps(classes, indent=2))

    cf = counterfactuals(model, attrs, classes, n=a.cf_samples)
    cf.to_csv(RESULTS / f"{a.tag}_counterfactuals.csv", index=False)
    print(f"\n{len(cf)} diagnosis-flipping counterfactuals found")
    print(cf.head(12).to_string(index=False))
    print(f"\nmodule 2 inputs written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1_swint_joint")
    ap.add_argument("--cf_samples", type=int, default=25)
    main(ap.parse_args())
