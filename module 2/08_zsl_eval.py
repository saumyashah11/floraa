"""
FLORA :: script 08 -- the evaluation that decides whether the thesis holds.

Reports, on the standard generalised zero-shot protocol:
  Acc_seen, Acc_unseen, harmonic mean H, and AUSUC over the calibration sweep,
  for each of the three routes (text, proto, fusion), plus image-to-monograph
  retrieval at rank 1 and 5.

Conventional zero-shot numbers on the unseen set alone are inflated and easy to
game. Report H. If a reviewer sees only unseen accuracy they will assume you
restricted the candidate set, because most papers do.

    !python 08_zsl_eval.py --tag m2_m1_swint_joint_mpnet
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS, CKPTS, CACHE
from flora_align import (FloraAligner, load_monographs, encode_texts,
                         build_attr_prototypes, normalise_attr)


def per_class_mean_acc(y, p, class_ids):
    """GZSL convention: average of per-class accuracies, not raw top-1."""
    accs = []
    for c in class_ids:
        m = y == c
        if m.sum():
            accs.append(float((p[m] == c).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def calibrated(sim, seen_pos, gamma):
    s = sim.copy()
    s[:, seen_pos] -= gamma
    return s


def sweep(sim, y, seen_pos, unseen_pos, n=81):
    lo, hi = float(np.percentile(sim, 1)), float(np.percentile(sim, 99))
    gammas = np.linspace(lo - (hi - lo), hi + (hi - lo) * 0.2, n)
    rows = []
    for g in gammas:
        p = calibrated(sim, seen_pos, g).argmax(1)
        s = per_class_mean_acc(y, p, seen_pos)
        u = per_class_mean_acc(y, p, unseen_pos)
        h = 0.0 if (np.isnan(s) or np.isnan(u) or s + u == 0) else 2 * s * u / (s + u)
        rows.append({"gamma": float(g), "seen": s, "unseen": u, "H": h})
    df = pd.DataFrame(rows).sort_values("seen")
    ausuc = float(np.trapz(df.unseen.fillna(0).values, df.seen.fillna(0).values))
    return df, abs(ausuc)


def retrieval(zi, zt, y, k=(1, 5)):
    sim = zi @ zt.T
    order = np.argsort(-sim, axis=1)
    return {f"R@{n}": float(np.mean([y[i] in order[i, :n] for i in range(len(y))]))
            for n in k}


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.load(CKPTS / f"{a.tag}_best.pt", map_location="cpu")
    seen, unseen, classes = s["seen"], s["unseen"], s["all_classes"]
    m1 = s["m1_tag"]

    model = FloraAligner(d_text=s["d_text"], d_shared=s["d_shared"],
                         use_features=s["use_features"], d_feat=s["d_feat"])
    model.load_state_dict(s["model"]); model.to(dev).eval()

    d = RESULTS / f"{m1}_full_export"
    meta = pd.read_csv(d / "meta.csv").reset_index().rename(columns={"index": "row"})
    attr = np.load(d / "attr_logits.npy")
    feat = np.load(d / "features.npy")

    # GZSL test pool: held-out images of seen classes PLUS every unseen image
    pool = meta[(meta.split == "test") | (meta.split == "zsl_test")]
    print(f"GZSL pool {len(pool):,} images over {len(classes)} candidate classes")
    print(pool.canonical_label.value_counts().to_string())

    y = np.array([classes.index(c) for c in pool.canonical_label])
    seen_pos = [classes.index(c) for c in seen]
    unseen_pos = [classes.index(c) for c in unseen]

    texts = load_monographs(classes)
    tb, tclasses = encode_texts(texts, s["encoder"], device=dev,
                                cache=CACHE / f"text_{s['encoder']}.npz")
    assert tclasses == classes, "text bank class order drifted"
    bank = tb.to(dev).mean(1)

    x = torch.from_numpy(attr[pool.row.values]).to(dev)
    f = torch.from_numpy(feat[pool.row.values]).to(dev)
    proto_n = F.normalize(build_attr_prototypes(classes).to(dev), dim=-1)

    with torch.no_grad():
        zi = model.embed_image(x, f)
        zt = model.embed_text(bank)
        sim_text = (zi @ zt.t()).cpu().numpy()
        sim_proto = (normalise_attr(x) @ proto_n.t()).cpu().numpy()

    def z(m):
        return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-8)

    routes = {"text": sim_text,
              "proto": sim_proto,
              "fusion": a.alpha * z(sim_text) + (1 - a.alpha) * z(sim_proto)}

    report = {}
    for name, sim in routes.items():
        raw = sim.argmax(1)
        df, ausuc = sweep(sim, y, seen_pos, unseen_pos)
        best = df.loc[df.H.idxmax()]
        report[name] = {
            "uncalibrated_seen": per_class_mean_acc(y, raw, seen_pos),
            "uncalibrated_unseen": per_class_mean_acc(y, raw, unseen_pos),
            "best_gamma": float(best.gamma),
            "seen_at_bestH": float(best.seen),
            "unseen_at_bestH": float(best.unseen),
            "H": float(best.H),
            "AUSUC": ausuc,
        }
        df.to_csv(RESULTS / f"{a.tag}_{name}_sweep.csv", index=False)
        print(f"\n=== route: {name} ===")
        print(f"  uncalibrated   seen {report[name]['uncalibrated_seen']:.4f}  "
              f"unseen {report[name]['uncalibrated_unseen']:.4f}")
        print(f"  at best gamma  seen {best.seen:.4f}  unseen {best.unseen:.4f}  "
              f"H {best.H:.4f}")
        print(f"  AUSUC {ausuc:.4f}")

    ret = retrieval(zi.cpu().numpy(), zt.cpu().numpy(), y)
    report["retrieval"] = ret
    print(f"\nimage-to-monograph retrieval: {ret}")

    # per-unseen-class breakdown on the fusion route -- the headline table
    p = calibrated(routes["fusion"], seen_pos,
                   report["fusion"]["best_gamma"]).argmax(1)
    rows = []
    for c in unseen:
        ci = classes.index(c); m = y == ci
        conf = pd.Series([classes[i] for i in p[m]]).value_counts().head(3).to_dict()
        rows.append({"unseen_class": c, "n": int(m.sum()),
                     "acc": float((p[m] == ci).mean()), "top_predictions": conf})
    unseen_tbl = pd.DataFrame(rows)
    print("\nper unseen class (fusion, calibrated):")
    print(unseen_tbl.to_string(index=False))
    unseen_tbl.to_csv(RESULTS / f"{a.tag}_unseen_breakdown.csv", index=False)

    (RESULTS / f"{a.tag}_gzsl_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {RESULTS / (a.tag + '_gzsl_report.json')}")
    print("\nSanity check: if 'proto' beats 'text' on unseen classes, your text "
          "tower has learnt nothing the priors did not already encode. Rewrite "
          "the monographs with symptom detail the prior table does not contain.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m2_m1_swint_joint_mpnet")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="fusion weight on the text route")
    main(ap.parse_args())
