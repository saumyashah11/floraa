"""
FLORA :: script 12 -- render Module 3's numbers into figures and a bundle
the demo UI can quote directly. Pure post-processing: everything it reads was
written by 09/10/11 from real data (real Module 1 photographs, a generated
but class-prior-grounded query corpus, a fusion gate trained on a disjoint
split). No numbers are invented here.

    python 12_make_report.py --encoder muril
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS, RESULTS  # noqa: E402

INK = "#1f2430"
MUTED = "#6b7280"
BLUE = "#3b6fd6"
ORANGE = "#e08a2c"
GREEN = "#2f9e6e"
RED = "#c0453a"
GRID = "#e4e4e7"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "font.size": 10.5, "axes.titlesize": 12,
    "axes.titleweight": "bold", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=150)
    plt.close(fig)
    print(f"  -> {name}")


def corpus_composition(encoder):
    df = pd.read_csv(CORPUS / "query_corpus.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    counts = df.canonical_label.value_counts().sort_values()
    ax[0].barh(counts.index, counts.values, color=BLUE)
    ax[0].set_title("Farmer queries per disease class")
    ax[0].set_xlabel("queries")
    ax[0].tick_params(axis="y", labelsize=8)

    mix = df.script_mix.value_counts()
    colors = [BLUE, ORANGE, GREEN]
    ax[1].pie(mix.values, labels=mix.index, autopct="%1.0f%%", colors=colors,
              startangle=90, textprops={"color": INK})
    ax[1].set_title("Script mix of the corpus")
    savefig(fig, "m3_corpus_composition.png")


def training_curves(encoder):
    hist = json.loads((RESULTS / f"m3_text_{encoder}_history.json").read_text())
    ep = [h["epoch"] for h in hist]
    loss = [h["loss"] for h in hist]
    acc = [h["val_mean_acc"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, loss, color=RED, lw=2, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color=RED)
    ax1.tick_params(axis="y", labelcolor=RED)
    ax2 = ax1.twinx()
    ax2.plot(ep, acc, color=BLUE, lw=2, label="val mean attribute accuracy")
    ax2.set_ylabel("val mean attribute accuracy", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylim(0, 1.02)
    ax2.grid(False)
    fig.suptitle(f"Module 3 text-head training ({encoder})", fontweight="bold")
    savefig(fig, "m3_training_curves.png")


def attribute_accuracy(encoder):
    te = json.loads((RESULTS / f"m3_text_{encoder}_test_report.json").read_text())
    heads = [k for k in te if k != "coverage_mae"]
    vals = [te[k] for k in heads]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(heads, vals, color=[BLUE if v >= 0.6 else ORANGE for v in vals])
    ax.axhline(1 / 8, color=MUTED, ls="--", lw=1, label="~chance (worst-case head)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held-out test accuracy")
    ax.set_title(f"Text-only attribute-head accuracy ({encoder}, held-out queries)")
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.text(0.99, 0.03, f"coverage MAE: {te['coverage_mae']:.3f}",
           transform=ax.transAxes, ha="right", color=MUTED, fontsize=9)
    ax.legend(loc="lower left", fontsize=8)
    savefig(fig, "m3_attribute_accuracy.png")


def route_comparison():
    df = pd.read_csv(RESULTS / "m3_route_comparison.csv")
    routes = df.route.unique().tolist()
    subsets = df.subset.unique().tolist()
    protocols = df.protocol.unique().tolist()
    x = np.arange(len(routes))
    w = 0.35
    colors = {"seen_test": BLUE, "zsl_unseen_test": ORANGE}
    titles = {"open_set_22": "Open-set: candidate = all 22 classes",
             "species_restricted": "Realistic: candidate = photo's known species"}

    fig, axes = plt.subplots(1, len(protocols), figsize=(8.5 * len(protocols), 5.5), sharey=True)
    if len(protocols) == 1:
        axes = [axes]
    for ax, proto_name in zip(axes, protocols):
        sub_df = df[df.protocol == proto_name]
        for i, s in enumerate(subsets):
            sub = sub_df[sub_df.subset == s].set_index("route").loc[routes]
            bars = ax.bar(x + (i - 0.5) * w, sub.accuracy, width=w,
                          label=s.replace("_", " "), color=colors.get(s, MUTED))
            for b, v in zip(bars, sub.accuracy):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([r.replace("_", "\n") for r in routes])
        ax.set_title(titles.get(proto_name, proto_name))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("disease accuracy")
    fig.suptitle("Image-only vs text-only vs fusion", fontweight="bold")
    savefig(fig, "m3_route_comparison.png")


def robustness_curve():
    df = pd.read_csv(RESULTS / "m3_robustness_sweep.csv")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df.noise_sigma, df.image_only_acc, "o-", color=RED, lw=2, label="image only")
    ax.plot(df.noise_sigma, df.fusion_confidence_acc, "s--", color=ORANGE, lw=2,
           label="fusion (confidence-weighted)")
    ax.plot(df.noise_sigma, df.fusion_gate_acc, "^-", color=GREEN, lw=2.2,
           label="fusion (prototype route)")
    if "fusion_learned_acc" in df.columns:
        ax.plot(df.noise_sigma, df.fusion_learned_acc, "D-", color=BLUE, lw=2.6,
               label="fusion (learned classifier)")
    ax.set_xlabel("synthetic noise added to the photo's attribute logits (σ)")
    ax.set_ylabel("disease accuracy")
    ax.set_title("A bad photo degrades faster than a bad sentence")
    ax.set_ylim(0, 1.05)
    ax.legend()
    savefig(fig, "m3_robustness_curve.png")


def confusion_plot():
    z = np.load(RESULTS / "m3_confusion.npz", allow_pickle=True)
    classes = list(z["classes"])
    short = [c.replace("_", "\n") for c in classes]
    keys = [k for k in ["image_only", "fusion_learned", "fusion_gate"] if k in z]
    titles = {"image_only": "Image only", "fusion_learned": "Fusion (learned classifier)",
             "fusion_gate": "Fusion (prototype route)"}
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 6.5))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        m = z[key].astype(float)
        row_sum = m.sum(1, keepdims=True); row_sum[row_sum == 0] = 1
        norm = m / row_sum
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(short, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(short, fontsize=7)
        ax.set_title(titles[key])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.grid(False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if m[i, j] > 0:
                    ax.text(j, i, int(m[i, j]), ha="center", va="center",
                           fontsize=6, color="white" if norm[i, j] > 0.5 else INK)
    fig.suptitle("Confusion matrices, row-normalised (seen_test + zsl_unseen_test)", fontweight="bold")
    savefig(fig, "m3_confusion_fusion.png")


def gate_weight_distribution():
    df = pd.read_csv(RESULTS / "m3_qualitative_full.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for subset, color in [("seen_test", BLUE), ("zsl_unseen_test", ORANGE)]:
        sub = df[df.subset == subset]
        ax.hist(sub.image_weight_in_fusion, bins=20, range=(0, 1), alpha=0.6,
               color=color, label=f"{subset.replace('_', ' ')} (n={len(sub)})")
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("gate weight on the image side (1 = trust photo fully, 0 = trust text fully)")
    ax.set_ylabel("count")
    ax.set_title("What the learned gate trusts, seen vs never-seen-in-images classes")
    ax.legend()
    savefig(fig, "m3_gate_weight_distribution.png")


def main(a):
    print("rendering Module 3 figures ->", RESULTS)
    corpus_composition(a.encoder)
    training_curves(a.encoder)
    attribute_accuracy(a.encoder)
    route_comparison()
    robustness_curve()
    confusion_plot()
    gate_weight_distribution()
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="muril")
    main(ap.parse_args())
