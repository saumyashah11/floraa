"""
FLORA :: module 3 core -- farmer free-text to the Module 1 symptom vocabulary.

MuRIL (or another Indic encoder) reads raw Hinglish/Marathi/Hindi text through
a set of attribute heads structurally identical to Module 1's, so a text
query and a photograph land in exactly the same attribute space defined by
attribute_schema.yaml -- same SPANS layout, same normalise_attr, same class
prototypes. FusionGate then merges the two attribute vectors, weighting each
side by how confident it is, before the fused vector is read off against
Module 2's class prototypes.

Nothing here predicts a disease directly from text. Module 1 doesn't either
(in bottleneck/joint mode): the disease name is read off the symptom
description, not guessed from raw input. Keeping that discipline in Module 3
is what makes the fusion meaningful instead of two black boxes voting.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "module 2"))

from config import LOSS_WEIGHTS                                   # noqa: E402
from flora_model import CAT_HEADS, CAT_VALUES, N_SEVERITY          # noqa: E402
from flora_align import SPANS, ATTR_DIM, normalise_attr, build_attr_prototypes, PRIORS  # noqa: E402

SPECIES_LIST = sorted(PRIORS["species"].unique().tolist())
SPECIES_IDX = {s: i for i, s in enumerate(SPECIES_LIST)}
ALL_CLASSES = sorted(PRIORS["canonical_label"].tolist())
SPECIES_OF = PRIORS.set_index("canonical_label")["species"].to_dict()


def species_onehot(species_values, device="cpu"):
    """species_values: iterable of species strings (rose/marigold/hibiscus/mixed)."""
    idx = torch.tensor([SPECIES_IDX[s] for s in species_values], dtype=torch.long)
    return F.one_hot(idx, num_classes=len(SPECIES_LIST)).float().to(device)


def species_restricted_mask(true_species, classes=None):
    """
    [N, len(classes)] boolean mask: True where a candidate class matches the
    plant species a photo (or a farmer's stated crop) is known to be, or is a
    species-agnostic 'generic_*' class. Shared by 11_fuse_and_evaluate.py and
    the local webapp so the "app already knows the crop" assumption is
    defined in exactly one place.
    """
    classes = list(classes) if classes is not None else ALL_CLASSES
    class_species = np.array([SPECIES_OF[c] for c in classes])
    mask = np.zeros((len(true_species), len(classes)), dtype=bool)
    for i, sp in enumerate(true_species):
        mask[i] = (class_species == sp) | (class_species == "mixed")
    return torch.from_numpy(mask)

TEXT_ENCODERS = {
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "marathi_bert": "l3cube-pune/marathi-bert-v2",
    "hi_mr_dev": "l3cube-pune/hindi-marathi-dev-bert",
}


@torch.no_grad()
def encode_texts(texts, encoder="muril", device="cpu", cache: Path = None,
                  batch_size=32, max_length=64):
    """
    Mean-pooled, L2-normalised sentence embeddings from a frozen Indic
    encoder. Frozen for the same reason Module 2 freezes its text tower: a
    few thousand sentences is not enough signal to fine-tune ~240M MuRIL
    parameters without memorising the corpus outright. 10_train_module3.py
    trains lightweight attribute heads on these fixed vectors instead --
    Module 1's frozen-DINOv2 linear-probe pattern, applied to text. Pass
    --finetune_backbone there to unfreeze end to end on a GPU.
    """
    texts = list(texts)
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["texts"]) == texts:
            return torch.from_numpy(z["emb"])

    from transformers import AutoTokenizer, AutoModel
    mid = TEXT_ENCODERS.get(encoder, encoder)
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModel.from_pretrained(mid).to(device).eval()

    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        b = tok(batch, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt").to(device)
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        embs.append(F.normalize(pooled, dim=-1).cpu())
    emb = torch.cat(embs, dim=0)
    if cache:
        np.savez(cache, emb=emb.numpy(), texts=np.array(texts, dtype=object))
    return emb


class FloraTextAttributeNet(nn.Module):
    """Frozen-embedding -> attribute heads, one head per axis in attribute_schema.yaml."""

    def __init__(self, d_in, hidden=256, drop=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop))
        self.cat_heads = nn.ModuleDict(
            {h: nn.Linear(hidden, len(CAT_VALUES[h])) for h in CAT_HEADS})
        self.severity_head = nn.Linear(hidden, N_SEVERITY)
        self.coverage_head = nn.Linear(hidden, 1)
        self.healthy_head = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.trunk(x)
        out = {h: self.cat_heads[h](f) for h in CAT_HEADS}
        out["severity"] = self.severity_head(f)
        out["coverage"] = self.coverage_head(f).squeeze(-1)
        out["is_healthy"] = self.healthy_head(f).squeeze(-1)
        out["attr_vec"] = torch.cat(
            [out[h] for h in CAT_HEADS]
            + [out["severity"], out["coverage"].unsqueeze(-1), out["is_healthy"].unsqueeze(-1)],
            dim=-1)
        return out


def text_loss(out, y, label_smooth=0.05):
    """Same weighted multi-task objective as flora_model.flora_loss, minus the
    disease term: Module 3 predicts symptoms, never a disease name directly."""
    parts = {}
    for h in CAT_HEADS:
        parts[h] = F.cross_entropy(out[h], y[h], label_smoothing=label_smooth)
    parts["severity"] = F.cross_entropy(out["severity"], y["severity"], label_smoothing=label_smooth)
    parts["coverage"] = F.smooth_l1_loss(torch.sigmoid(out["coverage"]), y["coverage"])
    parts["is_healthy"] = F.binary_cross_entropy_with_logits(out["is_healthy"], y["is_healthy"])
    total = sum(LOSS_WEIGHTS.get(k, 1.0) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def head_confidence(attr_probs: torch.Tensor) -> torch.Tensor:
    """
    8-number confidence fingerprint of a normalised attribute vector: the max
    softmax probability of each categorical head plus severity, then coverage
    and is_healthy verbatim (already in [0,1]). FusionGate reads sixteen of
    these (image side + text side) to decide who to trust.
    """
    parts = []
    for h in list(CAT_HEADS) + ["severity"]:
        s, e = SPANS[h]
        parts.append(attr_probs[:, s:e].max(dim=-1, keepdim=True).values)
    c0 = SPANS["coverage"][0]
    h0 = SPANS["is_healthy"][0]
    parts.append(attr_probs[:, c0:c0 + 1])
    parts.append(attr_probs[:, h0:h0 + 1])
    return torch.cat(parts, dim=-1)  # [N, 8]


class FusionGate(nn.Module):
    """
    Learned late fusion. sixteen confidence numbers (image side + text side)
    in, one scalar per sample out: how much weight the image gets in the
    convex combination of the two normalised attribute vectors. A crisp photo
    of a textbook lesion should dominate a vague text complaint; a precise
    text description should dominate a blurry or badly-lit photo. The gate
    learns that trade-off from data instead of using a fixed alpha, and is
    trained with random noise injected on the image side (see
    11_fuse_and_evaluate.py) so it learns to fall back on text when the photo
    degrades -- the actual failure mode farmers hit in the field.
    """

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, img_probs, txt_probs):
        ci = head_confidence(img_probs)
        ct = head_confidence(txt_probs)
        w = torch.sigmoid(self.net(torch.cat([ci, ct], dim=-1))).squeeze(-1)
        fused = w.unsqueeze(-1) * img_probs + (1 - w).unsqueeze(-1) * txt_probs
        return fused, w


def confidence_weighted_fusion(img_probs, txt_probs):
    """
    Deterministic control: weight each side by its own mean head confidence,
    no training involved. FusionGate is compared against this baseline, not
    just against the single-modality routes -- a learned gate that cannot
    beat the training-free heuristic would not be worth the extra parameters.
    """
    ci = head_confidence(img_probs).mean(dim=-1)
    ct = head_confidence(txt_probs).mean(dim=-1)
    w = ci / (ci + ct + 1e-9)
    fused = w.unsqueeze(-1) * img_probs + (1 - w).unsqueeze(-1) * txt_probs
    return fused, w


class DiseaseHead(nn.Module):
    """
    Learned disease read-off: fused attribute probabilities plus the plant
    species (a one-hot -- known in any real deployment, since a farmer's app
    already knows what crop it is looking at, exactly the assumption the
    species-restricted evaluation protocol makes) go in, disease logits over
    every candidate class come out.

    This exists because nearest-prototype matching (predict_disease below) is
    an *unlearned* heuristic: cosine similarity to a one-hot class prior
    cannot exploit anything beyond raw attribute agreement, so two classes
    that share a class_priors.csv row (a species-specific disease and its
    generic_* counterpart) are permanently indistinguishable to it. A trained
    classifier can still separate them via correlations the prior table
    doesn't encode -- coverage, severity, which species/organ combinations
    are actually seen together. The trade-off, and it is a real one: a
    disease with zero training examples in any modality is invisible to this
    head, whereas the prototype route needs no examples at all, only a
    written-down attribute profile. Report both.
    """
    def __init__(self, attr_dim, n_species, n_classes, hidden=128, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attr_dim + n_species, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, n_classes))

    def forward(self, fused_attr_probs, species_onehot):
        return self.net(torch.cat([fused_attr_probs, species_onehot], dim=-1))


def predict_disease(attr_probs, classes, device="cpu", allowed_mask=None):
    """
    Nearest-prototype read-off. Reuses Module 2's class-prior prototypes
    directly (the 'proto' route from flora_align.py) -- the cheapest, most
    literal way to turn a symptom vector into a disease name, and the same
    convention 07/08 use: no extra L2 norm on the probability side.

    allowed_mask, if given, is a [N, C] boolean tensor restricting the
    candidate set per sample -- e.g. "only diseases of the species the app
    already knows this photo is of". Many of class_priors.csv's 22 rows share
    an identical attribute combination (a species-specific disease and its
    generic_* counterpart), which no amount of symptom evidence, image or
    text, can tell apart; restricting the candidate set is how a real product
    resolves that, not a change to the underlying model.
    """
    proto = build_attr_prototypes(classes).to(device).float()
    proto_n = F.normalize(proto, dim=-1)
    sim = attr_probs @ proto_n.t()
    if allowed_mask is not None:
        sim = sim.masked_fill(~allowed_mask, float("-inf"))
    return sim.argmax(dim=-1), sim
