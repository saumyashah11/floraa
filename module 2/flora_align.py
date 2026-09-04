"""
FLORA :: module 2 core -- attribute-mediated vision-language alignment.

Three routes from an image to a disease name. The paper compares all three:

  text   : projected attribute vector  <->  projected monograph embedding
  proto  : normalised attribute vector <->  attribute prototype from the priors
  fusion : convex combination of the two similarity matrices

Only 'text' and 'fusion' can name a disease whose monograph was written but
whose images were never seen. That is the claim under test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from flora_model import CAT_HEADS, CAT_VALUES, N_SEVERITY

HERE = Path(__file__).parent
_PRIORS_PATH = HERE / "class_priors.csv"
if not _PRIORS_PATH.exists():
    _PRIORS_PATH = HERE.parent / "class_priors.csv"
PRIORS = pd.read_csv(_PRIORS_PATH)

TEXT_ENCODERS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "scibert": "allenai/scibert_scivocab_uncased",
}

# ------------------------------------------------------------- attr layout
def attr_spans():
    """Byte-for-byte the layout FloraAttributeNet.forward concatenates."""
    spans, i = {}, 0
    for h in CAT_HEADS:
        spans[h] = (i, i + len(CAT_VALUES[h])); i += len(CAT_VALUES[h])
    spans["severity"] = (i, i + N_SEVERITY); i += N_SEVERITY
    spans["coverage"] = (i, i + 1); i += 1
    spans["is_healthy"] = (i, i + 1); i += 1
    return spans, i


SPANS, ATTR_DIM = attr_spans()


def normalise_attr(vec: torch.Tensor) -> torch.Tensor:
    """
    Raw logits -> a probability-shaped attribute vector, so it is directly
    comparable with a one-hot class prototype. Softmax per categorical head,
    sigmoid on the two scalars.
    """
    out = torch.zeros_like(vec)
    for h in list(CAT_HEADS) + ["severity"]:
        s, e = SPANS[h]
        out[:, s:e] = torch.softmax(vec[:, s:e], dim=-1)
    for h in ("coverage", "is_healthy"):
        s, e = SPANS[h]
        out[:, s:e] = torch.sigmoid(vec[:, s:e])
    return out


def build_attr_prototypes(classes):
    """One-hot attribute vector per class, straight from class_priors.csv."""
    P = PRIORS.set_index("canonical_label")
    proto = np.zeros((len(classes), ATTR_DIM), dtype=np.float32)
    for ci, c in enumerate(classes):
        if c not in P.index:
            raise KeyError(f"{c} missing from class_priors.csv")
        r = P.loc[c]
        for h in CAT_HEADS:
            s, _ = SPANS[h]
            proto[ci, s + CAT_VALUES[h].index(str(r[h]))] = 1.0
        s, _ = SPANS["severity"]
        proto[ci, s + int(r["severity"])] = 1.0
        proto[ci, SPANS["coverage"][0]] = float(r["coverage"])
        proto[ci, SPANS["is_healthy"][0]] = float(r["is_healthy"])
    return torch.from_numpy(proto)


# ------------------------------------------------------------- monographs
def prior_sentence(label):
    """Auto-generated fourth variant, keeping text and priors consistent."""
    P = PRIORS.set_index("canonical_label")
    if label not in P.index:
        return ""
    r = P.loc[label]
    if int(r["is_healthy"]) == 1:
        return (f"Healthy {r['species']} {r['organ']} tissue with no lesion, "
                f"no discolouration and no surface growth of any kind.")
    return (f"On {r['species']} {r['organ']} tissue the affected area is "
            f"{str(r['colour']).replace('_', ' ')} in colour, with a "
            f"{str(r['margin']).replace('_', ' ')} margin, a "
            f"{str(r['distribution']).replace('_', ' ')} distribution across "
            f"the organ, and a {str(r['texture']).replace('_', ' ')} surface "
            f"texture. Typical severity is level {int(r['severity'])} of four, "
            f"covering about {float(r['coverage']) * 100:.0f} percent of the "
            f"visible tissue.")


def load_monographs(classes):
    raw = yaml.safe_load((HERE / "disease_monographs.yaml").read_text())
    missing = [c for c in classes if c not in raw]
    if missing:
        raise KeyError("no monograph written for: " + ", ".join(missing))
    texts = {}
    for c in classes:
        e = raw[c]
        vs = [" ".join(v.split()) for v in e["variants"]]
        head = f"{e['common_name']}, caused by {e['pathogen']}, on {e['species']}."
        vs = [f"{head} {v}" for v in vs]
        ps = prior_sentence(c)
        if ps:
            vs.append(f"{head} {ps}")
        texts[c] = vs
    return texts


@torch.no_grad()
def encode_texts(texts, encoder="mpnet", device="cpu", cache: Path = None):
    """
    Returns (embeddings [C, V, D], classes). The text tower is FROZEN by
    default: with only nineteen classes, fine-tuning it memorises the corpus in
    a handful of steps and the zero-shot number becomes meaningless.
    """
    classes = list(texts.keys())
    if cache and cache.exists():
        z = np.load(cache)
        if list(z["classes"]) == classes:
            return torch.from_numpy(z["emb"]), classes

    from transformers import AutoTokenizer, AutoModel
    mid = TEXT_ENCODERS.get(encoder, encoder)
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModel.from_pretrained(mid).to(device).eval()

    n_var = max(len(v) for v in texts.values())
    embs = []
    for c in classes:
        vs = texts[c] + [texts[c][-1]] * (n_var - len(texts[c]))
        b = tok(vs, padding=True, truncation=True, max_length=256,
                return_tensors="pt").to(device)
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        embs.append(F.normalize(pooled, dim=-1).cpu())
    emb = torch.stack(embs)
    if cache:
        np.savez(cache, emb=emb.numpy(), classes=np.array(classes))
    return emb, classes


# ------------------------------------------------------------- model
class Tower(nn.Module):
    def __init__(self, d_in, d_out, hidden=512, p=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(p), nn.Linear(hidden, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class FloraAligner(nn.Module):
    """
    image side  : ATTR_DIM -> shared space. Deliberately NOT the backbone
                  features. Feeding raw features lets the model bypass the
                  bottleneck and the zero-shot result becomes an artefact.
    text side   : monograph embedding -> the same shared space.
    """

    def __init__(self, d_text, d_shared=256, hidden=512, use_features=False,
                 d_feat=0, temp=0.07):
        super().__init__()
        d_img = ATTR_DIM + (d_feat if use_features else 0)
        self.use_features = use_features
        self.img = Tower(d_img, d_shared, hidden)
        self.txt = Tower(d_text, d_shared, hidden)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / temp), dtype=torch.float32))

    def embed_image(self, attr, feat=None):
        x = normalise_attr(attr)
        if self.use_features:
            x = torch.cat([x, F.normalize(feat, dim=-1)], dim=-1)
        return self.img(x)

    def embed_text(self, t):
        return self.txt(t)

    def forward(self, attr, text_bank, feat=None):
        zi = self.embed_image(attr, feat)
        zt = self.embed_text(text_bank)
        return self.logit_scale.exp().clamp(max=100.0) * zi @ zt.t()


def alignment_loss(logits, y, proto_sim=None, w_proto=0.0, smooth=0.05):
    """
    Cross-modal cross-entropy against the class text bank. Optionally add a
    prototype-consistency term that pulls the attribute vector toward its own
    class prior, which stabilises the 'proto' route.
    """
    loss = F.cross_entropy(logits, y, label_smoothing=smooth)
    parts = {"align": float(loss.detach())}
    if proto_sim is not None and w_proto > 0:
        lp = F.cross_entropy(proto_sim * 10.0, y, label_smoothing=smooth)
        loss = loss + w_proto * lp
        parts["proto"] = float(lp.detach())
    return loss, parts
