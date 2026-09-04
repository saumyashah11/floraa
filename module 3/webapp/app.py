"""
FLORA :: Module 3 local web app.

A real Flask server, not a static mockup. It loads the actual trained
checkpoints -- MuRIL, FloraTextAttributeNet, FusionGate, DiseaseHead -- once
at startup and runs genuine live inference on whatever a user types. The
"try it" decoder in the Claude-artifact version of this project used
JavaScript keyword matching because a static page has nowhere to run a
model; this app has a real Python process behind it, so it runs the real
model instead.

Run:
    pip install flask
    python "module 3/webapp/app.py"
    -> open http://127.0.0.1:5050

Requires the module 3 pipeline to have been run already (09 -> 10 -> 11 ->
12), so the checkpoints and result files this app reads actually exist.
"""
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_file, abort, render_template
from werkzeug.utils import secure_filename

HERE = Path(__file__).parent
M3 = HERE.parent
ROOT = M3.parent
sys.path.insert(0, str(M3))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module 2"))

from config import CORPUS, CKPTS, RESULTS, CACHE  # noqa: E402
from flora_align import normalise_attr, SPANS, ATTR_DIM  # noqa: E402
from flora_text import (FloraTextAttributeNet, FusionGate, DiseaseHead,
                        confidence_weighted_fusion, predict_disease,
                        species_onehot, species_restricted_mask,
                        SPECIES_LIST, ALL_CLASSES, TEXT_ENCODERS,
                        CAT_HEADS, CAT_VALUES, N_SEVERITY)  # noqa: E402

DEVICE = "cpu"
app = Flask(__name__)
UPLOAD_DIR = HERE / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ loading
print("loading MuRIL (this happens once, at startup)...")
from transformers import AutoTokenizer, AutoModel  # noqa: E402

_ENCODER_NAME = "muril"
_TOK = AutoTokenizer.from_pretrained(TEXT_ENCODERS[_ENCODER_NAME])
_MURIL = AutoModel.from_pretrained(TEXT_ENCODERS[_ENCODER_NAME]).to(DEVICE).eval()

_text_ck = torch.load(CKPTS / f"m3_text_{_ENCODER_NAME}_best.pt", map_location=DEVICE)
TEXT_MODEL = FloraTextAttributeNet(d_in=_text_ck["d_in"], hidden=_text_ck["hidden"]).to(DEVICE).eval()
TEXT_MODEL.load_state_dict(_text_ck["model"])

GATE = FusionGate().to(DEVICE).eval()
GATE.load_state_dict(torch.load(CKPTS / "m3_fusion_gate_best.pt", map_location=DEVICE)["model"])

_head_ck = torch.load(CKPTS / "m3_disease_head_best.pt", map_location=DEVICE)
HEAD = DiseaseHead(attr_dim=_head_ck["attr_dim"], n_species=len(_head_ck["species_list"]),
                  n_classes=_head_ck["n_classes"]).to(DEVICE).eval()
HEAD.load_state_dict(_head_ck["model"])

# class_priors.csv can grow (more diseases/species added) without anyone rerunning
# 11_fuse_and_evaluate.py. DiseaseHead's input/output widths are frozen at training
# time, so a grown live schema doesn't just mean "fewer classes recognised" -- it's
# an outright shape mismatch (crashes) or, if padded, a silently wrong class-index
# mapping (misdiagnoses). Detect the drift once at startup and disable the route
# rather than guess.
DISEASE_HEAD_STALE = (
    _head_ck["attr_dim"] != ATTR_DIM
    or _head_ck["n_classes"] != len(ALL_CLASSES)
    or list(_head_ck["species_list"]) != list(SPECIES_LIST)
)
if DISEASE_HEAD_STALE:
    print(
        "WARNING: m3_disease_head_best.pt was trained on a smaller class_priors.csv "
        f"({_head_ck['n_classes']} classes, {len(_head_ck['species_list'])} species) than the one "
        f"currently loaded ({len(ALL_CLASSES)} classes, {len(SPECIES_LIST)} species). The "
        "'fusion_learned' route is disabled until 11_fuse_and_evaluate.py is rerun on the expanded "
        "manifest -- the zero-shot 'fusion_prototype' route still covers every class."
    )

_export = RESULTS / "m1_resnet50_joint_full_export"
META = pd.read_csv(_export / "meta.csv").reset_index().rename(columns={"index": "row"})
ATTR_LOGITS = np.load(_export / "attr_logits.npy")
print(f"loaded {len(META):,} real Module 1 image records for the photo picker")

CORPUS_DF = pd.read_csv(CORPUS / "query_corpus.csv", encoding="utf-8-sig")

print("ready.")


# ------------------------------------------------------------------ inference helpers
@torch.no_grad()
def embed_live(text: str) -> torch.Tensor:
    b = _TOK([text], padding=True, truncation=True, max_length=64, return_tensors="pt")
    h = _MURIL(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, dim=-1)


def decode_categorical(logits_row, head):
    s, e = SPANS[head]
    probs = torch.softmax(logits_row[s:e], dim=-1)
    idx = int(probs.argmax())
    return CAT_VALUES[head][idx], float(probs[idx])


def top_k_from_logits(logits, classes, k=3, mask=None):
    logits = logits.clone()
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)[0]
    order = torch.argsort(probs, descending=True)[:k]
    return [{"cls": classes[i], "p": float(probs[i])} for i in order.tolist()]


def top_k_from_sim(sim, classes, k=3):
    order = torch.argsort(sim[0], descending=True)[:k]
    vals = sim[0][order]
    lo, hi = float(vals.min()), float(vals.max())
    spread = max(hi - lo, 1e-6)
    return [{"cls": classes[i], "p": float((v - lo) / spread)} for i, v in zip(order.tolist(), vals.tolist())]


@torch.no_grad()
def run_diagnosis(text: str, species_choice: str, image_id):
    text = text.strip()
    if not text:
        return {"error": "empty message"}

    emb = embed_live(text)
    text_out = TEXT_MODEL(emb)
    text_logits = text_out["attr_vec"]
    text_probs = normalise_attr(text_logits)

    image_used = image_id is not None
    true_label = None
    true_species = None
    image_path = None
    if image_used:
        row = META.iloc[int(image_id)]
        img_logits = torch.from_numpy(ATTR_LOGITS[int(image_id):int(image_id) + 1]).float()
        true_label = row.canonical_label
        true_species = row.species
        image_path = row.image_path
    else:
        img_logits = torch.zeros(1, ATTR_DIM)
    img_probs = normalise_attr(img_logits)

    species = species_choice if species_choice in SPECIES_LIST else "mixed"
    mask = species_restricted_mask([species], ALL_CLASSES) if species_choice in SPECIES_LIST else None

    fused, w = GATE(img_probs, text_probs)
    if DISEASE_HEAD_STALE:
        learned_top = None
    else:
        learned_logits = HEAD(fused, species_onehot([species]))
        learned_top = top_k_from_logits(learned_logits, ALL_CLASSES, k=3, mask=mask)

    _, sim_proto = predict_disease(fused, ALL_CLASSES, allowed_mask=mask)
    proto_top = top_k_from_sim(sim_proto, ALL_CLASSES, k=3)

    _, sim_img = predict_disease(img_probs, ALL_CLASSES, allowed_mask=mask)
    _, sim_txt = predict_disease(text_probs, ALL_CLASSES, allowed_mask=mask)
    img_only_top = top_k_from_sim(sim_img, ALL_CLASSES, k=3) if image_used else None
    txt_only_top = top_k_from_sim(sim_txt, ALL_CLASSES, k=1)

    detected = {}
    for h in CAT_HEADS:
        val, conf = decode_categorical(text_logits[0], h)
        if val != "not_applicable":
            detected[h] = {"value": val, "confidence": conf}
    sev_s, sev_e = SPANS["severity"]
    sev_probs = torch.softmax(text_logits[0][sev_s:sev_e], dim=-1)
    severity = int(sev_probs.argmax())
    is_healthy_prob = float(torch.sigmoid(text_logits[0][SPANS["is_healthy"][0]]))
    coverage = float(torch.sigmoid(text_logits[0][SPANS["coverage"][0]]))

    return {
        "detected": detected,
        "severity": severity,
        "is_healthy_prob": is_healthy_prob,
        "coverage": coverage,
        "image_used": image_used,
        "true_label": true_label,
        "true_species": true_species,
        "image_path": image_path,
        "image_id": image_id,
        "species_used": species,
        "gate_image_weight": float(w[0]),
        "learned_route_available": not DISEASE_HEAD_STALE,
        "learned_route_note": (
            None if not DISEASE_HEAD_STALE else
            "Disabled: class_priors.csv has grown since m3_disease_head_best.pt was trained "
            "(rerun 11_fuse_and_evaluate.py to restore it). Showing the zero-shot prototype route instead."
        ),
        "routes": {
            "fusion_learned": learned_top,
            "fusion_prototype": proto_top,
            "image_only": img_only_top,
            "text_only": txt_only_top,
        },
    }


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/examples")
def api_examples():
    """A curated spread of real Module 1 image records (thumbnails served
    from disk via /media/<row_id>) plus a few real corpus query strings."""
    picks = []
    for split, n in [("test", 2), ("zsl_test", 3)]:
        sub = META[META.split == split]
        for cls in sub.canonical_label.unique():
            rows = sub[sub.canonical_label == cls].head(n)
            for _, r in rows.iterrows():
                picks.append({"id": int(r.row), "label": r.canonical_label,
                             "species": r.species, "split": r.split})
    text_examples = (CORPUS_DF[CORPUS_DF.split == "test"]
                     .groupby("canonical_label").head(1)
                     .sample(min(8, len(CORPUS_DF)), random_state=0)[["canonical_label", "text"]]
                     .to_dict(orient="records"))
    return jsonify({"photos": picks, "texts": text_examples, "species_list": SPECIES_LIST})


@app.route("/media/<int:row_id>")
def media(row_id):
    if row_id < 0 or row_id >= len(META):
        abort(404)
    path = Path(META.iloc[row_id].image_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        abort(403)  # only ever serve files the app itself indexed under the project root
    if not path.exists():
        abort(404)
    return send_file(path)


def save_uploaded_image(upload_file):
    if upload_file is None or not upload_file.filename:
        return None
    original_name = Path(secure_filename(upload_file.filename)).name
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    unique_name = f"{stem}_{uuid4().hex[:8]}{suffix or '.png'}"
    path = UPLOAD_DIR / unique_name
    upload_file.save(path)
    return {
        "filename": original_name,
        "url": f"/uploads/{unique_name}",
        "size_bytes": int(path.stat().st_size),
    }


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    path = (UPLOAD_DIR / filename).resolve()
    try:
        path.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        abort(403)
    if not path.exists():
        abort(404)
    return send_file(path)


@app.route("/api/decode", methods=["POST"])
def api_decode():
    if request.content_type and "multipart/form-data" in request.content_type:
        text = request.form.get("text", "")
        species = request.form.get("species", "unknown")
        image_id_raw = request.form.get("image_id") or None
        image_id = int(image_id_raw) if image_id_raw not in (None, "", "unknown") else None
        uploaded = request.files.get("image")
        result = run_diagnosis(text, species, image_id)
        uploaded_info = save_uploaded_image(uploaded)
        if uploaded_info:
            result["uploaded_image"] = uploaded_info
            result["photo_note"] = (
                "Custom image preview received. The live disease pipeline is still text-first; "
                "the indexed dataset photos remain the source for direct image embeddings in this UI."
            )
        else:
            result["uploaded_image"] = None
        result["species_known"] = species not in (None, "", "unknown")
        return jsonify(result)

    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text", "")
    species = body.get("species", "unknown")
    image_id_raw = body.get("image_id")
    image_id = int(image_id_raw) if image_id_raw not in (None, "", "unknown") else None
    result = run_diagnosis(text, species, image_id)
    result["species_known"] = species not in (None, "", "unknown")
    return jsonify(result)


@app.route("/api/report")
def api_report():
    """Bundles the real result files 09-12 wrote to disk, for the dashboard."""
    def read_json(name):
        p = RESULTS / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def read_csv(name):
        p = RESULTS / name
        return pd.read_csv(p).to_dict(orient="records") if p.exists() else None

    corpus_counts = CORPUS_DF.canonical_label.value_counts().to_dict()
    script_mix = CORPUS_DF.script_mix.value_counts().to_dict()

    z = np.load(RESULTS / "m3_confusion.npz", allow_pickle=True)
    confusion = {
        "classes": list(z["classes"]),
        "image_only": z["image_only"].tolist(),
        "fusion_gate": z["fusion_gate"].tolist(),
        "fusion_learned": z["fusion_learned"].tolist() if "fusion_learned" in z else None,
    }

    qdf = pd.read_csv(RESULTS / "m3_qualitative_full.csv", encoding="utf-8-sig")
    bins = np.linspace(0, 1, 17)
    gate_hist = {}
    for subset in ["seen_test", "zsl_unseen_test"]:
        sub = qdf[qdf.subset == subset]
        counts, edges = np.histogram(sub.image_weight_in_fusion, bins=bins)
        gate_hist[subset] = {"counts": counts.tolist(), "edges": edges.round(3).tolist(), "n": int(len(sub))}

    return jsonify({
        "corpus_total": int(len(CORPUS_DF)),
        "corpus_per_class": corpus_counts,
        "corpus_script_mix": script_mix,
        "train_curve": read_json(f"m3_text_{_ENCODER_NAME}_history.json"),
        "attr_accuracy": read_json(f"m3_text_{_ENCODER_NAME}_test_report.json"),
        "route_comparison": read_csv("m3_route_comparison.csv"),
        "robustness_sweep": read_csv("m3_robustness_sweep.csv"),
        "confusion": confusion,
        "gate_weight_hist": gate_hist,
        "qualitative_examples": pd.read_csv(RESULTS / "m3_qualitative_examples.csv",
                                            encoding="utf-8-sig").to_dict(orient="records"),
        "fusion_report": read_json("m3_fusion_report.json"),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
