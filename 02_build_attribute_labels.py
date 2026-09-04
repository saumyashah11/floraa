"""
FLORA :: script 02  -- attach the language-bottleneck attribute targets.

Pass 1 (default)  : class-prior expansion. Costs nothing, covers every image.
Pass 2 (--vlm)    : Qwen2-VL-2B in 4-bit refines ONLY the instance-varying
                    fields (organ, severity, coverage). Cached to Drive by
                    phash so a dropped Colab session costs you nothing.

    !python 02_build_attribute_labels.py
    !python 02_build_attribute_labels.py --vlm --limit 6000
"""
import argparse, json, sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import MANIFESTS, CACHE, IMG_SIZE

HERE = Path(__file__).parent
SCHEMA = yaml.safe_load((HERE / "attribute_schema.yaml").read_text())
PRIORS = pd.read_csv(HERE / "class_priors.csv")

CAT_HEADS = list(SCHEMA["categorical"].keys())


def build_priors(df):
    missing = set(df.canonical_label) - set(PRIORS.canonical_label)
    if missing:
        raise SystemExit(
            "These canonical labels have no row in class_priors.csv:\n  "
            + "\n  ".join(sorted(missing))
            + "\nAdd a row for each, or remap them in label_map.csv.")
    cols = CAT_HEADS + ["severity", "coverage", "is_healthy", "species"]
    out = df.merge(PRIORS[["canonical_label"] + cols], on="canonical_label", how="left")
    out["refined"] = 0
    return out


# ------------------------------------------------------------------ VLM
QUESTIONS = {
    "organ": ("Which single plant part fills most of this photograph? "
              "Answer with exactly one word: leaf, petal, bud, sepal, or stem."),
    "severity": ("How much of the visible plant tissue shows disease symptoms "
                 "such as spots, discolouration, powder or rot? Answer with "
                 "exactly one word: none, trace, mild, moderate, or severe."),
}
SEV_MAP = {"none": 0, "trace": 1, "mild": 2, "moderate": 3, "severe": 4}
ORGANS = SCHEMA["categorical"]["organ"]["values"]


def load_vlm():
    import torch
    from transformers import (Qwen2VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    mid = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        mid, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    proc = AutoProcessor.from_pretrained(mid, min_pixels=256 * 28 * 28,
                                         max_pixels=512 * 28 * 28)
    model.eval()
    return model, proc


def ask(model, proc, image, question):
    from PIL import Image as PILImage
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True).strip().lower()


def refine(df, limit):
    from PIL import Image as PILImage
    cache_path = CACHE / "vlm_attributes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # refine a stratified slice: the model only needs enough to learn the spread
    todo = (df[df.split != "zsl_test"]
            .groupby("canonical_label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, limit // df.canonical_label.nunique())),
                                      random_state=0)))
    todo = todo[~todo.phash.isin(cache.keys())]
    print(f"{len(todo):,} images to label, {len(cache):,} already cached")
    if len(todo) == 0:
        return apply_cache(df, cache)

    model, proc = load_vlm()
    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo))):
        try:
            im = PILImage.open(r.image_path).convert("RGB")
            im.thumbnail((IMG_SIZE * 2, IMG_SIZE * 2))
            organ = ask(model, proc, im, QUESTIONS["organ"])
            sev = ask(model, proc, im, QUESTIONS["severity"])
            cache[r.phash] = {
                "organ": next((o for o in ORGANS if o in organ), None),
                "severity": next((v for k, v in SEV_MAP.items() if k in sev), None),
            }
        except Exception as e:
            cache[r.phash] = {"organ": None, "severity": None, "error": str(e)}
        if i % 200 == 0:
            cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    return apply_cache(df, cache)


def apply_cache(df, cache):
    hit_o = hit_s = 0
    for idx, ph in df.phash.items():
        c = cache.get(ph)
        if not c:
            continue
        if c.get("organ"):
            df.at[idx, "organ"] = c["organ"]; hit_o += 1
        if c.get("severity") is not None and df.at[idx, "is_healthy"] == 0:
            df.at[idx, "severity"] = c["severity"]; hit_s += 1
            df.at[idx, "coverage"] = round(c["severity"] / 4 * 0.8 + 0.05, 3)
        df.at[idx, "refined"] = 1
    print(f"refined organ on {hit_o:,} rows, severity on {hit_s:,} rows")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--limit", type=int, default=6000)
    a = ap.parse_args()

    df = pd.read_csv(MANIFESTS / "manifest.csv")
    df = build_priors(df)
    if a.vlm:
        df = refine(df, a.limit)

    out = MANIFESTS / "manifest_attr.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df):,} rows)")
    for h in CAT_HEADS:
        print(f"\n[{h}]\n{df[h].value_counts().to_string()}")
    print(f"\n[severity]\n{df.severity.value_counts().sort_index().to_string()}")
