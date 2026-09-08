"""
FLORA :: module 4 core -- retrieval-augmented treatment/remedy answers.

Modules 1-3 name a disease. Module 4 answers "what do I do about it?" by
retrieving grounded passages from treatment_monographs.yaml and having a
small local instruction model phrase them as an answer -- optionally
targeted at a farmer's free-text follow-up question. Retrieval and
generation are both frozen, off-the-shelf components (like Module 2's text
tower): nothing here is fine-tuned, so the corpus stays the single source of
truth and the model has no way to invent a dosage or product that isn't in
it.
"""
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).parent

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def wants_hindi_reply(query):
    """
    Devanagari script, or Hinglish (Latin-script Hindi/Marathi) -- Module 3's
    farmer queries are predominantly one of these two, per its own corpus
    stats (README: 88% mixed-script, 12% Hinglish, <1% pure Devanagari).
    Detecting Hinglish precisely needs a real classifier; this is a cheap
    heuristic good enough to pre-select the language dropdown, not to decide
    translation quality.

    History: an earlier approach asked the generator (Qwen2.5-1.5B-Instruct)
    to write its answer natively in Hindi. That produced Devanagari script
    but frequently incoherent content (invented non-words, no real connection
    to the source passages) -- see git history / README for the specific
    failure case. The actual solution (see `translate_text` below) is a
    separate translation pass: generate in English as normal, then translate
    the finished, already-grounded answer with a dedicated MT model. That
    keeps the property that matters (the English generation is what's
    verified against `eval_questions.yaml`) while still producing Hindi
    output.
    """
    if not query:
        return False
    if _DEVANAGARI.search(query):
        return True
    hinglish_markers = (" hai", " hain", " nahi", " kya", " kaise", " karo",
                        " patti", " paudha", " dikh", " upay", " ilaj",
                        " sanga", " zyada", " thoda")
    q = f" {query.lower()} "
    return any(m in q for m in hinglish_markers)


# ------------------------------------------------------------- translation
TRANSLATION_MODELS = {
    "hi": "facebook/nllb-200-distilled-600M",
}
NLLB_LANG_CODES = {"hi": "hin_Deva"}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_translator_cache = {}


def load_translator(language="hi"):
    if language in _translator_cache:
        return _translator_cache[language]
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    mid = TRANSLATION_MODELS[language]
    tok = AutoTokenizer.from_pretrained(mid, src_lang="eng_Latn")
    mdl = AutoModelForSeq2SeqLM.from_pretrained(mid).eval()
    _translator_cache[language] = (tok, mdl)
    return tok, mdl


@torch.no_grad()
def translate_text(text, language="hi", device="cpu"):
    """
    English -> `language` via NLLB-200-distilled-600M, sentence by sentence
    (keeps quality and latency predictable on multi-sentence answers; NLLB's
    quality holds up well on real treatment-advice text -- verified live,
    correctly handling technical vocabulary like "fungicide" and "copper-
    based" that broke both an earlier native-Hindi-generation attempt and a
    smaller MarianMT (opus-mt-en-hi) translation attempt).
    """
    if language == "en" or not text or not text.strip():
        return text
    if language not in TRANSLATION_MODELS:
        raise ValueError(f"no translation model configured for {language!r}")
    tok, mdl = load_translator(language)
    mdl = mdl.to(device)
    lang_id = tok.convert_tokens_to_ids(NLLB_LANG_CODES[language])

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return text
    batch = tok(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    out = mdl.generate(**batch, forced_bos_token_id=lang_id, max_length=400)
    translated = tok.batch_decode(out, skip_special_tokens=True)
    return " ".join(translated)

TEXT_ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

GEN_MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

SECTIONS = ["cultural", "organic", "chemical", "prevention", "severity_note"]

_encoder_cache = {}
_gen_cache = {}


# ------------------------------------------------------------- look-alikes
_symptom_monographs = None


def load_symptom_monographs():
    """Module 2's disease_monographs.yaml -- the `look_alikes` field on each
    entry names the classes this one is most often confused with. Reused
    here (not duplicated) so a look-alike edit in Module 2 automatically
    reaches Module 4."""
    global _symptom_monographs
    if _symptom_monographs is None:
        path = HERE.parent / "module 2" / "disease_monographs.yaml"
        _symptom_monographs = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _symptom_monographs


def look_alike_notes(disease, max_alikes=2):
    """
    Short symptom-differentiation notes for the diagnosed disease's most
    common look-alikes, sourced from Module 2's monographs (not this
    module's treatment corpus, since telling two diseases apart is a
    symptom question, not a treatment one). Returns [] if the disease has
    no look_alikes entry -- most healthy states and a few classes don't.
    """
    raw = load_symptom_monographs()
    entry = raw.get(disease)
    if not entry or not entry.get("look_alikes"):
        return []
    own_desc = " ".join(entry["variants"][0].split())
    notes = []
    for alike in entry["look_alikes"][:max_alikes]:
        alike_entry = raw.get(alike)
        if not alike_entry:
            continue
        notes.append({
            "look_alike": alike,
            "look_alike_common_name": alike_entry.get("common_name", alike),
            "this_disease_symptoms": own_desc,
            "look_alike_symptoms": " ".join(alike_entry["variants"][0].split()),
        })
    return notes


# ------------------------------------------------------------- corpus
def load_corpus():
    raw = yaml.safe_load((HERE / "treatment_monographs.yaml").read_text(encoding="utf-8"))
    passages = []
    for label, entry in raw.items():
        common = entry.get("common_name", label)
        sources = entry.get("sources", [])
        for sec in SECTIONS:
            text = entry.get(sec)
            if not text:
                continue
            passages.append({
                "label": label,
                "common_name": common,
                "section": sec,
                "text": " ".join(text.split()),
                "sources": sources,
            })
    return passages


# ------------------------------------------------------------- retrieval
def _load_encoder(encoder, device):
    if encoder in _encoder_cache:
        return _encoder_cache[encoder]
    from transformers import AutoTokenizer, AutoModel
    mid = TEXT_ENCODERS.get(encoder, encoder)
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModel.from_pretrained(mid).to(device).eval()
    _encoder_cache[encoder] = (tok, mdl)
    return tok, mdl


@torch.no_grad()
def _mean_pool(texts, encoder, device):
    tok, mdl = _load_encoder(encoder, device)
    embs = []
    bs = 16
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        b = tok(batch, padding=True, truncation=True, max_length=256,
                return_tensors="pt").to(device)
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        embs.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(embs, dim=0)


def embed_passages(passages, encoder="minilm", device="cpu", cache: Path = None):
    texts = [p["text"] for p in passages]
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["texts"]) == texts:
            return torch.from_numpy(z["emb"])
    emb = _mean_pool(texts, encoder, device)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, emb=emb.numpy(), texts=np.array(texts, dtype=object))
    return emb


def embed_query(query, encoder="minilm", device="cpu"):
    return _mean_pool([query], encoder, device)[0]


def retrieve(passages, emb, disease=None, query=None, k=4, encoder="minilm"):
    """
    disease: canonical_label to restrict retrieval to. The typical case --
    Modules 1-3 already named the disease, so retrieval only has to fetch
    ITS treatment passages, not search the whole corpus.
    query: optional free-text farmer follow-up (e.g. "koi organic upay hai?")
    used to rank passages by relevance when given; falls back to monograph
    order (cultural, organic, chemical, prevention) otherwise.
    """
    idx = list(range(len(passages)))
    if disease is not None:
        idx = [i for i in idx if passages[i]["label"] == disease]
        if not idx:
            raise KeyError(f"no treatment entry for {disease!r}")

    if query:
        qv = embed_query(query, encoder=encoder)
        sims = emb[idx] @ qv
        order = torch.argsort(sims, descending=True).tolist()
        idx = [idx[i] for i in order]

    return [passages[i] for i in idx[:k]]


# ------------------------------------------------------------- generation
def build_prompt(disease_common_name, passages, query=None, look_alikes=None):
    """Always builds an English prompt -- generation stays English (that's
    the config verified against eval_questions.yaml); see translate_text()
    for how a Hindi *output* is produced from this same English answer."""
    ctx = "\n".join(f"- ({p['section']}) {p['text']}" for p in passages)
    if query:
        ask = (f'The farmer also asked: "{query}"\n'
               f"Answer that question specifically, using only the passages above.")
    else:
        ask = "Summarise practical treatment advice from the passages above."

    confusable = ""
    lookalike_instruction = ""
    if look_alikes:
        rows = "\n".join(
            f"  - {n['look_alike_common_name']}: {n['look_alike_symptoms']}"
            for n in look_alikes)
        confusable = (
            f"\n\nThe diagnosis is not fully certain. This disease's symptoms "
            f"({look_alikes[0]['this_disease_symptoms']}) can be confused with:\n{rows}")
        lookalike_instruction = (
            f" Your answer MUST start with one sentence naming the most likely "
            f"look-alike ({look_alikes[0]['look_alike_common_name']}) and the single "
            f"clearest visual difference the farmer can check, before any treatment advice.")

    return (
        f"You are an agricultural assistant. A plant has been diagnosed with: "
        f"{disease_common_name}.\n\n"
        f"Reference passages (the only source of truth -- do not add advice "
        f"that isn't grounded in them):\n{ctx}"
        f"{confusable}\n\n"
        f"{ask} Be concise and practical. Do not invent specific chemical "
        f"doses beyond what is written above -- tell the farmer to follow the "
        f"product label for exact rates.{lookalike_instruction}"
    )


def load_generator(model="qwen1.5b", device="cpu"):
    if model in _gen_cache:
        return _gen_cache[model]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = GEN_MODELS.get(model, model)
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    _gen_cache[model] = (tok, mdl)
    return tok, mdl


@torch.no_grad()
def generate_answer(prompt, model="qwen1.5b", device="cpu", max_new_tokens=220):
    tok, mdl = load_generator(model, device)
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True).to(device)
    out = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


# ------------------------------------------------------------- end to end
CONFIDENCE_THRESHOLD = 0.6


def answer(disease, query=None, k=4, encoder="minilm", gen_model="qwen1.5b",
           passages=None, emb=None, device="cpu", confidence=None, language="en"):
    """Retrieve + generate. Pass cached `passages`/`emb` (see
    13_build_treatment_corpus.py) to avoid re-embedding the corpus per call.

    confidence: the diagnosis's own top-class probability from Modules 1-3
    (e.g. res.routes.fusion_learned[0].p in the webapp). When given and below
    CONFIDENCE_THRESHOLD, look-alike differentiation notes (Module 2's
    look_alikes field) are pulled in so the answer flags "this could also be
    X, here's how to tell them apart" instead of treating a shaky diagnosis
    as settled fact.

    language: "en" (default) or "hi". Generation always happens in English
    (see build_prompt's docstring for why); when language != "en", the
    finished English answer and any look-alike notes are run through
    translate_text() before being returned. `answer_en` always carries the
    untranslated English text too, so a UI can offer both.
    """
    if passages is None or emb is None:
        passages = load_corpus()
        emb = embed_passages(passages, encoder=encoder, device=device)
    top = retrieve(passages, emb, disease=disease, query=query, k=k, encoder=encoder)
    common_name = top[0]["common_name"] if top else disease

    alikes = look_alike_notes(disease) if (confidence is not None
                                           and confidence < CONFIDENCE_THRESHOLD) else []
    prompt = build_prompt(common_name, top, query=query, look_alikes=alikes)
    text = generate_answer(prompt, model=gen_model, device=device)

    answer_text = text
    alikes_out = alikes
    common_name_out = common_name
    if language != "en":
        answer_text = translate_text(text, language=language, device=device)
        common_name_out = translate_text(common_name, language=language, device=device)
        alikes_out = [
            dict(n, look_alike_common_name=translate_text(
                     n["look_alike_common_name"], language=language, device=device),
                 look_alike_symptoms=translate_text(
                     n["look_alike_symptoms"], language=language, device=device))
            for n in alikes
        ]

    return {"disease": disease, "common_name": common_name_out, "passages": top,
            "answer": answer_text, "answer_en": text, "language": language,
            "look_alikes": alikes_out}
