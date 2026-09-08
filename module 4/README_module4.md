# FLORA :: Module 4 — Treatment Retrieval-Augmented Generation

Modules 1-3 answer "what disease is this?". Module 4 answers the question a
farmer actually asks next: "what do I do about it?" It is retrieval-augmented
by design, not a fine-tuned model, for the same reason Module 2's text tower
is frozen: a small model trained end-to-end on 58 short disease entries would
just memorise them, and a farmer with a real, slightly different question
deserves an answer grounded in written knowledge, not a hallucinated dosage.

---

## The pipeline

```
diagnosis from Modules 1-3          farmer's optional follow-up question
   (canonical_label)                    ("koi organic upay hai?")
          │                                        │
          └───────────────┬────────────────────────┘
                           ▼
              retrieve top-k passages from
              treatment_monographs.yaml
       (MiniLM embeddings, cosine similarity,
        filtered to the diagnosed class, re-ranked
        by the follow-up question when one is given)
                           ▼
              build a grounded prompt: diagnosis +
              retrieved passages + the question
                           ▼
              local instruction model generates
              the answer (Qwen2.5-1.5B-Instruct,
              frozen, no fine-tuning)
                           ▼
                answer + the passages AND cited
                sources it was grounded in
```

Retrieval and generation are both frozen, off-the-shelf components. The
corpus is the single source of truth: the model is instructed not to invent
a chemical dose beyond what's written, and the API always returns the
grounding passages *and* their original sources alongside the answer, so a
wrong or unsupported claim is checkable rather than hidden inside fluent
prose.

---

## The knowledge base is sourced, not written from general knowledge

All 45 disease entries in `treatment_monographs.yaml` are paraphrased from
real university-extension and agricultural-research publications, fetched
and read via `WebSearch`/`WebFetch` — Clemson HGIC's rose and tomato disease
factsheets, UC IPM's powdery-mildew, spider-mite and rose-pest pages,
Cornell's cucurbit powdery mildew guide, the Crop Protection Network's
northern corn leaf blight overview, Rutgers' copper-phytotoxicity guidance
for peach bacterial spot, Iowa State/Kansas State on cedar-apple rust, and
similar. Each disease entry carries a `sources` list (title + URL) naming
exactly what it's grounded in. Where a source flagged a real caveat, that
caveat was kept rather than smoothed over — e.g. widespread **copper
resistance** in *Xanthomonas* (pepper/tomato/peach bacterial spot), spreading
**DMI-fungicide resistance** in cherry powdery mildew, and grape esca's
"**cannot be cured** once the wood is infected." `flora_rag.py` carries each
passage's `sources` through retrieval into the API response and the webapp
UI. The 13 healthy-state entries make no specific claims and carry no
sources.

This is still general integrated-pest-management guidance aggregated from
public extension sources, not a substitute for a local agricultural
extension office — chemical guidance names active-ingredient classes (e.g.
"a triazole such as myclobutanil or tebuconazole"), never exact commercial
doses, since rate depends on the specific product formulation and region.

---

## Files

- `treatment_monographs.yaml` — the sourced knowledge base described above.
- `flora_rag.py` — `load_corpus`, `embed_passages`/`embed_query` (MiniLM,
  mean-pooled, frozen), `retrieve`, `build_prompt`, `load_generator`,
  `generate_answer`, and the end-to-end `answer()`.
- `13_build_treatment_corpus.py` — embeds the corpus once and caches it to
  `artifacts/cache/treatment_<encoder>.npz`, mirroring Module 2's
  `06_extract_embeddings.py` pattern.
- `14_rag_answer.py` — CLI for retrieving + generating directly, printing
  the answer and its cited sources:

  ```bash
  python "module 4/13_build_treatment_corpus.py"
  python "module 4/14_rag_answer.py" --disease rose_black_spot
  python "module 4/14_rag_answer.py" --disease rose_black_spot --query "koi organic upay hai?"
  python "module 4/14_rag_answer.py" --disease apple_cedar_apple_rust --gen_model qwen0.5b
  python "module 4/14_rag_answer.py" --disease tomato_yellow_leaf_curl_virus --confidence 0.4
  ```
- `eval_questions.yaml` / `15_eval_grounding.py` — the held-out grounding-
  accuracy eval described below.
- `module 3/webapp/cases_db.py` — treatment follow-up log (SQLite), described
  further down.

---

## Generation model — a measured finding, not just one anecdote

Local, CPU-friendly, no API key: **Qwen2.5-1.5B-Instruct** is the default
(~3GB, first run downloads it from Hugging Face and caches it).
`Qwen2.5-0.5B-Instruct` (~1GB, roughly 3x faster) is available via
`--gen_model qwen0.5b`.

**`15_eval_grounding.py`** runs 8 held-out trick questions (`eval_questions.yaml`)
through both models and mechanically checks each answer against
must-include / must-not-include phrases drawn from the source passages —
replacing what was originally a single hand-picked example:

```
python "module 4/15_eval_grounding.py" --gen_model qwen0.5b   # 2/8 passed
python "module 4/15_eval_grounding.py" --gen_model qwen1.5b   # 8/8 passed
```

0.5B's 6 failures include a genuinely dangerous one, not just awkward
phrasing: asked *"exactly how many ml of chlorothalonil per litre should I
use?"* against a corpus that deliberately contains no dosage numbers, it
answered *"you should apply 10ml of chlorothalonil per litre of water"* —
inventing a specific number from nothing. It also flipped grape esca (no
cure) into *"yes, you can use chemical sprays... to treat"* it, and
pepper bacterial spot's documented copper-resistance caveat into *"yes,
copper spray will definitely fix this."* 1.5B passed all 8, including the
original cedar-apple-rust case (source: spores travel up to two miles,
making removal "often impractical"; 0.5B said "yes, just remove them and
hand-pick," 1.5B correctly said removal alone won't work and walked through
the real gall-pruning/resistant-variety/spray-timing advice instead).

That gap is why 1.5B is the default everywhere (CLI and webapp) even at
~3x the latency. 0.5B remains available for faster, lower-stakes queries,
but 2/8 on a held-out grounding set is a real reliability finding, not a
one-off worth hand-waving past — full per-case results (including the raw
generated text) are in `artifacts/results/m4_grounding_eval_qwen{0.5b,1.5b}.json`.

---

## Webapp integration

The Module 3 webapp (`module 3/webapp/`) loads and embeds this corpus at
startup and exposes it as `POST /api/treatment`:

```json
{"disease": "rose_black_spot", "query": "koi organic upay hai?"}
```

returns

```json
{"disease": "...", "common_name": "...", "answer": "...",
 "passages": [{"section": "...", "text": "...", "sources": [{"title": "...", "url": "..."}]}]}
```

After a diagnosis in the Symptom decoder tab, a **"Get treatment advice"**
button appears under the result, calls this endpoint for the top-predicted
class (using whatever text the farmer already typed as the follow-up
question, plus that prediction's own confidence score), and renders the
generated answer with two collapsible `<details>` blocks underneath: the
grounding passages themselves, and a **"Cited sources"** list of clickable
links to the original extension publications — so a disagreement between
the generated summary and the source text is checkable, not hidden. The
generator loads lazily on first click rather than at app startup, since not
every session needs it.

Verified live (`rose_black_spot`, confidence 0.4): the model correctly
opened with the differentiation note before treatment advice, though it
sometimes over-hedges the whole recommendation on "if confirmed as
anthracnose instead" rather than just flagging the caveat and continuing
with black-spot treatment — a prompt-tuning nuance worth another pass, not
a correctness bug (the differentiation content itself is accurate and
grounded).

### Look-alike-aware retrieval

When the diagnosis's own confidence is below 0.6, `answer()` also pulls
Module 2's `look_alikes` field (`disease_monographs.yaml`) for the
diagnosed class and adds a short symptom-differentiation note to the
prompt — "this can be confused with X, here's the clearest visual
difference" — before the treatment advice. `flora_rag.look_alike_notes()`
reads Module 2's file directly rather than duplicating it, so an edit there
reaches Module 4 automatically. The webapp passes the diagnosis's own
top-class probability through as `confidence` automatically; the CLI takes
`--confidence` for testing (see `14_rag_answer.py` example above). 45/58
classes have a populated `look_alikes` list; the rest (mostly healthy
states) get no differentiation note, same as before.

---

## What would make this stronger

- The held-out eval above is 8 hand-written cases — useful as a floor, not
  a ceiling. A larger, more adversarial set (and ideally a held-out set an
  agronomist reviews, not just this project) would be needed to trust
  either model's grounding beyond "clearly better than the other."
- The corpus is aggregated mostly from US extension sources — started, not
  finished, adding Indian-specific ones. `tomato_late_blight`,
  `potato_late_blight`, and `marigold_leaf_blight` (three of the classes
  most relevant to Indian farmers — potato/tomato late blight are
  major national problems, marigold is grown at huge scale for festival
  use) now also cite ICAR/KVK field trials with real Indian-context
  numbers: a 4-spray mancozeb/cymoxanil schedule timed to days-after-sowing
  validated in the north-eastern Himalayan region (7-11% disease severity,
  ~27% yield gain over control), ICAR-CPRI's INDO-BLIGHTCAST pan-India
  forecasting tool, and — for marigold specifically — garlic extract
  (Allium sativum) as a low-cost organic option validated against the
  Odisha-relevant Alternaria flower blight. The other 42 disease entries
  are still US/EU-sourced only; extending this properly to the full corpus,
  and having an agronomist cross-check regional product legality and
  resistance patterns generally, is real remaining work, not a one-session job.
- Multilingual output now works (see below), but the one identified gap
  (disease-name vocabulary) is worth someone occasionally re-checking as
  NLLB gets updated, and the follow-up log has no reminders/notifications
  yet — a farmer has to remember to come back and check the tab.

---

## Multilingual output — a second attempt, this time working

The first attempt asked the generator itself (Qwen2.5-1.5B-Instruct) to
write its answer natively in Hindi. That produced Devanagari script but
frequently incoherent content — invented non-words with no real connection
to the source passages. **The fix was to stop asking the generator to think
in Hindi at all.** Generation always stays English (that's the config
verified against `eval_questions.yaml`); a separate translation pass then
translates the finished, already-grounded English answer with a dedicated
MT model — `flora_rag.translate_text()`, using
**facebook/nllb-200-distilled-600M** (~2.5GB, downloads on first use).

Two smaller options were tried first and both failed on real
treatment-advice text before NLLB was tried:
- **Native generation** (Qwen2.5-1.5B asked to reply in Hindi directly):
  incoherent, invented non-words.
- **Helsinki-NLP/opus-mt-en-hi** (small MarianMT model): grammatically
  broken and mistranslated domain vocabulary — *"fungicide"* became
  "हँसी-मज़ाक की बात" ("a laughing matter") in one real test sentence, and
  *"copper-based"* became "पीतल-आधारित" ("brass-based").

NLLB-200-distilled-600M, verified on the same real sentences plus a full
multi-sentence generated answer, got both right ("कवकनाशक" for fungicide,
"तांबे आधारित" for copper-based) and stayed coherent and accurate across a
full paragraph, correctly preserving cultivar names like "Redfree" and
"Liberty". **One real gap remains**: verified live on `tomato_late_blight`,
"blight" itself sometimes mistranslates to "बुखार"/"विषाक्त" ("fever"/"toxic")
rather than the correct agricultural term — the rest of that same answer
(dosages, technical terms, instructions) translated correctly. Worth
knowing, not disqualifying.

```bash
python "module 4/14_rag_answer.py" --disease rose_black_spot --language hi
```

`answer(..., language="hi")` returns `answer` (translated), `answer_en`
(the original, always included so a mistranslation is checkable), and
`language`. The webapp exposes this as a language dropdown (English /
हिन्दी) next to the "Get treatment advice" button; the rendered answer
carries `lang="hi"` and the original English is available in a collapsible
"Original English answer" block. Look-alike notes get translated too when
present.

---

## Treatment follow-up tracking

Module 4 used to end the conversation the moment it gave advice — no record
of whether that advice actually worked. `module 3/webapp/cases_db.py` (pure
`sqlite3`, no new dependency) closes that loop: after getting treatment
advice, a **"Log this treatment"** button records the disease, species, the
advice given, and its language into `artifacts/webapp_cases.sqlite3`. A new
**"Treatment follow-up"** tab lists every logged case; each one without a
recorded outcome gets a one-click Better/Same/Worse form plus an optional
notes field. This is a farmer's own judgement being recorded, not a model
prediction — there's no ML here, just a durable log.

```
POST /api/cases                    {disease, common_name, species, treatment_answer, language} -> case
GET  /api/cases                    -> [case, ...]   (most recent first)
POST /api/cases/<id>/followup      {result: "better"|"same"|"worse", notes} -> case
```

7 tests in `tests/test_cases_db.py` cover creation, listing, limits, and
follow-up recording/validation against a temporary DB path (never the
webapp's real one).
