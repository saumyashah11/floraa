"""
FLORA :: module 4 script 15 -- mechanical grounding-accuracy eval.

Runs every case in eval_questions.yaml through the RAG pipeline for a given
--gen_model, and scores each answer against must_include/must_not_include
phrase lists (case-insensitive substring match) -- a blunt but repeatable
check, replacing the single hand-picked example the README used to justify
picking qwen1.5b over qwen0.5b.

    python "module 4/15_eval_grounding.py" --gen_model qwen0.5b
    python "module 4/15_eval_grounding.py" --gen_model qwen1.5b
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flora_rag import load_corpus, embed_passages, answer  # noqa: E402
from config import CACHE, RESULTS  # noqa: E402

HERE = Path(__file__).parent


def score_case(case, generated_answer):
    text = generated_answer.lower()
    hits = [p for p in case.get("must_include", []) if p.lower() in text]
    violations = [p for p in case.get("must_not_include", []) if p.lower() in text]
    has_include_check = bool(case.get("must_include"))
    include_ok = (not has_include_check) or bool(hits)
    no_violation = not violations
    passed = include_ok and no_violation
    return {
        "id": case["id"],
        "passed": passed,
        "include_hits": hits,
        "include_expected": case.get("must_include", []),
        "violations": violations,
        "answer": generated_answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="qwen1.5b", choices=["qwen0.5b", "qwen1.5b"])
    ap.add_argument("--encoder", default="minilm", choices=["minilm", "mpnet"])
    args = ap.parse_args()

    cases = yaml.safe_load((HERE / "eval_questions.yaml").read_text(encoding="utf-8"))
    passages = load_corpus()
    cache = CACHE / f"treatment_{args.encoder}.npz"
    emb = embed_passages(passages, encoder=args.encoder, cache=cache)

    results = []
    for case in cases:
        r = answer(case["disease"], query=case["query"], k=4, encoder=args.encoder,
                   gen_model=args.gen_model, passages=passages, emb=emb)
        scored = score_case(case, r["answer"])
        results.append(scored)
        status = "PASS" if scored["passed"] else "FAIL"
        print(f"[{status}] {case['id']}")
        if not scored["passed"]:
            print(f"    expected any of: {scored['include_expected']}")
            print(f"    violations found: {scored['violations']}")
            print(f"    answer: {r['answer'][:200]}...")

    n_pass = sum(r["passed"] for r in results)
    print(f"\n{n_pass}/{len(results)} passed ({args.gen_model})")

    out = RESULTS / f"m4_grounding_eval_{args.gen_model}.json"
    out.write_text(json.dumps({
        "gen_model": args.gen_model, "n_pass": n_pass, "n_total": len(results),
        "cases": results,
    }, indent=2))
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
