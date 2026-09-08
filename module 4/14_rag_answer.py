"""
FLORA :: module 4 script 14 -- retrieve + generate a treatment answer.

Usage:
  python "module 4/14_rag_answer.py" --disease rose_black_spot
  python "module 4/14_rag_answer.py" --disease rose_black_spot --query "koi organic upay hai?"
  python "module 4/14_rag_answer.py" --disease tomato_late_blight --gen_model qwen0.5b
  python "module 4/14_rag_answer.py" --disease rose_black_spot --language hi
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flora_rag import load_corpus, embed_passages, answer  # noqa: E402
from config import CACHE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True,
                     help="canonical_label, e.g. rose_black_spot (see class_priors.csv)")
    ap.add_argument("--query", default=None, help="optional farmer follow-up question")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--encoder", default="minilm", choices=["minilm", "mpnet"])
    ap.add_argument("--gen_model", default="qwen1.5b", choices=["qwen0.5b", "qwen1.5b"])
    ap.add_argument("--confidence", type=float, default=None,
                     help="diagnosis confidence (0-1); below 0.6 pulls in look-alike notes")
    ap.add_argument("--language", default="en", choices=["en", "hi"],
                     help="output language; 'hi' translates the finished English answer via NLLB")
    args = ap.parse_args()

    passages = load_corpus()
    cache = CACHE / f"treatment_{args.encoder}.npz"
    emb = embed_passages(passages, encoder=args.encoder, cache=cache)

    result = answer(args.disease, query=args.query, k=args.k,
                     encoder=args.encoder, gen_model=args.gen_model,
                     passages=passages, emb=emb, confidence=args.confidence,
                     language=args.language)

    print(f"\n=== {result['common_name']} ===")
    print("\nRetrieved passages:")
    for p in result["passages"]:
        print(f"  [{p['section']}] {p['text'][:100]}...")
    if result["look_alikes"]:
        print("\nLook-alike notes (confidence was below threshold):")
        for n in result["look_alikes"]:
            print(f"  vs {n['look_alike_common_name']}: {n['look_alike_symptoms'][:100]}...")
    print("\nGenerated answer:\n" + result["answer"])
    sources = {s["url"]: s["title"] for p in result["passages"] for s in p.get("sources", [])}
    if sources:
        print("\nSources:")
        for url, title in sources.items():
            print(f"  - {title}\n    {url}")


if __name__ == "__main__":
    main()
