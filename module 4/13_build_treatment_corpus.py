"""FLORA :: module 4 script 13 -- embed the treatment corpus and cache it.

Mirrors module 2's 06_extract_embeddings.py pattern: precompute once, every
later retrieval call (14_rag_answer.py, the webapp) just loads the cache.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flora_rag import load_corpus, embed_passages  # noqa: E402
from config import CACHE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="minilm", choices=["minilm", "mpnet"])
    args = ap.parse_args()

    passages = load_corpus()
    n_classes = len(set(p["label"] for p in passages))
    print(f"{len(passages)} treatment passages across {n_classes} classes")

    cache = CACHE / f"treatment_{args.encoder}.npz"
    emb = embed_passages(passages, encoder=args.encoder, cache=cache)
    print(f"embedded -> {cache} ({tuple(emb.shape)})")


if __name__ == "__main__":
    main()
