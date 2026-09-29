#!/usr/bin/env python3
"""Chroma 검색기 평가: eval_embed.py와 같은 방식(조문 단위, gold 중 하나라도 적중).

    python scripts/build_index.py      # 색인 먼저
    python eval/eval_retriever.py
    python eval/eval_retriever.py --misses

합격 기준 (FAISS 정확 검색 결과 대비): R@1 >= 0.81, R@5 >= 0.93, MRR@10 >= 0.87
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))
sys.path.insert(0, str(EVAL_DIR))

from eval_embed import KS, load_jsonl, score  # noqa: E402
from law_rag.vectorstore import search  # noqa: E402

EXPECTED = {"R@1": 0.844, "R@3": None, "R@5": 0.969, "R@10": None, "MRR@10": 0.902}
THRESHOLD = {"R@1": 0.81, "R@5": 0.93, "MRR@10": 0.87}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", type=Path, default=EVAL_DIR / "eval_set.jsonl")
    ap.add_argument("--misses", action="store_true", help="top-5에서 틀린 질문 출력")
    args = ap.parse_args()

    evalset = load_jsonl(args.evalset)
    k = max(KS)
    ranked = [[d.metadata["article_label"] for d, _ in search(e["q"], k=k)] for e in evalset]
    res = score(ranked, evalset)

    print(f"평가 질문 {len(evalset)}개 (Chroma, 조문 단위)\n")
    print(f"{'지표':<8} {'Chroma':>8} {'기대값':>8} {'기준':>8}  판정")
    passed = True
    for name, v in res.items():
        exp, th = EXPECTED.get(name), THRESHOLD.get(name)
        ok = th is None or v >= th
        passed &= ok
        print(
            f"{name:<8} {v:>8.3f} {exp if exp is not None else '-':>8} "
            f"{('>=' + str(th)) if th else '-':>8}  {'' if th is None else ('OK' if ok else 'FAIL')}"
        )
    print(f"\n{'합격' if passed else '불합격'}")

    if args.misses:
        print("\ntop-5 밖:")
        for e, r in zip(evalset, ranked):
            if not set(e["gold"]) & set(r[:5]):
                print(f"  Q: {e['q']}\n     정답 {e['gold']}  /  검색 {r[:5]}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
