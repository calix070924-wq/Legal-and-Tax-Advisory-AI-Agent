"""CLI: python -m upstage_rag "질문" [--laws 소득세법 부가가치세법] [--show-context]"""

import argparse
import sys

from upstage_rag.answer import answer


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--laws", nargs="+", help="찾아볼 법령을 직접 지정 (생략하면 Solar가 고름)")
    ap.add_argument("--show-context", action="store_true", help="검색된 근거와 유사도 출력")
    args = ap.parse_args()

    r = answer(args.question, laws=args.laws)
    if args.show_context:
        print("찾아본 법령:", r["laws"])
        for i, h in enumerate(r["hits"], 1):
            print(f"  [{i}] {h['score']:.3f}  {h['source']}")
        print()
    print(r["answer"])
    if r["usage"]:
        u = r["usage"]
        print(f"\n(답변 토큰: 입력 {u['prompt_tokens']} / 출력 {u['completion_tokens']})")


if __name__ == "__main__":
    main()
