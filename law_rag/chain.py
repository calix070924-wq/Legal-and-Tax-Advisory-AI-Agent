"""답변 생성 체인: 질문 → 검색·조문 확장 → Solar Pro 4 → 인용 검증·출처·기준일·면책.

    python -m law_rag.chain "서비스 탈퇴한 회원 정보는 언제까지 지워야 해?"
    python -m law_rag.chain "..." --show-context     # LLM에 들어간 근거 전체 출력
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from law_rag.context import Evidence, format_evidence, retrieve_evidence
from law_rag.llm import get_llm
from law_rag.postprocess import postprocess
from law_rag.prompts import ANSWER_PROMPT


@dataclass
class AnswerResult:
    question: str
    answer: str  # 후처리 끝난 최종 텍스트
    raw_answer: str
    evidence: list[Evidence]
    cited: list[int]
    invalid_citations: list[int]
    latency_s: float
    usage: dict = field(default_factory=dict)  # input_tokens, output_tokens, total_tokens

    @property
    def context(self) -> str:
        return format_evidence(self.evidence)


def _finish(x: dict) -> dict:
    p = postprocess(x["raw_answer"], x["evidence"])
    return {**x, "answer": p.answer, "cited": p.cited, "invalid_citations": p.invalid_citations}


@lru_cache(maxsize=1)
def build_chain():
    generate = ANSWER_PROMPT | get_llm() | StrOutputParser()
    return (
        RunnablePassthrough.assign(evidence=lambda x: retrieve_evidence(x["question"]))
        | RunnablePassthrough.assign(context=lambda x: format_evidence(x["evidence"]))
        | RunnablePassthrough.assign(raw_answer=generate)
        | RunnableLambda(_finish)
    )


def answer(question: str) -> AnswerResult:
    t0 = time.time()
    with get_usage_metadata_callback() as cb:
        out = build_chain().invoke({"question": question})
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for u in cb.usage_metadata.values():
        for k in usage:
            usage[k] += u.get(k, 0)
    return AnswerResult(
        question=question,
        answer=out["answer"],
        raw_answer=out["raw_answer"],
        evidence=out["evidence"],
        cited=out["cited"],
        invalid_citations=out["invalid_citations"],
        latency_s=round(time.time() - t0, 2),
        usage=usage,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="법령 근거 기반 답변")
    ap.add_argument("question")
    ap.add_argument("--show-context", action="store_true", help="LLM에 들어간 근거 전체 출력")
    args = ap.parse_args()

    r = answer(args.question)
    if args.show_context:
        print("=" * 20, "근거", "=" * 20)
        print(r.context)
        print("=" * 46, "\n")
    print(r.answer)
    u = r.usage
    print(f"\n({r.latency_s:.1f}s | 입력 {u['input_tokens']} / 출력 {u['output_tokens']} 토큰)")
    if r.invalid_citations:
        print(f"(제거된 잘못된 인용: {r.invalid_citations})")


if __name__ == "__main__":
    main()
