"""Solar Pro 4 연결 (Upstage OpenAI 호환 API → langchain-openai ChatOpenAI).

    python -m law_rag.llm                 # 연결 테스트 1회
    python -m law_rag.llm --compare       # reasoning_effort 기본값 vs "low" 비교
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from langchain_openai import ChatOpenAI

from law_rag.config import LLM_MAX_TOKENS, LLM_TEMPERATURE, SOLAR_MODEL, UPSTAGE_BASE_URL


def _api_key() -> str:
    key = os.environ.get("UPSTAGE_API_KEY", "").strip()
    if not key:
        sys.exit(".env에 UPSTAGE_API_KEY를 설정하세요.")
    return key


def get_llm(**overrides) -> ChatOpenAI:
    kwargs = dict(
        model=SOLAR_MODEL,
        base_url=UPSTAGE_BASE_URL,
        api_key=_api_key(),
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=60,
        max_retries=2,
    )
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


def _probe(llm: ChatOpenAI, prompt: str) -> None:
    t0 = time.time()
    res = llm.invoke(prompt)
    dt = time.time() - t0
    meta = res.response_metadata or {}
    print(f"  모델: {meta.get('model_name')}")
    print(f"  응답: {res.content}")  # 추론 과정(reasoning)은 출력하지 않는다
    print(f"  시간: {dt:.2f}s")
    print(f"  토큰: {res.usage_metadata}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Solar Pro 4 연결 테스트")
    ap.add_argument("--compare", action="store_true", help='reasoning_effort 기본값 vs "low" 비교')
    args = ap.parse_args()

    prompt = "한 문장으로 자기소개해"
    print("[기본값]")
    _probe(get_llm(), prompt)
    if args.compare:
        print('\n[reasoning_effort="low"]')
        try:
            _probe(get_llm(reasoning_effort="low"), prompt)
        except Exception as e:  # 400이면 허용값이 아님 → 기본값 유지
            print(f"  실패: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
