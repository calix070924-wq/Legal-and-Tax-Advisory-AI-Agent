"""Upstage API 클라이언트 (OpenAI 호환) + 사용량 기록.

같은 키로 Solar Pro 4(채팅)와 Embedding(임베딩)을 모두 쓴다.
모든 호출의 토큰 수를 data/usage.jsonl 에 남겨서 "질문 1건당 비용"을 계산할 수 있게 한다.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np
from openai import OpenAI

from upstage_rag import config


@lru_cache(maxsize=1)
def client() -> OpenAI:
    if not config.UPSTAGE_API_KEY:
        raise RuntimeError(".env에 UPSTAGE_API_KEY가 없습니다")
    return OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL)


def log_usage(kind: str, model: str, usage) -> dict:
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    config.USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def chat(messages: list[dict], **kwargs) -> tuple[str, dict]:
    """Solar Pro 4 호출 → (답변 텍스트, 사용량)."""
    res = client().chat.completions.create(
        model=config.SOLAR_MODEL, messages=messages, temperature=config.TEMPERATURE, **kwargs
    )
    return res.choices[0].message.content or "", log_usage("chat", config.SOLAR_MODEL, res.usage)


def embed(texts: list[str], model: str) -> np.ndarray:
    """텍스트 → 정규화된 벡터 (n, dim). 정규화했으니 내적 = 코사인 유사도."""
    vecs = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = [t[: config.EMBED_MAX_CHARS] for t in texts[i : i + config.EMBED_BATCH]]
        res = client().embeddings.create(model=model, input=batch)
        vecs += [d.embedding for d in sorted(res.data, key=lambda d: d.index)]
        log_usage("embed", model, res.usage)
    v = np.array(vecs, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_query(text: str) -> np.ndarray:
    return embed([text], config.EMBED_QUERY_MODEL)[0]


def embed_passages(texts: list[str]) -> np.ndarray:
    return embed(texts, config.EMBED_PASSAGE_MODEL)
