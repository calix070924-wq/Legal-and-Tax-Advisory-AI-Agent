"""Chroma 벡터스토어 (로컬 persist, 코사인 거리) + 조문 단위 검색.

    python -m law_rag.vectorstore "탈퇴한 회원 정보 언제까지 지워야 해?"
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from law_rag.config import ARTICLES_PATH, CHROMA_DIR, COLLECTION
from law_rag.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    # Chroma 기본 거리는 L2. 컬렉션을 처음 만들 때만 적용되고, 이미 있는 컬렉션은 바뀌지 않는다.
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def distance_space(vs: Chroma) -> str:
    """컬렉션에 실제 적용된 거리 함수."""
    cfg = vs._collection.configuration or {}
    hnsw = cfg.get("hnsw") or {}
    return hnsw.get("space") or (vs._collection.metadata or {}).get("hnsw:space") or "l2"


@lru_cache(maxsize=1)
def _vs() -> Chroma:
    return get_vectorstore()


def get_retriever(k: int = 5, filter: dict | None = None):
    return _vs().as_retriever(search_kwargs={"k": k, "filter": filter})


def search(query: str, k: int = 5, filter: dict | None = None) -> list[tuple[Document, float]]:
    """조문 단위로 중복 제거한 top-k. 점수는 코사인 거리(작을수록 가까움)."""
    hits = _vs().similarity_search_with_score(query, k=k * 3, filter=filter)
    seen, out = set(), []
    for doc, dist in hits:
        uid = doc.metadata["article_uid"]
        if uid in seen:
            continue
        seen.add(uid)
        out.append((doc, dist))
        if len(out) == k:
            break
    return out


@lru_cache(maxsize=1)
def _articles() -> dict[str, dict]:
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        return {a["id"]: a for a in map(json.loads, f) if a}


def get_article(article_uid: str) -> dict:
    """분할 청크가 검색됐을 때 LLM에 넘길 조문 전체."""
    return _articles()[article_uid]


def main() -> None:
    ap = argparse.ArgumentParser(description="법령 조문 검색")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    for doc, dist in search(args.query, k=args.k):
        m = doc.metadata
        print(f"{m['article_label']} | {m['article_title']} | {dist:.4f} | {m['url']}")


if __name__ == "__main__":
    main()
