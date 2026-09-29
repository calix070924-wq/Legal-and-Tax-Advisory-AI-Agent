"""Chroma 벡터스토어 (로컬 persist, 코사인 거리)."""

from __future__ import annotations

from langchain_chroma import Chroma

from law_rag.config import CHROMA_DIR, COLLECTION
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

