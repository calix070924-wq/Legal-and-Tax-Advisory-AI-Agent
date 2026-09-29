"""arctic-ko 임베딩 (LangChain HuggingFaceEmbeddings).

질문에만 "query: " 접두어가 붙어야 한다 (prompt_name="query").
접두어가 빠져도 에러는 나지 않고 성능만 떨어지므로 check_query_prompt()로 확인한다.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from law_rag.config import EMBED_MODEL, MAX_SEQ_LEN


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": pick_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},  # 문서용
        query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},  # 질문용
    )
    emb._client.max_seq_length = MAX_SEQ_LEN
    return emb


def check_query_prompt(emb: HuggingFaceEmbeddings | None = None) -> float:
    """같은 문장의 질문/문서 임베딩 코사인. 1.0이면 접두어가 적용되지 않은 것."""
    import numpy as np

    emb = emb or get_embeddings()
    q = np.array(emb.embed_query("테스트"))
    d = np.array(emb.embed_documents(["테스트"])[0])
    return float(q @ d)  # 둘 다 정규화됨


if __name__ == "__main__":
    e = get_embeddings()
    print("device:", e._client.device, "| max_seq_length:", e._client.max_seq_length)
    print("prompts:", e._client.prompts)
    print(f"cos(embed_query, embed_documents) = {check_query_prompt(e):.4f}  (1보다 작아야 정상)")
