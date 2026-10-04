from sentence_transformers import SentenceTransformer


# BGE-M3 모델 로드
model = SentenceTransformer("BAAI/bge-m3")


def embed_texts(texts):
    """
    여러 문장을 BGE-M3를 이용해 벡터로 변환합니다.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings