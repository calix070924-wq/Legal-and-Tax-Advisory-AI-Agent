import os
import numpy as np

from bge_rag.embedding import embed_texts


CACHE_DIR = "embedding_cache"

# 캐시 폴더가 없으면 자동 생성
os.makedirs(CACHE_DIR, exist_ok=True)


def build_embeddings(documents, cache_name="law_embeddings"):
    """
    법령 조문을 임베딩하고 파일로 저장합니다.
    이미 저장된 임베딩이 있으면 다시 계산하지 않습니다.
    """

    cache_path = os.path.join(
        CACHE_DIR,
        f"{cache_name}.npy"
    )

    # 이미 임베딩 파일이 있으면 불러오기
    if os.path.exists(cache_path):
        print("저장된 법령 임베딩을 불러옵니다.")
        return np.load(cache_path)

    print("저장된 임베딩이 없습니다.")
    print("법령 조문을 처음부터 임베딩합니다.")

    document_texts = [
        document["text"]
        for document in documents
    ]

    embeddings = embed_texts(document_texts)

    # 임베딩 저장
    np.save(cache_path, embeddings)

    print("법령 임베딩 저장 완료!")

    return embeddings


def retrieve_top_k(
    query,
    documents,
    document_embeddings,
    top_k=5
):
    """
    사용자 질문과 가장 유사한 법령 조문 Top-K를 반환합니다.
    """

    if not documents:
        return []

    # 질문만 새로 임베딩
    query_embedding = embed_texts([query])[0]

    # cosine similarity
    # normalize_embeddings=True이므로 dot product 사용 가능
    similarities = np.dot(
        document_embeddings,
        query_embedding
    )

    # 유사도 높은 순으로 정렬
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        document = documents[index].copy()

        document["score"] = float(similarities[index])

        results.append(document)

    return results