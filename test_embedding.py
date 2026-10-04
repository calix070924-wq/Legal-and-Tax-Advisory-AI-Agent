from bge_rag.embedding import embed_texts


texts = [
    "근로자는 임금을 지급받을 권리가 있습니다.",
    "사업소득에 대한 원천징수에 관한 내용입니다.",
    "부가가치세가 면제되는 용역에 관한 내용입니다."
]


embeddings = embed_texts(texts)


print("\nBGE-M3 임베딩 성공!")

print("문장 개수:", len(embeddings))
print("벡터 차원:", embeddings.shape[1])

print("\n전체 shape:", embeddings.shape)