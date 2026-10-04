from bge_rag.law_api import (
    search_law,
    get_law_detail,
    parse_articles
)

from bge_rag.retriever import (
    build_embeddings,
    retrieve_top_k
)


# 1. 소득세법 검색
search_result = search_law("소득세법")

laws = search_result["LawSearch"]["law"]


# 2. 정확히 "소득세법" 찾기
income_tax_law = None

for law in laws:
    if law["법령명한글"] == "소득세법":
        income_tax_law = law
        break


if income_tax_law is None:
    raise ValueError("소득세법을 찾지 못했습니다.")


# 3. 법령 본문 가져오기
mst = income_tax_law["법령일련번호"]

detail = get_law_detail(mst)


# 4. 조문 파싱
documents = parse_articles(detail)

print("소득세법 조문 수:", len(documents))


# 5. 법령 임베딩 생성 또는 기존 파일 불러오기
document_embeddings = build_embeddings(
    documents,
    cache_name=f"income_tax_{mst}"
)


# 6. 테스트 질문
query = "프리랜서가 돈을 받을 때 원천징수하는 세금에 관한 규정은?"

print("\n질문:")
print(query)


# 7. 관련 조문 Top-5 검색
results = retrieve_top_k(
    query=query,
    documents=documents,
    document_embeddings=document_embeddings,
    top_k=5
)


# 8. 결과 출력
print("\n===== BGE-M3 검색 결과 Top-5 =====")

for rank, result in enumerate(results, start=1):

    print(f"\n[{rank}위]")
    print("조문번호:", result["article_number"])
    print("유사도:", round(result["score"], 4))
    print("내용:", result["text"])