from bge_rag.law_api import (
    search_law,
    get_law_detail,
    parse_articles
)

from bge_rag.retriever import (
    build_embeddings,
    retrieve_top_k
)

from bge_rag.generator import generate_answer


# =========================================
# 1. 테스트 질문
# =========================================

question = "프리랜서가 돈을 받을 때 원천징수하는 세금에 관한 규정은?"

print("\n===== 사용자 질문 =====")
print(question)


# =========================================
# 2. 소득세법 검색
# =========================================

search_result = search_law("소득세법")

laws = search_result["LawSearch"]["law"]

income_tax_law = None

for law in laws:
    if law["법령명한글"] == "소득세법":
        income_tax_law = law
        break


if income_tax_law is None:
    raise ValueError("소득세법을 찾지 못했습니다.")


mst = income_tax_law["법령일련번호"]


# =========================================
# 3. 법령 본문 가져오기 + 조문 파싱
# =========================================

detail = get_law_detail(mst)

documents = parse_articles(detail)

print("\n소득세법 조문 수:", len(documents))


# =========================================
# 4. 저장된 BGE-M3 임베딩 불러오기
# =========================================

document_embeddings = build_embeddings(
    documents,
    cache_name=f"income_tax_{mst}"
)


# =========================================
# 5. BGE-M3 Top-5 검색
# =========================================

hits = retrieve_top_k(
    query=question,
    documents=documents,
    document_embeddings=document_embeddings,
    top_k=5
)


print("\n===== BGE-M3 검색 결과 =====")

for rank, hit in enumerate(hits, start=1):

    print(f"\n[{rank}위]")

    print("조문번호:", hit["article_number"])
    print("유사도:", round(hit["score"], 4))
    print("내용:", hit["text"])


# =========================================
# 6. Top-5 근거를 Solar Pro 4에 전달
# =========================================

print("\n===== Solar Pro 4 답변 생성 =====")

answer = generate_answer(
    question=question,
    hits=hits
)


# =========================================
# 7. 최종 답변 출력
# =========================================

print("\n===== 최종 답변 =====")
print(answer)