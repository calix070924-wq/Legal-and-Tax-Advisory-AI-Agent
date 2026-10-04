from bge_rag.law_api import (
    search_law,
    get_law_detail,
    parse_articles
)


# 1. "소득세법" 검색
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


# 4. 실제 조문 단위로 변환
documents = parse_articles(detail)


# 5. 결과 확인
print("찾은 법령:", income_tax_law["법령명한글"])
print("MST:", mst)

print("\n파싱된 조문 수:", len(documents))


print("\n===== 앞의 조문 5개 =====")

for document in documents[:5]:

    print("\n조문번호:", document["article_number"])
    print("내용:", document["text"])