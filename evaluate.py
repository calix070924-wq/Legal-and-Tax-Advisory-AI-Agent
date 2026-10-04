from retriever import vectorstore


# --------------------------------------------------
# 1. 평가 데이터
# --------------------------------------------------
# question: 검색 질문
# answer_articles: 정답으로 인정할 소득세법 조문 번호
#
# 아래는 평가 코드가 정상 작동하는지 확인하기 위한 예시입니다.
# 최종 성능 비교에서는 팀 공통 평가셋으로 교체해야 합니다.

EVAL_DATA = [
    {
        "question": "사업을 해서 번 돈은 소득세법에서 어떤 소득으로 분류되나요?",
        "answer_articles": ["19"],
    },
    {
        "question": "회사에서 월급이나 상여금을 받은 경우 어떤 소득에 해당하나요?",
        "answer_articles": ["20"],
    },
    {
        "question": "사업을 하면서 사용한 비용을 사업소득 계산에서 어떻게 처리하나요?",
        "answer_articles": ["27"],
    },
    {
        "question": "종합소득 과세표준은 어떻게 계산하나요?",
        "answer_articles": ["14"],
    },
    {
        "question": "종합소득세 과세표준을 신고해야 하는 규정은 어디에 있나요?",
        "answer_articles": ["70"],
    },
    {
        "question": "소득세를 원천징수해야 하는 사람에 관한 규정은?",
        "answer_articles": ["127"],
    },
    {
        "question": "사업소득을 지급할 때 원천징수하는 방법은?",
        "answer_articles": ["144"],
    },
    {
        "question": "근로소득을 지급할 때 원천징수하는 방법은?",
        "answer_articles": ["134"],
    },
    {
        "question": "이자나 배당소득을 지급할 때 원천징수하는 방법은?",
        "answer_articles": ["130"],
    },
    {
        "question": "사업소득에 대한 원천징수영수증은 어떻게 발급하나요?",
        "answer_articles": ["144"],
    },
]


# --------------------------------------------------
# 2. 평가
# --------------------------------------------------

recall_at_1 = 0
recall_at_5 = 0
reciprocal_rank_sum = 0.0


for item in EVAL_DATA:

    question = item["question"]
    correct_articles = set(item["answer_articles"])

    # MRR@10까지 계산하기 위해 10개 검색
    results = vectorstore.similarity_search(
        question,
        k=10
    )

    retrieved_articles = [
        str(doc.metadata.get("article_num"))
        for doc in results
    ]

    print("\n===================================")
    print(f"질문: {question}")
    print(f"정답 조문: {sorted(correct_articles)}")
    print(f"검색 결과: {retrieved_articles}")

    # Recall@1
    if any(
        article in correct_articles
        for article in retrieved_articles[:1]
    ):
        recall_at_1 += 1

    # Recall@5
    if any(
        article in correct_articles
        for article in retrieved_articles[:5]
    ):
        recall_at_5 += 1

    # MRR@10
    for rank, article in enumerate(
        retrieved_articles[:10],
        start=1
    ):
        if article in correct_articles:
            reciprocal_rank_sum += 1 / rank
            break


# --------------------------------------------------
# 3. 최종 결과
# --------------------------------------------------

total = len(EVAL_DATA)

recall_at_1 /= total
recall_at_5 /= total
mrr_at_10 = reciprocal_rank_sum / total


print("\n\n========== KURE-v1 평가 결과 ==========")

print(f"평가 질문 수 : {total}")
print(f"Recall@1    : {recall_at_1:.3f}")
print(f"Recall@5    : {recall_at_5:.3f}")
print(f"MRR@10      : {mrr_at_10:.3f}")

print("=======================================")