import os

from dotenv import load_dotenv
from openai import OpenAI

from retriever import retriever


# --------------------------------------------------
# 환경변수
# --------------------------------------------------

load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

if not UPSTAGE_API_KEY:
    raise ValueError(
        ".env 파일에서 UPSTAGE_API_KEY를 찾을 수 없습니다."
    )


# --------------------------------------------------
# Solar Pro 4 Client
# --------------------------------------------------

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1"
)


# --------------------------------------------------
# KURE Retriever
# --------------------------------------------------

def retrieve_laws(query):

    documents = retriever.invoke(query)

    contexts = []

    for doc in documents:

        law_name = doc.metadata.get(
            "law_name",
            "소득세법"
        )

        article_num = doc.metadata.get(
            "article_num",
            ""
        )

        context = (
            f"[{law_name} 제{article_num}조]\n"
            f"{doc.page_content}"
        )

        contexts.append(context)

    return documents, contexts


# --------------------------------------------------
# Solar Pro 4 답변 생성
# --------------------------------------------------

def generate_answer(query):

    documents, contexts = retrieve_laws(query)

    context_text = "\n\n".join(contexts)

    system_prompt = """
당신은 대한민국 법률·세무 정보를 제공하는 AI 어시스턴트입니다.

반드시 제공된 법령 근거만 사용하여 답변하세요.

규칙:
1. 제공된 법령에 없는 내용을 임의로 만들지 마세요.
2. 답변의 근거가 되는 법령명과 조문 번호를 명시하세요.
3. 법령 근거만으로 답변하기 어려우면
   "제공된 법령 근거로는 확인 불가합니다."라고 답변하세요.
4. 사용자가 이해하기 쉬운 한국어로 설명하세요.
5. 확정적인 법률·세무 자문처럼 표현하지 마세요.
""".strip()

    user_prompt = f"""
[사용자 질문]

{query}


[검색된 법령 근거]

{context_text}


위 법령 근거를 바탕으로 사용자의 질문에 답변하세요.
""".strip()

    response = client.chat.completions.create(
        model="solar-pro4-260806",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return answer, documents


# --------------------------------------------------
# 실행 테스트
# --------------------------------------------------

if __name__ == "__main__":

    query = (
        "프리랜서가 돈을 받을 때 "
        "원천징수하는 세금에 관한 규정은?"
    )

    answer, documents = generate_answer(query)

    print("\n========== KURE 검색 결과 ==========")

    for i, doc in enumerate(documents, 1):

        law_name = doc.metadata.get(
            "law_name",
            "소득세법"
        )

        article_num = doc.metadata.get(
            "article_num",
            ""
        )

        print(
            f"{i}. {law_name} 제{article_num}조"
        )

    print("\n========== Solar Pro 4 답변 ==========")

    print(answer)

    print("\n======================================")