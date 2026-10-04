import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from law_loader import fetch_law_data


# --------------------------------------------------
# 설정
# --------------------------------------------------

DB_PATH = "./chroma_db_kure_income_tax"


# --------------------------------------------------
# 1. KURE-v1 임베딩 모델
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="nlpai-lab/KURE-v1",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# --------------------------------------------------
# 2. 기존 Chroma DB 확인
# --------------------------------------------------

if os.path.exists(DB_PATH) and os.path.exists(
    os.path.join(DB_PATH, "chroma.sqlite3")
):

    print("저장된 KURE 소득세법 DB를 불러옵니다.")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

else:

    print("저장된 DB가 없습니다.")
    print("소득세법 조문을 처음부터 임베딩합니다.")

    # 소득세법 데이터 가져오기
    documents = fetch_law_data("소득세법")

    if not documents:
        raise RuntimeError(
            "법령 데이터를 가져오지 못했습니다. "
            "법령명이나 API 응답을 확인하세요."
        )

    print(f"소득세법 조문 수: {len(documents)}")

    # 조문 전체 단위로 임베딩 후 DB 생성
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("KURE 소득세법 DB 저장 완료!")


# --------------------------------------------------
# 3. Retriever 생성
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# --------------------------------------------------
# 4. 검색 테스트
# --------------------------------------------------

if __name__ == "__main__":

    query = "프리랜서가 돈을 받을 때 원천징수하는 세금에 관한 규정은?"

    retrieved_docs = retriever.invoke(query)

    print(
        f"\n=== '{query}' 실제 법령 검색 결과 ==="
    )

    for i, doc in enumerate(
        retrieved_docs,
        1
    ):

        print(
            f"\n[{i}] 출처: "
            f"{doc.metadata.get('law_name')} "
            f"(제{doc.metadata.get('article_num')}조)"
        )

        print(doc.page_content[:500])