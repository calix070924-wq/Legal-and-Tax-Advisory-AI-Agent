import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. .env 파일 로드 (OPENAI_API_KEY 활성화)
load_dotenv()

# 2. 문서 로드 (법률/세무 테스트용 문서 예시)
# 실제 프로젝트에서는 가지고 계신 .txt, .pdf 문서 경로를 지정하시면 됩니다.
loader = TextLoader("sample_legal_doc.txt", encoding="utf-8")
documents = loader.load()

# 3. 문서 분할 (Text Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 4. OpenAI text-embedding-3-small 모델 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 5. Vector Store 생성 및 문서 저장
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"  # DB 저장 경로
)

# 6. Retriever 객체 생성 (유사도 기반 검색기)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 질문과 가장 유사한 문서 3개 검색
)

# 7. 검색 테스트
if __name__ == "__main__":
    query = "세금 감면 조건이 어떻게 되나요?"
    retrieved_docs = retriever.invoke(query)
    
    print(f"=== '{query}' 검색 결과 ===")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n[{i}] {doc.page_content}")