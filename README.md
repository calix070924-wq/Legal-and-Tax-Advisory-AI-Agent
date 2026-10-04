# Legal-and-Tax-Advisory-AI-Agent — `bge_rag` (박다연)

프리랜서·1인 사업자를 위한 법률·세무 상담 RAG. 질문이 들어오면 법제처 Open API로 필요한 법령을 가져오고, **BGE-M3 Embedding**으로 관련 조문을 찾은 뒤, **Solar Pro 4**가 검색된 근거 조문을 바탕으로 답한다.

> 이 프로젝트의 답변은 법령 정보를 안내하는 참고용이며 법률·세무 자문이 아니다.

```text
질문 → 법제처 API(법령 검색·본문 조회) → 조문 파싱 → BGE-M3 Embedding(캐시)
     → 코사인 유사도 Top-5 → (점수 미달 시 "확인 불가") → Solar Pro 4 답변
```

## 사용 모델

- Embedding: `BAAI/bge-m3`
- Embedding Dimension: `1024`
- Similarity: Cosine Similarity
- Top-K: `5`
- LLM: `solar-pro4-260806`
- Temperature: `0.2`
- Minimum Similarity Score: `0.25`

BGE-M3 임베딩은 `normalize_embeddings=True`로 정규화하며, 정규화된 벡터의 내적을 이용해 코사인 유사도를 계산한다.

## 설치

```bash
pip install requests python-dotenv numpy sentence-transformers torch openai
```

프로젝트 루트의 `.env`에 다음 값을 설정한다.

```env
LAW_OC=
UPSTAGE_API_KEY=
```

- `LAW_OC`: 국가법령정보센터 공동활용 Open API 인증값
- `UPSTAGE_API_KEY`: Solar Pro 4 호출용 Upstage API Key

`.env`는 Git에 업로드하지 않는다.

## 실행

```bash
python test_law_api.py       # 법령 API 및 조문 파싱 테스트
python test_embedding.py     # BGE-M3 임베딩 테스트
python test_retriever.py     # BGE-M3 Top-5 검색 테스트
python test_rag.py           # 전체 RAG 파이프라인 테스트
```

## 구현 구조

```text
bge_rag/
├── __init__.py
├── law_api.py       # 법령 검색·본문 조회·조문 파싱
├── embedding.py     # BGE-M3 임베딩
├── retriever.py     # 임베딩 캐시·코사인 Top-5 검색
└── generator.py     # Solar Pro 4 근거 기반 답변 생성
```

법령 조문 임베딩은 최초 실행 시 `.npy` 파일로 캐싱한다. 이후 질문에서는 저장된 법령 임베딩을 재사용하고 질문만 새로 임베딩하여 검색 시간을 줄인다.

## Retriever 테스트 예시

```text
질문: 프리랜서가 돈을 받을 때 원천징수하는 세금에 관한 규정은?

BGE-M3
→ 사용자 질문 임베딩
→ 법령 조문과 코사인 유사도 계산
→ 관련 조문 Top-5 검색
→ Solar Pro 4에 근거 전달
→ 근거 기반 최종 답변
```

본 구현은 동일한 법령 데이터와 생성 모델을 사용하는 RAG 환경에서 **BGE-M3 기반 Retriever의 검색 성능을 비교·평가하기 위한 구현**이다.