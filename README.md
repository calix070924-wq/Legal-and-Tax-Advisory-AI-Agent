# Legal-and-Tax-Advisory-AI-Agent — `upstage_rag` (정준영)

프리랜서·1인 사업자를 위한 법률·세무 상담 RAG. 질문이 들어오면 법제처 Open API로 필요한 법령을 가져오고, **Upstage Embedding**으로 관련 조문을 찾은 뒤, **Solar Pro 4**가 근거 조문을 인용해 답한다.

> 처음 보는 사람은 [docs/설명_기초부터_RAG까지.md](docs/설명_기초부터_RAG까지.md)부터 읽는다.
> 이 프로젝트의 답변은 법령 정보를 안내하는 참고용이며 법률·세무 자문이 아니다.

```
질문 → Solar가 법령 선택 → 법제처 API(조문 파싱, 캐시) → Upstage Embedding(query/passage, 캐시)
     → 코사인 Top-5 → (점수 미달 시 "확인 불가") → Solar Pro 4 답변 → 인용 검증·출처·면책
```

## 설치 (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # LAW_OC, UPSTAGE_API_KEY 채우기
pytest -q
```

## 실행

```bash
python -m upstage_rag "프리랜서 외주 대금에서 3.3% 떼는 근거가 뭐야?" --show-context
python -m upstage_rag.retriever "질문"      # 검색만
python -m upstage_rag.law_api 소득세법       # 법령 수집·파싱만
```

API 사용량은 `data/usage.jsonl`에 쌓인다 (멘토링 비용 보고용).
