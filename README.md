# Legal-and-Tax-Advisory-AI-Agent

1인 창업가·프리랜서를 위한 법률·세무 상담 RAG 에이전트. 법제처 국가법령정보 Open API로 법령을 모아 조문 단위로 검색하고, Solar Pro 4가 근거 조문을 인용해 답한다.

> 현재 색인은 **개인정보 보호법**뿐이다. 이 프로젝트의 답변은 법령 정보를 안내하는 참고용이며 법률·세무 자문이 아니다.

## 구조

```
질문
 → Chroma 검색 (arctic-ko 임베딩, 코사인, 조문 단위 중복 제거, top-5)
 → 조문 확장 (분할 청크 → 조문 전체, 근거 번호 [1]~[n], 미시행 규정 표시)
 → Solar Pro 4 (근거 안에서만 답변, 문장마다 [n] 인용)
 → 후처리 (인용 검증 · 출처 목록 · 기준 시행일 · 면책 문구 — 코드가 붙임)
```

```
law_rag/
  config.py        경로·모델명·상수, .env 로드
  fetch_law.py     법제처 API 수집 (현행법령 목록·본문 JSON)
  parse_law.py     조문 단위 청킹 (1,500자 초과 시 항·호 단위 분할)
  embeddings.py    arctic-ko 임베딩 (질문에만 "query: " 접두어)
  vectorstore.py   Chroma 색인·조문 단위 검색, 검색 CLI
  llm.py           Solar Pro 4 연결 (langchain-openai + Upstage base_url)
  context.py       근거 조립 (small-to-big, 글자 수 예산)
  prompts.py       시스템 프롬프트
  postprocess.py   인용 검증·출처·기준일·면책
  chain.py         LCEL 답변 체인, 답변 CLI
scripts/build_index.py   chunks.jsonl → Chroma
eval/                    검색·답변 평가 스크립트와 결과
tests/                   후처리 단위 테스트 (API 호출 없음)
docs/tasks/              작업지시서 01·02
```

## 설치

Python 3.13, Apple Silicon(MPS)에서 확인했다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip check                          # No broken requirements found.
cp .env.example .env               # 값 채우기
```

`.env`:

| 변수 | 내용 |
|---|---|
| `LAW_OC` | 법제처 Open API OC (신청한 이메일의 @ 앞부분) |
| `UPSTAGE_API_KEY` | Upstage 콘솔에서 발급한 키 |
| `SOLAR_MODEL` | (선택) 기본값 `solar-pro4-260806` |

주의:
- `huggingface-hub`를 2.x로 올리지 않는다 (`pip install -U huggingface_hub` 금지). transformers와 충돌한다.
- `langchain-upstage`를 설치하지 않는다. `tokenizers<0.21`을 요구해서 transformers와 충돌한다. Solar는 OpenAI 호환 API라 `ChatOpenAI`로 붙인다.
- 임베딩 모델이 `~/.cache/huggingface/hub`에 있으면 `export HF_HUB_OFFLINE=1`로 네트워크 없이 실행할 수 있다.

## 실행 순서

```bash
python -m law_rag.fetch_law --query 개인정보 --detail     # data/raw/law/*.json
python -m law_rag.parse_law data/raw/law/*.json           # data/parsed/chunks.jsonl, articles.jsonl (청크 131개)
python scripts/build_index.py                             # Chroma 색인 (--rebuild: 새로 생성)
python -m law_rag.vectorstore "탈퇴한 회원 정보 언제까지 지워야 해?"   # 검색만
python -m law_rag.chain "서비스 탈퇴한 회원 정보는 언제까지 지워야 해?" # 답변 (--show-context: 근거 출력)
```

답변 예시 (요약):

```
**결론**: … 처리 목적이 달성되어 회원 정보가 불필요해졌다면 지체 없이 파기해야 합니다 [1]. …

출처
[1] 개인정보 보호법 제21조(개인정보의 파기) — 시행 2026.09.11 — https://www.law.go.kr/법령/개인정보보호법/제21조

※ 이 답변은 2026.09.11 시행 법령 기준입니다.

※ 이 답변은 법령 정보를 안내하는 참고용이며 법률·세무 자문이 아닙니다. 구체적인 사안은 변호사·세무사와 상담하세요.
```

## 평가

```bash
python eval/eval_embed.py        # 임베딩 모델 비교 (FAISS 정확 검색)
python eval/eval_retriever.py    # Chroma 검색기 검증
python eval/eval_answer.py       # 답변 평가 → eval/results_answer.jsonl, eval/answer_report.md
pytest -q                        # 후처리 단위 테스트
```

### 임베딩 모델 비교 (`eval/eval_set.jsonl` 32문항, 조문 단위 적중)

| 모델 | R@1 | R@5 | MRR@10 |
|---|---|---|---|
| **dragonkue/snowflake-arctic-embed-l-v2.0-ko** | **0.844** | 0.969 | **0.902** |
| BAAI/bge-m3 | 0.812 | 1.000 | 0.893 |
| dragonkue/BGE-m3-ko | 0.719 | 0.969 | 0.822 |
| nlpai-lab/KURE-v1 | 0.688 | 0.969 | 0.812 |
| BM25 (kiwi) | 0.406 | 0.656 | 0.522 |

### Chroma 검색기 (`eval_retriever.py`, 같은 32문항)

| 지표 | Chroma | FAISS 기대값 | 허용 범위 |
|---|---|---|---|
| R@1 | 0.844 | 0.844 | ≥ 0.81 |
| R@3 | 0.938 | – | – |
| R@5 | 0.969 | 0.969 | ≥ 0.93 |
| R@10 | 1.000 | – | – |
| MRR@10 | 0.902 | 0.902 | ≥ 0.87 |

FAISS 정확 검색과 같은 값이다 (131개 규모에서는 HNSW 근사 오차가 없었다).

### 답변 (`eval_answer.py`, 15문항: 범위 안 12 · 범위 밖 세법 2 · 지시 무시 시도 1)

| 체크 | 대상 | 통과 |
|---|---|---|
| 형식 (인용 1개 이상) | 범위 안 | 11/12 |
| 인용 유효성 (범위 밖 번호 없음) | 전체 | 15/15 |
| 인용 적중 (정답 조문 인용) | 범위 안 | **11/12** |
| 근거 부족 처리 ("확인하기 어렵다", 조문 지어내지 않음) | 범위 밖·지시 무시 | 3/3 |
| 조문 지어내기 없음 | 전체 | 15/15 |
| 면책 문구 | 전체 | 15/15 |

합격 기준(인용 유효성·면책 100%, 인용 적중 10/12 이상, 범위 밖 조문 지어내지 않음)을 통과했다. 실패 1건은 아래 제35조 문항이며, 근거가 없어 "확인하기 어렵다"로 답한 결과다. 평균 응답 5.7초, 입력 약 2,800 / 출력 약 350 토큰. 문항별 답변 전문은 [`eval/answer_report.md`](eval/answer_report.md).

프롬프트는 두 차례 조정했다: (1) 조문을 본문에 언급할 때도 `[n]`을 달게 함, (2) 근거가 질문에 직접 답하지 못하면 결론을 내리지 않고 다른 조문을 짐작해 쓰지 않게 함. 1차 조정 후 제35조 문항에서 근거에 없는 조문 번호를 지어낸 것을 발견해 "조문 지어내기 없음" 체크를 추가했다.

## 주요 결정

| 결정 | 이유 |
|---|---|
| 임베딩 `snowflake-arctic-embed-l-v2.0-ko` | 자체 평가 32문항에서 R@1·MRR@10 1위 |
| BM25 하이브리드 제외 | 동일 가중치 RRF로 섞으면 모든 모델 성능 하락 (arctic-ko MRR 0.902 → 0.726) |
| Chroma, 코사인 거리 | 로컬 persist, 메타데이터 필터, id 단위 upsert/delete. 기본값 L2 대신 코사인을 명시 |
| 문서 id = 청크 id, 법령 단위 교체 | 재색인해도 중복 없음. 개정으로 청크 수가 줄어도 옛 청크가 남지 않음 |
| `langchain-upstage` 대신 `ChatOpenAI` | `langchain-upstage`는 transformers와 충돌. Solar는 OpenAI 호환 API |
| 모델 스냅샷 `solar-pro4-260806` 고정 | 별칭(`solar-pro4`)은 업데이트되면 바뀐다. 참고로 `solar-pro` 별칭은 Pro 2로 연결된다 |
| `temperature=0.2` | Solar Pro 4 기본값 1.0은 법률 답변에 너무 높다 |
| **`reasoning_effort` 지정 안 함 (추론 끔)** | 아래 참고 |
| 출처·기준일·면책은 코드가 붙임 | LLM이 빠뜨려도 항상 나가야 한다. 범위 밖 인용 번호도 코드가 지운다 |

### `reasoning_effort`를 기본값으로 둔 이유

solar-pro4는 `reasoning_effort`를 지정하지 않으면 추론을 하지 않는다. 같은 질문("한 문장으로 자기소개해")으로 비교했다 (2026-09-29, 각 1회):

| | 기본값 (지정 안 함) | `reasoning_effort="low"` |
|---|---|---|
| 응답 시간 | **0.50초** | 10.42초 |
| 출력 토큰 | 12 (추론 0) | 430 (추론 412) |

이 체인에서 LLM이 하는 일은 이미 주어진 근거 조문을 쉬운 말로 요약하고 번호를 다는 것이라 추론 없이도 충분하다고 판단했다. 실제로 기본값으로 답변 평가 합격 기준을 통과했다. 추론을 켜면 응답이 약 20배 느려지고 출력 토큰도 늘어난다. 답변 품질이 기준에 못 미치는 경우가 생기면 `get_llm(reasoning_effort="low")`로 같은 평가를 돌려 비교한다. 추론 과정(`reasoning`)은 사용자에게 노출하지 않는다.

## 한계

- **색인 범위:** 개인정보 보호법만 있다. 세법(부가가치세법, 소득세법 등)·시행령·판례는 아직 없다. 세법 질문에는 "확인하기 어렵다"고 답한다.
- **일상어 질의:** "고객이 자기 정보를 보여달라고 하면?"의 정답은 제35조(개인정보의 열람)인데, 조문 용어("열람")와 달라 네 임베딩 모델 모두 top-5 밖이다. 답변은 결론 없이 "확인하기 어렵다"로 처리된다. 다음 작업(질의 재작성)에서 다룬다.
- **미시행 규정 시행일 미검증:** 프롬프트는 미시행 규정을 쓸 때 시행일을 밝히게 하지만, 색인의 미시행 규정(제32조의2제1항 단서, 제75조제2항제15호, 2027.07.01 시행)을 묻는 평가 문항이 없어 검증하지 못했다.
- **내용 정확성은 자동 체크 밖:** 평가는 인용 번호와 조문 번호만 검사한다. 금액·요건을 근거와 맞게 옮겼는지는 사람이 봐야 한다. 예: 지시 무시 문항에서 제75조를 요약하며 처리방침 위반을 3천만원 이하 항목에도 넣은 오류가 있었다(실제 1천만원 이하).
- **법률 자문 아님:** 모든 답변은 참고용이며 구체적인 사안은 변호사·세무사와 상담해야 한다.

## 다음 작업

- 질의 분석·재작성 (일상어 → 법률 용어, 법령명 추출)
- 조건부 BM25 (질문에 "제○조"나 법령명이 있을 때만, 또는 낮은 가중치) + 용어형 평가 문항
- 세법·시행령 수집, 판례·해석례(`target=prec`, `expc`) 파서
- 리랭커 (`bge-reranker-v2-m3`)
- 평가셋 50~100문항으로 확대
