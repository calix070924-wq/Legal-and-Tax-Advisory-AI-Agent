# 작업 지시서 01 — 법령 Retriever: 코드 이관 + Chroma 색인 + 검색기

> 이 문서를 처음부터 끝까지 읽고 단계 순서대로 진행한다.
> 각 단계가 끝나면 체크리스트를 확인하고, "멈추고 보고"라고 적힌 곳에서는 사용자에게 결과를 보여 준 뒤 진행한다.

- 레포: https://github.com/calix070924-wq/Legal-and-Tax-Advisory-AI-Agent (현재 `main`에 README만 있음)
- 작업 브랜치: `feat/retriever-index` (`main`에서 분기)
- 작업 환경: MacBook Air (Apple Silicon, 24GB RAM, GPU 없음 → MPS 사용), Python 3.13
- 기존 작업물 위치: `~/law_api/` (레포 밖, 아래 2단계에서 이관)

---

## 0. 배경 — 지금까지 정해진 것

1인 창업가·프리랜서 대상 법률·세무 상담 RAG 에이전트를 만든다. 이번 작업은 **법령 검색기(Retriever)** 까지다. 답변 생성(Solar Pro 4)은 다음 작업이다.

| 항목 | 결정 | 근거 |
|---|---|---|
| 데이터 | 법제처 국가법령정보 Open API (JSON) | `fetch_law.py`로 수집 완료 (현재 개인정보 보호법) |
| 청킹 | 조문 단위, 1,500자 초과 시 항·호 단위 분할 | `parse_law.py` 완료, 개인정보 보호법 → 청크 131개, 원문 대조 누락 0건 |
| 임베딩 | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | 자체 평가 32문항에서 1위 (아래 표) |
| 벡터 DB | Chroma (로컬 persist), 코사인 거리 | 메타데이터 필터, id 단위 upsert/delete 필요 |
| BM25 하이브리드 | **이번엔 넣지 않음** | 동일 가중치 RRF로 섞으면 모든 모델 성능 하락 (MRR 0.902 → 0.726) |
| LangChain | 이번 단계부터 사용 | `HuggingFaceEmbeddings`, `Chroma`, `VectorStoreRetriever` |

임베딩 모델 평가 결과 (`eval/eval_set.jsonl`, 32문항, 조문 단위 적중):

| 모델 | R@1 | R@5 | MRR@10 |
|---|---|---|---|
| **snowflake-arctic-embed-l-v2.0-ko** | **0.844** | 0.969 | **0.902** |
| BAAI/bge-m3 | 0.812 | 1.000 | 0.893 |
| dragonkue/BGE-m3-ko | 0.719 | 0.969 | 0.822 |
| nlpai-lab/KURE-v1 | 0.688 | 0.969 | 0.812 |
| BM25 (kiwi) | 0.406 | 0.656 | 0.522 |

**이 숫자는 5단계의 합격 기준이다.** Chroma에 옮긴 뒤에도 arctic-ko가 이 성능을 내야 한다.

---

## 1. 브랜치 만들기

```bash
git checkout main
git pull origin main
git checkout -b feat/retriever-index
```

- [ ] `git branch --show-current` 가 `feat/retriever-index`

---

## 2. 기존 코드 이관 + 비밀값 제거

### 2-1. 목표 구조

```
Legal-and-Tax-Advisory-AI-Agent/
├── law_rag/
│   ├── __init__.py
│   ├── config.py          # 신규: 경로·모델명·컬렉션명 상수, .env 로드
│   ├── fetch_law.py       # ~/law_api/fetch_law.py 이관
│   ├── parse_law.py       # ~/law_api/parse_law.py 이관
│   ├── embeddings.py      # 신규 (3단계)
│   └── vectorstore.py     # 신규 (3~4단계)
├── scripts/
│   └── build_index.py     # 신규 (3단계)
├── eval/
│   ├── eval_set.jsonl     # ~/law_api/eval/eval_set.jsonl 이관
│   ├── eval_embed.py      # ~/law_api/eval_embed.py 이관
│   └── eval_retriever.py  # 신규 (5단계)
├── data/                  # 전부 .gitignore
│   ├── raw/law/           # 법제처 원본 JSON (~/law_api/out/detail/*.json 복사)
│   ├── parsed/            # chunks.jsonl, articles.jsonl
│   ├── emb/               # eval_embed.py 임베딩 캐시
│   └── chroma/            # Chroma persist 디렉터리
├── docs/tasks/01_retriever_색인_작업지시서.md   # 이 문서
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### 2-2. 이관 작업

1. `~/law_api/fetch_law.py`, `parse_law.py`, `eval_embed.py`, `eval/eval_set.jsonl`을 위 위치로 **복사**한다. `~/law_api` 원본은 건드리지 않는다.
2. `~/law_api/out/detail/*.json` → `data/raw/law/`로 복사한다.
3. 각 파일의 기본 경로(`out/...`)를 `law_rag/config.py` 상수를 참조하도록 바꾼다.
   - 원본 JSON: `data/raw/law/`
   - 파싱 결과: `data/parsed/`
   - 임베딩 캐시: `data/emb/`
4. 모듈 실행을 지원한다: `python -m law_rag.parse_law data/raw/law/*.json`, `python -m law_rag.fetch_law --query ...`

### 2-3. ⚠️ 법제처 API 키(OC) 제거 — 첫 커밋 전에 반드시

`~/law_api/fetch_law.py` 상단의 `OC = "..."` 줄에 **실제 키가 하드코딩되어 있다.** 레포가 공개 레포이므로 절대 커밋하지 않는다.

- `OC` 기본값을 빈 문자열로 바꾸고, `LAW_OC` 환경변수만 쓰게 한다 (`python-dotenv`로 `.env` 로드).
- `.env.example`에는 `LAW_OC=` 와 `UPSTAGE_API_KEY=` 를 빈 값으로 둔다.
- 실제 키는 사용자에게 `.env`에 직접 넣어 달라고 요청한다. **키 값을 이 레포의 어떤 파일에도 옮겨 적지 않는다.**
- `data/raw/`의 목록 JSON(`list_*.json`)은 `법령상세링크` 필드에 OC 키를 포함하고 있다 → `data/` 전체를 `.gitignore`에 넣는 이유 중 하나.

### 2-4. `.gitignore`

```
.env
.venv/
data/
__pycache__/
*.pyc
.DS_Store
```

- [ ] `git status`에 `data/`, `.env`가 보이지 않음
- [ ] `grep -rn "OC = \"" law_rag/` 결과에 실제 키 값이 없음
- [ ] `python -m law_rag.parse_law data/raw/law/*.json` → `청크 131개` 출력, `data/parsed/`에 두 파일 생성

---

## 3. 환경 점검 + 패키지 설치

### 3-1. 현재 상태부터 확인 (설치 전에)

LangChain 관련 패키지는 **아직 설치되지 않았을 가능성이 높다.** 설치하기 전에 무엇이 있는지 먼저 확인하고 결과를 기록한다.

```bash
which python3 && python3 --version
python3 -m pip list 2>/dev/null | grep -iE "langchain|chroma|sentence|transformers|huggingface|torch|tokenizers|faiss|kiwi|rank"
python3 -c "import langchain_core" 2>&1 | tail -1
python3 -c "import langchain_huggingface" 2>&1 | tail -1
python3 -c "import langchain_chroma" 2>&1 | tail -1
python3 -c "import chromadb" 2>&1 | tail -1
```

### 3-2. 가상환경

레포 루트에 `.venv`를 만들어 쓴다. 시스템 Python에는 설치하지 않는다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3-3. `requirements.txt`

2026-09-29 기준 PyPI에서 확인한 버전 제약이다.

```
# 임베딩
torch
sentence-transformers
transformers
huggingface-hub>=1.5.0,<2.0      # 2.0.0은 transformers와 충돌 (실제로 한 번 깨졌음)
hf_xet

# LangChain
langchain-core>=1.2.31,<2
langchain-huggingface>=1.2,<2    # huggingface-hub<2 요구 → 위와 호환
langchain-chroma>=1.1,<2
chromadb>=1.3.5,<2               # langchain-chroma 1.1이 요구

# 평가 (eval_embed.py)
faiss-cpu
kiwipiepy
rank-bm25

# 기타
python-dotenv
requests
numpy
```

설치 후:

```bash
pip install -r requirements.txt
pip check                                  # "No broken requirements found." 이어야 함
python -c "import huggingface_hub as h; print(h.__version__)"   # 2.x 이면 안 됨
python -c "import torch; print(torch.backends.mps.is_available())"   # True 기대
```

모든 버전이 확정되면 `pip freeze`로 실제 설치 버전을 확인하고, 위 패키지들을 `==`로 고정한 버전을 `requirements.txt`에 반영한다.

### 3-4. 하지 말 것

- `pip install -U huggingface_hub` 금지. 2.0.0이 설치되면서 transformers가 깨진 적이 있다.
- **`langchain-upstage` 설치 금지.** `tokenizers<0.21`을 요구해서 transformers와 충돌한다. Solar Pro 4는 OpenAI 호환 API라서, 다음 작업에서 `langchain-openai`의 `ChatOpenAI(base_url="https://api.upstage.ai/v1", model="solar-pro4")`로 붙일 예정이다. 이번 작업 범위는 아니다.
- 모델을 다시 받지 않는다. 4개 모델이 이미 `~/.cache/huggingface/hub`에 있다. 네트워크 접근을 막으려면 `export HF_HUB_OFFLINE=1`로 실행한다.

- [ ] 3-1 확인 결과와 3-3 설치 결과(`pip check`, hub 버전, MPS 여부)를 사용자에게 보여 준다 — **멈추고 보고**

---

## 4. 색인 구현

### 4-1. `law_rag/config.py`

- `ROOT`, `DATA_DIR`, `RAW_LAW_DIR`, `PARSED_DIR`, `CHROMA_DIR`
- `EMBED_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"`
- `COLLECTION = "law_articles"`
- `MAX_SEQ_LEN = 1024` (청크 최대 약 1,600자라 충분)
- `.env` 로드 (`load_dotenv()`)

### 4-2. `law_rag/embeddings.py` — `get_embeddings()`

`langchain_huggingface.HuggingFaceEmbeddings`로 만든다.

- `model_kwargs={"device": <mps|cuda|cpu 자동 선택>}`
- `encode_kwargs={"normalize_embeddings": True, "batch_size": 8}` — 문서용
- `query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True}` — 질문용
- 로드 후 내부 SentenceTransformer의 `max_seq_length = 1024`

**핵심: 질문에만 `"query: "` 접두어가 붙어야 한다.** 이 모델은 접두어가 빠지면 성능이 떨어지는데 에러는 나지 않는다.

- 설치된 `langchain_huggingface` 소스에서 `embed_query`가 `query_encode_kwargs`를 실제로 사용하는지 먼저 확인한다.
- 안 쓰면 `langchain_core.embeddings.Embeddings`를 상속한 얇은 래퍼를 직접 만든다. `embed_documents`는 `encode(texts, normalize_embeddings=True)`, `embed_query`는 `encode([text], prompt_name="query", normalize_embeddings=True)`로 구현한다.
- 검증: `embed_query("테스트")`와 `embed_documents(["테스트"])[0]`의 코사인 유사도가 **1보다 확실히 작아야** 한다. 1.0이면 접두어가 적용되지 않은 것이다.

### 4-3. `law_rag/vectorstore.py`

- `get_vectorstore()`: `langchain_chroma.Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(), persist_directory=str(CHROMA_DIR), ...)`
- **거리 함수를 코사인으로 설정한다.** Chroma 기본값은 L2다.
  - 설치된 `langchain-chroma` 버전에 맞는 방식을 쓴다: `collection_configuration={"hnsw": {"space": "cosine"}}` 또는 `collection_metadata={"hnsw:space": "cosine"}`.
  - 생성 후 실제 컬렉션 설정(`vs._collection.configuration` 또는 `.metadata`)을 출력해서 cosine인지 확인한다.
  - 이미 L2로 만들어진 컬렉션은 설정을 바꿀 수 없으니 지우고 다시 만든다.

### 4-4. `scripts/build_index.py`

```
python scripts/build_index.py                 # data/parsed/chunks.jsonl → Chroma
python scripts/build_index.py --rebuild       # 컬렉션 삭제 후 새로 생성
```

- 입력: `data/parsed/chunks.jsonl` (`parse_law.load_documents()` 재사용)
- **문서 id = 청크의 `id` 필드** (예: `011357-0015001-p1`). uuid를 새로 만들지 않는다.
- 법령 단위 교체: `law_id`별로 `delete(where={"law_id": ...})`를 먼저 한 뒤 `add_documents(docs, ids=ids)`로 넣는다. 개정으로 조각 수가 줄었을 때 옛 청크가 남지 않게 하려는 것이다.
- 메타데이터는 str/int/float/bool만 허용한다. 그 외 타입이나 None은 문자열로 바꾸거나 제거한다.
- 끝에 출력할 것: 컬렉션 문서 수, 법령별 문서 수, 거리 함수, 소요 시간.

- [ ] 1회 실행 후 문서 수 131
- [ ] 한 번 더 실행해도 131 (중복 없음)
- [ ] 거리 함수 cosine 확인

---

## 5. 검색기 + 검증

### 5-1. 검색 함수 (`law_rag/vectorstore.py`)

- `get_retriever(k=5, filter=None)` → `vs.as_retriever(search_kwargs={"k": k, "filter": filter})`
- `search(query, k=5, filter=None) -> list[tuple[Document, float]]`
  - `similarity_search_with_score`로 넉넉히 가져온다 (k×3).
  - 같은 조문의 분할 청크(`metadata["article_uid"]` 동일)는 **조문 단위로 중복 제거**하고 k개를 반환한다.
- `get_article(article_uid) -> dict`: `data/parsed/articles.jsonl`에서 조문 전체를 찾아 반환한다. 다음 작업에서 분할 청크가 검색됐을 때 조문 전체를 LLM에 넘기는 데 쓴다.
- CLI: `python -m law_rag.vectorstore "탈퇴한 회원 정보 언제까지 지워야 해?"` → top-5를 `조문라벨 | 제목 | 점수 | url`로 출력한다.

### 5-2. `eval/eval_retriever.py` — 합격 기준

`eval/eval_set.jsonl` 32문항을 **Chroma 검색기로** 돌려 `eval_embed.py`와 같은 방식(조문 단위, gold 중 하나라도 적중)으로 R@1, R@3, R@5, R@10, MRR@10을 계산한다.

| 지표 | 기대값 (FAISS 정확 검색 결과) | 허용 범위 |
|---|---|---|
| R@1 | 0.844 | 0.81 이상 |
| R@5 | 0.969 | 0.93 이상 |
| MRR@10 | 0.902 | 0.87 이상 |

- 허용 범위보다 확실히 낮으면 먼저 의심할 곳: ① 질문 접두어 미적용(4-2), ② 거리 함수 L2(4-3), ③ 정규화 누락.
- 허용 범위 안에서 1문항 정도 차이는 HNSW 근사 검색 때문일 수 있다. 결과표에 그렇게 적는다.

### 5-3. 알려진 약점 (고치지 말고 기록만)

"고객이 우리가 가지고 있는 자기 정보를 보여달라고 하면 보여줘야 돼?" → 정답은 제35조(개인정보의 열람)인데, 네 모델 모두 top-5 밖이었다. 조문에는 "열람"이라고 적혀 있어서다. 다음 작업의 **질의 재작성**(Solar로 일상어 → 법률 용어 변환)에서 다룬다. 이번 작업에서 평가셋이나 청크 텍스트를 이 문항에 맞춰 바꾸지 않는다.

- [ ] 5-2 결과표와 CLI 예시 출력 3개를 사용자에게 보여 준다 — **멈추고 보고**

---

## 6. 문서화 · 커밋 · PR

### 6-1. README.md

- 프로젝트 한 줄 소개
- 설치: venv, `requirements.txt`, `.env` 설정 (`LAW_OC`)
- 파이프라인 실행 순서: `fetch_law` → `parse_law` → `build_index` → `vectorstore` CLI
- 평가: `eval_embed.py`(모델 비교), `eval_retriever.py`(색인 검증)와 0장의 결과표

### 6-2. 커밋

작업 단위로 나눈다:
1. `chore: 프로젝트 구조, requirements, gitignore, env 설정`
2. `feat: 법제처 API 수집·조문 파서 이관 (OC 키 환경변수화)`
3. `feat: arctic-ko 임베딩 + Chroma 색인 스크립트`
4. `feat: 조문 단위 검색기와 CLI`
5. `test: Chroma 검색기 평가 스크립트`
6. `docs: README`

**매 커밋 전에 확인한다:**
- `git diff --cached`에 API 키 문자열이 없는지
- `data/`, `.env`, `.venv/`가 스테이징되지 않았는지

### 6-3. PR

```bash
git push -u origin feat/retriever-index
```

- `main` 대상으로 PR을 만든다. 제목은 `법령 Retriever: Chroma 색인 + 검색기`로 한다.
- 본문에 넣을 것: 변경 요약, 5-2 결과표, 알려진 약점(제35조), 다음 작업 목록(아래 7장).
- **머지는 하지 않는다.** 사용자가 리뷰 후 머지한다.

---

## 7. 이번 범위가 아닌 것 (다음 작업)

- Solar Pro 4 답변 생성 체인 (`langchain-openai` + Upstage base_url, 근거 번호 인용, 시행일·면책 문구)
- 질의 분석·재작성 (일상어 → 법률 용어, 법령명 추출)
- BM25를 조건부로 추가 (질문에 "제○조"나 법령명이 있을 때만, 또는 낮은 가중치) + 용어형 평가 문항 추가
- 시행령·세법(부가가치세법, 소득세법 등) 수집, 판례·해석례(`target=prec`, `expc`) 파서
- 리랭커 (`bge-reranker-v2-m3`)
- 평가셋 50~100문항으로 확대
