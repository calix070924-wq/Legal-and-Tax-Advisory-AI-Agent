"""경로·모델명·컬렉션명 상수. 모든 모듈은 여기 값을 기본값으로 쓴다."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv 미설치 환경에서도 파서는 돌 수 있게
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_LAW_DIR = DATA_DIR / "raw" / "law"  # 법제처 법령 본문 JSON (lawService.do)
RAW_DIR = DATA_DIR / "raw"  # 목록 JSON (list_*.json) 저장 위치
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_PATH = PARSED_DIR / "chunks.jsonl"
ARTICLES_PATH = PARSED_DIR / "articles.jsonl"
EMB_DIR = DATA_DIR / "emb"  # eval_embed.py 임베딩 캐시
CHROMA_DIR = DATA_DIR / "chroma"

EMBED_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
COLLECTION = "law_articles"
MAX_SEQ_LEN = 1024  # 청크 최대 약 1,600자라 충분

LAW_OC = os.environ.get("LAW_OC", "")

# LLM (Solar Pro 4, Upstage OpenAI 호환 API)
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_MODEL = os.getenv("SOLAR_MODEL", "solar-pro4-260806")  # 스냅샷 고정 (별칭 solar-pro4는 업데이트되면 바뀜)
LLM_TEMPERATURE = 0.2  # Solar Pro 4 기본값은 1.0 → 법률 답변은 낮게
LLM_MAX_TOKENS = 4096  # 추론 토큰도 여기에 포함될 수 있어 넉넉히
# reasoning_effort는 지정하지 않는다 (solar-pro4 기본값 = 추론 없음).
# 2026-09-29 측정: 기본값 0.50s·추론 0토큰 vs "low" 10.42s·추론 412토큰.
# 근거가 주어진 요약·인용 작업이라 추론 없이 시작하고, 답변 평가 미달 시 "low"를 비교한다.
RETRIEVE_K = 5
CONTEXT_CHAR_BUDGET = 12000  # 근거 전체 글자 수 상한
ARTICLE_CHAR_LIMIT = 4000  # 조문 1개 상한 (넘으면 검색된 청크만 사용)
