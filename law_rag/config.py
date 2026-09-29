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

