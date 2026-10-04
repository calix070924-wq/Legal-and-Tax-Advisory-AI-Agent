"""설정값. 모든 모듈은 여기 값을 기본값으로 쓴다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 비밀값 (.env)
LAW_OC = os.getenv("LAW_OC", "")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "")

# 캐시 경로 (git에 올리지 않음)
CACHE_DIR = ROOT / "data" / "cache"
LAW_CACHE_DIR = CACHE_DIR / "law"  # 법령 원문 JSON
EMB_CACHE_DIR = CACHE_DIR / "emb"  # 조문 임베딩 .npy
USAGE_LOG = ROOT / "data" / "usage.jsonl"  # API 사용량 기록 (건당 비용 보고용)

# Upstage
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_MODEL = os.getenv("SOLAR_MODEL", "solar-pro4-260806")  # 별칭(solar-pro4)은 업데이트되면 바뀌어서 스냅샷 고정
EMBED_QUERY_MODEL = os.getenv("EMBED_QUERY_MODEL", "embedding-query")
EMBED_PASSAGE_MODEL = os.getenv("EMBED_PASSAGE_MODEL", "embedding-passage")
EMBED_BATCH = 100  # 임베딩 요청 1회당 최대 문장 수
EMBED_MAX_CHARS = 3000  # 임베딩 입력 상한 (긴 조문은 앞부분만 임베딩, 답변 근거에는 전문 사용)
TEMPERATURE = 0.2  # 법률 답변이라 낮게

# 검색
TOP_K = 5  # Solar에 넘길 조문 수
MIN_SCORE = 0.25  # 1등 조문의 유사도가 이보다 낮으면 "확인 불가" (평가 후 조정)
MAX_LAWS = 3  # 질문 1개당 불러올 법령 수
CONTEXT_CHAR_BUDGET = 12000  # 근거 전체 글자 수 상한

# MVP(프리랜서 용역계약) 범위에서 자주 쓰는 법령. 검색어 추출 프롬프트에 후보로 준다.
DEFAULT_LAWS = [
    "소득세법",
    "소득세법 시행령",
    "부가가치세법",
    "부가가치세법 시행령",
    "국세기본법",
    "지방세법",  # 3.3% 중 0.3%(지방소득세)의 근거
    "민법",
    "하도급거래 공정화에 관한 법률",
    "근로기준법",
]
