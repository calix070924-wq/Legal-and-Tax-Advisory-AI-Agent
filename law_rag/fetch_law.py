#!/usr/bin/env python3
"""국가법령정보 공동활용 OPEN API - 현행법령(target=law) JSON 수집 스크립트.

사용 전 준비
-----------
레포 루트 .env 에 LAW_OC=<본인 OC(신청한 이메일의 @ 앞부분)> 를 넣으세요.
(.env.example 참고. 키 값은 절대 코드에 적지 않습니다.)

사용 예
------
    # 법령명에 '개인정보'가 들어가는 현행법령 목록만 저장
    python -m law_rag.fetch_law --query 개인정보

    # 목록 + 각 법령 본문까지 저장
    python -m law_rag.fetch_law --query 개인정보 --detail

    # 현행법령 전체 목록 (검색어 없이 전 페이지 순회)
    python -m law_rag.fetch_law --all

    # 특정 법령 하나의 본문만
    python -m law_rag.fetch_law --mst 267581
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from law_rag.config import RAW_DIR, RAW_LAW_DIR

OC = os.environ.get("LAW_OC", "")  # .env 는 law_rag.config 에서 로드

BASE = "https://www.law.go.kr/DRF"
TARGET = "law"  # 현행법령
DISPLAY = 100  # 한 페이지 최대 100건
SLEEP = 0.3  # 연속 호출 간 대기(초)


def _get(path, params):
    """API 호출 후 JSON 반환. JSON이 아니면 원문을 보여주고 종료."""
    params = {"OC": OC, "target": TARGET, "type": "JSON", **params}
    r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    try:
        data = r.json()
    except json.JSONDecodeError:
        body = r.text.strip()
        sys.exit(
            "JSON이 아닌 응답을 받았습니다. OC 값이나 활용 신청 승인 여부를 확인하세요.\n"
            f"요청 URL: {r.url}\n"
            f"응답 앞부분:\n{body[:500]}"
        )

    # 인증 실패 시에도 HTTP 200 + JSON으로 오므로 여기서 걸러낸다.
    if isinstance(data, dict) and "result" in data and "msg" in data:
        sys.exit(
            f"API 인증/권한 오류: {data['result']}\n"
            f"안내: {data['msg']}\n"
            f"요청 URL: {r.url}"
        )

    return data


def _as_list(value):
    """결과가 1건이면 dict로 오는 API 특성 보정."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def search_page(query=None, page=1, display=DISPLAY, **extra):
    """현행법령 목록 1페이지 조회."""
    params = {"page": page, "display": display, **extra}
    if query:
        params["query"] = query
    return _get("lawSearch.do", params)


def search_all(query=None, max_pages=None, **extra):
    """totalCnt를 보고 전 페이지를 순회하며 목록을 모두 수집."""
    first = search_page(query=query, page=1, **extra)
    root = first.get("LawSearch", {})
    total = int(root.get("totalCnt", 0) or 0)
    items = _as_list(root.get("law"))
    print(f"총 {total}건 / 1페이지에서 {len(items)}건 수집", file=sys.stderr)

    if total == 0:
        return []

    last_page = (total + DISPLAY - 1) // DISPLAY
    if max_pages:
        last_page = min(last_page, max_pages)

    for page in range(2, last_page + 1):
        time.sleep(SLEEP)
        res = search_page(query=query, page=page, **extra)
        got = _as_list(res.get("LawSearch", {}).get("law"))
        items.extend(got)
        print(f"  {page}/{last_page} 페이지 … 누적 {len(items)}건", file=sys.stderr)
        if not got:
            break

    return items


def get_detail(mst):
    """법령일련번호(MST)로 법령 본문 조회."""
    return _get("lawService.do", {"MST": str(mst)})


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"저장: {path}", file=sys.stderr)


def safe_name(text):
    """파일명으로 쓸 수 없는 문자 제거."""
    bad = '/\\:*?"<>|\n\t'
    return "".join("_" if c in bad else c for c in str(text)).strip()[:120]


def main():
    ap = argparse.ArgumentParser(description="현행법령 JSON 수집")
    ap.add_argument("--query", help="법령명 검색어")
    ap.add_argument("--all", action="store_true", help="검색어 없이 현행법령 전체 목록 수집")
    ap.add_argument("--detail", action="store_true", help="목록의 각 법령 본문도 함께 저장")
    ap.add_argument("--mst", help="법령일련번호로 본문만 조회")
    ap.add_argument("--max-pages", type=int, help="목록 수집 최대 페이지 수")
    ap.add_argument("--limit", type=int, help="본문을 받아올 법령 개수 상한")
    ap.add_argument("--out", default=str(RAW_DIR), help="목록 JSON 저장 디렉터리 (기본: data/raw)")
    ap.add_argument("--detail-out", default=str(RAW_LAW_DIR), help="본문 JSON 저장 디렉터리 (기본: data/raw/law)")
    args = ap.parse_args()

    if not OC:
        sys.exit("LAW_OC 가 비어 있습니다. 레포 루트 .env 에 LAW_OC=... 를 설정하세요.")

    out = Path(args.out)
    detail_out = Path(args.detail_out)

    # 본문 단건 조회
    if args.mst:
        detail = get_detail(args.mst)
        name = (
            detail.get("법령", {}).get("기본정보", {}).get("법령명_한글")
            or detail.get("법령", {}).get("기본정보", {}).get("법령명한글")
            or args.mst
        )
        save_json(detail, detail_out / f"{safe_name(name)}_{args.mst}.json")
        return

    if not args.query and not args.all:
        ap.error("--query 또는 --all 중 하나를 지정하세요 (또는 --mst).")

    # 목록 조회
    items = search_all(query=args.query, max_pages=args.max_pages)
    if not items:
        print("검색 결과가 없습니다.", file=sys.stderr)
        return

    label = safe_name(args.query) if args.query else "현행법령_전체"
    save_json(items, out / f"list_{label}.json")

    for it in items[:5]:
        print(
            f"  - {it.get('법령명한글')} "
            f"(MST={it.get('법령일련번호')}, 시행 {it.get('시행일자')})",
            file=sys.stderr,
        )

    # 본문 조회
    if args.detail:
        targets = items[: args.limit] if args.limit else items
        print(f"본문 {len(targets)}건 수집 시작", file=sys.stderr)
        for i, it in enumerate(targets, 1):
            mst = it.get("법령일련번호")
            if not mst:
                continue
            try:
                detail = get_detail(mst)
            except requests.HTTPError as e:
                print(f"  [{i}] MST={mst} 실패: {e}", file=sys.stderr)
                continue
            fname = f"{safe_name(it.get('법령명한글', mst))}_{mst}.json"
            save_json(detail, detail_out / fname)
            time.sleep(SLEEP)


if __name__ == "__main__":
    main()
