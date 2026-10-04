"""법제처 국가법령정보 Open API → 조문 목록.

- lawSearch.do  : 법령명으로 현행법령 검색
- lawService.do : 법령 본문(JSON)
원문 JSON은 data/cache/law/ 에 저장해서 같은 법령을 다시 받지 않는다.

API 특이사항
- 항/호/목이 1건이면 list 대신 dict 로 온다 → as_list 로 정규화
- 조문여부 == "전문" 은 장·절 제목 → 조문으로 만들지 않는다
- "제8조 삭제 <2020.2.4>" 같은 삭제 조문은 제외
- 텍스트가 문자열/리스트/중첩 리스트로 섞여 온다 → flatten
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import requests

from upstage_rag.config import LAW_CACHE_DIR, LAW_OC

BASE = "https://www.law.go.kr/DRF"
AMEND_TAG = re.compile(r"\s*<(?:개정|신설|전문개정|제목개정|타법개정|본조신설)[^>]*>")
DELETED = re.compile(r"^제\d+조(?:의\d+)?\s*(?:\([^)]*\))?\s*삭제")


class LawAPIError(RuntimeError):
    pass


def _get(path: str, params: dict) -> dict:
    if not LAW_OC:
        raise LawAPIError(".env에 LAW_OC가 없습니다 (open.law.go.kr에서 Open API 신청 후 발급)")
    r = requests.get(f"{BASE}/{path}", params={"OC": LAW_OC, "target": "law", "type": "JSON", **params}, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    try:
        data = r.json()
    except json.JSONDecodeError:
        raise LawAPIError(f"JSON이 아닌 응답입니다. OC 값이나 활용 신청 승인 여부를 확인하세요.\n{r.text[:300]}")
    if isinstance(data, dict) and "result" in data and "msg" in data:  # 인증 실패도 HTTP 200으로 온다
        raise LawAPIError(f"법령 API 오류: {data.get('msg')}")
    return data


def as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(t for t in (flatten(v) for v in value) if t)
    return str(value)


def clean(value: Any) -> str:
    s = AMEND_TAG.sub("", flatten(value))
    lines = [re.sub(r"[ \t　]+", " ", ln).strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ──────────────────────────────────────────────────────────────


def search_law(name: str) -> dict | None:
    """법령명 → 검색 결과 1건 (이름이 정확히 같은 것 우선). 없으면 None."""
    res = _get("lawSearch.do", {"query": name, "display": 20})
    items = as_list(res.get("LawSearch", {}).get("law"))
    if not items:
        return None
    key = name.replace(" ", "")
    exact = [it for it in items if it.get("법령명한글", "").replace(" ", "") == key]
    return (exact or items)[0]


def fetch_law(mst: str) -> dict:
    """법령일련번호(MST) → 본문 JSON. 디스크 캐시 사용."""
    path = LAW_CACHE_DIR / f"{mst}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get("lawService.do", {"MST": mst})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def parse_articles(data: dict) -> list[dict]:
    """본문 JSON → 조문 리스트. 1조문 = 1검색 단위."""
    law = data["법령"]
    info = law.get("기본정보", {})
    law_name = info.get("법령명_한글") or info.get("법령명한글") or ""
    effective = info.get("시행일자", "")
    out = []
    for u in as_list(law.get("조문", {}).get("조문단위")):
        if u.get("조문여부") == "전문":
            continue
        head = clean(u.get("조문내용"))
        if DELETED.match(head):
            continue
        label = f"제{u.get('조문번호')}조" + (f"의{u['조문가지번호']}" if u.get("조문가지번호") else "")
        lines = [head]
        for hang in as_list(u.get("항")):
            lines.append(clean(hang.get("항내용")))
            for ho in as_list(hang.get("호")):
                lines.append(clean(ho.get("호내용")))
                for mok in as_list(ho.get("목")):
                    lines.append(clean(mok.get("목내용")))
        text = "\n".join(ln for ln in lines if ln)
        title = clean(u.get("조문제목"))
        out.append({
            "id": f"{law_name}|{label}",
            "law": law_name,
            "label": label,
            "title": title,
            "source": f"{law_name} {label}" + (f"({title})" if title else ""),
            "text": text,
            "effective": effective,
            "url": f"https://www.law.go.kr/법령/{law_name.replace(' ', '')}/{label}",
        })
    return out


def load_law(name: str) -> tuple[str, list[dict]]:
    """법령명 → (캐시 키, 조문 리스트). 못 찾으면 (\"\", [])."""
    hit = search_law(name)
    if not hit:
        return "", []
    mst = str(hit.get("법령일련번호"))
    return mst, parse_articles(fetch_law(mst))


if __name__ == "__main__":  # python -m upstage_rag.law_api 소득세법
    name = " ".join(sys.argv[1:]) or "소득세법"
    mst, arts = load_law(name)
    print(f"{name}: MST={mst}, 조문 {len(arts)}개")
    for a in arts[:3]:
        print(f"\n[{a['source']}]\n{a['text'][:300]}")
