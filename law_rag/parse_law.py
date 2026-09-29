#!/usr/bin/env python3
"""법제처 현행법령 본문 JSON(lawService.do, target=law) → 조문 단위 RAG 청크.

입력:  data/raw/law/*.json   (python -m law_rag.fetch_law --detail 로 받은 파일)
출력:  data/parsed/articles.jsonl  조문 전체 1줄 = 1조문 (답변 생성 시 원문 확장용)
       data/parsed/chunks.jsonl    임베딩/BM25 색인용 청크 (긴 조문은 항·호 단위로 분할)

사용 예
------
    python -m law_rag.parse_law data/raw/law/*.json
    python -m law_rag.parse_law data/raw/law/*.json --max-chars 1200 --out data/parsed
    python -m law_rag.parse_law data/raw/law/*.json --show 제15조     # 특정 조문 청크 미리보기

LangChain 에서 쓰기
------------------
    from law_rag.parse_law import load_documents
    docs = load_documents()   # 기본: data/parsed/chunks.jsonl   # list[Document]

처리하는 API 특이사항
--------------------
- 항/호/목이 1건이면 list 대신 dict 로 온다 → 항상 list 로 정규화
- 항번호 없이 {"호": [...]} 만 있는 항 (예: 제2조 정의) → 조문 본문 바로 아래 호로 처리
- 조문여부 == "전문" 은 장/절 제목 → 청크로 만들지 않고 이후 조문의 chapter/section 메타데이터로 사용
- "제8조 삭제 <2020.2.4>", "3. 삭제 <...>" 같은 삭제 조문·호는 제외
- 조문가지번호: 제7조의10 처럼 "의" 조문 표기
- 기본정보.조문시행일자문자열 (예: "20270701:제32조의2제1항 단서,...") → 아직 시행 전인 부분이 있는 조문에 표시
- 텍스트 필드가 문자열/리스트/중첩 리스트로 섞여 올 수 있음 → 평탄화
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from law_rag.config import CHUNKS_PATH, PARSED_DIR

# ──────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────

AMEND_TAG = re.compile(r"\s*<(?:개정|신설|전문개정|제목개정|타법개정|본조신설)[^>]*>")
DELETED = re.compile(r"^(?:제\d+조(?:의\d+)?|[^\s]{1,6})\s*삭제\s*(?:<[^>]*>)?\s*$")
CHAPTER = re.compile(r"^제\d+(?:의\d+)?장\b")
SECTION = re.compile(r"^제\d+(?:의\d+)?절\b")
ARTICLE_REF = re.compile(r"제\d+조(?:의\d+)?")


def as_list(value: Any) -> list:
    """API가 1건이면 dict, 여러 건이면 list로 주는 것을 보정."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def flatten_text(value: Any) -> str:
    """문자열 / 리스트 / 중첩 리스트를 줄바꿈으로 이어 붙인다."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(t for t in (flatten_text(v) for v in value) if t)
    return str(value)


def clean(text: Any, keep_amend_tags: bool = False) -> str:
    """공백 정리 + (기본) <개정 ...> 류 이력 태그 제거."""
    s = flatten_text(text)
    if not keep_amend_tags:
        s = AMEND_TAG.sub("", s)
    lines = [re.sub(r"[ \t　]+", " ", ln).strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def is_deleted(text: str) -> bool:
    return bool(DELETED.match(text.strip()))


def article_label(no: str, branch: str | None) -> str:
    return f"제{no}조" + (f"의{branch}" if branch else "")


def law_url(law_name: str, label: str | None = None) -> str:
    """법제처 한글주소. 법령명은 공백 제거 형태를 사용."""
    base = f"https://www.law.go.kr/법령/{law_name.replace(' ', '')}"
    return f"{base}/{label}" if label else base


def fmt_date(yyyymmdd: str | None) -> str:
    s = (yyyymmdd or "").strip()
    return f"{s[:4]}.{s[4:6]}.{s[6:8]}" if len(s) == 8 and s.isdigit() else s


def parse_future_effective(raw: str | None) -> dict[str, list[str]]:
    """'20270701:제32조의2제1항 단서,제75조제2항제15호' → {'제32조의2': ['2027.07.01 시행: 제32조의2제1항 단서'], ...}"""
    out: dict[str, list[str]] = {}
    if not raw:
        return out
    for date, targets in re.findall(r"(\d{8}):(.*?)(?=\|?\d{8}:|$)", raw):
        for part in re.split(r"[,|]", targets):
            part = part.strip()
            m = ARTICLE_REF.match(part)
            if m:
                out.setdefault(m.group(0), []).append(f"{fmt_date(date)} 시행: {part}")
    return out


# ──────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────


@dataclass
class Block:
    """분할 최소 단위. leads = 이 블록 위에 붙어야 하는 상위 문장들 (조문 첫 줄, 항 첫 줄)."""

    leads: tuple[str, ...]
    text: str


@dataclass
class Article:
    uid: str
    law_name: str
    label: str  # 제15조, 제7조의10
    title: str
    header: str  # "개인정보 보호법 제15조(개인정보의 수집ㆍ이용)"
    blocks: list[Block]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return render_blocks(self.blocks)


@dataclass
class Chunk:
    id: str
    text: str  # 임베딩/BM25에 들어가는 텍스트 (문맥 헤더 포함)
    metadata: dict[str, Any]


def render_blocks(blocks: Iterable[Block]) -> str:
    """블록들을 이어 쓰되, 같은 lead는 한 번만 출력."""
    lines: list[str] = []
    prev: tuple[str, ...] = ()
    for b in blocks:
        common = 0
        while common < min(len(prev), len(b.leads)) and prev[common] == b.leads[common]:
            common += 1
        for depth, lead in enumerate(b.leads[common:], start=common):
            lines.append(("  " * depth) + lead)
        lines.append(("  " * len(b.leads)) + b.text.replace("\n", "\n" + "  " * len(b.leads)))
        prev = b.leads
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 파서
# ──────────────────────────────────────────────────────────────


def _render_ho(ho: dict, keep_tags: bool) -> str | None:
    text = clean(ho.get("호내용"), keep_tags)
    if not text or is_deleted(text):
        return None
    lines = [text]
    for mok in as_list(ho.get("목")):
        mt = clean(mok.get("목내용"), keep_tags)
        if mt and not is_deleted(mt):
            lines.append("  " + mt.replace("\n", "\n  "))
    return "\n".join(lines)


def build_blocks(unit: dict, keep_tags: bool) -> list[Block]:
    body = clean(unit.get("조문내용"), keep_tags)
    blocks: list[Block] = []
    for hang in as_list(unit.get("항")):
        hang_text = clean(hang.get("항내용"), keep_tags)
        if hang_text and (is_deleted(hang_text) or hang_text in body):
            # 삭제된 항이거나, 항이 1개라 조문내용에 이미 들어있는 경우
            hang_text = ""
        hos = [t for t in (_render_ho(h, keep_tags) for h in as_list(hang.get("호"))) if t]

        if hos:
            leads = (body, hang_text) if hang_text else (body,)
            blocks += [Block(leads, t) for t in hos]
        elif hang_text:
            blocks.append(Block((body,), hang_text))

    if not blocks:  # 항이 없는 단문 조문 (예: 제1조)
        blocks.append(Block((), body))
    return blocks


def parse_law_json(data: dict, mst: str | None = None, keep_tags: bool = False) -> list[Article]:
    law = data.get("법령", data)
    info = law.get("기본정보", {})
    law_name = (info.get("법령명_한글") or info.get("법령명한글") or "").strip()
    law_id = info.get("법령ID", "")
    kind = info.get("법종구분")
    kind = kind.get("content") if isinstance(kind, dict) else kind
    ministry = info.get("소관부처")
    ministry = ministry.get("content") if isinstance(ministry, dict) else ministry
    future = parse_future_effective(info.get("조문시행일자문자열"))

    base_meta = {
        "source_type": "law",
        "law_name": law_name,
        "law_id": law_id,
        "mst": mst or "",
        "law_kind": kind or "",
        "ministry": ministry or "",
        "promulgation_date": info.get("공포일자", ""),
        "effective_date": info.get("시행일자", ""),
        "amend_type": info.get("제개정구분", ""),
    }

    chapter = section = ""
    articles: list[Article] = []
    for unit in as_list(law.get("조문", {}).get("조문단위")):
        body = clean(unit.get("조문내용"))
        if unit.get("조문여부") == "전문":
            if CHAPTER.match(body):
                chapter, section = body, ""
            elif SECTION.match(body):
                section = body
            continue
        if not body or is_deleted(body):
            continue

        no = str(unit.get("조문번호", "")).strip()
        branch = str(unit.get("조문가지번호", "") or "").strip() or None
        label = article_label(no, branch)
        title = (unit.get("조문제목") or "").strip()
        header = f"{law_name} {label}" + (f"({title})" if title else "")

        meta = {
            **base_meta,
            "article_no": int(no) if no.isdigit() else no,
            "article_branch": int(branch) if branch and branch.isdigit() else 0,
            "article_label": label,
            "article_title": title,
            "article_key": unit.get("조문키", ""),
            "article_effective_date": unit.get("조문시행일자", ""),
            "chapter": chapter,
            "section": section,
            "amend_history": unit.get("조문제개정일자문자열", "") or "",
            "reference_note": unit.get("조문참고자료", "") or "",
            "future_effective": " / ".join(future.get(label, [])),
            "url": law_url(law_name, label),
        }
        uid = f"{law_id or law_name}-{unit.get('조문키') or label}"
        articles.append(
            Article(uid, law_name, label, title, header, build_blocks(unit, keep_tags), meta)
        )
    return articles


def parse_addendum(data: dict, mst: str | None = None) -> Article | None:
    """가장 최근 부칙 1건 (시행일·적용례 질문 대응용)."""
    law = data.get("법령", data)
    info = law.get("기본정보", {})
    units = as_list(law.get("부칙", {}).get("부칙단위"))
    if not units:
        return None
    latest = max(units, key=lambda u: u.get("부칙공포일자", ""))
    text = clean(latest.get("부칙내용"))
    if not text:
        return None
    name = (info.get("법령명_한글") or "").strip()
    label = f"부칙({fmt_date(latest.get('부칙공포일자'))})"
    meta = {
        "source_type": "law_addendum",
        "law_name": name,
        "law_id": info.get("법령ID", ""),
        "mst": mst or "",
        "article_label": label,
        "article_title": "부칙",
        "effective_date": info.get("시행일자", ""),
        "promulgation_date": latest.get("부칙공포일자", ""),
        "url": law_url(name),
    }
    lines = text.splitlines()
    blocks = [Block((lines[0],), ln) for ln in lines[1:]] or [Block((), text)]
    return Article(f"{info.get('법령ID', name)}-addendum-{latest.get('부칙키', '')}",
                   name, label, "부칙", f"{name} {label}", blocks, meta)


# ──────────────────────────────────────────────────────────────
# 청킹
# ──────────────────────────────────────────────────────────────


def context_header(a: Article) -> str:
    """임베딩 품질을 위해 청크 맨 앞에 붙이는 한 줄 문맥."""
    path = " > ".join(p for p in (a.law_name, a.meta.get("chapter"), a.meta.get("section")) if p)
    eff = fmt_date(a.meta.get("article_effective_date") or a.meta.get("effective_date"))
    head = f"[{path}] (시행 {eff})" if eff else f"[{path}]"
    if a.meta.get("future_effective"):
        head += f"\n※ 아직 시행 전인 규정 포함 — {a.meta['future_effective']}"
    return head


def chunk_article(a: Article, max_chars: int = 1500) -> list[Chunk]:
    head = context_header(a)
    full = a.full_text
    parts: list[list[Block]] = []
    if len(full) <= max_chars:
        parts = [a.blocks]
    else:
        cur: list[Block] = []
        for b in a.blocks:
            if cur and len(render_blocks(cur + [b])) > max_chars:
                parts.append(cur)
                cur = []
            cur.append(b)
        if cur:
            parts.append(cur)

    chunks = []
    for i, blocks in enumerate(parts, 1):
        body = render_blocks(blocks)
        meta = {**a.meta, "article_uid": a.uid, "part": i, "parts": len(parts)}
        chunks.append(Chunk(f"{a.uid}-p{i}", f"{head}\n{body}", meta))
    return chunks


# ──────────────────────────────────────────────────────────────
# 입출력
# ──────────────────────────────────────────────────────────────


def mst_from_filename(path: Path) -> str | None:
    m = re.search(r"_(\d+)\.json$", path.name)
    return m.group(1) if m else None


def parse_files(paths: Iterable[Path], max_chars: int, keep_tags: bool, addendum: bool):
    articles: list[Article] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if "법령" not in data:
            print(f"건너뜀 (법령 본문 JSON 아님): {p}", file=sys.stderr)
            continue
        mst = mst_from_filename(Path(p))
        arts = parse_law_json(data, mst, keep_tags)
        if addendum and (ad := parse_addendum(data, mst)):
            arts.append(ad)
        print(f"{Path(p).name}: 조문 {len(arts)}개", file=sys.stderr)
        articles += arts
    chunks = [c for a in articles for c in chunk_article(a, max_chars)]
    return articles, chunks


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {path}", file=sys.stderr)


def load_documents(chunks_path: str | Path = CHUNKS_PATH):
    """chunks.jsonl → LangChain Document 리스트 (langchain-core 필요)."""
    from langchain_core.documents import Document

    docs = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append(Document(page_content=r["text"], metadata=r["metadata"], id=r["id"]))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description="법령 본문 JSON → 조문 청크 JSONL")
    ap.add_argument("files", nargs="+", type=Path, help="data/raw/law/*.json")
    ap.add_argument("--out", type=Path, default=PARSED_DIR)
    ap.add_argument("--max-chars", type=int, default=1500, help="청크 최대 글자 수 (초과 시 항·호 단위 분할)")
    ap.add_argument("--keep-amend-tags", action="store_true", help="<개정 2020.2.4> 같은 이력 태그 유지")
    ap.add_argument("--no-addendum", action="store_true", help="최신 부칙 청크 제외")
    ap.add_argument("--show", help="해당 조문(예: 제15조)의 청크를 출력")
    args = ap.parse_args()

    articles, chunks = parse_files(args.files, args.max_chars, args.keep_amend_tags, not args.no_addendum)

    write_jsonl(
        ({"id": a.uid, "header": a.header, "text": a.full_text, "metadata": a.meta} for a in articles),
        args.out / "articles.jsonl",
    )
    write_jsonl((asdict(c) for c in chunks), args.out / "chunks.jsonl")

    lens = sorted(len(c.text) for c in chunks)
    split = sum(1 for a in articles if len(chunk_article(a, args.max_chars)) > 1)
    print(
        f"\n조문 {len(articles)}개 → 청크 {len(chunks)}개 (분할된 조문 {split}개)\n"
        f"청크 길이: 최소 {lens[0]} / 중앙 {lens[len(lens)//2]} / 최대 {lens[-1]}자",
        file=sys.stderr,
    )

    if args.show:
        for c in chunks:
            if c.metadata["article_label"] == args.show:
                print(f"\n── {c.id} ──\n{c.text}")


if __name__ == "__main__":
    main()
