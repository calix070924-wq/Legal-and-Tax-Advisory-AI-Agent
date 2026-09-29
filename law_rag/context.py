"""근거 조립: 검색 → 조문 전체로 확장(small-to-big) → 근거 번호 [1]~[n] 포맷."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from law_rag.config import ARTICLE_CHAR_LIMIT, CONTEXT_CHAR_BUDGET, RETRIEVE_K
from law_rag.parse_law import fmt_date

FUTURE_PREFIX = "※ 아직 시행 전인 규정 포함"


@dataclass
class Evidence:
    n: int
    law_name: str
    article_label: str
    article_title: str
    effective_date: str  # "2026.09.11"
    future_effective: str  # "2027.07.01 시행: 제32조의2제1항 단서" 또는 ""
    url: str
    text: str
    score: float  # 코사인 거리 (작을수록 가까움)

    @property
    def title(self) -> str:
        return f"{self.law_name} {self.article_label}({self.article_title})"

    def to_dict(self) -> dict:
        return asdict(self)


def strip_context_header(chunk_text: str) -> str:
    """청크 맨 앞의 문맥 헤더([법령 > 장] (시행 …), ※ 미시행 안내)를 뺀다. 번호 줄에 같은 정보가 있다."""
    lines = chunk_text.split("\n")
    if lines and lines[0].startswith("["):
        lines = lines[1:]
    if lines and lines[0].startswith(FUTURE_PREFIX):
        lines = lines[1:]
    return "\n".join(lines)


def build_evidence(hits, get_article=None) -> list[Evidence]:
    """search() 결과 [(Document, score)] → 번호 붙은 근거 목록.

    조문 전체가 ARTICLE_CHAR_LIMIT을 넘으면 검색된 청크만 쓰고,
    누적 글자 수가 CONTEXT_CHAR_BUDGET을 넘으면 낮은 순위부터 뺀다.
    """
    if get_article is None:
        from law_rag.vectorstore import get_article

    out: list[Evidence] = []
    used = 0
    for doc, score in hits:
        m = doc.metadata
        text = strip_context_header(doc.page_content)
        try:
            full = get_article(m["article_uid"])["text"]
            if len(full) <= ARTICLE_CHAR_LIMIT:
                text = full
        except KeyError:
            pass  # articles.jsonl에 없으면 청크 텍스트 사용
        if used + len(text) > CONTEXT_CHAR_BUDGET:
            break
        used += len(text)
        out.append(
            Evidence(
                n=len(out) + 1,
                law_name=m.get("law_name", ""),
                article_label=m.get("article_label", ""),
                article_title=m.get("article_title", ""),
                effective_date=fmt_date(m.get("article_effective_date") or m.get("effective_date")),
                future_effective=m.get("future_effective", ""),
                url=m.get("url", ""),
                text=text,
                score=float(score),
            )
        )
    return out


def retrieve_evidence(question: str, k: int = RETRIEVE_K) -> list[Evidence]:
    from law_rag.vectorstore import search

    return build_evidence(search(question, k=k))


def format_evidence(evidence: list[Evidence]) -> str:
    blocks = []
    for e in evidence:
        head = f"[{e.n}] {e.title}"
        if e.effective_date:
            head += f" | 시행 {e.effective_date}"
        if e.future_effective:
            head += f"\n{FUTURE_PREFIX} — {e.future_effective}"
        blocks.append(f"{head}\n{e.text}")
    return "\n\n".join(blocks)
