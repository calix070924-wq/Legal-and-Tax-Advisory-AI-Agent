"""LLM이 빠뜨려도 항상 나가야 하는 부분: 인용 검증, 출처 목록, 기준일, 면책 문구."""

from __future__ import annotations

import re
from dataclasses import dataclass

from law_rag.context import Evidence

DISCLAIMER = "※ 이 답변은 법령 정보를 안내하는 참고용이며 법률·세무 자문이 아닙니다. 구체적인 사안은 변호사·세무사와 상담하세요."

# [1], [12], [1, 3] 형태. [1, 3]은 [1][3]으로 풀어 쓴다.
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Postprocessed:
    answer: str
    cited: list[int]
    invalid_citations: list[int]


def check_citations(raw: str, n_evidence: int) -> tuple[str, list[int], list[int]]:
    """범위 밖 번호는 지우고, (정리된 답변, cited, invalid) 반환."""
    cited: set[int] = set()
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        keep = []
        for s in m.group(1).split(","):
            i = int(s)
            if 1 <= i <= n_evidence:
                cited.add(i)
                keep.append(f"[{i}]")
            elif i not in invalid:
                invalid.append(i)
        return "".join(keep)

    text = CITATION_RE.sub(repl, raw)
    text = re.sub(r"[ \t]+([.,。])", r"\1", text)  # 인용 제거 후 남은 "문장 ." 정리
    return text, sorted(cited), invalid


def source_lines(evidence: list[Evidence], cited: list[int]) -> list[str]:
    by_n = {e.n: e for e in evidence}
    return [
        f"[{n}] {by_n[n].title} — 시행 {by_n[n].effective_date} — {by_n[n].url}"
        for n in cited
        if n in by_n
    ]


def base_date(evidence: list[Evidence], cited: list[int]) -> str:
    """인용된 근거 중 가장 최근 시행일 (YYYY.MM.DD는 문자열 비교로 정렬됨)."""
    dates = [e.effective_date for e in evidence if e.n in cited and e.effective_date]
    return max(dates) if dates else ""


def postprocess(raw: str, evidence: list[Evidence]) -> Postprocessed:
    text, cited, invalid = check_citations(raw, len(evidence))
    parts = [text.strip()]

    sources = source_lines(evidence, cited)
    if sources:
        parts.append("출처\n" + "\n".join(sources))
    date = base_date(evidence, cited)
    if date:
        parts.append(f"※ 이 답변은 {date} 시행 법령 기준입니다.")
    parts.append(DISCLAIMER)

    return Postprocessed("\n\n".join(parts), cited, invalid)
