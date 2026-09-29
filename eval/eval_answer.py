#!/usr/bin/env python3
"""답변 평가: eval/answer_set.jsonl 15문항을 체인으로 돌려 자동 체크.

    python eval/eval_answer.py
    → eval/results_answer.jsonl (문항별), eval/answer_report.md (보고서)

체크
----
형식        in_scope: 인용 1개 이상
인용 유효성  전체: invalid_citations 없음
인용 적중    in_scope: gold 조문 중 하나 이상이 cited에 포함
근거 부족    out_of_scope·injection: 근거에 없는 조문 번호를 쓰지 않고 "확인하기 어렵" 문구 포함
조문 지어내기 없음  전체: 답변에 나온 조문 번호가 모두 근거(제목·본문)에 있음
면책        전체: 면책 문구 포함
(참고) 시행일  cited 근거에 미시행 규정이 있으면 답변에 그 시행 연도가 나오는지

합격 기준: 인용 유효성·면책 100%, 인용 적중 in_scope 12개 중 10개 이상, 범위 밖 2개 근거 부족 통과,
          조문 지어내기 없음 100% (지시서 기준에 추가)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from law_rag.chain import answer  # noqa: E402
from law_rag.config import SOLAR_MODEL  # noqa: E402
from law_rag.postprocess import DISCLAIMER  # noqa: E402

ARTICLE_RE = re.compile(r"제\d+조(?:의\d+)?")
NO_EVIDENCE_RE = re.compile(r"확인하기\s*어렵")
CHECKS = ["형식", "인용 유효성", "인용 적중", "근거 부족", "조문 지어내기 없음", "면책", "(참고) 시행일"]


def run_checks(item: dict, r) -> dict:
    t = item["type"]
    # 근거 본문이 다른 조문을 참조("제15조제2항에 따라")하면 그 번호도 근거 안에 있는 것으로 본다
    known = {e.article_label for e in r.evidence} | {a for e in r.evidence for a in ARTICLE_RE.findall(e.text)}
    cited_labels = {e.article_label for e in r.evidence if e.n in r.cited}
    mentioned = set(ARTICLE_RE.findall(r.raw_answer))
    c: dict[str, bool | None] = {k: None for k in CHECKS}  # None = 해당 없음

    c["인용 유효성"] = not r.invalid_citations
    c["면책"] = DISCLAIMER in r.answer
    c["조문 지어내기 없음"] = mentioned <= known
    if t == "in_scope":
        c["형식"] = len(r.cited) >= 1
        c["인용 적중"] = bool(set(item["gold"]) & cited_labels)
        future = [e.future_effective for e in r.evidence if e.n in r.cited and e.future_effective]
        if future:
            years = {y for f in future for y in re.findall(r"(20\d\d)\.", f)}
            c["(참고) 시행일"] = any(y in r.raw_answer for y in years)
    else:
        c["근거 부족"] = mentioned <= known and bool(NO_EVIDENCE_RE.search(r.raw_answer))
    return c


def rate(rows: list[dict], name: str) -> tuple[int, int]:
    vals = [r["checks"][name] for r in rows if r["checks"][name] is not None]
    return sum(vals), len(vals)


def write_report(rows: list[dict], path: Path) -> None:
    n_in = [r for r in rows if r["type"] == "in_scope"]
    hit, _ = rate(n_in, "인용 적중")
    oos_ok = all(r["checks"]["근거 부족"] for r in rows if r["type"] == "out_of_scope")
    passed = (
        rate(rows, "인용 유효성")[0] == len(rows)
        and rate(rows, "면책")[0] == len(rows)
        and hit >= 10
        and oos_ok
        and rate(rows, "조문 지어내기 없음")[0] == len(rows)
    )
    lat = sum(r["latency_s"] for r in rows) / len(rows)
    tok_in = sum(r["usage"]["input_tokens"] for r in rows) / len(rows)
    tok_out = sum(r["usage"]["output_tokens"] for r in rows) / len(rows)

    lines = [
        "# 답변 평가 보고서",
        "",
        f"- 실행: {datetime.now():%Y-%m-%d %H:%M} / 모델: `{SOLAR_MODEL}` / 문항 {len(rows)}개 "
        f"(in_scope {len(n_in)}, out_of_scope 2, injection 1)",
        f"- 평균 응답 시간 {lat:.1f}s, 평균 토큰 입력 {tok_in:.0f} / 출력 {tok_out:.0f}",
        f"- **판정: {'합격' if passed else '불합격'}** "
        "(인용 유효성·면책·조문 지어내기 없음 100%, 인용 적중 10/12 이상)",
        "",
        "| 체크 | 대상 | 통과 | 통과율 |",
        "|---|---|---|---|",
    ]
    target = {"형식": "in_scope", "인용 유효성": "전체", "인용 적중": "in_scope",
              "근거 부족": "범위 밖·지시 무시", "조문 지어내기 없음": "전체", "면책": "전체", "(참고) 시행일": "미시행 규정 인용 시"}
    for name in CHECKS:
        ok, n = rate(rows, name)
        lines.append(f"| {name} | {target[name]} | {ok}/{n} | {ok / n:.0%} |" if n else f"| {name} | {target[name]} | - | - |")

    lines += [
        "",
        "## 한계",
        "",
        "- **미시행 규정 시행일(규칙 6) 미검증:** 색인의 미시행 규정은 제32조의2제1항 단서, 제75조제2항제15호"
        "(2027.07.01 시행) 두 개뿐이고, 평가 문항 중 이 조항을 묻는 문항이 없다. "
        "'(참고) 시행일' 체크는 해당 조문(제75조 등)이 인용되기만 해도 대상이 되므로, "
        "다른 항·호를 쓴 답변도 실패로 잡힌다(오탐). 합격 기준에는 넣지 않았다.",
        "- **제35조(열람) 검색 실패:** 일상어 질문이라 정답 조문이 top-5 밖이다. "
        "답변은 '확인하기 어렵다'로 처리되며, 다음 작업(질의 재작성)에서 다룬다.",
        "- **내용 정확성은 자동 체크 밖:** 인용 번호·조문 번호만 검사하고, 금액·요건을 근거와 맞게 옮겼는지는 보지 않는다. "
        "예: 2026-09-29 실행의 지시 무시 문항(#15)은 규칙 무시는 거절했지만, 제75조를 요약하면서 "
        "'개인정보 처리방침 미작성·미공개'를 3천만원 이하 항목에도 넣었다(실제로는 제4항제8호, 1천만원 이하). "
        "같은 내용을 직접 물은 #12는 1천만원 이하로 정확히 답했다.",
    ]
    lines += ["", "## 문항별 요약", "", "| # | 유형 | 질문 | 정답 | 인용 | 실패한 체크 |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        fails = [k for k, v in r["checks"].items() if v is False]
        lines.append(
            f"| {i} | {r['type']} | {r['q']} | {', '.join(r['gold']) or '-'} | "
            f"{', '.join(r['cited_labels']) or '-'} | {', '.join(fails) or '✅'} |"
        )

    lines += ["", "## 문항별 답변 전문", ""]
    for i, r in enumerate(rows, 1):
        lines += [
            f"### {i}. [{r['type']}] {r['q']}",
            "",
            f"- 정답: {', '.join(r['gold']) or '-'} / 검색된 근거: {', '.join(r['evidence_labels'])}",
            f"- 인용: {', '.join(r['cited_labels']) or '-'} / 잘못된 인용: {r['invalid_citations'] or '-'}",
            f"- {r['latency_s']}s, 입력 {r['usage']['input_tokens']} / 출력 {r['usage']['output_tokens']} 토큰",
            "",
            *[f"> {ln}" if ln else ">" for ln in r["answer"].splitlines()],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'합격' if passed else '불합격'} — 인용 적중 {hit}/{len(n_in)}")


def main() -> None:
    items = [json.loads(ln) for ln in open(EVAL_DIR / "answer_set.jsonl", encoding="utf-8") if ln.strip()]
    rows = []
    for i, item in enumerate(items, 1):
        r = answer(item["q"])
        checks = run_checks(item, r)
        rows.append({
            **item,
            "answer": r.answer,
            "raw_answer": r.raw_answer,
            "evidence_labels": [e.article_label for e in r.evidence],
            "cited": r.cited,
            "cited_labels": [e.article_label for e in r.evidence if e.n in r.cited],
            "invalid_citations": r.invalid_citations,
            "checks": checks,
            "latency_s": r.latency_s,
            "usage": r.usage,
        })
        fails = [k for k, v in checks.items() if v is False]
        print(f"[{i:>2}/{len(items)}] {item['type']:<12} {r.latency_s:>5.1f}s  {'FAIL ' + ','.join(fails) if fails else 'ok'}  {item['q']}")

    with open(EVAL_DIR / "results_answer.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name in CHECKS:
        ok, n = rate(rows, name)
        print(f"  {name:<10} {ok}/{n}" if n else f"  {name:<10} -")
    write_report(rows, EVAL_DIR / "answer_report.md")


if __name__ == "__main__":
    main()
