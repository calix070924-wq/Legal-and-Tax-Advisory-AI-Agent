"""후처리·근거 포맷 단위 테스트 (API 호출 없음)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from law_rag.config import ARTICLE_CHAR_LIMIT  # noqa: E402
from law_rag.context import Evidence, build_evidence, format_evidence, strip_context_header  # noqa: E402
from law_rag.postprocess import DISCLAIMER, postprocess  # noqa: E402


def ev(n, label="제21조", title="개인정보의 파기", eff="2026.09.11", future=""):
    return Evidence(n, "개인정보 보호법", label, title, eff, future,
                    f"https://www.law.go.kr/법령/개인정보보호법/{label}", "본문", 0.5)


EVIDENCE = [ev(1), ev(2, "제36조", "개인정보의 정정ㆍ삭제", "2025.10.02"), ev(3, "제39조의3", "특례", "2026.09.11")]


def test_out_of_range_citation_removed():
    p = postprocess("지체 없이 파기해야 합니다[1][9].", EVIDENCE)
    assert p.invalid_citations == [9]
    assert p.cited == [1]
    assert "[9]" not in p.answer


def test_consecutive_citations():
    p = postprocess("파기 [1][3] 그리고 정정 [2].", EVIDENCE)
    assert p.cited == [1, 2, 3]
    assert p.invalid_citations == []


def test_comma_citations_expanded():
    p = postprocess("둘 다 해당합니다[1, 3].", EVIDENCE)
    assert p.cited == [1, 3]
    assert "[1][3]" in p.answer


def test_no_citation_only_disclaimer():
    p = postprocess("제공된 법령 근거로는 확인하기 어렵습니다.", EVIDENCE)
    assert p.cited == []
    assert "출처" not in p.answer
    assert "시행 법령 기준" not in p.answer
    assert p.answer.endswith(DISCLAIMER)


def test_disclaimer_always_included():
    for raw in ["", "답[1]", "답[9]"]:
        assert DISCLAIMER in postprocess(raw, EVIDENCE).answer


def test_sources_only_cited_in_order():
    p = postprocess("정정[2], 파기[1].", EVIDENCE)
    src = p.answer.split("출처\n", 1)[1].split("\n\n")[0].splitlines()
    assert [s[:3] for s in src] == ["[1]", "[2]"]
    assert "제39조의3" not in p.answer


def test_base_date_is_latest_cited():
    assert "※ 이 답변은 2025.10.02 시행 법령 기준입니다." in postprocess("정정[2].", EVIDENCE).answer
    assert "※ 이 답변은 2026.09.11 시행 법령 기준입니다." in postprocess("[1][2]", EVIDENCE).answer


def test_format_includes_future_effective():
    e = ev(1, "제32조의2", "개인정보 보호 인증", future="2027.07.01 시행: 제32조의2제1항 단서")
    out = format_evidence([e, ev(2)])
    assert "[1] 개인정보 보호법 제32조의2(개인정보 보호 인증) | 시행 2026.09.11" in out
    assert "※ 아직 시행 전인 규정 포함 — 2027.07.01 시행: 제32조의2제1항 단서" in out
    assert out.count("※ 아직 시행 전인") == 1


def test_strip_context_header():
    t = "[개인정보 보호법 > 제4장] (시행 2026.09.11)\n※ 아직 시행 전인 규정 포함 — 2027.07.01 시행: x\n제32조의2(인증)\n  ① 본문"
    assert strip_context_header(t) == "제32조의2(인증)\n  ① 본문"


def _hit(uid, text="[법 > 장] (시행 2026.09.11)\n청크 본문"):
    meta = {"law_name": "개인정보 보호법", "article_label": uid, "article_title": "t",
            "effective_date": "20260911", "future_effective": "", "url": "u", "article_uid": uid}
    return Document(page_content=text, metadata=meta), 0.5


def test_build_evidence_expands_and_limits():
    articles = {"a": {"text": "조문 전체"}, "long": {"text": "x" * (ARTICLE_CHAR_LIMIT + 1)}}
    out = build_evidence([_hit("a"), _hit("long"), _hit("missing")], get_article=articles.__getitem__)
    assert [e.text for e in out] == ["조문 전체", "청크 본문", "청크 본문"]
    assert [e.n for e in out] == [1, 2, 3]
    assert out[0].effective_date == "2026.09.11"


def test_build_evidence_budget_drops_lower_ranks(monkeypatch):
    import law_rag.context as ctx

    monkeypatch.setattr(ctx, "CONTEXT_CHAR_BUDGET", 10)
    articles = {"a": {"text": "x" * 6}, "b": {"text": "y" * 6}}
    out = build_evidence([_hit("a"), _hit("b")], get_article=articles.__getitem__)
    assert [e.article_label for e in out] == ["a"]
