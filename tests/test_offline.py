"""API 호출 없이 도는 테스트: 법령 JSON 파싱, 근거 조립, 후처리."""

from upstage_rag import answer, law_api

SAMPLE = {
    "법령": {
        "기본정보": {"법령명_한글": "소득세법", "시행일자": "20260101"},
        "조문": {
            "조문단위": [
                {"조문여부": "전문", "조문내용": "제1장 총칙"},
                {
                    "조문여부": "조문", "조문번호": "127", "조문제목": "원천징수의무",
                    "조문내용": "제127조(원천징수의무) <개정 2020.12.29>",
                    "항": {  # 1건이면 dict로 온다
                        "항내용": "① 국내에서 거주자에게 다음 각 호의 소득을 지급하는 자는 소득세를 원천징수하여야 한다.",
                        "호": [{"호내용": "3. 원천징수대상 사업소득"}, {"호내용": ["4. 근로소득", ""]}],
                    },
                },
                {"조문여부": "조문", "조문번호": "128", "조문내용": "제128조 삭제 <2010.12.27>"},
                {"조문여부": "조문", "조문번호": "129", "조문가지번호": "2", "조문내용": "제129조의2(예시) 본문"},
            ]
        },
    }
}


def test_parse_articles():
    arts = law_api.parse_articles(SAMPLE)
    assert [a["label"] for a in arts] == ["제127조", "제129조의2"]  # 장 제목·삭제 조문 제외
    a = arts[0]
    assert a["source"] == "소득세법 제127조(원천징수의무)"
    assert "<개정" not in a["text"]
    assert "원천징수대상 사업소득" in a["text"] and "4. 근로소득" in a["text"]
    assert a["url"].endswith("/법령/소득세법/제127조")


HITS = [
    {"source": "소득세법 제127조(원천징수의무)", "url": "u1", "effective": "20260101", "text": "본문1"},
    {"source": "소득세법 제129조(원천징수세율)", "url": "u2", "effective": "20260101", "text": "본문2"},
]


def test_build_context_numbers():
    ctx = answer.build_context(HITS)
    assert ctx.startswith("[1] 소득세법 제127조") and "\n\n[2] 소득세법 제129조" in ctx


def test_postprocess_removes_bad_citations_and_adds_sources():
    text, cited = answer.postprocess("원천징수합니다 [1]. 세율은 3%입니다 [2]. 지어낸 근거 [7].", HITS, "3.3% 근거?")
    assert cited == [1, 2]
    assert "[7]" not in text
    assert "출처\n[1] 소득세법 제127조(원천징수의무) — u1" in text
    assert text.endswith(answer.DISCLAIMER)
    assert answer.ESCALATE not in text


def test_postprocess_escalates_high_risk():
    text, _ = answer.postprocess(answer.NO_EVIDENCE, [], "클라이언트가 대금을 안 줘서 소송하려고요")
    assert answer.ESCALATE in text and "출처" not in text
