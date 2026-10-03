"""공공 API 기반 Retriever.

질문 → ① Solar가 관련 법령명 고르기 → ② 법제처 API로 법령 조문 가져오기
     → ③ Upstage Embedding으로 조문·질문 임베딩 → ④ 코사인 유사도 Top-k

미리 만든 벡터 DB 없이, 질문에 필요한 법령만 그때그때 가져온다.
대신 법령 원문과 조문 임베딩은 캐시해서 같은 법령은 다시 받거나 임베딩하지 않는다.
"""

from __future__ import annotations

import json
import re

import numpy as np

from upstage_rag import config, law_api, upstage

PLAN_PROMPT = """사용자 질문에 답하려면 어떤 한국 법령을 찾아봐야 하는지 고르세요.
- 정식 법령명만 쓰세요 (예: "소득세법", "부가가치세법 시행령").
- 아래 후보 중에서 고르되, 후보에 없는 법령이 꼭 필요하면 추가해도 됩니다.
- 최대 {max_laws}개, 관련 높은 순서로.
- 법률·세무와 무관한 질문이면 빈 배열.
- JSON 배열만 출력하세요. 설명 금지.

후보: {candidates}"""


def plan_laws(question: str) -> list[str]:
    """① 질문 → 찾아볼 법령명 목록."""
    text, _ = upstage.chat([
        {"role": "system", "content": PLAN_PROMPT.format(max_laws=config.MAX_LAWS, candidates=", ".join(config.DEFAULT_LAWS))},
        {"role": "user", "content": question},
    ])
    m = re.search(r"\[.*?\]", text, re.S)
    try:
        names = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        names = []
    return [n.strip() for n in names if isinstance(n, str) and n.strip()][: config.MAX_LAWS]


def law_index(name: str) -> tuple[list[dict], np.ndarray]:
    """② + ③ 법령명 → (조문 리스트, 조문 임베딩). 임베딩은 MST(법령 버전) 단위로 캐시."""
    mst, articles = law_api.load_law(name)
    if not articles:
        return [], np.zeros((0, 0), dtype="float32")
    path = config.EMB_CACHE_DIR / f"{mst}.npy"
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(articles):
            return articles, emb
    # 조문 제목까지 넣어 임베딩하면 "원천징수" 같은 제목 키워드가 검색에 잡힌다
    emb = upstage.embed_passages([f"{a['source']}\n{a['text']}" for a in articles])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return articles, emb


def retrieve(question: str, laws: list[str] | None = None, k: int = config.TOP_K) -> dict:
    """질문 → {"laws": 찾아본 법령, "hits": Top-k 조문(점수 포함), "best": 최고 점수}."""
    laws = laws if laws is not None else plan_laws(question)
    pool, vecs = [], []
    for name in laws:
        arts, emb = law_index(name)
        if arts:
            pool += arts
            vecs.append(emb)
    if not pool:
        return {"laws": laws, "hits": [], "best": 0.0}

    q = upstage.embed_query(question)  # ③ 질문은 query 모델로
    scores = np.vstack(vecs) @ q  # ④ 정규화된 벡터의 내적 = 코사인 유사도
    order = np.argsort(-scores)[:k]
    hits = [{**pool[i], "score": float(scores[i])} for i in order]
    return {"laws": laws, "hits": hits, "best": hits[0]["score"]}


if __name__ == "__main__":  # python -m upstage_rag.retriever "질문"
    import sys

    r = retrieve(" ".join(sys.argv[1:]) or "프리랜서 외주 대금에서 3.3% 떼는 근거가 뭐야?")
    print("찾아본 법령:", r["laws"])
    for h in r["hits"]:
        print(f"{h['score']:.3f}  {h['source']}")
