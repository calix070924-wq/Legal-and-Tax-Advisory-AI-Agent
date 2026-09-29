#!/usr/bin/env python3
"""임베딩 모델 비교: 법령 청크 검색 성능 (Recall@k, MRR@10) + BM25 / 하이브리드(RRF).

준비
----
    pip install sentence-transformers faiss-cpu kiwipiepy rank-bm25
    python -m law_rag.parse_law data/raw/law/*.json   # data/parsed/chunks.jsonl 생성

실행
----
    python eval/eval_embed.py                           # 기본 후보 4개 + BM25 + 하이브리드
    python eval/eval_embed.py --models dragonkue/snowflake-arctic-embed-l-v2.0-ko nlpai-lab/KURE-v1
    python eval/eval_embed.py --misses                  # 모델별로 틀린 질문 출력

평가 단위는 "조문"이다. 청크 검색 결과를 조문 단위로 중복 제거한 뒤,
정답 조문(gold) 중 하나라도 top-k 안에 있으면 적중으로 본다.
임베딩은 data/emb/<모델>.npy 에 캐시되어, 나중에 FAISS 색인을 만들 때 재사용할 수 있다.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from law_rag.config import CHUNKS_PATH, EMB_DIR  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko",  # 쿼리에 "query: " 접두어 (prompt_name="query")
    "nlpai-lab/KURE-v1",                           # bge-m3 한국어 파인튜닝
    "dragonkue/BGE-m3-ko",                         # bge-m3 한국어 파인튜닝
    "BAAI/bge-m3",                                 # 베이스라인
]
KS = (1, 3, 5, 10)


# ──────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_articles(chunk_ids: list[int], chunks: list[dict], k: int = 10) -> list[str]:
    """청크 순위 → 조문 라벨 순위 (중복 제거)."""
    seen, out = set(), []
    for i in chunk_ids:
        label = chunks[i]["metadata"]["article_label"]
        if label not in seen:
            seen.add(label)
            out.append(label)
            if len(out) == k:
                break
    return out


def score(ranked: list[list[str]], evalset: list[dict]) -> dict:
    res = {f"R@{k}": 0.0 for k in KS}
    mrr = 0.0
    for r, ex in zip(ranked, evalset):
        gold = set(ex["gold"])
        for k in KS:
            res[f"R@{k}"] += any(a in gold for a in r[:k])
        rank = next((i for i, a in enumerate(r[:10], 1) if a in gold), None)
        mrr += 1 / rank if rank else 0
    n = len(evalset)
    res = {k: v / n for k, v in res.items()}
    res["MRR@10"] = mrr / n
    return res


# ──────────────────────────────────────────────────────────────
# 검색기
# ──────────────────────────────────────────────────────────────


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dense_rankings(model_name: str, chunks, evalset, batch: int, max_len: int, emb_dir: Path):
    import faiss
    from sentence_transformers import SentenceTransformer

    device = pick_device()
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_len
    load_s = time.time() - t0

    cache = emb_dir / (model_name.replace("/", "__") + ".npy")
    t0 = time.time()
    if cache.exists() and np.load(cache).shape[0] == len(chunks):
        doc_emb = np.load(cache)
    else:
        doc_emb = model.encode(
            [c["text"] for c in chunks], batch_size=batch,
            normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
        ).astype("float32")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, doc_emb)
    doc_s = time.time() - t0

    # 모델 설정에 query 프롬프트가 있으면 사용 (arctic 계열: "query: ")
    q_kwargs = {"prompt_name": "query"} if "query" in (model.prompts or {}) else {}
    t0 = time.time()
    q_emb = model.encode(
        [e["q"] for e in evalset], batch_size=batch,
        normalize_embeddings=True, convert_to_numpy=True, **q_kwargs,
    ).astype("float32")
    q_ms = (time.time() - t0) / len(evalset) * 1000

    index = faiss.IndexFlatIP(doc_emb.shape[1])  # 정규화된 벡터의 내적 = 코사인 유사도
    index.add(doc_emb)
    _, ids = index.search(q_emb, 50)

    info = {
        "device": device, "dim": doc_emb.shape[1], "load_s": round(load_s, 1),
        "doc_encode_s": round(doc_s, 1), "query_ms": round(q_ms, 1),
        "query_prompt": bool(q_kwargs),
    }
    del model
    gc.collect()
    try:
        import torch

        if device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass
    return [list(row) for row in ids], info


def bm25_rankings(chunks, evalset):
    from kiwipiepy import Kiwi
    from rank_bm25 import BM25Okapi

    kiwi = Kiwi()
    keep = ("NN", "VV", "VA", "XR", "SL", "SN", "SH")  # 명사·동사·형용사 어간·어근·외국어·숫자·한자

    def tok(text: str) -> list[str]:
        return [t.form for t in kiwi.tokenize(text) if t.tag.startswith(keep)]

    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    out = []
    for e in evalset:
        s = bm25.get_scores(tok(e["q"]))
        out.append(list(np.argsort(-s)[:50]))
    return out


def rrf(*rankings: list[list[int]], k: int = 60) -> list[list[int]]:
    """Reciprocal Rank Fusion: 여러 검색기의 순위를 점수 없이 합친다."""
    fused = []
    for per_query in zip(*rankings):
        s: dict[int, float] = {}
        for ranking in per_query:
            for r, doc in enumerate(ranking):
                s[doc] = s.get(doc, 0) + 1 / (k + r + 1)
        fused.append(sorted(s, key=s.get, reverse=True))
    return fused


# ──────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    ap.add_argument("--evalset", type=Path, default=EVAL_DIR / "eval_set.jsonl")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024, help="토큰 상한 (청크 최대 ~1600자라 1024면 충분)")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--misses", action="store_true", help="틀린 질문 출력")
    args = ap.parse_args()

    chunks = load_jsonl(args.chunks)
    evalset = load_jsonl(args.evalset)
    print(f"청크 {len(chunks)}개, 평가 질문 {len(evalset)}개\n")

    results: dict[str, dict] = {}
    rankings: dict[str, list[list[str]]] = {}

    bm25_ids = None
    if not args.no_bm25:
        bm25_ids = bm25_rankings(chunks, evalset)
        rankings["BM25 (kiwi)"] = [to_articles(r, chunks) for r in bm25_ids]
        results["BM25 (kiwi)"] = score(rankings["BM25 (kiwi)"], evalset)

    for name in args.models:
        print(f"▶ {name}")
        try:
            ids, info = dense_rankings(name, chunks, evalset, args.batch, args.max_len, EMB_DIR)
        except Exception as e:  # 모델 다운로드 실패 등
            print(f"  실패: {e}\n")
            continue
        print(f"  {info}\n")
        short = name.split("/")[-1]
        rankings[short] = [to_articles(r, chunks) for r in ids]
        results[short] = {**score(rankings[short], evalset), **info}
        if bm25_ids is not None:
            hyb = f"{short} + BM25"
            rankings[hyb] = [to_articles(r, chunks) for r in rrf(ids, bm25_ids)]
            results[hyb] = score(rankings[hyb], evalset)

    cols = [f"R@{k}" for k in KS] + ["MRR@10"]
    width = max(len(n) for n in results) + 2
    print("\n" + "모델".ljust(width) + "".join(c.rjust(9) for c in cols) + "   문서인코딩(s)  쿼리(ms)")
    print("─" * (width + 9 * len(cols) + 26))
    for name, r in sorted(results.items(), key=lambda x: -x[1]["MRR@10"]):
        extra = f"   {r['doc_encode_s']:>10}  {r['query_ms']:>8}" if "doc_encode_s" in r else ""
        print(name.ljust(width) + "".join(f"{r[c]:9.3f}" for c in cols) + extra)

    out = EVAL_DIR / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")

    if args.misses:
        for name, ranked in rankings.items():
            miss = [(e, r) for e, r in zip(evalset, ranked) if not set(e["gold"]) & set(r[:5])]
            if not miss:
                continue
            print(f"\n■ {name} — top-5 밖 {len(miss)}건")
            for e, r in miss:
                print(f"  Q: {e['q']}\n     정답 {e['gold']}  /  검색 {r[:5]}")


if __name__ == "__main__":
    main()
