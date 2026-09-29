#!/usr/bin/env python3
"""data/parsed/chunks.jsonl → Chroma 색인.

    python scripts/build_index.py              # 법령 단위로 교체(upsert)
    python scripts/build_index.py --rebuild    # 컬렉션 삭제 후 새로 생성
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from law_rag.config import CHUNKS_PATH  # noqa: E402
from law_rag.parse_law import load_documents  # noqa: E402
from law_rag.vectorstore import distance_space, get_vectorstore  # noqa: E402

META_TYPES = (str, int, float, bool)


def clean_metadata(meta: dict) -> dict:
    """Chroma는 str/int/float/bool만 허용. None은 빼고 그 외 타입은 문자열로."""
    return {k: (v if isinstance(v, META_TYPES) else str(v)) for k, v in meta.items() if v is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description="조문 청크 → Chroma 색인")
    ap.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    ap.add_argument("--rebuild", action="store_true", help="컬렉션 삭제 후 새로 생성")
    args = ap.parse_args()

    t0 = time.time()
    docs = load_documents(args.chunks)
    for d in docs:
        d.metadata = clean_metadata(d.metadata)

    vs = get_vectorstore()
    if args.rebuild:
        vs.delete_collection()
        vs = get_vectorstore()

    space = distance_space(vs)
    if space != "cosine":
        sys.exit(f"컬렉션 거리 함수가 {space}입니다. --rebuild로 다시 만드세요.")

    by_law: dict[str, list] = {}
    for d in docs:
        by_law.setdefault(d.metadata["law_id"], []).append(d)

    # 법령 단위 교체: 개정으로 조각 수가 줄어도 옛 청크가 남지 않게 먼저 지운다.
    for law_id, law_docs in by_law.items():
        vs.delete(where={"law_id": law_id})
        vs.add_documents(law_docs, ids=[d.id for d in law_docs])
        print(f"  {law_docs[0].metadata['law_name']} ({law_id}): {len(law_docs)}개", file=sys.stderr)

    got = vs._collection.get(include=["metadatas"])
    per_law = Counter(m["law_name"] for m in got["metadatas"])
    print(f"\n컬렉션 문서 수: {len(got['ids'])}")
    for name, n in per_law.most_common():
        print(f"  {name}: {n}")
    print(f"거리 함수: {distance_space(vs)}")
    print(f"소요 시간: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
