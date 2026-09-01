"""검색 성능 측정. 지표 선정 근거는 docs/adr/ADR-000-evaluation-design.md.

    python -m scripts.evaluate --index db_ada
    python -m scripts.evaluate --index db_3small --hybrid
    python -m scripts.evaluate --index db_3small --rerank --out docs/eval-rerank.json
"""

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CUTS = (5, 10, 20)
PRIMARY_CUT = 5  # 의도별 분해와 요약에 쓰는 기준

# 인덱스 빌드에 쓴 모델과 일치 필수 - 불일치 시 좌표계가 달라 검색 무의미
INDEX_MODEL = {
    "db_ada": "text-embedding-ada-002",
    "db_3small": "text-embedding-3-small",
    "db_smoke": "text-embedding-3-small",
}

# 논문이 쓴 모델. baseline 재현용 기본값
DEFAULT_RERANKER = "BAAI/bge-reranker-base"

# rank_bm25 기본 전처리는 공백 분리 - 조사 붙은 어절이 별개 토큰 ("감염의" != "감염")
# 끝 글자만 떼면 "정의" -> "정" 이 되므로 3글자 이상만 적용. 활용형은 미처리
_JOSA = re.compile(r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만|까지|부터)$")
_TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")


def ko_tokenize(text):
    return [_JOSA.sub("", t) if len(t) >= 3 else t for t in _TOKEN.findall(text)]


def load_golden(path, limit=None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


def is_hit(meta, gold):
    return (meta.get("disease_name") == gold["disease_name"]
            and meta.get("intention") == gold["intention"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="db_ada")
    ap.add_argument("--golden", default=str(ROOT / "data" / "goldenset.jsonl"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--bm25-weight", type=float, default=0.25)
    ap.add_argument("--bm25-tokenizer", choices=["ko", "whitespace"], default="ko")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = INDEX_MODEL.get(args.index)
    if not model:
        raise SystemExit(f"{args.index} 의 임베딩 모델을 모릅니다. INDEX_MODEL에 추가하세요.")

    golden = load_golden(Path(args.golden), args.limit)
    vs = FAISS.load_local(
        str(ROOT / args.index),
        OpenAIEmbeddings(model=model),
        allow_dangerous_deserialization=True,
    )
    print(f"골든셋 {len(golden)}건 | {args.index} ({model}) | 문서 {vs.index.ntotal} | k={args.k}")

    retriever = vs.as_retriever(search_kwargs={"k": args.k})

    if args.hybrid:
        from langchain_community.retrievers import BM25Retriever
        try:  # langchain 1.x 에서 langchain_classic 으로 이동
            from langchain_classic.retrievers import EnsembleRetriever
        except ImportError:
            from langchain.retrievers import EnsembleRetriever

        docs_all = list(vs.docstore._dict.values())
        print(f"BM25 인덱싱 {len(docs_all)}건 (tokenizer={args.bm25_tokenizer})", flush=True)
        t0 = time.time()
        if args.bm25_tokenizer == "ko":
            bm25 = BM25Retriever.from_documents(docs_all, preprocess_func=ko_tokenize)
        else:
            bm25 = BM25Retriever.from_documents(docs_all)
        bm25.k = args.k
        print(f"BM25 완료 {time.time() - t0:.0f}s")
        retriever = EnsembleRetriever(
            retrievers=[bm25, retriever],
            weights=[args.bm25_weight, 1 - args.bm25_weight],
        )

    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        print(f"리랭커 로드 {args.reranker_model}", flush=True)
        reranker = CrossEncoder(args.reranker_model, max_length=512)

    prec = {c: [] for c in CUTS}
    intent_rate = {c: [] for c in CUTS}
    disease_rate = {c: [] for c in CUTS}
    rr = []
    per_intention = defaultdict(list)
    per_query = []  # 대응표본 검정용
    t0 = time.time()

    for i, g in enumerate(golden, 1):
        docs = retriever.invoke(g["question"])
        if not docs:
            continue

        # 전체 재정렬 - cut별 비교를 같은 조건으로 맞추기 위함
        if reranker:
            scores = reranker.predict([[g["question"], d.page_content] for d in docs])
            docs = [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]

        full_hits = [is_hit(d.metadata, g) for d in docs]
        rr.append(1 / (full_hits.index(True) + 1) if True in full_hits else 0.0)

        for c in CUTS:
            top = docs[:c]
            if not top:
                continue
            hits = full_hits[:c]
            prec[c].append(sum(hits) / len(top))
            intent_rate[c].append(sum(d.metadata.get("intention") == g["intention"] for d in top) / len(top))
            disease_rate[c].append(sum(d.metadata.get("disease_name") == g["disease_name"] for d in top) / len(top))

        per_intention[g["intention"]].append(prec[PRIMARY_CUT][-1])
        per_query.append({
            "source": g["source"],
            "intention": g["intention"],
            "disease_name": g["disease_name"],
            "precision": {f"@{c}": prec[c][-1] for c in CUTS},
            "intention_rate": {f"@{c}": intent_rate[c][-1] for c in CUTS},
            "disease_rate": {f"@{c}": disease_rate[c][-1] for c in CUTS},
            "rr": rr[-1],
        })

        if i % 20 == 0:
            print(f"  {i}/{len(golden)} ({time.time() - t0:.0f}s)", flush=True)

    label = args.index
    if args.hybrid:
        label += f" +bm25({args.bm25_weight},{args.bm25_tokenizer})"
    if args.rerank:
        label += f" +rerank({args.reranker_model.split('/')[-1]})"

    result = {
        "label": label,
        "index": args.index,
        "embedding_model": model,
        "k": args.k,
        "n": len(prec[PRIMARY_CUT]),
        "hybrid": args.hybrid,
        "bm25_weight": args.bm25_weight if args.hybrid else None,
        "bm25_tokenizer": args.bm25_tokenizer if args.hybrid else None,
        "rerank": args.rerank,
        "reranker_model": args.reranker_model if args.rerank else None,
        "precision": {f"@{c}": statistics.mean(prec[c]) for c in CUTS if prec[c]},
        "intention_rate": {f"@{c}": statistics.mean(intent_rate[c]) for c in CUTS if intent_rate[c]},
        "disease_rate": {f"@{c}": statistics.mean(disease_rate[c]) for c in CUTS if disease_rate[c]},
        "mrr": statistics.mean(rr) if rr else 0.0,
        "per_intention_at5": {k: statistics.mean(v) for k, v in per_intention.items()},
        "elapsed_sec": round(time.time() - t0, 1),
        "per_query": per_query,
    }

    print()
    print("=" * 62)
    print(f"{label}   (질의 {result['n']}건)")
    print("=" * 62)
    print(f"  {'':<22}{'@5':>10}{'@10':>10}{'@20':>10}")
    for name, key in (("Precision (질환+의도)", "precision"),
                      ("의도 일치율", "intention_rate"),
                      ("질환 일치율", "disease_rate")):
        row = "".join(f"{result[key].get(f'@{c}', float('nan')):>10.4f}" for c in CUTS)
        print(f"  {name:<22}{row}")
    print()
    print(f"  MRR {result['mrr']:.4f}   무작위 기대값 0.0016   소요 {result['elapsed_sec']}s")
    print()
    print(f"  의도별 Precision@{PRIMARY_CUT}:")
    for k, v in sorted(result["per_intention_at5"].items(), key=lambda x: -x[1]):
        print(f"    {k:<10} {v:.4f}  (n={len(per_intention[k])})")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
