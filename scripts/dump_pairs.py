"""재순위화 입력 덤프. Colab GPU 용.

    python -m scripts.dump_pairs
    python -m scripts.dump_pairs --intent-filter --out data/pairs-intent.json
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from scripts.evaluate import INDEX_MODEL, add_retrieval_args, build_search

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    add_retrieval_args(ap)
    ap.set_defaults(index="db_3small")
    ap.add_argument("--golden", default=str(ROOT / "data" / "goldenset.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "pairs.json"))
    args = ap.parse_args()

    model = INDEX_MODEL[args.index]
    golden = [json.loads(l) for l in open(args.golden, encoding="utf-8")]
    vs = FAISS.load_local(
        str(ROOT / args.index),
        OpenAIEmbeddings(model=model),
        allow_dangerous_deserialization=True,
    )
    retriever = vs.as_retriever(search_kwargs={"k": args.k})
    search = build_search(vs, retriever, args)
    print(f"골든셋 {len(golden)}건 | {args.index} | k={args.k} | intent={args.intent_filter}")

    rows = []
    for i, g in enumerate(golden, 1):
        docs, intent_pred, intent_conf = search(g["question"])
        rows.append({
            "source": g["source"],
            "question": g["question"],
            "gold_disease": g["disease_name"],
            "gold_intention": g["intention"],
            "intent_pred": intent_pred,
            "intent_conf": intent_conf,
            "docs": [{
                "text": d.page_content,
                "disease_name": d.metadata.get("disease_name"),
                "intention": d.metadata.get("intention"),
                "source": d.metadata.get("source"),
            } for d in docs],
        })
        if i % 50 == 0:
            print(f"  {i}/{len(golden)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "index": args.index,
        "embedding_model": model,
        "k": args.k,
        "intent_filter": args.intent_filter,
        "queries": rows,
    }, ensure_ascii=False), encoding="utf-8")

    n_pairs = sum(len(r["docs"]) for r in rows)
    print(f"완료 - 질의 {len(rows)} / 쌍 {n_pairs} / {out.stat().st_size/1e6:.1f}MB -> {out}")


if __name__ == "__main__":
    main()
