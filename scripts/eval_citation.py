"""인용 유효성 측정. LLM 채점 없이 정규식과 문자열 비교만 사용.

    python -m scripts.eval_citation --index db_3small --intent-filter --limit 50
"""

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from prompts.rag_prompts import build_prompt
from scripts.evaluate import INDEX_MODEL, add_retrieval_args, build_search

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CITE = re.compile(r"\[(\d+)\]")
SENT = re.compile(r"(?<=[.!?])\s+")
DISCLAIMER = "전문의와 상담"


def sentences(text):
    return [s for s in SENT.split(text.strip()) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    add_retrieval_args(ap)
    ap.set_defaults(index="db_3small")
    ap.add_argument("--golden", default=str(ROOT / "data" / "goldenset.jsonl"))
    ap.add_argument("--top-n", type=int, default=config.RERANK_TOP_N)
    # 요청당 약 2000토큰 - gpt-4o TPM 30000 기준 동시 3이 상한
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--no-citation", dest="citation", action="store_false")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    golden = [json.loads(l) for l in open(args.golden, encoding="utf-8")]
    # 골든셋이 의도별 정렬 - 앞에서 자르면 한 의도에 편중
    if args.limit and args.limit < len(golden):
        step = len(golden) / args.limit
        golden = [golden[int(i * step)] for i in range(args.limit)]

    model = INDEX_MODEL[args.index]
    vs = FAISS.load_local(str(ROOT / args.index), OpenAIEmbeddings(model=model),
                          allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": args.k})
    search = build_search(vs, retriever, args)
    print(f"골든셋 {len(golden)}건 | {args.index} | top_n={args.top_n} | intent={args.intent_filter}")

    contexts, inputs = [], []
    for i, g in enumerate(golden, 1):
        docs, _, _ = search(g["question"])
        docs = docs[:args.top_n]
        contexts.append(docs)
        block = "\n\n---\n\n".join(f"[{n}] {d.page_content}" for n, d in enumerate(docs, 1))
        inputs.append({"context": block, "question": g["question"]})
        if i % 50 == 0:
            print(f"  검색 {i}/{len(golden)}")

    prompt = build_prompt(with_citation=args.citation)
    llm = ChatOpenAI(model_name=config.RAG_MODEL_NAME, temperature=config.RAG_TEMPERATURE, max_retries=10)
    chain = prompt | llm | StrOutputParser()
    print(f"생성 시작 (동시 {args.concurrency})", flush=True)
    t0 = time.time()
    answers = chain.batch(inputs, config={"max_concurrency": args.concurrency})
    print(f"생성 완료 {time.time() - t0:.0f}s")

    rows = []
    for g, docs, answer in zip(golden, contexts, answers):
        nums = [int(n) for n in CITE.findall(answer)]
        valid = [n for n in nums if 1 <= n <= len(docs)]
        cited_docs = [docs[n - 1] for n in valid]

        body = [s for s in sentences(answer) if DISCLAIMER not in s]
        rows.append({
            "source": g["source"],
            "intention": g["intention"],
            "disease_name": g["disease_name"],
            "n_sentences": len(body),
            "n_cited_sentences": sum(1 for s in body if CITE.search(s)),
            "n_citations": len(nums),
            "n_valid": len(valid),
            "n_disease_match": sum(1 for d in cited_docs if d.metadata.get("disease_name") == g["disease_name"]),
            "n_intention_match": sum(1 for d in cited_docs if d.metadata.get("intention") == g["intention"]),
            "refused": any(k in answer for k in config.FALLBACK_KEYWORDS),
            "answer": answer,
        })

    def rate(num, den):
        n, d = sum(r[num] for r in rows), sum(r[den] for r in rows)
        return n / d if d else 0.0

    result = {
        "label": f"{args.index}{' +intent' if args.intent_filter else ''}"
                 f"{' +citation' if args.citation else ' (인용 규칙 없음)'}",
        "citation_rule": args.citation,
        "n": len(rows),
        "top_n": args.top_n,
        "intent_filter": args.intent_filter,
        "sentence_citation_rate": rate("n_cited_sentences", "n_sentences"),
        "valid_citation_rate": rate("n_valid", "n_citations"),
        "cited_disease_match": rate("n_disease_match", "n_valid"),
        "cited_intention_match": rate("n_intention_match", "n_valid"),
        "citations_per_answer": statistics.mean(r["n_citations"] for r in rows),
        "uncited_answers": sum(1 for r in rows if r["n_citations"] == 0),
        "refused_answers": sum(1 for r in rows if r["refused"]),
        "per_query": rows,
    }

    print()
    print("=" * 62)
    print(f"{result['label']}   (질의 {result['n']}건)")
    print("=" * 62)
    print(f"  문장 인용률       {result['sentence_citation_rate']:.4f}   규칙 준수")
    print(f"  유효 인용률       {result['valid_citation_rate']:.4f}   없는 번호를 지어내지 않는가")
    print(f"  인용 질환 일치율  {result['cited_disease_match']:.4f}   deceptive grounding 노출도")
    print(f"  인용 의도 일치율  {result['cited_intention_match']:.4f}")
    print()
    print(f"  답변당 인용 {result['citations_per_answer']:.1f}개   "
          f"인용 없는 답변 {result['uncited_answers']}건   거부 {result['refused_answers']}건")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
