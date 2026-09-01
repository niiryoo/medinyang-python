"""재순위화 점수 분포에서 폴백 임계값 산출.

    python -m scripts.choose_threshold docs/rerank-base.json
"""

import argparse
import json
import statistics
from pathlib import Path


def rows(path, top_n):
    for q in json.loads(Path(path).read_text(encoding="utf-8"))["per_query"]:
        yield q["scores"][0], any(q["hits"][:top_n])


def score(data, tau):
    tp = sum(1 for s, ok in data if s >= tau and ok)
    fp = sum(1 for s, ok in data if s >= tau and not ok)
    fn = sum(1 for s, ok in data if s < tau and ok)
    tn = sum(1 for s, ok in data if s < tau and not ok)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def fmt(s):
    tp, fp, fn, tn, prec, rec, f1 = s
    return f"정밀도 {prec:.3f} 재현율 {rec:.3f} F1 {f1:.3f} (오답변 {fp}건, 놓친 답변 {fn}건)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--target-precision", type=float, default=0.90)
    args = ap.parse_args()

    data = list(rows(args.path, args.top_n))
    label = json.loads(Path(args.path).read_text(encoding="utf-8"))["label"]
    ok = [s for s, hit in data if hit]
    no = [s for s, hit in data if not hit]
    base = len(ok) / len(data)

    print(f"{label}   질의 {len(data)}건   top{args.top_n} 정답 포함 {len(ok)}건 ({base:.1%})")
    print()
    for name, xs in (("정답", ok), ("오답", no)):
        if xs:
            print(f"  top1 점수 {name}  n={len(xs):<4} 중앙값 {statistics.median(xs):>8.3f}  "
                  f"범위 [{min(xs):.3f}, {max(xs):.3f}]")
    if ok and no:
        overlap = sum(1 for s in no if s >= min(ok)) / len(no)
        print(f"  오답 중 정답 최저점 이상 비율 {overlap:.1%}")
    print()

    cands = sorted({round(s, 3) for s, _ in data})
    print(f"  {'tau':>8}{'정밀도':>9}{'재현율':>9}{'F1':>8}{'답변':>7}{'폴백':>7}{'오답변':>8}")
    print("  " + "-" * 56)
    for tau in cands[::max(1, len(cands) // 15)]:
        tp, fp, fn, tn, prec, rec, f1 = score(data, tau)
        print(f"  {tau:>8.3f}{prec:>9.3f}{rec:>9.3f}{f1:>8.3f}{tp+fp:>7}{fn+tn:>7}{fp:>8}")

    best = max(cands, key=lambda t: score(data, t)[6])
    print()
    print(f"  F1 최대      tau={best:.3f}  {fmt(score(data, best))}")

    reached = [t for t in cands if score(data, t)[4] >= args.target_precision]
    if reached:
        t = min(reached)
        print(f"  정밀도 {args.target_precision:.0%} 이상  tau={t:.3f}  {fmt(score(data, t))}")
    else:
        print(f"  정밀도 {args.target_precision:.0%} 를 만족하는 tau 없음")

    if score(data, best)[4] <= base + 0.02:
        print()
        print("  최대 정밀도가 기저율과 차이가 없다. 이 점수로는 게이팅이 성립하지 않는다.")


if __name__ == "__main__":
    main()
