"""두 측정 결과 대응표본 비교.

    python -m scripts.compare docs/pq-dense.json docs/pq-hybrid-ko.json
"""

import argparse
import json
import math
import statistics
from pathlib import Path

Z95 = 1.96
Z80 = 0.8416
CUTS = ("@5", "@10", "@20")
METRICS = (("precision", "Precision"), ("intention_rate", "의도 일치율"), ("disease_rate", "질환 일치율"))


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d, {q["source"]: q for q in d["per_query"]}


def paired(a_rows, b_rows, metric, cut):
    keys = sorted(set(a_rows) & set(b_rows))
    return [a_rows[k][metric][cut] - b_rows[k][metric][cut] for k in keys]


def stats(diffs):
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    lo, hi = mean - Z95 * se, mean + Z95 * se
    mde = (Z95 + Z80) * se
    return mean, lo, hi, mde, ("유의" if (lo > 0 or hi < 0) else "구분불가")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()

    a, a_rows = load(args.a)
    b, b_rows = load(args.b)
    n = len(set(a_rows) & set(b_rows))

    print(f"A: {a['label']}")
    print(f"B: {b['label']}")
    print(f"대응 질의 {n}건   (A - B)")
    print()
    print(f"  {'지표':<14}{'cut':>5}{'차이':>10}{'95% CI':>20}{'MDE':>9}  판정")
    print("  " + "-" * 66)

    for key, name in METRICS:
        for cut in CUTS:
            mean, lo, hi, mde, sig = stats(paired(a_rows, b_rows, key, cut))
            ci = f"[{lo*100:+.2f}, {hi*100:+.2f}]"
            print(f"  {name:<14}{cut:>5}{mean*100:>+9.2f}%p{ci:>20}{mde*100:>8.2f}%p  {sig}")
        print()

    mde5 = stats(paired(a_rows, b_rows, "precision", "@5"))[3]
    if mde5 > 0:
        need = math.ceil(n * (mde5 / 0.02) ** 2)
        print(f"  Precision@5 검출 한계 {mde5*100:.2f}%p")
        print(f"  2%p 검출에 필요한 골든셋 약 {need}건 (현재 {n}건)")


if __name__ == "__main__":
    main()
