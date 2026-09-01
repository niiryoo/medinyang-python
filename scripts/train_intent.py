"""질의 의도 분류기 학습. 골든셋 질문은 학습에서 제외.

    python -m scripts.train_intent
"""

import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union

ROOT = Path(__file__).resolve().parents[1]

SRC_ZIP = os.getenv(
    "SRC_ZIP",
    "C:/Users/User/Downloads/120.초거대AI 사전학습용 헬스케어 질의응답 데이터"
    "/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL.zip",
)
Q_PREFIX = os.getenv("Q_PREFIX", "1.질문/감염성질환/")
GOLDEN = Path(os.getenv("OUT_GOLDEN", ROOT / "data" / "goldenset.jsonl"))
OUT = Path(os.getenv("INTENT_MODEL", ROOT / "models" / "intent.joblib"))
# 최다 클래스가 13,673건이라 12,000이면 사실상 전량
# 1000/3000/6000/12000 스윕에서 정확도 0.8532 -> 0.8706 -> 0.8875 -> 0.9000
PER_INTENTION = int(os.getenv("TRAIN_PER_INTENTION", "12000"))
SEED = int(os.getenv("SEED", "42"))


def load(z, names):
    for n in names:
        try:
            d = json.loads(z.read(n).decode("utf-8-sig"))
        except Exception:
            continue
        q = (d.get("question") or "").strip()
        if q:
            yield q


def run():
    excluded = set()
    if GOLDEN.exists():
        excluded = {json.loads(l)["source"] for l in open(GOLDEN, encoding="utf-8")}
    print(f"[intent] 골든셋 제외 대상 {len(excluded)}건")

    z = zipfile.ZipFile(SRC_ZIP)
    by_intention = defaultdict(list)
    for n in z.namelist():
        if not (n.startswith(Q_PREFIX) and n.endswith(".json")):
            continue
        if os.path.basename(n) in excluded:
            continue
        parts = n.split("/")
        if len(parts) >= 4:
            by_intention[parts[3]].append(n)

    rnd = random.Random(SEED)
    X, y = [], []
    for intention, files in sorted(by_intention.items()):
        k = min(PER_INTENTION, len(files))
        picked = rnd.sample(files, k)
        qs = list(load(z, picked))
        X += qs
        y += [intention] * len(qs)
        print(f"  {intention:<10} {len(files):>6}건 중 {len(qs)}개")

    print(f"[intent] 학습 표본 {len(X)}건 / 클래스 {len(set(y))}종")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 어절 단위만으로는 어미 변화를 못 잡아 자모 n-gram을 함께 사용
    features = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True),
    )
    clf = make_pipeline(features, LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1))
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    print()
    print(classification_report(yte, pred, digits=4))

    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels)
    print("  주요 혼동 (실제 -> 예측, 20건 이상):")
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and cm[i][j] >= 20:
                print(f"    {a} -> {b}  {cm[i][j]}건")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "labels": labels}, OUT)
    print(f"\n[intent] 저장 -> {OUT}")


if __name__ == "__main__":
    run()
