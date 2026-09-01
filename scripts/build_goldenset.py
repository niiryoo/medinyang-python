"""질문 JSON -> 골든셋 JSONL. 의도별 층화 추출.

정답 라벨은 (disease_name, intention) 조합. 파일 ID 매칭 불가
(질문 HC-Q-7자리 / 답변 HC-A-8자리, 교집합 0).

    python -m scripts.build_goldenset
"""

import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC_ZIP = os.getenv(
    "SRC_ZIP",
    "C:/Users/User/Downloads/120.초거대AI 사전학습용 헬스케어 질의응답 데이터"
    "/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL.zip",
)
Q_PREFIX = os.getenv("Q_PREFIX", "1.질문/감염성질환/")
OUT = Path(os.getenv("OUT_GOLDEN", ROOT / "data" / "goldenset.jsonl"))
PER_INTENTION = int(os.getenv("PER_INTENTION", "10"))  # 의도당 표본 수
SEED = int(os.getenv("SEED", "42"))


def run():
    z = zipfile.ZipFile(SRC_ZIP)
    names = [n for n in z.namelist() if n.startswith(Q_PREFIX) and n.endswith(".json")]
    print(f"[golden] 질문 후보 {len(names)}건")

    # 경로에서 의도를 얻어 층화 - 전체를 열지 않기 위함
    by_intention = defaultdict(list)
    for n in names:
        parts = n.split("/")
        if len(parts) >= 4:
            by_intention[parts[3]].append(n)

    rnd = random.Random(SEED)
    picked = []
    for intention, files in sorted(by_intention.items()):
        k = min(PER_INTENTION, len(files))
        picked += rnd.sample(files, k)
        print(f"  {intention:<10} {len(files):>6}건 중 {k}개")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for nm in picked:
            try:
                d = json.loads(z.read(nm).decode("utf-8-sig"))
                q = (d.get("question") or "").strip()
                disease = (d.get("disease_name") or {}).get("kor", "")
                intention = d.get("intention", "")
                if not (q and disease and intention):
                    bad += 1
                    continue
                out.write(json.dumps({
                    "question": q,
                    "disease_name": disease,
                    "intention": intention,
                    "source": os.path.basename(nm),
                }, ensure_ascii=False) + "\n")
                ok += 1
            except Exception:
                bad += 1

    print(f"[golden] 완료 {ok}건 (제외 {bad}) -> {OUT}")


if __name__ == "__main__":
    run()
