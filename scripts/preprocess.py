"""AI Hub 답변 JSON -> 단일 JSONL. jq 스키마 적용 지점.

개별 파일 10만 개 열기는 1.4 it/s 로 붕괴 (열거 1.7초, 열기가 병목).
zip 직접 읽기는 987 it/s.

    python -m scripts.preprocess
"""

import json
import os
import time
import zipfile
from pathlib import Path

import jq

ROOT = Path(__file__).resolve().parents[1]

SRC_ZIP = os.getenv(
    "SRC_ZIP",
    "C:/Users/User/Downloads/120.초거대AI 사전학습용 헬스케어 질의응답 데이터"
    "/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL.zip",
)
SRC_PREFIX = os.getenv("SRC_PREFIX", "2.답변/감염성질환/")  # 논문의 감염성 질환 서브셋
OUT_JSONL = Path(os.getenv("OUT_JSONL", ROOT / "data" / "answers.jsonl"))

JQ_SCHEMA = (
    '{ text: ('
    '"질병명: " + .disease_name.kor + "\n" + '
    '"진료과: " + (.department[0] // "정보 없음") + "\n" + '
    '"목적: " + .intention + "\n\n" + '
    '"답변: " + .answer.intro + " " + .answer.body + " " + .answer.conclusion'
    '), '
    'disease_name: .disease_name.kor, '
    'department: (.department[0] // "정보 없음"), '
    'intention: .intention, '
    'disease_category: .disease_category }'
)


def run():
    prog = jq.compile(JQ_SCHEMA)
    z = zipfile.ZipFile(SRC_ZIP)
    names = [n for n in z.namelist() if n.startswith(SRC_PREFIX) and n.endswith(".json")]
    print(f"[preprocess] 대상 {len(names)}건 <- {SRC_PREFIX}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = bad = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for nm in names:
            try:
                rec = prog.input(json.loads(z.read(nm).decode("utf-8-sig"))).first()
                rec["source"] = os.path.basename(nm)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception:
                bad += 1

    el = time.time() - t0
    mb = OUT_JSONL.stat().st_size / 1e6
    print(f"[preprocess] 완료 {ok}건 (실패 {bad}) / {el:.1f}s ({ok/el:.0f} it/s) / {mb:.1f}MB")
    print(f"[preprocess] -> {OUT_JSONL}")


if __name__ == "__main__":
    run()
