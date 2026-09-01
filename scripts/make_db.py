"""JSONL -> FAISS 인덱스.

    python -m scripts.make_db
    EMBEDDING_MODEL=text-embedding-3-small DB_OUT=db_3small python -m scripts.make_db
"""

import json
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# --- 설정 ---
SRC_JSONL = Path(os.getenv("SRC_JSONL", ROOT / "data" / "answers.jsonl"))
DB_OUT = Path(os.getenv("DB_OUT", ROOT / "db"))
BATCH_SIZE = 500       # 배치당 약 220K 토큰
DELAY_SECONDS = float(os.getenv("EMBED_DELAY_SECONDS", "12"))  # TPM 1M 기준
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
META_FIELDS = ("disease_name", "department", "intention", "disease_category", "source")
# ------------


def load_documents(path: Path) -> list[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(
                Document(
                    page_content=rec["text"],
                    metadata={k: rec.get(k) for k in META_FIELDS},
                )
            )
    return docs


def create_and_save_db():
    if not SRC_JSONL.exists():
        print(f"[오류] {SRC_JSONL} 없음. 먼저 python -m scripts.preprocess 실행")
        return

    print(f"[1/4] 문서 로드: {SRC_JSONL}")
    t0 = time.time()
    docs = load_documents(SRC_JSONL)
    print(f"      문서 {len(docs)}개 ({time.time() - t0:.1f}s)")

    print("[2/4] 문서 분할")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"      청크 {len(chunks)}개 (분할 발생 {len(chunks) - len(docs)}건)")

    print(f"[3/4] 임베딩 및 인덱스 생성 (모델: {EMBEDDING_MODEL})")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    total = len(chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"      배치 {total_batches}개 (크기 {BATCH_SIZE}, 지연 {DELAY_SECONDS}s)")

    t0 = time.time()
    vectorstore = FAISS.from_documents(documents=chunks[:BATCH_SIZE], embedding=embeddings)
    print(f"      [1/{total_batches}] 완료 ({time.time() - t0:.0f}s)")

    if total_batches > 1 and DELAY_SECONDS > 0:
        time.sleep(DELAY_SECONDS)

    for i in range(BATCH_SIZE, total, BATCH_SIZE):
        vectorstore.add_documents(chunks[i:i + BATCH_SIZE])
        print(f"      [{i // BATCH_SIZE + 1}/{total_batches}] 완료 ({time.time() - t0:.0f}s)")

        if i + BATCH_SIZE < total and DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

    print(f"[4/4] 인덱스 저장: {DB_OUT}")
    if DB_OUT.exists():
        shutil.rmtree(DB_OUT)
    vectorstore.save_local(str(DB_OUT))

    print()
    print(f"완료 - 문서 {len(docs)} / 청크 {total} / 모델 {EMBEDDING_MODEL} / 경로 {DB_OUT.name}")


if __name__ == "__main__":
    create_and_save_db()
