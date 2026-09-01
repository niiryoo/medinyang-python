import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드
load_dotenv()

# --- LangSmith 설정 ---
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "김지훈")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# --- OpenAI 설정 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # .env에 OPENAI_API_KEY도 있어야 합니다.

# --- 모델 설정 ---
RAG_MODEL_NAME = "gpt-4o"
FALLBACK_MODEL_NAME = "gpt-4o-mini"
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "gpt-4o")

RAG_TEMPERATURE = 0
FALLBACK_TEMPERATURE = 0.7
IMAGE_TEMPERATURE = 0

# 클라이언트가 보내는 이력을 그대로 넘기면 컨텍스트 초과와 비용 증가
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# --- 벡터 DB ---
# 인덱스와 임베딩 모델은 함께 교체 (불일치 시 검색 무의미)
DB_PATH = os.getenv("DB_PATH", "db_3small")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# --- 폴백 판정 ---
# 프롬프트 지시 문구와 아래 탐지의 단일 출처
# 두 곳에 따로 적으면 프롬프트 수정 시 탐지가 조용히 사망
REFUSAL_MARKER = "모르겠어요."

# "없습니다" 제외 - 정상 답변에도 흔함
FALLBACK_KEYWORDS = [
    REFUSAL_MARKER.rstrip("."),
    "제공된 문서에서는 해당 정보를 찾을 수 없습니다",
    "모르는 내용입니다.",
]

# 재순위화 top1 점수가 이 값 미만이면 생성 없이 폴백
# 정답/오답 점수 분포 중첩으로 비활성 유지 (AUC 0.59, scripts/choose_threshold.py)
RELEVANCE_THRESHOLD = None

BASE_K = 20        # 벡터 검색 후보 수
RERANK_TOP_N = 5   # 재순위화 후 LLM에 넘길 문서 수

# --- 의도 필터 ---
# scripts/train_intent.py 산출물. 파일이 없으면 필터 없이 동작
INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "models/intent.joblib")
# 필터가 후보를 약 1/11로 줄이므로 넉넉히 확보
INTENT_FETCH_K = int(os.getenv("INTENT_FETCH_K", "8000"))

# --- 재순위화 ---
# 의도 필터 도입 후 이득 구분불가(+0.82%p, CI [-0.62, +2.25]) - ADR-007
# CPU 질의당 14.1s / GPU 0.44s. 기본 비활성은 CPU 서빙 전제, GPU면 켤 만함
USE_RERANKER = os.getenv("USE_RERANKER", "0") == "1"



# 키 없이 os.environ 대입 시 TypeError - import 단계에서 서버 사망
def setup_langsmith():
    if not LANGCHAIN_API_KEY:
        print("[config] LANGCHAIN_API_KEY 미설정 - 트레이싱 비활성화")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    
    
    