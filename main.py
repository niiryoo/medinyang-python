from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router as api_router
from services.rag_service import get_engine


# 첫 요청이 인덱스 적재를 떠안지 않도록 기동 시 미리 만든다
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    yield


app = FastAPI(
    title="메디냥 RAG API",
    description="문서 기반 질의응답 및 폴백 기능을 제공하는 AI API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/", summary="Health Check")
def read_root():
    """
    서버가 정상적으로 실행 중인지 확인하는 헬스 체크 엔드포인트입니다.
    """
    return {"status": "ok", "message": "메디냥 RAG API가 정상적으로 실행 중입니다."}
