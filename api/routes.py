from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


# 핵심 로직이 담긴 서비스 모듈을 가져옵니다.
from services.rag_service import get_answer

from services.image_analysis_service import summarize_medical_image


# 이 파일은 /ask 경로를 담당하는 라우터입니다.
router = APIRouter()

# 요청 바디(Request Body) 모델 정의
class QuestionRequest(BaseModel):
    question: str
    history: list[tuple[str, str]]

# 응답 바디(Response Body) 모델 정의
class AnswerResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    사용자 질문을 받아 RAG 서비스에 전달하고 답변을 반환합니다.
    """
    # 모든 복잡한 로직은 get_answer 함수에 위임합니다.
    final_answer = get_answer(request.history, request.question)
    
    return AnswerResponse(answer=final_answer)

class ImageUrlRequest(BaseModel):
    imageUrl: str

@router.post("/image", response_model=AnswerResponse)
async def summarize_image_route(request: ImageUrlRequest):
    """
    [멀티모달 기능] 업로드된 의료 이미지(처방전, 결과지)를 LLM이 분석하고 요약합니다.
    """
    # 1. 서비스 로직 호출
    try:
        summary_answer = summarize_medical_image(request.imageUrl)
        return AnswerResponse(answer=summary_answer)
    except Exception as e:
        # LLM 호출 실패 등 서비스 내부 오류 처리
        raise HTTPException(status_code=500, detail=f"Service processing error: {e}")
