from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO


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

def encode_file_to_base64(file: UploadFile) -> str:
    """업로드된 파일을 읽어 Base64 문자열로 인코딩합니다."""
    try:
        # 파일 내용을 메모리에 읽어옵니다.
        file_bytes = file.file.read()
        
        # Base64로 인코딩하고 UTF-8 디코딩합니다.
        return base64.b64encode(file_bytes).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File encoding error: {e}")
    
    
@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    사용자 질문을 받아 RAG 서비스에 전달하고 답변을 반환합니다.
    """
    # 모든 복잡한 로직은 get_answer 함수에 위임합니다.
    final_answer = get_answer(request.history, request.question)
    
    return AnswerResponse(answer=final_answer)

@router.post("/image", response_model=AnswerResponse)
async def summarize_image_route(
    file: UploadFile = File(..., description="처방전, 건강검진 결과지 등 의료 기록 이미지"),
):
    """
    [멀티모달 기능] 업로드된 의료 이미지(처방전, 결과지)를 LLM이 분석하고 요약합니다.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    
    # 1. 이미지 파일을 Base64로 인코딩
    base64_image = encode_file_to_base64(file)
    
    # 2. 서비스 로직 호출
    try:
        # services/image_analysis_service에서 임포트한 함수를 호출
        summary_answer = summarize_medical_image(base64_image)
        return AnswerResponse(answer=summary_answer)
    except Exception as e:
        # LLM 호출 실패 등 서비스 내부 오류 처리
        raise HTTPException(status_code=500, detail=f"Service processing error: {e}")
