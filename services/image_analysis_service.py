# services/image_analysis_service.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import config

from prompts.ocr_prompt import ocr_prompt 


# 이미지 분석 LLM은 서비스 초기화 시 config.RAG_MODEL_NAME을 사용하여 설정됩니다.
try:
    
    IMAGE_ANALYSIS_LLM = ChatOpenAI(model_name="gpt-4.1", temperature=0)
    IMAGE_SERVICE_READY = True
    print("서비스 모듈: 이미지 분석 LLM 설정 완료.")
except Exception as e:
    print(f"FATAL: 이미지 분석 서비스 초기화 실패: {e}")
    IMAGE_ANALYSIS_LLM = None
    IMAGE_SERVICE_READY = False

def summarize_medical_image(base64_image: str) -> str:
    """
    Base64 인코딩된 의료 이미지(처방전/결과지)를 LLM에 전달하여 요약합니다.
    """
    if not IMAGE_SERVICE_READY:
        return "오류: 이미지 분석 서비스가 올바르게 초기화되지 않았습니다."

    # 텍스트와 Base64 이미지를 포함하는 HumanMessage 생성
    message = HumanMessage(
        content=[
            {"type": "text", "text": ocr_prompt},
            # Base64 문자열을 GPT API 형식에 맞게 URL로 변환하여 전달합니다.
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    )

    try:
        response = IMAGE_ANALYSIS_LLM.invoke([message])
        return response.content
    except Exception as e:
        print(f"이미지 분석 LLM 호출 실패: {e}")
        return "죄송합니다. 이미지 분석 중 오류가 발생했습니다."