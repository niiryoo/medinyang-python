# services/image_analysis_service.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import config

from prompts.ocr_prompt import ocr_prompt 


try:
    IMAGE_ANALYSIS_LLM = ChatOpenAI(model_name=config.IMAGE_MODEL_NAME, temperature=config.IMAGE_TEMPERATURE)
    IMAGE_SERVICE_READY = True
    print("서비스 모듈: 이미지 분석 LLM 설정 완료.")
except Exception as e:
    print(f"FATAL: 이미지 분석 서비스 초기화 실패: {e}")
    IMAGE_ANALYSIS_LLM = None
    IMAGE_SERVICE_READY = False

def summarize_medical_image(image_url: str) -> str:
    """
    이미지 URL(처방전/검사결과 등)을 LLM에 전달하여 요약합니다.
    """
    if not IMAGE_SERVICE_READY:
        return "오류: 이미지 분석 서비스가 올바르게 초기화되지 않았습니다."

    prompt_text = ocr_prompt.format()

    # 텍스트 + 이미지 URL을 포함한 HumanMessage 생성
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    try:
        response = IMAGE_ANALYSIS_LLM.invoke([message])
        return response.content
    except Exception as e:
        print(f"이미지 분석 LLM 호출 실패: {e}")
        return "죄송합니다. 이미지 분석 중 오류가 발생했습니다."
