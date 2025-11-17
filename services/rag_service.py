# services/rag_service.py

import torch
from sentence_transformers import CrossEncoder 
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Fallback 전용 프롬프트 임포트
# 💡 수정됨: 파일 이름까지 명시하여 정확한 모듈 경로를 지정합니다.
from prompts.rag_prompts import medi_nyang_prompt 
from prompts.fallback_prompt import fallback_prompt 

# 우리가 만든 설정과 프롬프트를 가져옵니다.
import config

# LangSmith 설정 실행
config.setup_langsmith()

# BgeReranker 클래스는 그대로 유지
class BgeReranker:
    """HuggingFace BGE reranker 래퍼"""
    def __init__(self, model_name="BAAI/bge-reranker-base", top_n=5, device=None):
        self.model = CrossEncoder(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_n = top_n

    def rerank(self, query, docs):
        """문서 리스트를 rerank하고 상위 top_n 리턴"""
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_n]]


try:
    print("서비스 모듈: 데이터베이스 로드를 시도합니다...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        config.DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("서비스 모듈: FAISS 벡터스토어 로드 성공.")

    # 1. Base Retriever 생성 (검색)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": config.BASE_K})
    print(f"서비스 모듈: Base Retriever (K={config.BASE_K}) 생성 완료.")

    # 2. Reranker 초기화
    reranker = BgeReranker(top_n=config.RERANK_TOP_N)
    print("서비스 모듈: BGE Reranker 초기화 완료.")

    # 3. 검색 및 Rerank 결합 함수
    def retrieve_and_rerank(question):
        docs = base_retriever.invoke(question)
        return reranker.rerank(question, docs)

    # 4. RAG Chain 생성 (메인)
    rag_llm = ChatOpenAI(model_name=config.RAG_MODEL_NAME, temperature=0)
    rag_chain = (
        {"context": retrieve_and_rerank, "question": RunnablePassthrough()}
        | medi_nyang_prompt
        | rag_llm
        | StrOutputParser()
    )
    print("서비스 모듈: RAG + Rerank 체인 생성 완료.")

    # 5. Fallback Chain 생성 (새로 추가)
    fallback_llm = ChatOpenAI(model_name=config.FALLBACK_MODEL_NAME, temperature=0.7)
    fallback_chain = (
        {"question": RunnablePassthrough()}
        | fallback_prompt # 👈 Fallback 전용 프롬프트 사용
        | fallback_llm
        | StrOutputParser()
    )
    print("서비스 모듈: Fallback Chain 생성 완료.")


except Exception as e:
    print(f"FATAL: RAG 서비스 초기화 실패: {e}")
    rag_chain = None
    fallback_chain = None


def get_answer(history: list[tuple[str, str]], current_question: str) -> str:
    """
    history: [(question, answer), ...]  최대 10개
    current_question: 지금 사용자가 입력한 질문
    """

    if not rag_chain or not fallback_llm:
        return "오류: RAG 서비스가 올바르게 초기화되지 않았습니다. 관리자에게 문의하세요."

    # 1️⃣ 프롬프트 메시지 구성
    messages = []

    # 이전 Q/A를 conversation history 메시지로 넣기
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    # 현재 질문은 마지막 user 메시지에 배치
    messages.append({"role": "user", "content": current_question})

    # 2️⃣ RAG 호출 (current_question를 반영한 질문 메시지 전달)
    rag_response = rag_chain.invoke(current_question)

    # rag_chain.invoke() 결과 타입이 문자열일 수도, 객체일 수도 있으므로 정리
    if hasattr(rag_response, "content"):
        rag_text = rag_response.content
    else:
        rag_text = str(rag_response)

    # 3️⃣ RAG 문서에서 답변이 없는 경우 → fallback LLM 호출
    if any(keyword in rag_text.strip() for keyword in config.FALLBACK_KEYWORDS):
        print("⚠️ 문서에 답변이 없어 일반 GPT 모델(폴백)을 호출합니다...")

        fallback_response = fallback_llm.invoke(messages)

        final_answer = (
            fallback_response.content
            if hasattr(fallback_response, "content")
            else str(fallback_response)
        )
    else:
        final_answer = rag_text

    return final_answer
