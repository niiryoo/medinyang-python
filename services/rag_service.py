# services/rag_service.py

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts.rag_prompts import medi_nyang_prompt
from prompts.fallback_prompt import fallback_prompt

import config

config.setup_langsmith()


class BgeReranker:
    """HuggingFace BGE reranker 래퍼"""

    # torch/sentence-transformers는 사용할 때만 import (ADR-007 기본 비활성)
    def __init__(self, model_name="BAAI/bge-reranker-base", top_n=5, device=None):
        import torch
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_n = top_n

    def rerank(self, query, docs):
        if not docs:
            return []
        scores = self.model.predict([[query, doc.page_content] for doc in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:self.top_n]]


def format_docs(scored):
    return "\n\n---\n\n".join(
        f"[{i}] {doc.page_content}" for i, (doc, _) in enumerate(scored, 1)
    )


def to_messages(history):
    messages = []
    for question, answer in history[-config.MAX_HISTORY_TURNS:]:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


class RagEngine:
    """인덱스와 모델을 보유. 생성 비용이 크므로 get_engine()으로만 만든다."""

    def __init__(self):
        print("서비스 모듈: 데이터베이스 로드를 시도합니다...")
        embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        self.vectorstore = FAISS.load_local(config.DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("서비스 모듈: FAISS 벡터스토어 로드 성공.")

        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.BASE_K})

        self.reranker = BgeReranker(top_n=config.RERANK_TOP_N) if config.USE_RERANKER else None
        print(f"서비스 모듈: 재순위화 {'활성' if self.reranker else '비활성 (ADR-007)'}")

        self.intent_clf = None
        intent_path = Path(config.INTENT_MODEL_PATH)
        if intent_path.exists():
            import joblib
            self.intent_clf = joblib.load(intent_path)["model"]
            print(f"서비스 모듈: 의도 분류기 로드 {intent_path}")
        else:
            print(f"서비스 모듈: 의도 분류기 없음 ({intent_path}) - 필터 비활성")

        rag_llm = ChatOpenAI(model_name=config.RAG_MODEL_NAME, temperature=config.RAG_TEMPERATURE)
        self.rag_chain = medi_nyang_prompt | rag_llm | StrOutputParser()

        fallback_llm = ChatOpenAI(model_name=config.FALLBACK_MODEL_NAME, temperature=config.FALLBACK_TEMPERATURE)
        self.fallback_chain = fallback_prompt | fallback_llm | StrOutputParser()

        print(f"서비스 모듈: 체인 생성 완료 (k={config.BASE_K}, top_n={config.RERANK_TOP_N}, "
              f"threshold={config.RELEVANCE_THRESHOLD}).")

    def retrieve(self, question):
        if self.intent_clf is None:
            return self.base_retriever.invoke(question)
        intention = self.intent_clf.predict([question])[0]
        docs = self.vectorstore.similarity_search(
            question, k=config.BASE_K, fetch_k=config.INTENT_FETCH_K,
            filter={"intention": intention},
        )
        return docs or self.base_retriever.invoke(question)

    def retrieve_and_rerank(self, question):
        docs = self.retrieve(question)
        if self.reranker:
            return self.reranker.rerank(question, docs)
        return [(doc, None) for doc in docs[:config.RERANK_TOP_N]]


_engine = None
_init_error = None


def get_engine():
    """첫 호출에서 적재. 기동 시 main.py가 미리 부른다."""
    global _engine, _init_error
    if _engine is None and _init_error is None:
        try:
            _engine = RagEngine()
        except Exception as e:
            _init_error = e
            print(f"FATAL: RAG 서비스 초기화 실패: {e}")
    return _engine


def get_answer(history: list[tuple[str, str]], current_question: str) -> str:
    """
    history: [(question, answer), ...]  최대 config.MAX_HISTORY_TURNS 개
    current_question: 지금 사용자가 입력한 질문
    """
    engine = get_engine()
    if engine is None:
        return "오류: RAG 서비스가 올바르게 초기화되지 않았습니다. 관리자에게 문의하세요."

    messages = to_messages(history)

    def fallback(reason):
        print(f"[rag] 폴백: {reason}")
        return engine.fallback_chain.invoke({"history": messages, "question": current_question})

    scored = engine.retrieve_and_rerank(current_question)
    if not scored:
        return fallback("검색 결과 없음")

    top_score = scored[0][1]  # 재순위화 비활성이면 None
    threshold = config.RELEVANCE_THRESHOLD
    if threshold is not None and top_score is not None and top_score < threshold:
        return fallback(f"최고 점수 {top_score:.3f} < {threshold}")

    answer = engine.rag_chain.invoke({"context": format_docs(scored), "question": current_question})

    if any(keyword in answer for keyword in config.FALLBACK_KEYWORDS):
        return fallback("문서로 답변 불가")

    return answer
