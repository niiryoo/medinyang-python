"""예외 없이 조용히 깨지는 계약만 검사.

    pytest -q
"""

import config
from prompts.fallback_prompt import fallback_prompt
from prompts.rag_prompts import build_prompt
from services.rag_service import format_docs, to_messages


class Doc:
    def __init__(self, text, meta=None):
        self.page_content = text
        self.metadata = meta or {}


def test_refusal_marker_shared_by_prompt_and_detector():
    """프롬프트 문구만 바꾸면 폴백 탐지가 죽던 결함 (ADR-005)"""
    template = build_prompt(with_citation=True).template
    assert config.REFUSAL_MARKER in template
    assert any(k in config.REFUSAL_MARKER for k in config.FALLBACK_KEYWORDS)


def test_normal_answer_does_not_trigger_fallback():
    """"없습니다"가 키워드에 있어 정상 답변이 폴백으로 대체되던 결함"""
    normal = "특별한 부작용은 없습니다. 이 정보는 참고용이며, 정확한 진단은 전문의와 상담하세요."
    assert not any(k in normal for k in config.FALLBACK_KEYWORDS)


def test_history_truncated_to_configured_turns():
    """이력 제한이 독스트링에만 있고 강제되지 않던 결함"""
    history = [(f"q{i}", f"a{i}") for i in range(config.MAX_HISTORY_TURNS + 15)]
    messages = to_messages(history)
    assert len(messages) == config.MAX_HISTORY_TURNS * 2
    assert messages[-1].content == history[-1][1]


def test_format_docs_numbers_each_document():
    """인용 표기의 전제 (ADR-008)"""
    block = format_docs([(Doc("가"), 0.9), (Doc("나"), 0.8)])
    assert "[1] 가" in block and "[2] 나" in block


def test_citation_rule_toggles():
    assert "[1] 형식" in build_prompt(with_citation=True).template
    assert "[1] 형식" not in build_prompt(with_citation=False).template


def test_fallback_prompt_carries_system_and_history():
    """system 없이 호출돼 안전 지침이 우회되던 결함 (ADR-005)"""
    messages = fallback_prompt.invoke({"history": [], "question": "두통"}).to_messages()
    assert messages[0].type == "system"
    assert "전문의와 상담" in messages[0].content
    assert set(fallback_prompt.input_variables) >= {"history", "question"}


def test_index_and_embedding_model_match():
    """어긋나면 예외 없이 좌표계가 달라짐 (ADR-001)"""
    from scripts.evaluate import INDEX_MODEL
    assert INDEX_MODEL[config.DB_PATH] == config.EMBEDDING_MODEL_NAME
