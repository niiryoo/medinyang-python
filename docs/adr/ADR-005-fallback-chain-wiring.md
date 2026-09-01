# ADR-005: 폴백 경로에 안전 지침 복구

## 관측

`fallback_chain`이 정의만 되고 `.invoke()`가 한 번도 호출되지 않았다.
실제 폴백은 아래 한 줄로 나갔다.

```python
fallback_response = fallback_llm.invoke(messages)
```

`messages`는 대화 이력과 현재 질문뿐이다. **system 메시지가 없다.**

그 결과 `prompts/fallback_prompt.py`에 적힌 지침이 전부 우회됐다.

- 추측 금지 / 확정적 표현 금지
- 증상 지속 시 전문의 상담 권고
- 안전 고지 문구 필수 포함

즉 **검색이 실패했을 때, 안전 장치가 가장 필요한 순간에, 안전 장치가 없는
범용 LLM이 의료 질문에 답하고 있었다.**

## 원인

`fallback_chain`을 만든 커밋과 `get_answer`에서 폴백을 호출하는 커밋이 따로 있었고,
병합 과정에서 후자가 남았다. 두 경로가 공존한 채로 하나만 동작했다.

프롬프트 구조에도 원인이 있다. `fallback_prompt`가 `PromptTemplate`에
`{question}` 하나만 받는 형태라 **대화 이력을 실을 자리가 없었다.**
이력을 유지하려면 체인을 우회하는 수밖에 없는 구조였다.

## 고려한 대안

| 안 | 장점 | 단점 |
|---|---|---|
| (a) `messages` 앞에 system 딕셔너리를 수동으로 끼워넣기 | 최소 변경 | 프롬프트가 두 곳에 흩어짐 |
| (b) 이력을 버리고 기존 `fallback_chain` 사용 | 변경 작음 | 멀티턴 대화가 깨짐 |
| (c) **`ChatPromptTemplate` + `MessagesPlaceholder`로 재작성** | 지침과 이력을 한 체인에서 처리 | 프롬프트 파일 수정 필요 |

## 결정

(c). `fallback_prompt`를 `ChatPromptTemplate`로 바꾸고 `MessagesPlaceholder("history")`를 넣었다.

```python
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", FALLBACK_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
```

`get_answer`는 이제 `fallback_chain.invoke({"history": ..., "question": ...})` 하나만 쓴다.
경로가 둘이었던 것이 문제였으므로 하나로 합쳤다.

## 함께 고친 것

`FALLBACK_KEYWORDS`에 `"없습니다"`가 들어 있었다. 부분 문자열 일치라
**"특별한 부작용은 없습니다" 같은 정상 답변이 폴백으로 넘어간다.**
근거 있는 답변을 근거 없는 답변으로 바꿔치기하는 동작이라 제거했다.

RAG 프롬프트가 답을 못 할 때 `"모르겠어요."`로만 답하도록 지시하므로,
그 표지 위주로 남겼다.

## 트레이드오프

system 메시지가 붙으면서 폴백 호출의 토큰이 늘었다 (약 455자). 폴백은
전체 질의 중 일부에서만 발생하므로 감수할 수준이다.

그리고 `ChatPromptTemplate`은 `PromptTemplate`보다 문자열 이스케이프에 민감하다.
프롬프트 본문에 `{`가 들어가면 변수로 해석되므로, 지침 문구를 수정할 때 주의가 필요하다.

## 2차 효과

폴백 경로가 하나로 정리되면서 **폴백 사유를 한 곳에서 기록할 수 있게 됐다.**
현재 세 가지 사유(검색 결과 없음 / 점수 미달 / 문서로 답변 불가)를 구분해 로깅한다.
ADR-004에서 점수 게이팅이 기각됐지만 이 분기 구조는 남겨뒀다 —
의도 분류기가 붙으면 네 번째 사유가 여기에 들어간다.

또한 `rag_chain`에서 검색을 분리해 `get_answer`로 끌어올렸다. 원래는 체인 안에
retriever가 묶여 있어 **생성 전에 검색 결과를 볼 수 없었다.** 폴백 게이팅을 하려면
검색과 생성 사이에 판단 지점이 필요하고, 그러려면 둘을 분리해야 한다.
게이팅은 기각됐지만 이 분리는 유지한다 — 로깅과 측정이 여기에 의존한다.

## 검증

```
폴백 프롬프트 메시지: [('system', 455), ('human', 7)]
```

이전에는 system 메시지가 0개였다.
