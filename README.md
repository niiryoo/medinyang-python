# MediNyang RAG 서버

> [KNU-OldMan/medinyang-python](https://github.com/KNU-OldMan/medinyang-python)의 포크.
> 논문 게재 후 코드 감사에서 식별한 결함을 수정하고, 검색 파이프라인을 측정 기반으로 재구성했다.

원 프로젝트는 「JSON 데이터 구조화 및 이중 재순위화 기법을 적용한 의료용 RAG 챗봇 시스템 구현」
(정보통신논문지 Vol.30, 2026.02)의 구현체다. 게재 시점 코드에는 검색 품질을 재는 수단이 없었고,
논문이 서술한 임계값 기반 폴백이 구조적으로 동작하지 않는 상태였다.

---

## 개선 결과

골든셋 220건(의도 11종 × 20건 층화 추출) 기준, 논문 설정 대비 대응표본 비교.

| 지표 | 논문 설정 | 최종 | 차이 | 95% CI | 판정 |
|---|---|---|---|---|---|
| Precision@5 | 0.4800 | **0.7645** | +28.45%p | [+22.75, +34.16] | 유의 |
| 의도 일치@5 | 0.5555 | 0.8864 | +33.09%p | [+27.23, +38.95] | 유의 |
| 질환 일치@5 | 0.8527 | 0.8718 | +1.91%p | [-2.32, +6.14] | 구분불가 |
| MRR | 0.5581 | 0.7765 | — | — | — |

무작위 기대값은 0.0016이다 (조합당 정답 문서 중앙값 160건 / 전체 102,008건).

개선이 사실상 전부 의도 축에서 나왔다. 질환 축은 임베딩 교체로 얻은 만큼을
의도 필터가 반납해 상쇄됐다.

### 주요 수정

| 변경 | 근거 |
|---|---|
| 질의 의도 분류기 + 메타데이터 필터 | Precision@5 +19.73%p · [ADR-006](docs/adr/ADR-006-intent-classifier.md) |
| 근거 인용 `[n]` 표기 + 기계적 검증 | 유효 인용률 1.0000, 인용 질환 일치율 1.0000 · [ADR-008](docs/adr/ADR-008-citation.md) |
| 폴백 경로에 system 메시지 복구 | 검색 실패 시 안전 지침 없는 LLM이 의료 질문에 답하던 상태 · [ADR-005](docs/adr/ADR-005-fallback-chain-wiring.md) |
| 임베딩 ada-002 → text-embedding-3-small | Precision@5 +4.64%p, 가격 1/5 · [ADR-001](docs/adr/ADR-001-embedding-model.md) |
| 재순위화 점수 반환, 검색/생성 분리 | 점수를 버리면 측정도 게이팅도 불가 · [ADR-004](docs/adr/ADR-004-fallback-gating.md) |
| 골든셋 생성기 + 평가 스크립트 | 품질 측정 수단 부재 · [ADR-000](docs/adr/ADR-000-evaluation-design.md) |

### 검토 후 기각

| 대안 | 측정 결과 |
|---|---|
| BM25 하이브리드 | dense 단독 대비 13.09%p 열위 · [ADR-002](docs/adr/ADR-002-hybrid-retrieval.md) |
| 한국어 특화 리랭커 | 9개 지표 전부 구분불가, 비용 3.5배 · [ADR-003](docs/adr/ADR-003-reranker-model.md) |
| 재순위화 점수 임계값 게이팅 | 정답/오답 점수 분포 중첩, 최고 AUC 0.692 · [ADR-004](docs/adr/ADR-004-fallback-gating.md) |
| 의도 soft filter (상위 2개 허용) | hard filter 대비 Precision@5 15.09%p 열위 · [ADR-006](docs/adr/ADR-006-intent-classifier.md) |
| 분류 신뢰도 기반 필터 해제 | 걸수록 악화. 불확실해도 필터가 무필터보다 나음 · [ADR-006](docs/adr/ADR-006-intent-classifier.md) |
| 재순위화 (의도 필터 도입 후) | +0.82%p 구분불가, CPU 질의당 14.1초. 기본 비활성 · [ADR-007](docs/adr/ADR-007-reranker-removal.md) |

### 병목을 찾은 방법

Precision은 질환과 의도를 동시에 맞혀야 오르는 지표라, 두 축으로 분해하면 구조가 드러난다.

```
질환 0.9336 × 의도 0.5982 = 0.5585      실측 Precision@5 = 0.5564
```

거의 정확히 일치한다. 두 오차가 독립적으로 곱해지고 있었다. 임베딩 교체·하이브리드·재순위화
중 어느 것도 의도 축을 유의하게 올리지 못했고, 세 실험이 같은 결론을 가리켰다 —
검색기 교체로 풀리는 문제가 아니었다. 의도 분류기를 붙여 0.5982를 0.8864로 올렸다.

### 검색 지표가 재지 못하는 것

인용을 도입하면서 생성 단계를 함께 쟀더니, Precision@5가 답변 가능성을 과대평가한다는 것이
드러났다 (44건 표본).

| | 값 |
|---|---|
| Precision@5 | 0.7645 |
| **실제 답변율** | **0.5682** (25/44) |

거부 19건 중 검색 실패로 설명되는 건 6건뿐이다. 나머지 13건은 Precision@5가 1.0인데도
거부됐다 — `(질환, 의도)` 라벨이 맞는 문서라도 *"어떤 의료기관을 찾아가야 하나요"* 같은
질문에 답할 정보는 없기 때문이다. 상세는 [ADR-008](docs/adr/ADR-008-citation.md).

### 남은 것

- 의도 일치율(0.8864)이 분류기 정확도(0.9000)에 묶여 있음. 혼동 쌍은 전부 의미적으로 인접해
  라벨 정의가 겹치는 구간이 한계로 보임
- 필터로 반납한 질환 일치율 (0.9336 → 0.8718) 회복
- 답변율 56.8%. 나머지는 폴백으로 넘어가 문서 근거 없이 생성됨
- 인용 내용 충실도 미측정. 인용한 문서가 그 문장을 실제로 뒷받침하는지는 범위 밖

### 문서

- [docs/EVALUATION.md](docs/EVALUATION.md) — 측정 설계와 실험 9종
- [docs/adr/](docs/adr/) — 결정 기록 9건
- [docs/results/](docs/results/) — 질의별 원자료 21개. 신뢰구간 재계산 가능, AI Hub 원문 미포함

> 게재 시점 인덱스는 수작업 선별이라 재현 불가. 본 측정은 동일 데이터셋에서 재현 가능한
> 규칙으로 구성한 인덱스 기준이며, 절대값이 아닌 상대 비교로만 해석한다.

---

## 실행

AI Hub 「초거대AI 사전학습용 헬스케어 질의응답 데이터」는 신청·승인이 필요하고,
인덱스 빌드에 약 48분과 $2.7이 든다. 데이터 없이는 테스트까지만 돌아간다.

```bash
poetry install                            # Python 3.11.2 고정
cp .env.example .env                      # OPENAI_API_KEY 필수
poetry run pytest -q

poetry run python -m scripts.preprocess   # zip -> data/answers.jsonl
poetry run python -m scripts.make_db      # JSONL -> FAISS
poetry run python -m scripts.build_goldenset
poetry run python -m scripts.train_intent

poetry run uvicorn main:app --reload      # http://127.0.0.1:8000/docs
```

인덱스와 임베딩 모델은 함께 바꾼다. 어긋나도 예외가 나지 않고 그럴듯한 오답이 나온다.

### 측정 재실행

```bash
poetry run python -m scripts.evaluate --index db_3small --intent-filter --out docs/results/eval.json
poetry run python -m scripts.compare docs/results/eval.json docs/results/g220-E1-ada.json
poetry run python -m scripts.eval_citation --intent-filter --limit 44
```

재순위화 측정은 CPU에서 질의당 25초가 걸린다. GPU가 없으면
[notebooks/rerank_colab.ipynb](notebooks/rerank_colab.ipynb)를 Colab에서 실행한다.

---

<sub>1차 개선분: [PR #1](https://github.com/niiryoo/medinyang-python/pull/1). 이후 작업으로 수치가 갱신됐다.</sub>
