# KB Classification — pgvector 4-Layer Evaluation

Generated 2026-07-01T23:04:41 · gold `2.0-pgvector` · 59 cases

Live thresholds: `{'CONFIDENT_SCORE': 0.6, 'MIN_DISPLAY_SCORE': 0.55, 'FOLLOWUP_FLOOR': 0.15, 'SPREAD_THRESHOLD': 0.08, 'MIN_FIELD_SCORE': 0.35, 'MAX_ROUNDS': 5, 'RERANK_K': 10}`

## Gate

| Check | Target | Actual | Status |
|---|---|---|---|
| layer1_recall_at_k | >=0.9 | 95.7 | PASS |
| layer1_top1 | >=0.85 | 91.5 | PASS |
| layer2_display_precision | >=0.9 | 100.0 | PASS |
| layer2_out_of_kb_rejection | >=0.8 | 100.0 | PASS |
| layer3_clarification_trigger | >=0.7 | 78.6 | PASS |
| layer3_resolved_after_followup | >=0.7 | 71.4 | PASS |
| layer4_end_to_end_top1 | >=0.85 | 97.9 | PASS |
| layer4_out_of_kb_correct | >=0.8 | 91.7 | PASS |

## Layer 1 — Retrieval
- recall@3: **95.7%** · top-1: **91.5%** · MRR: 0.942 (47 in-KB cases)
- per-doc top-1: {'KB0010265': 87.5, 'KB0010863': 100.0, 'KB0010865': 100.0, 'KB0013608': 83.3, 'KB0015622': 100.0, 'KB0015711': 100.0, 'KB0016162': 100.0, 'KB0016493': 66.7}
- not-#1 cases: 4 (see eval_results.json for the list)

## Layer 2 — Threshold / routing
- display precision: **100.0%** · display recall (strong): **84.8%** · out-of-KB rejection: **100.0%**
- CONFIDENT_SCORE in use: 0.6 · turn-1 resolves: 31 (false: 0)

  CONFIDENT_SCORE sweep (correct vs false turn-1 resolves):

  | CONFIDENT_SCORE | correct | false |
  |---|---|---|
  | 0.4 | 34 | 3 |
  | 0.45 | 34 | 0 |
  | 0.5 | 34 | 0 |
  | 0.55 | 33 | 0 |
  | 0.6 | 31 | 0 |
  | 0.65 | 31 | 0 |
  | 0.7 | 26 | 0 |
  | 0.75 | 21 | 0 |
  | 0.8 | 12 | 0 |
  | 0.85 | 6 | 0 |

## Layer 3 — Clarification (weak + ambiguous)
- clarification trigger: **78.6%** · grounded phrases: 81.8% · resolved after 1 follow-up: **71.4%** (14 cases)

## Layer 4 — End-to-end
- end-to-end top-1: **97.9%** (47 in-KB) · out-of-KB correct: **91.7%** (12) · avg rounds: 2.49
