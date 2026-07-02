# Hyperparameter Tuning — cutoff thresholds

Gold `2.0-pgvector` · 59 cases · in-memory grid over precomputed rankings.

## Recommended settings
- **CONFIDENT_SCORE = 0.6**  (was 0.6)
- **MIN_DISPLAY_SCORE = 0.55**  (was 0.4)
- **SPREAD_THRESHOLD = 0.08**  (was 0.1)
- **FOLLOWUP_FLOOR = 0.1**  (was 0.15) · **MIN_FIELD_SCORE = 0.3**  (was 0.4)

## Baseline vs tuned (end-to-end over all cases)

| | in-KB top-1 | out-of-KB correct | display precision | turn-1 resolves | avg rounds |
|---|---|---|---|---|---|
| baseline | 100.0% | 33.3% | 100.0% | 27 | 2.66 |
| **tuned** | 97.9% | 91.7% | 100.0% | 31 | 2.49 |

## Top combinations (C = CONFIDENT_SCORE, MD = MIN_DISPLAY_SCORE, ST = SPREAD_THRESHOLD)

| C | MD | ST | correct | in-KB | out-of-KB | disp.prec | t1 | rounds |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 0.55 | 0.08 | 57/59 | 97.9% | 91.7% | 100.0% | 31 | 2.49 |
| 0.65 | 0.55 | 0.08 | 57/59 | 97.9% | 91.7% | 100.0% | 31 | 2.54 |
| 0.55 | 0.55 | 0.1 | 57/59 | 97.9% | 91.7% | 100.0% | 29 | 2.53 |
| 0.6 | 0.55 | 0.1 | 57/59 | 97.9% | 91.7% | 100.0% | 27 | 2.66 |
| 0.65 | 0.55 | 0.1 | 57/59 | 97.9% | 91.7% | 100.0% | 27 | 2.71 |
| 0.7 | 0.55 | 0.08 | 57/59 | 97.9% | 91.7% | 100.0% | 26 | 2.93 |
| 0.7 | 0.55 | 0.1 | 57/59 | 97.9% | 91.7% | 100.0% | 25 | 3.0 |
| 0.55 | 0.55 | 0.08 | 57/59 | 97.9% | 91.7% | 97.1% | 34 | 2.29 |
| 0.55 | 0.55 | 0.15 | 56/59 | 95.7% | 91.7% | 100.0% | 26 | 2.78 |
| 0.6 | 0.55 | 0.15 | 56/59 | 95.7% | 91.7% | 100.0% | 25 | 2.85 |
| 0.65 | 0.55 | 0.15 | 56/59 | 95.7% | 91.7% | 100.0% | 25 | 2.9 |
| 0.7 | 0.55 | 0.15 | 56/59 | 95.7% | 91.7% | 100.0% | 24 | 3.07 |

## Grounded follow-up sweep (FOLLOWUP_FLOOR x MIN_FIELD_SCORE -> grounded % on weak+ambiguous)

| FOLLOWUP_FLOOR | MIN_FIELD_SCORE | grounded % |
|---|---|---|
| 0.1 | 0.3 | 100.0% |
| 0.1 | 0.35 | 85.7% |
| 0.1 | 0.4 | 64.3% |
| 0.1 | 0.45 | 57.1% |
| 0.1 | 0.5 | 42.9% |
| 0.15 | 0.3 | 100.0% |
| 0.15 | 0.35 | 85.7% |
| 0.15 | 0.4 | 64.3% |
| 0.15 | 0.45 | 57.1% |
| 0.15 | 0.5 | 42.9% |
| 0.2 | 0.3 | 100.0% |
| 0.2 | 0.35 | 85.7% |
| 0.2 | 0.4 | 64.3% |
| 0.2 | 0.45 | 57.1% |
| 0.2 | 0.5 | 42.9% |
| 0.25 | 0.3 | 100.0% |
| 0.25 | 0.35 | 85.7% |
| 0.25 | 0.4 | 64.3% |
| 0.25 | 0.45 | 57.1% |
| 0.25 | 0.5 | 42.9% |

## Parameters NOT grid-searched (accounted for)
- `TOP_FIELDS_K` (3): max grounded phrases returned — verbosity, not accuracy.
- `MAX_ROUNDS` (5): rounds before the cap — affects latency/avg-rounds, not final accuracy in this sim.
- `VECTOR_K`/`KEYWORD_K`/`RRF_K`/`RETURN_K`/`RERANK_K`: retrieval breadth — on an 8-doc/17-chunk KB everything is already retrieved, so they don't move the metrics; revisit at larger corpus scale.
