# M6 — Veritas transfer protocol

Protocol v1 was committed **before** the committed Polarity predictions on the Veritas 75-pair set
were produced. Git supports protocol-before-score chronology and shows no later tracked training or
configuration change. It cannot independently prove the author's record that scoring happened only
once. Protocol v2 below is a post-evaluation serialization erratum derived only from the same
committed validation predictions; it does not rescore either dataset.

## Model under test (frozen)

| | |
|---|---|
| experiment | `xp-006-deberta-transformers-fixed` (M3.3) |
| recipe fingerprint | `pl1:04857dd3f1522863cc9e99378569403ab76f72f5f64d7fc59abc98e38a125c17` |
| execution fingerprint | `pl1:bc55b62acce7235c6442b53e2b3cf1147241619cac801899d7993db7ee289cec` |
| artifact-bundle fingerprint | `pl1:0dfcb787d1b5ea811b7405606f8a1e1ae95d50802053c0227324d16cb3c33df1` |
| checkpoint sha256 | `8d765a7e55c94bb5a8c7754eb705ddf8dc7e6c74cc72e1b0ce4c9110e7a0f355` (`artifacts/xp-006/model.safetensors`) |
| provenance | author retrain under the recorded M3.3 recipe (Transformers 4.57.6, Torch 2.6.0+cu124, fp32, seed 42); max score diff 0.0 vs committed M3.3 on 1,608 internal-test + 1,644 validation rows |
| model / tokenizer revision | `microsoft/deberta-v3-base` @ `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` |

The recorded scores agree exactly on those rows, which provides continuity for the committed
metrics. It does not establish a byte-identical checkpoint or a fully reproducible retrain. The
exact historical Python/container is unknown; see `docs/RELEASE-INTEGRITY.md`.

## Frozen decision rule (validation-derived; never Veritas)

Thresholds are on the raw score `P(CACHE_EQUIVALENT)`, selected on the model's **validation**
predictions at the pre-registered operating points, then applied unchanged to Veritas. A target FPR
names the **validation selection rule**; it is not a claim that a different evaluation set realizes
the same FPR.

| operating point | full-precision frozen τ | validation recall | validation FP / negatives | observed validation FPR |
|---|--:|--:|--:|--:|
| 0-observed-FP (`zero_fp`) | **0.9996990000000001** | 341/781 = 0.436620 | **0 / 863** | 0.000000 |
| target FPR ≤ 1% (`fpr_le_0.01`) | **0.994833** | 707/781 = 0.905250 | 8 / 863 | 0.009270 |
| target FPR ≤ 5% (`fpr_le_0.05`) | **0.035936** | 752/781 = 0.962868 | 43 / 863 | 0.049826 |

Protocol v1 serialized the strict zero-observed-FP threshold to six decimals (`0.999699`), equal
to the maximum validation-negative score. Because decisions use `score >= tau`, that rounded value
could admit the very negative it was meant to exclude. Protocol v2 persists the exact
`nextafter(0.999699, +∞)` float shown above, records the selection split/rule/budget and observed
selection counts/rates for every point, and validates those fields on load. JSON round-trip tests
prove the restored threshold remains strictly greater than the maximum validation negative. Raw
validation and Veritas score artifacts are unchanged.

Calibrator (for probability quality / ECE, reported alongside): **isotonic**, fit on validation,
17 breakpoints, frozen in `reports/m6/protocol-xp-006-deberta-transformers-fixed.json`. The
decision uses the raw threshold; the calibrated probability is reported, not used to decide.

## Metrics (fixed)

Positive class = `CACHE_EQUIVALENT`; a **false positive** = a `NOT_CACHE_EQUIVALENT` pair predicted
`CACHE_EQUIVALENT` (serving a wrong cached answer). Reported at each frozen operating point:
precision, recall, false-positive **count**, FPR, confusion matrix, and the **per-pair-type**
breakdown. Ties (`score == τ`) count as predicted-positive (`>=`), matching the M2/M3 convention.

## Veritas set (fixed; do not add/remove)

75 pairs: 31 equivalent, 44 not. Pair types: exact-repeat 10, paraphrase-should-hit 21,
negation-must-not-hit 22, different-answer-must-not-hit 22. Every slice is reported separately —
the load-bearing question is whether paraphrase recall transfers **without** accepting
negation / different-answer negatives.

## Baselines (comparison; provenance-labelled)

- **Frozen embedding cosine** on Veritas — recomputed or historical, labelled as such.
- **Historical Haiku LLM-judge guard** — recall 0.97 @ 0 observed FP, τ=0.72, model
  `claude-haiku-4-5`, committed verdicts `data/golden-guard.json` (veritas repo, commit `a482d4a`,
  2026-06-30). **HISTORICAL — not re-run in M6.**
- **Off-the-shelf NLI cross-encoder** (M2) — if scored on Veritas, labelled as an M6 computation.

Polarity does **not** need to beat the historical LLM guard to be useful: a local prefilter or a
gain over the embedding/deterministic baselines are valid outcomes. No controlled cost or latency
comparison with that guard is part of this protocol.

## Statistics (N=75 is small — report finite-sample honestly)

Exact counts foremost; Wilson/bootstrap CIs where appropriate; paired McNemar vs a baseline where
useful. Small-N slices show N prominently. No significance overstatement.

## Disposition handling

Protocol v1 pre-registered a five-level transfer taxonomy before evaluation. Its labels used broad
out-of-distribution language that the completed evidence does not support: Veritas was
author-created, present from M0, and drawn from the motivating target domain. Protocol v2 retains
the frozen numerical rules but retires those labels from public claims. The result is reported as
**bounded target-domain transfer** with exact counts and realized rates, followed by the weak MRPC
external-transfer result. No post-result threshold, calibration, or raw score was substituted.

The M7 augmentation idea is banked rather than executed. Any future work would require a new,
pre-registered and independently authored final evaluation set before construction or scoring.
