# M2 METHODOLOGY — frozen before results

This protocol is fixed **before** any baseline is run. It governs `docs/M2-REPORT.md`. M2 is a
**falsification milestone**: evaluate the frozen set of NO-TRAINING baselines and decide whether a
custom trained model is still justified. No training, no GPU, no paid API, no Veritas evaluation.

## Data & splits (M1 is authoritative)

- **Selection (validation):** `manifests/primary/validation.jsonl` — sha256 `6b31c102…`, 1,644 rows.
- **Final reporting (internal_test):** `manifests/primary/internal_test.jsonl` — sha256 `2f674ccb…`, 1,608 rows.
- Recipe = **primary** (PAWS-Wiki + CANNOT). MNLI excluded (M1 verdict). **The sealed Veritas
  75-pair set is NOT used in M2** (reserved for M6). Pair text is reconstructed from `data/raw`
  and verified against each manifest row's committed `a_sha`/`b_sha` (fail closed on mismatch).

**Thresholds, mappings, calibrators, and any hyperparameter are selected on validation ONLY and
frozen before internal_test is scored. internal_test is never used for selection.**

## Primary metric

Positive class = **CACHE_EQUIVALENT**. A **false positive = NOT_CACHE_EQUIVALENT predicted
CACHE_EQUIVALENT** (score ≥ τ) — the operationally dangerous error (a wrong cached answer).
Primary curve = **recall vs. false-positive rate**; accuracy/AUC are secondary.

Every baseline emits a per-pair **score ∈ [0,1] = P(cache-equivalent)**; higher ⇒ more likely
equivalent. Reports print the FP definition next to every FP/FPR table.

## Operating points (report on internal_test at validation-frozen thresholds)

- **Validation zero-observed-FP threshold** — maximize validation recall subject to zero observed
  validation false positives, freeze that threshold, then apply it unchanged to internal_test.
  Report the realized internal-test FP count and FPR separately; never relabel the resulting test
  measurement as "recall at 0 FP" unless the test count is itself zero.
- **Validation-selected target-FPR thresholds** — select thresholds on validation for supported
  target budgets (0.1% / 0.5% / 1%), freeze them, then report the realized internal-test FP count,
  FPR, and recall. A validation target is not a guarantee about realized internal-test FPR.
- **Descriptive internal-test sweep** — the full frozen-score curve may separately report the
  recall achievable where actual internal-test FPR is at most a stated budget. These
  threshold-free summaries are descriptive only and are not selection or deployment results.
- Full FP-vs-recall curve, precision–recall curve, threshold table, confusion matrix. Internal test
  has 883 negatives, so its realized FPR granularity is 1/883 ≈ 0.11%.

## Per-category reporting (mandatory)

Report per category **with N**: paraphrase, equivalent_negation, explicit_negation, hard_paraphrase,
lexical_contradiction, quantifier/numeric/scope (sparse — shown separately, never claimed robust),
exact_repeat, unknown. **`equivalent_negation` is the headline slice**: a model that merely rejects
negation must fail there visibly.

## Calibration

Fit any calibrator (temperature / Platt / isotonic) on **validation only**; evaluate the frozen
calibrator on internal_test. Report ECE, Brier, reliability bins. State whether calibration
actually improves the thresholded decision; do not force a method.

## Statistical uncertainty

Percentile bootstrap CIs (seeded) on headline points; paired bootstrap + McNemar for
baseline-vs-baseline comparisons on internal_test. Statistics distinguish real gaps from
small-slice noise; no p-value flood.

## Ties, missing, malformed

Score ties at τ: `score ≥ τ` counts as predicted-positive (consistent with `eval/metrics.py`).
A missing/`NaN`/out-of-range score is a hard error (fail closed), never silently coerced.

## Baselines (frozen set)

1. **exact** — `text_a == text_b`.
2. **normalized** — `textnorm.normalize(a) == normalize(b)` (casefold + strip punctuation +
   collapse whitespace only; **keeps** negation tokens, numbers, entities, modality, quantifiers).
3. **embedding-cosine** — `BAAI/bge-small-en-v1.5` @ `5c38ec7c…` (MIT), score `= (cos+1)/2`.
4. **adaptive (vCache-inspired, per-neighborhood)** — offline reduction of vCache
   (arXiv:2502.03771): per-neighborhood logistic `P(equiv|sim)=expit(γ(s−t))` over each pair's k
   nearest validation anchors; δ / neighborhood handling selected on validation. Honestly labeled
   an approximation (online adaptation, stochastic explore/exploit, live judge, and the δ-guarantee
   validity are NOT reproduced). Not called "vCache".
5. **nli-cross-encoder** — `cross-encoder/nli-deberta-v3-base` @ `6c749ce3…` (Apache-2.0, SNLI+MNLI,
   no ANLI), `score = min(P_entail(a,b), P_entail(b,a))` (bidirectional; entailment index from
   `config.label2id`, never hardcoded). Mapping frozen on validation.

## Models (pinned; free; CPU; no `trust_remote_code`)

Recorded in `reports/m2/models.json` (id, revision, license, pooling/normalize, params, device).
Weights are downloaded to the HF cache (gitignored), never committed. Excluded: MoritzLaurer NLI
(ANLI = CC-BY-NC training data — same policy as M1), quora-roberta (Quora ToS).

## Latency

Local CPU only: device, batch size, warm-up, #runs, p50/p95, throughput for the embedding encode
and the NLI cross-encoder. **"Local inference benchmark only; API comparison deferred"** — no
comparable historical Haiku latency measurement exists.

## Artifacts & reproducibility

Per-baseline **predictions are first-class** (`predictions/m2/<name>/{validation,internal_test}.jsonl`)
and committed. All metrics/curves/decisions are recomputed from predictions alone by
`polarity baseline report --check`, **byte-for-byte** (drift gate); CI never re-runs model inference.

## Decision

The report ends with exactly one disposition: `CUSTOM_TRAINING_JUSTIFIED` /
`OFF_THE_SHELF_BASELINE_SUFFICIENT` / `DATASET_OR_TASK_DEFINITION_NEEDS_REVISION` /
`BASELINE_RESULTS_INCONCLUSIVE` — from validation + internal_test evidence only, never because
training was on the roadmap.
