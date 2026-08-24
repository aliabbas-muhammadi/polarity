# METHODOLOGY — splits, evaluation hierarchy, metrics, calibration, statistics

The point of this document is that later training results are **scientifically defensible**,
not merely visually impressive. Everything here is fixed before training.

## 1. Leakage-safe splitting

A sentence that appears in both train and eval through a different transformation invalidates
the measurement. Rules:

- **Split by source family, before augmentation.** The unit of assignment is the `group`
  (source family), never the individual pair. Every augmented variant of one source sentence
  shares a group and lands in the same split. Implemented in
  `src/polarity/data/split.py::assign_splits`; a group's split is a pure function of
  `(seed, group)`, independent of dataset order.
- **Connected components as the family check.** `union_find_groups` merges pairs that share
  any text node; the leakage audit uses it to catch declared groups that are secretly the same
  family.
- **Immutable manifests.** Splits are frozen to committed manifests (`manifests/`) recording
  `pair_id, source, group, split, pair_sha`. `pair_sha` is a content hash of `(text_a, text_b,
  label)`, so a silent edit to a committed pair is detectable. Once training begins, manifests
  are not silently regenerated.

## 2. Evaluation hierarchy

Three conceptually distinct layers. They answer different questions; the small author-created
target-domain layer is not automatically a stronger generalization claim than an external corpus.

- **A — in-domain held-out.** Group-safe held-out split of the training-source families.
  Measures training-task generalization only.
- **B — cross-dataset transfer.** Datasets/domains excluded from training. Measures external
  generalization; negation robustness is known not to transfer (NevIR reproduction, SIGIR
  2025), so report per-slice.
- **C — the Veritas 75-pair target-domain holdout.** The author-created cache set that motivated
  the project (`data/snapshots/veritas-golden/`). It is governed strictly below. Because the
  examples were created by the model author and present from M0, it is protocol-held-out
  target-domain evidence, not an independently blinded benchmark.

## 3. Final-holdout governance (G1 / G2 / G3)

- **G1 — sealed snapshot.** Pair text + labels only; **no** embeddings, **no** guard verdicts
  imported as features. Never used for training, threshold selection, calibration,
  augmentation, or architecture/hyperparameter/checkpoint choice.
- **G2 — one-way discipline.** During M0–M2 the holdout is touched only for schema / checksum /
  leakage validation. Its per-pair model performance is **not** inspected while choosing data
  recipes, baselines, or category weights. The pipeline is:

  ```
  train  ->  validation (threshold + calibration selected here)  ->  internal held-out test
         ->  FREEZE model + calibration  ->  Veritas target-domain evaluation (M6)
  ```

- **G3 — acknowledged prior knowledge and authorship.** The *historical* Veritas
  embedding/guard/judge numbers
  are already known (they motivate the project). That is unavoidable prior knowledge and is
  stated as a limitation; the set is not pretended to be epistemically untouched. The author also
  created the 75 examples. What G2 protects is protocol-before-score chronology and the absence of
  a tracked Polarity model/threshold fit against the set. Git cannot technically enforce how many
  times an external scoring command was run.

## 4. Metrics (primary: recall vs. false-positive rate)

Positive = CACHE_EQUIVALENT; `false positive = NOT_CACHE_EQUIVALENT predicted CACHE_EQUIVALENT
(score >= tau)` (G5). Implemented in `src/polarity/eval/metrics.py`:

- **FP-rate-vs-recall curve** (primary) and **precision–recall curve**.
- **Confusion matrix** and **threshold table** across an operating grid.
- **Per-category error rates** — no aggregate may mask a category failure.
- **Validation-selected operating points**: select max recall at **zero observed validation FP** or
  at a supported validation FPR target, then freeze the threshold. Every evaluation set reports
  its own realized FP count and FPR; a selection target is never presented as a realized-test
  guarantee. Separate threshold-free sweeps may describe recall at an actual evaluation-set FPR
  budget, but are labelled descriptive. With `N` negatives the realized FPR granularity is `1/N`.
- ROC/AUC only as a **secondary** view — the literature shows AUC-good, deploy-bad cross-encoders.
- **Confidence intervals** via seeded percentile bootstrap (`bootstrap_recall_ci`).

## 5. Calibration (M4; foundations in M0)

The deliverable is a *thresholded* accept/reject, and cross-encoders trained with BCE/MNRL can
be badly miscalibrated at the operating threshold even with strong ranking AUC
(`docs/LITERATURE.md`). So we report and fix calibration:

- Metrics: **ECE**, **Brier**, reliability diagram (`src/polarity/calibration/` ships ECE +
  Brier now; reliability plotting at M4).
- Calibrator: temperature / isotonic / Platt, **selected on validation only** — never on the
  sealed holdout.

## 6. Statistics

- Headline metrics carry bootstrap confidence intervals.
- Model-vs-baseline on the *same* evaluation set: **paired bootstrap** + **McNemar's test**.
- Small differences are never reported without an interval.

## 7. Baselines (must be credible, not straw men)

`configs/eval/operating-points.yaml` fixes the baseline matrix: exact/normalized match; the
frozen embedding + single threshold (the Veritas `text-embedding-3-small` config where
recoverable); a **vCache-style per-prompt adaptive threshold** (the strong baseline to beat);
an off-the-shelf NLI cross-encoder without Polarity fine-tuning; and the **historical** Veritas
LLM-judge result (provenance-labelled, not re-run in early milestones). Beating only a static
cutoff is not a result.
