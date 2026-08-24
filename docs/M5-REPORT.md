# M5 REPORT — Training Signal Audit ($0, local, no GPU)

A debugging/falsification milestone. The author-run controls isolated a strong
Transformers-version-sensitive DeBERTa failure and led to a guarded release profile. **$0** —
Apple-Silicon CPU/MPS, no GPU rental, no paid API, Veritas untouched.

> **Release-hardening note:** committed predictions and reports support collapse under
> Transformers 5.15.1 and success under 4.57.6. The decisive small diagnostic runs below do not
> have machine-readable committed logs, so their causal experiment cannot be independently
> replayed from this repository. Treat “root cause” statements below as the author's diagnosis,
> not as independently reproduced library attribution.

## Question

Why does a 184M DeBERTa-v3-base classifier converge to a constant, input-invariant predictor
despite stable gradients (M3.2: fp32, no NaN, max grad-norm 8.87, smooth loss)?

## Current failure

M3/M3.1/M3.2 all produced (near-)constant scores: validation `n_unique → 1`, class separation 0,
internal_test pAUC 0, recall 0 at every FP budget. bf16 (M4 hypothesis) was refuted by M3.2 (fp32
collapsed too). The signature — *stable optimization converging to the class prior* — pointed at
the signal path, not precision.

## Hypotheses

Twenty pre-registered hypotheses (`docs/M5-DIAGNOSTIC-PLAN.md`), H1–H19 covering
label/encoding/optimizer/loss/score/checkpoint/sampler/LR, H20 = a DeBERTa-specific
implementation/library issue. The strategy: a single model-swap control eliminates most at once.

## Overfit-one-batch result

The load-bearing test. 16 pairs, repeatedly trained (no early stop, no collapse gate), asking
`CAN_THE_PIPELINE_MEMORIZE?` Pre-registered `OVERFIT_PASS`: train accuracy ≥ 0.95, CE < 0.35, and
prob-spread > 0.5.

| setup (deberta-v3-base) | transformers | final acc | final loss | prob spread | verdict |
|---|---|--:|--:|--:|:--:|
| trivial synthetic, lr 2e-5 | 5.15.1 | 0.500 | 0.83 | 0.000 | **FAIL** |
| real 16-pair, lr 2e-5 | 5.15.1 | 0.500 | 0.69 | 0.001 | **FAIL** |
| trivial, lr 5e-6 (stable grads) | 5.15.1 | 0.500 | 0.72 | 0.000 | **FAIL** |
| trivial, lr 1e-5 | 5.15.1 | 0.500 | 0.72 | 0.000 | **FAIL** |

The pipeline **cannot memorize even 16 trivial, mechanically-separable pairs** under transformers
5.15.1. That is not a data problem — it is a machinery problem.

## Toy-signal result

The trivial batch is deliberately artificial: `("alpha token number i", "alpha token number i") →
equivalent` vs `("alpha token number i", "zeta symbol distinct i") → not`. A correctly wired model
memorizes this trivially. Under 5.15.1 it cannot (acc 0.500) — so the **dataset is exonerated**;
the failure is upstream of the data.

## Label audit

`label_index(CACHE_EQUIVALENT) = 1`, `label_index(NOT_CACHE_EQUIVALENT) = 0` — one shared
convention (`predict.POSITIVE_INDEX`) used by both `labels_tensor` (training) and the score writer
(eval). No inversion. Unit-tested (`test_diagnose.py`). **Healthy.**

## Encoding audit

`tokenizer([a…],[b…], truncation, max_length, padding)` yields **16/16 distinct `input_ids`**,
both sequences present, seq-len 11 for the trivial batch (no truncation). Changing text_b changes
the encoding. **Healthy.**

## Trainable parameter audit

184,423,682 parameters, all trainable (nothing frozen). Backbone-layer-0 weight moved by 0.41 and
the classifier head by 3.4 after training. **Backbone and head both train. Healthy.**

## Optimizer audit

One AdamW group over `model.parameters()` (decay/no-decay split by dim); the classifier head is
included. **Healthy.**

## Parameter update proof

Snapshots before/after steps: Δhead-weight-norm 3.4, Δbackbone-layer0-norm 0.41 — updates are
**large, not negligible**. Gradients are computed *and* applied. **Healthy** (rules out H9).

## Gradient-flow audit

Gradients reach the backbone (non-zero Δ there). Under no-warmup lr 2e-5 the raw grad-norm spikes
to ~57 (clipped to 1.0) with an oscillating loss; with warmup it settles to ~3–11. Instability is
a *symptom* of the steep degenerate landscape, not the cause (a lower, stable LR still collapses).

## Logit/loss audit

Raw logit rows are all distinct (16/16), but the **decision margin `l1−l0` is near-constant across
examples** (std ~0.004). The softmax therefore collapses even though the common-mode logit varies.
The model encodes *something* per input but cannot express a per-example *decision*.

## Score extraction audit

`logits → softmax[:, POSITIVE_INDEX] → PredictionRecord` is correct: the distilbert control (below)
produces cleanly separated scores (0.001…0.999) through the **identical** transform. **Healthy.**

## Checkpoint roundtrip

Not the culprit — the collapse is visible in the **live in-memory model** before any save/load, so
checkpoint save/restore is not implicated (H16/H17 N/A).

## Sampler audit

Not implicated — the control memorizes the **same fixed batch** through the same batching. A
broken sampler would fail the control too.

## Learning-rate / warmup probe

LR ∈ {5e-6, 1e-5, 2e-5}, with and without warmup, on the trivial batch under 5.15.1: **all FAIL**
(acc 0.500). A stable low LR (5e-6, grad-norm ~1–2) collapses just like the unstable high LR. So
LR is **not** the root cause — it only modulates how violently it fails.

## Dataset-signal controls

Two author-run controls form the version-sensitive diagnosis:

1. **Model swap (same pipeline).** `distilbert-base-uncased` through the identical
   encode/label/loss/score path **memorizes the 16 trivial pairs in <30 steps** (acc 1.0, loss
   0.0013, spread 0.998) under transformers 5.15.1. → The pipeline is healthy; the problem is
   model-family-specific.
2. **transformers-version swap (only variable).** Same deberta-v3-base @ the same pinned revision,
   same LR/warmup:

   | transformers | overfit | pooled-repr std (across inputs) |
   |---|:--:|--:|
   | 5.15.1 | **FAIL** (acc 0.500) | **0.004** (constant) |
   | 4.46.3 | **PASS** (acc 1.0, loss 0.006) | 0.42 (discriminative) |
   | 4.57.6 (repo path) | **PASS** (acc 1.0, loss 0.006) | 0.41 |

## Author diagnosis

The author diagnosed **a Transformers-5.x-sensitive DeBERTa-v2 fine-tuning failure**. In the
recorded diagnostics, the 5.x DeBERTa-v2 path left the pooled `[CLS]` representation nearly
input-invariant during training, while the 4.x and DistilBERT controls learned. That pattern is
consistent with the M3/M3.1/M3.2 constant predictors and with M3.3 success under 4.57.6. Without
committed machine-readable diagnostic logs or a bisected upstream change, the repository cannot
independently prove a specific library defect or exclude every interaction with the environment.

Historical author disposition: **`PIPELINE_HEALTHY_LIBRARY_REGRESSION_FOUND`**. That label records
the decision made from the author-run controls; it is not independently reproduced attribution to
a specific Transformers defect. The release claim is the narrower observed version-sensitive
failure, with the upstream cause unproven.

## Fix

Pin the training path to a DeBERTa-safe transformers major and enforce it:

1. **`pyproject.toml`** xp-006 `train` profile → Transformers exactly `4.57.6` and Torch
   `>=2.6,<2.7`. The separate `models` extra keeps its broader range, so `train` and `models` are
   declared **conflicting extras** in `[tool.uv]`.
2. **Runtime guard.** `diagnose.assert_transformers_supported()` runs at the start of the paid
   `train run` and **refuses to start** on an out-of-range version — the enforcement that would
   have prevented all three wasted runs. Verified: it raises under 5.15.1, passes under 4.57.6.
3. **Diagnostic CLI.** `polarity diagnose overfit [--kind trivial|real]` reproduces the audit and
   emits a machine-readable verdict.

## Regression evidence

- The failing diagnostic now passes: `polarity diagnose overfit --kind trivial` under 4.57.6 →
  **OVERFIT_PASS** (acc 1.0, loss 0.0057, spread 0.995, pooled-repr std 0.412).
- **Larger real-data sanity check** (§25): overfit 32 real train pairs → **OVERFIT_PASS** (acc 1.0,
  loss 0.0028, spread 0.997, pooled-repr std 0.423, max grad-norm 3.7).
- The behavior suite, Ruff, formatting, and strict mypy gates pass; the version-guard matrix is
  unit-tested, with 5.15.1 rejected and 4.57.6 accepted for the xp-006 profile.

## Remaining uncertainty

- The exact 4.x→5.x upstream change was not bisected. The release profile deliberately preserves
  the observed successful version rather than generalizing beyond the recorded evidence.
- The author observed the compatibility change on CPU/MPS at small scale. Full-scale behavior was
  then evaluated in M3.3. The missing committed diagnostic logs remain a replay limitation.

## Whether another GPU run is justified

**Historical decision: yes.** The author-run diagnostics showed that the pipeline could memorize a
tiny labelled set in the compatible environment and motivated a guard against the observed bad
profile. That was sufficient to authorize the later single M3.3 run, while remaining short of an
independently replayable upstream root-cause proof.

## Proposed next experiment

**M3.3 — the corrected paid re-run** (owner-gated, separate cost packet, ~A$0.35): the M3.1 recipe
(bf16 or fp32 — precision was never the issue; keep bf16 for speed, or fp32 for one clean run)
**under transformers <5**, with the collapse gate active as a safety net (it should *not* fire).
Success is now the ordinary bar: a non-collapsed model, then comparison with NLI at each model's
validation-selected target-FPR≤1% threshold, with realized internal-test FP/FPR reported
separately. This requires explicit owner approval; M5 does **not** spend.

## Reproduce / gates

`uv run pytest` (incl. `test_diagnose.py`) · `uv run polarity diagnose overfit --kind trivial
--device mps` (author-run, needs `uv sync --extra train`) · `$0`, no GPU, no network beyond the
one-time model download.
