# M3 PLAN — pre-registered before training (the frozen protocol)

**No spend is authorized by this document.** It freezes the single controlled experiment so its
result is interpretable whether it succeeds or fails, then hands the compute/cost decision to
`docs/M3-COST-PACKET.md`. Nothing below may change after seeing internal_test.

## Research question

**Primary:** can task-specific fine-tuning of a `microsoft/deberta-v3-base` cross-encoder on the
leakage-safe **PAWS-Wiki + CANNOT** cache-equivalence data materially improve recall at a tightly
controlled unsafe-false-positive budget relative to the strongest M2 off-the-shelf cross-encoder?
At its threshold selected to target validation FPR≤1%, the NLI model achieved **0.073
internal-test recall** (95% CI 0.055–0.094) with **16/883 false positives = 1.81% realized
internal-test FPR**. A separate descriptive internal-test sweep at actual FPR≤1% gave 0.047 recall.
**Secondary:** does it improve recognition of **equivalent-negation** positives (baseline 0.052 on
n=58) **without** increasing unsafe acceptance of **explicit-negation** contradictions? The model must
learn *semantic equivalence*, not the presence of "not".

## Dataset — Recipe A (committed capped primary manifests)

Train on the committed `manifests/primary/train.jsonl` (12,731 pairs: 5,819 equivalent / 6,912 not;
equivalent-negation 565, explicit-negation 984), select on `validation.jsonl` (1,644), report once on
`internal_test.jsonl` (1,608; 725 pos / 883 neg). **Not** a full rebuild — a rebuild would change the
manifest sha256s (a dataset-version change requiring separate review) and break comparability with the
M2 baselines. The three approved shas are pinned in the config and re-verified by the refuse-to-start
gate:

- train `656625d6…`, validation `6b31c102…`, internal_test `2f674ccb…`

MNLI, QQP, PAWS-QQP, ANLI, NaN-NLI, NevIR, and the sealed Veritas set are **excluded** (config +
`materialize.py` guards refuse them). Veritas stays sealed until M6.

## Model & recipe

`microsoft/deberta-v3-base` @ `8ccc9b6f…` (MIT, native `deberta-v2`, no `trust_remote_code`), ordinary
2-class head (index 1 = CACHE_EQUIVALENT). No architectural novelty — the experiment tests
task-specific **data + objective**. The full frozen recipe (LR 2e-5, AdamW eps 1e-6, linear/6% warmup,
eff-batch 32, 4 epochs, grad-clip 1.0, bf16, plain cross-entropy, max_length 128, seed 42) and its
evidence are in `configs/train/m3-deberta-v3-base.yaml` + `docs/M3-TRAINING-RESEARCH.md`.

## Checkpoint selection — VALIDATION only (frozen)

Lexicographic: (1) prefer steps achieving validation FPR ≤ 1% (`select.threshold_at_fpr(val, 0.01)`);
(2) maximize validation recall at that threshold; tie-break earliest step, then lowest validation loss.
A **pAUC of the FP–recall curve over FPR∈[0, 2%]** is reported as a stability cross-check (the FPR≤1%
point is noisy — pinned by ~top-1% of 863 validation negatives). Accuracy / F1 / internal_test /
Veritas are **never** used for selection. **Internal_test is intended to be one-way:** the plan
permits one score after the checkpoint, threshold, and calibrator are frozen; it never becomes a
second validation set. Git records chronology but does not technically enforce how many local
scoring invocations occurred. A failed M3 does not license "retrain until it looks good" — a new
hypothesis becomes a new experiment.

## Calibration — fit on validation, applied frozen

Fit temperature / Platt / (isotonic if `n ≥ isotonic_min_obs`) on validation, select the lowest
validation NLL, freeze, apply unchanged to internal_test. Report ECE / Brier / reliability + the effect
on the low-FPR operating point. Temperature is monotonic, so it does not move the operating point.

## Success criteria (frozen — uncertainty-aware paired)

Positive = CACHE_EQUIVALENT; **false positive = NOT predicted CACHE_EQUIVALENT** (the unsafe error).
All comparisons are against the M2 NLI baseline on the **identical** internal_test pairs.

- **PRIMARY (material):** the paired-bootstrap Δrecall (M3 − NLI) at each model's threshold selected
  to target validation FPR≤1% has a **95% CI excluding 0**, **AND** M3's recall lower-CI-bound
  exceeds the NLI point estimate **0.073** at that validation-selected threshold. Realized
  internal-test FPR is reported separately for each model. (A move like 0.073 → 0.078 with a wide
  CI is **not** material.)
- **SECONDARY:** equivalent-negation recall (n=58) exceeds **0.052 + 0.05** with reported uncertainty
  (no unrealistic significance demanded at n=58).
- **GUARDRAIL:** explicit-negation unsafe FP rate must not rise by more than 0.05 over the baseline —
  equivalent-negation gains must not come from becoming permissive. Explicit-negation FP is reported
  beside every equivalent-negation number.

## Pre-registered dispositions

- **STRONG_SUCCESS** — primary material AND secondary material AND guardrail holds.
- **PARTIAL_SUCCESS** — primary XOR secondary material, guardrail holds.
- **NO_MATERIAL_GAIN** — neither primary nor secondary material (≈ off-the-shelf NLI).
- **REGRESSION** — worse low-FP frontier (paired Δrecall CI high < 0) or a guardrail breach.
- **DATA_OR_OBJECTIVE_PROBLEM** — a label/shortcut/calibration pathology invalidates interpretation
  (a manual call stated with rationale in `docs/M3-REPORT.md`).

Negative outcomes are valid results (house policy) and will be reported as such.

## Statistics, artifacts, cost

Paired bootstrap Δrecall + McNemar vs the M2 NLI baseline; bootstrap recall CI; per-category with N.
Run artifacts land in the immutable, gitignored `runs/<experiment-id>/` (config, fingerprint, env, git
SHA, manifest hashes, telemetry, checkpoints); the compact predictions/reports that reproduce the
headline are promoted to `predictions/m3/` + `reports/m3/` and committed, so the keyless drift gate
(`polarity train report --check`) reproduces every number without training. Cost caps
(`MAX_RUNTIME_HOURS`, `MAX_ESTIMATED_COST_AUD`) refuse an over-budget start; see the cost packet.
