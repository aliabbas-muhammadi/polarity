# EXPERIMENTS — registry, run metadata, milestones

No "try a bunch of stuff and keep the best" without a record. Every experiment is registered
before it runs; every run is immutable.

## Experiment registry

One row per experiment. Run facts remain append-only. On 2026-08-23, release hardening corrected
decision prose whose scientific or serialization claims were known to be false or too strong. The
corrected, release-safe rows below replace those claims in the source-only candidate; the
unmodified original decisions remain preserved in the private audit history and are intentionally
absent from the candidate. See the dated errata immediately below.

| Field | Meaning |
|---|---|
| `id` | short stable slug, e.g. `xp-003-deberta-cannot` |
| `purpose` | one line |
| `hypothesis` | the falsifiable prediction |
| `train_data` | dataset ids + versions in the mix |
| `eval_data` | which eval layer(s) (A / B; C only at M6) |
| `split_hash` | sha of the split manifest used |
| `model` | base model + revision |
| `hyperparameters` | lr, batch, epochs, max_len, objective |
| `seed` | int |
| `cost` | measured $ + wall time |
| `result` | headline metrics with CIs (link to `runs/<id>/`) |
| `decision` | keep / discard / iterate, and why |

### Release-hardening errata — 2026-08-23

| Registry entry | Corrected category | Release-safe correction |
|---|---|---|
| M5 | causal certainty | Narrowed a specific-library attribution to an observed version-sensitive failure; the upstream cause remains unproven. |
| M6 | threshold serialization and OOD verdict | Preserved the exact validation-selected threshold, separated it from realized Veritas FPR, and limited the conclusion to the small author-created target-domain set. |
| M6.1 | causal and M7 conclusion | Kept the probe/MRPC observations without assigning one proved cause or necessary remedy; banked M7 pending an independently authored untouched final set. |

### Registered experiments

| field | value |
|---|---|
| `id` | **`xp-003-deberta-primary-capped`** |
| `purpose` | first controlled cache-equivalence fine-tune (M3) |
| `hypothesis` | task-specific training beats the M2 NLI baseline at each model's validation-selected target-FPR≤1% threshold, especially on equivalent negation, without inflating explicit-negation FP. NLI reference: 0.073 internal-test recall with 16/883 FP (1.81% realized FPR); its descriptive actual-test-FPR≤1% recall is 0.047. |
| `train_data` | primary capped: PAWS-Wiki + CANNOT (`manifests/primary/train.jsonl`, 12,731) |
| `eval_data` | validation (select) + internal_test (report once). **Not** Veritas (M6) |
| `split_hash` | train `656625d6…` · validation `6b31c102…` · internal_test `2f674ccb…` |
| `model` | `microsoft/deberta-v3-base` @ `8ccc9b6f…` (MIT) |
| `hyperparameters` | lr 2e-5, eff-batch 32, 4 epochs, max_len 128, bf16, plain CE (see config) |
| `seed` | 42 |
| `fingerprint` | `pl1:7dbb3c69…` (`polarity train fingerprint`) |
| `cost` | **US$0.247 ≈ A$0.35** · RTX 4090 secure (EUR-NO-1), bf16, ~2 min train · 2026-08-21 |
| `result` | At the validation-selected thresholds, internal_test recall was **0.000** for xp-003 (tau 1.0, degenerate) versus NLI 0.073; NLI's realized FPR there was 1.81%. The old `_pauc` reported 0.097 vs 0.048, but M4 retracted that metric and recomputed xp-003 step-pAUC[0,2%] as **0.000**, with collapsed scores. ECE was 0.021. Early-stopped at 1/4 epochs. `docs/M3-REPORT.md` |
| `decision` | Historical **`DATA_OR_OBJECTIVE_PROBLEM`** — the floored-metric/early-stop diagnosis led to M3.1. M4 superseded the claim that this was not a model failure by establishing collapse under the corrected metric. |

| field | value |
|---|--:|
| `id` | **`xp-004-deberta-pauc-select`** (M3.1) |
| `purpose` | corrected selection: pAUC checkpoint selection + full 4 epochs |
| `train_data` / `model` / `seed` | same as xp-003 (only selection+stopping changed) |
| `fingerprint` | `pl1:8c9a5517…` |
| `cost` | **US$0.237 ≈ A$0.35** · RTX 4090 secure, bf16, full 1,592 updates · 2026-08-21 |
| `result` | At the validation-selected target-FPR≤1% threshold, internal_test recall was **0.000** again (tau 1.0). The obsolete `_pauc` selected degenerate update-300 over graded update-700; M4 then recomputed xp-004 step-pAUC[0,2%] as **0.000** and established collapse. ECE was 0.037. `docs/M3.1-REPORT.md` |
| `decision` | Historical **`DATA_OR_OBJECTIVE_PROBLEM`**, superseded by M4 — exposing the `_pauc` bug correctly triggered a $0 measurement fix before further paid work; the corrected result is a collapsed model, not merely a measurement confound. |

| field | value |
|---|--:|
| `id` | **`xp-005-deberta-fp32`** (M3.2) |
| `purpose` | controlled precision experiment: test whether bf16 instability caused the M3/M3.1 collapse |
| `train_data` / `model` / `seed` | same as xp-004 (**only `precision` changed: bf16 → fp32**) |
| `fingerprint` | `pl1:04857dd3…` (fingerprint diff vs xp-004 = the precision field, nothing else) |
| `cost` | **≈ A$0.37** · RTX 4090 secure (US-NC-1), **fp32**, torch 2.6.0+cu124, stopped at update 400 · 2026-08-21 |
| `result` | **collapsed identically to bf16** with **stable gradients** (max grad-norm 8.87, no ~70 spikes, no NaN); constant output (val n_unique→1); threshold-free internal_test pAUC 0.0 and recall 0.0 at every actual-FPR budget. At each model's validation-selected threshold, paired Δrecall was −0.073 (95% CI [−0.092,−0.055]) vs NLI; NLI realized 16/883 FP (1.81% FPR). `docs/M3.2-REPORT.md` |
| `decision` | **`FP32_HYPOTHESIS_NOT_SUFFICIENT`** — bf16 instability **refuted** as the dominant cause (fp32 gradients were stable yet it still collapsed). Root cause is precision-independent (optimization/objective/encoding). **No further paid run before a $0 root-cause diagnostic** |

| field | value |
|---|--:|
| `id` | **M5 — Training Signal Audit** (diagnostic milestone, $0, local, no GPU) |
| `question` | why does a 184M classifier converge to a constant predictor despite stable gradients? |
| `method` | overfit-one-batch + model-swap control + transformers-version swap (`docs/M5-DIAGNOSTIC-PLAN.md`) |
| `result` | pipeline **cannot memorize 16 trivial pairs** under transformers 5.15.1 (acc 0.500); `distilbert` memorizes through the **identical** pipeline (acc 1.0); deberta-v3-base memorizes under transformers **4.46.3/4.57.6** (acc 1.0) — pooled `[CLS]` std 0.004 (5.x) vs 0.42 (4.x) |
| `author diagnosis` | **Transformers-version-sensitive DeBERTa-v2 fine-tuning failure.** Committed predictions support collapse under 5.15.1 and success under 4.57.6, but the decisive small diagnostic runs lack machine-readable committed logs and the upstream change was not bisected. The causal experiment is not independently replayable. |
| `fix` | xp-006 release profile pins Transformers `4.57.6` and Torch `>=2.6,<2.7` (train/models are conflicting extras); the runtime guard rejects incompatible release-profile versions. The author diagnostic reported that overfit passes on trivial + 32 real pairs. `docs/M5-REPORT.md` |
| `decision` | Historical author disposition **`PIPELINE_HEALTHY_LIBRARY_REGRESSION_FOUND`** — used to authorize M3.3 from the author-run controls. It records an observed version-sensitive failure, not proof that a specific Transformers defect caused it. |

| field | value |
|---|--:|
| `id` | **`xp-006-deberta-transformers-fixed`** (M3.3) |
| `purpose` | the corrected re-run — same recipe as M3.2, **only transformers 5.x → 4.57.6** (the M5 fix) |
| `train_data` / `model` / `seed` / `precision` | same as xp-005 (fp32); stable **recipe** fingerprint `pl1:04857dd3…` equals xp-005, while the execution fingerprints differ because the library environments differ |
| `cost` | **≈ A$0.38** · RTX 4090 secure, fp32, torch 2.6.0+cu124, **transformers 4.57.6**, full 4 epochs (1,592 updates, 278s) · 2026-08-21 |
| `result` | **non-collapsed** (internal_test 799 distinct scores, class-sep 0.913; collapse gate never fired). Threshold-free: **pAUC[0,2%] 0.716 vs NLI 0.048; recall at actual FPR≤1% 0.763 vs 0.047; recall at actual FPR≤5% 0.968 vs 0.232.** At each model's validation-selected target-FPR≤1% threshold: xp-006 recall 0.903 with 21/883 FP (2.38%) vs NLI 0.073 with 16/883 FP (1.81%); paired Δrecall +0.830 (95% CI [0.80, 0.86]), McNemar 552. At xp-006's validation-selected tau, equivalent-negation recall was 0.983 vs 0.052 and explicit-negation FPR was 1/107=0.009. `docs/M3.3-REPORT.md` |
| `decision` | **`STRONG_SUCCESS`** (in-distribution) — clears both pre-registered bars and is consistent with the M5 version-sensitive diagnosis at GPU scale. **Caveat: internal_test is same-distribution (PAWS-Wiki+CANNOT).** M6 adds small author-created target-domain evidence; M6.1 MRPC transfer is weak. |

| field | value |
|---|--:|
| `id` | **M6 — Veritas transfer arm** (the sealed 75-pair holdout, evaluated once) |
| `model` | frozen `xp-006` checkpoint (max score diff 0.0 vs committed M3.3 validation/internal-test predictions; weight sha256 `8d765a7e…`); protocol commit predates committed Veritas scores (`docs/M6-TRANSFER-PROTOCOL.md`) |
| `result` | At validation-selected `tau=0.994833`: Veritas recall **31/31**, precision 0.939, **2/44 false accepts = 4.55% realized FPR**, negation slice **0/22 false accepts**, paraphrase 21/21, exact-repeat 10/10. At the validation zero-observed-FP threshold: recall 22/31, 0 false accepts. A separate descriptive sweep at actual Veritas FPR≤1% gives recall 24/31=0.774. `docs/M6-REPORT.md` |
| `decision` | **Bounded target-domain evidence.** Veritas is a small author-created, protocol-held-out cache-query set, not an independently blinded/general benchmark. Git verifies protocol-before-score chronology but cannot enforce the author's one-score claim. The observed counts stand; broader transfer claims do not. Transfer inference was local; author telemetry estimates the checkpoint retrain at about A$0.35, without receipts. |

| field | value |
|---|--:|
| `id` | **M6.1 — failure-mode probe + 2nd transfer set** (diagnostics, $0 local, frozen xp-006) |
| `part 1 (different-answer probe)` | 44 post-hoc author-curated pairs, partly reusing Veritas material. At validation-selected `tau=0.994833`: measurement 4/6, temporal 2/6, numeric 1/6 false accepts; entity/comparative/operation 0/6; controls 8/8 recalled. Descriptive localization only, not rates or cause. `docs/M6.1-DIFFERENT-ANSWER-PROBE.md` |
| `part 2 (MRPC external transfer)` | 1,725 news-paraphrase pairs, M6-frozen protocol. ROC-AUC **0.639870**; repository-defined step-pAUC **0.031938**; at `tau=0.994833`, TP/FN/FP/TN=`898/249/332/246`, realized FPR **0.574394**. `docs/M6.1-MRPC-TRANSFER.md` |
| `decision` | MRPC exhibits score saturation alongside distribution/calibration shift, semantic overacceptance, and weak ranking; the artifacts do not isolate which observation caused the result, and overlap alone does not explain it. The model is not a general paraphrase detector. The M7 augmentation hypothesis is banked, not recommended now; reopening it requires an independently authored untouched final set. |

### Cumulative spend posture

Historical rows above preserve author-recorded per-run estimates. Across all author-run paid
training and checkpoint-recovery activity, the current aggregate is **approximately A$1.78,
author-reported; no receipts**. It must not be presented as an independently verified exact spend.

## Run metadata layout (immutable)

```
runs/<run-id>/
  config.json         # frozen experiment config
  fingerprint.txt     # content-addressed identity (src/polarity/fingerprint.py)
  env.json            # python, deps (uv.lock hash), platform, GPU
  git_sha.txt         # code version
  manifest_hashes.txt # split-manifest shas used
  predictions.jsonl   # per-pair scores (a first-class artifact)
  metrics.json        # computed by src/polarity/eval/metrics.py (deterministic)
  logs/               # training/eval logs
```

`runs/` is git-ignored; the **finished** artifacts that reproduce a headline (predictions +
metrics) are promoted to `predictions/` and `reports/` and committed, so CI reproduces the
numbers **without** training (the keyless drift gate). Runs are never overwritten.

## Fingerprints

An experiment now carries two identities. The stable **recipe fingerprint** hashes the model/data/
hyperparameter/seed/split recipe, so xp-005 and xp-006 intentionally share it. The distinct
**execution fingerprint** covers Python, platform, Torch, Transformers, CUDA, and exact base and
tokenizer revisions, so those runs differ. Historical values that were not recorded are represented
as `unknown:not-recorded`, not reconstructed after the fact.

## Milestones

| M | Name | Paid? | Gate to enter |
|---|---|---|---|
| **M0** | research + reproducible foundation | no | — (this milestone) |
| **M1** | deterministic keyless public-data builders + leakage-safe manifests | no | owner go; only `license_status != LICENSE_UNRESOLVED` sources |
| **M2** | baseline reproduction (embedding threshold, vCache-style adaptive, off-the-shelf NLI) | no | M1 manifests + leakage audit green |
| **M3** | first controlled DeBERTa training run | **yes (GPU)** | owner-approved cost cap; M2 baselines recorded |
| **M4** | evaluation + calibration + error analysis | no | M3 checkpoint frozen |
| **M5** | robustness / limited augmentation | maybe (LLM) | M3/M4 evidence justifies it; owner go |
| **M6** | Veritas transfer arm (`armXenc`) — the sealed 75-pair holdout, evaluated **once** | no | model + calibration frozen (governance G2) |
| **Release hardening** | source-only model card, provenance, history sanitation, security gates | no | owner approval; no weights/raw data; separate approval before public visibility |
| **M7 (banked)** | possible attribute-substitution experiment | maybe | not scheduled; preregister a new independently authored final set before construction/scoring |

Paid milestones do **not** auto-advance. Each requires an explicit owner approval with the cost
packet from `docs/COSTS.md`.

## Negative-result policy

A partial or negative outcome is registered and reported like any other: "improves negation but
regresses ordinary paraphrase", "cannot reach 0 FP on the transfer set", "off-the-shelf NLI
already closes most of the gap", "cannot beat the vCache-style adaptive threshold at equal FP".
A dead direction gets its registry row updated with the honest outcome — a documented negative
beats a silently abandoned experiment.
