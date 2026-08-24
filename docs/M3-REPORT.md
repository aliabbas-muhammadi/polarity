# M3 REPORT — first controlled training run (xp-003)

> **Correction (M4, superseding the original interpretation below):** the pAUC figures produced by
> the old `_pauc` (for example, 0.097 vs 0.048) over-credit degenerate all-or-nothing curves. M4
> recomputed the committed predictions with the corrected step metric: xp-003 pAUC[0,2%] is
> **0.000**, its scores are collapsed, and it does not beat NLI at any reported low-FP budget. Old
> values remain only as an audit trail of the retracted M3 interpretation. Calibration (ECE) is
> unaffected but does not rescue a non-discriminating model.

**Historical disposition at M3: `DATA_OR_OBJECTIVE_PROBLEM` (superseded by M4).** The first paid run
executed the frozen recipe on a real GPU. At the time, the author attributed the result to a
premature early-stop and a degenerate operating-point selection, and read the old pAUC 0.097 vs
0.048 as possible signal. M4 showed that interpretation was wrong: the corrected pAUC is 0.000 and
the checkpoint is collapsed. ECE 0.021 is retained as a calibration measurement, not evidence of
useful discrimination.
The mechanical `suggested_disposition` in `reports/m3/disposition-xp-003-deberta-primary-capped.json`
is `REGRESSION` (it reads the raw operating-point recall); the manual, evidence-based call is
`DATA_OR_OBJECTIVE_PROBLEM`.
Negative/inconclusive first runs are house policy (`docs/EXPERIMENTS.md`).

## What ran (evidence)

- **Compute:** RunPod Secure, RTX 4090 (24 GB), bf16, DC EUR-NO-1. Wall ~2 min train + ~30 s
  predict. **Cost: US$0.247 ≈ A$0.35** (balance 15.00 → 14.75). Pod torn down; `pod list` empty.
- **Model:** `microsoft/deberta-v3-base` @ `8ccc9b6f…`, **184,423,682 params**, precision **bf16**.
- **Gate:** `train fingerprint` on the pod matched the approved shas (`pl1:7dbb3c69…`) before training.
- **Data:** trained on `manifests/primary/train.jsonl` (12,731), selected on validation (1,644),
  evaluated once on internal_test (1,608 = 725 pos / 883 neg). Corpora re-fetched at pinned
  revisions on the pod; the sealed Veritas set was **never transferred** (governance).
- **Artifacts:** `predictions/m3/xp-003-deberta-primary-capped/{validation,internal_test}.jsonl`,
  `reports/m3/{metrics,disposition}-xp-003-deberta-primary-capped.json`, `runs/xp-003/logs/metrics.csv`
  (telemetry). The report reproduces byte-for-byte from the committed predictions (`train report --check`).

## The two issues this run exposed

**1. Early-stop aborted training at ~1 epoch of 4.** The selection metric (validation recall at
FPR≤1%) was **0.0 at every eval** (updates 100/200/300/400), so patience-3 early-stopping stopped the
run at update 400 (best checkpoint = update 100), even as training loss fell 1.45 → 0.027. Stopping on
a metric sitting at a **floor** aborts a still-improving model.

**2. The FPR≤1% operating point is degenerate on the small validation negative set.** To hold
FPR ≤ 1% on ~863 validation negatives (≈ ≤8 FP), `threshold_at_fpr` is forced to **tau = 1.0** — a few
hard negatives score near 1.0, so any lower threshold admits >1% FP. At tau = 1.0 the model accepts
almost nothing → **internal_test recall 0.0** (this is exactly the thin-operating-point noise flagged
in `docs/M3-TRAINING-RESEARCH.md` §12).

A third, now-fixed bug: `train predict` defaulted to CPU (`predict_split` didn't pass `device`); CPU
inference with the pod's 120-thread host thrashed and hung. Fixed (`--device cuda`); the committed
predictions are from correct GPU inference.

## Internal-test numbers (frozen checkpoint, val-selected operating point)

| metric | M3 (xp-003) | M2 NLI baseline |
|---|--:|--:|
| recall at each validation-selected target-FPR≤1% threshold | **0.000** (tau 1.0) | 0.073 (16/883 FP; 1.81% realized test FPR) |
| recall 95% CI | 0.000–0.000 | 0.055–0.094 |
| **obsolete `_pauc`, FPR∈[0,2%] (retracted; corrected xp-003 value 0.000)** | **0.097** | 0.048 |
| calibration (ECE, selected=platt) | **0.021** | 0.287 (raw) |
| equivalent-negation recall (n=58) | 0.000 | 0.052 |
| explicit-negation FP-rate (n=107) | 0.000 | 0.000 |

Paired bootstrap Δrecall (M3 − NLI) at each model's validation-selected threshold:
**−0.073 (95% CI −0.092 to −0.055)**; McNemar 53 vs 16 discordant (stat 18.8). NLI's 0.073 recall
there came with 16/883 false accepts (1.81% realized internal-test FPR); it is not a result at
actual internal-test FPR≤1%. The historical pAUC interpretation below is superseded by the
correction banner and M4.

## Historical rationale for `DATA_OR_OBJECTIVE_PROBLEM` (retracted by M4)

The original decision treated old `_pauc` 0.097 vs 0.048 and ECE 0.021 as evidence that measurement
and stopping, rather than the model, had failed. M4 falsified the load-bearing pAUC premise: the
corrected xp-003 pAUC is 0.000 and the scores are collapsed. This section records why the project
continued to M3.1; it is not the current scientific conclusion.

## Internal_test is one-way — this does not license "retrain until it looks good"

Per the author record and `docs/M3-PLAN.md`, internal_test was scored once for this frozen
checkpoint; the repository does not technically enforce that count. Any corrected run is a **new
pre-registered experiment (M3.1)** with its own intended single internal_test evaluation — its recipe
chosen from the diagnosis above and validation/training curves, **not** by tuning against
internal_test.

## Proposed M3.1 (new experiment — needs owner approval; ~A$0.35, setup already done)

Same model/data/seed/loss; change only the **selection + stopping** objective to fix the exposed
degeneracy, and record it as a fresh frozen pre-registration:

1. **Select the checkpoint by validation pAUC over FPR∈[0,2%]** (robust to the thin FPR≤1% point),
   not by floored recall@FPR≤1%.
2. **Stop on training-loss plateau (or run the full 4 epochs)** — never patience on a floored metric.
3. Report the same FP-vs-recall / negation-slice / calibration / paired protocol on internal_test.

Historical hypothesis: with full training plus a more robust selection metric, M3.1 would beat the
NLI baseline. Its premise was the old pAUC reading, later retracted by M4. M3.1 is therefore retained
as part of the debugging chronology, not as confirmation that useful signal was already present.

## Reproduce / cost / governance

`uv run polarity train report --check` (keyless, byte-for-byte from committed predictions) · $0 to
reproduce · run cost A$0.35, pod removed · Veritas never transferred, still sealed until M6 · no
GitHub remote.
