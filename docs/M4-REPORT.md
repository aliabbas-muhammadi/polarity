# M4 REPORT — measurement fix + failure diagnosis ($0, no GPU)

A local analysis pass triggered by two confounded training runs (`docs/M3-REPORT.md`,
`docs/M3.1-REPORT.md`). It fixed the low-FP metric, re-scored the committed predictions, and — the
important part — **overturned the earlier "promising signal" reading**. No GPU, no paid calls, $0.

## 1. The pAUC bug is fixed (and it changes the conclusion)

`src/polarity/eval/curves.py` replaces the old `_pauc` with a **step-integrated** partial AUC:
recall achievable at a false-positive budget `b` is the recall of the best operating point with
FPR ≤ b, and **no recall is credited beyond the last point at or below `fmax`**. A degenerate
"reject-everything-then-accept-a-burst" curve now scores ~0, not ~0.5. Unit-tested
(`tests/test_curves.py`): degenerate → 0, collapsed-constant → 0, separable → high.

Re-scoring the **committed** M3/M3.1 predictions with the corrected metric (the recall rows are
threshold-free sweeps constrained to actual internal-test FPR):

| internal_test | M3 (xp-003) | M3.1 (xp-004) | M2 NLI |
|---|--:|--:|--:|
| pAUC[0,2%] (corrected) | **0.000** | **0.000** | 0.048 |
| recall @ FPR≤0.5% | 0.000 | 0.000 | 0.039 |
| recall @ FPR≤1% | 0.000 | 0.000 | 0.047 |
| recall @ FPR≤2% | 0.000 | 0.000 | 0.080 |
| recall @ FPR≤5% | 0.000 | 0.000 | 0.232 |

The trained model recovers **zero** cache-equivalent pairs at **any** budget up to 5% FP —
**strictly worse than the off-the-shelf NLI baseline everywhere**. The earlier "M3 pAUC 0.097 vs
0.048, ~2× signal" was **entirely the metric bug** and is retracted (M3-REPORT.md carries a
correction banner).

## 2. Root cause: the model collapsed to a constant output

Score distributions on internal_test (1,608 pairs) pin it exactly:

- **M3 (xp-003):** every score ≈ **0.472** (2 distinct values across all 1,608 pairs); positives and
  negatives are **indistinguishable** (both mean 0.472).
- **M3.1 (xp-004):** **one** distinct value — **0.488 for all 1,608 pairs**.
- **NLI baseline** (for contrast): positive median **0.989**, negative median **0.0003**.

The constant ≈ the class base rate (positives ≈ 0.457). **The model learned the label prior and
nothing input-dependent** — classic mode collapse. That is why recall is 0 at every budget: with
(near-)constant scores no threshold separates anything. The FPR≤1% "degeneracy" of M3/M3.1 was not a
thin-operating-point artifact — the model has **no discrimination at all**.

## 3. Historical M4 hypothesis: training instability (bf16; later refuted)

This was the hypothesis formed at M4. M3.2 later collapsed in fp32 with stable gradients, so bf16
was not a sufficient explanation and the precision hypothesis was rejected. The observations below
are retained to show the debugging chronology, not as the current root-cause conclusion.

New telemetry (`grad_norm`, corrected `train_loss`) makes the mechanism visible. Even a 3-step fp32
CPU smoke shows the pre-clip gradient norm spiking to **~70** (clip is 1.0), and the full bf16 runs'
loss oscillated between ~random and ~perfect before collapsing. DeBERTa-v3 + bf16 over ~1,600 updates
is the prime suspect: the M3-training-research already flagged a source arguing DeBERTa should train
in **fp32** (bf16's reduced mantissa can destabilize its disentangled attention). The local fp32
smoke does not collapse; the bf16 full runs did.

## 4. What is (and isn't) trustworthy from M3/M3.1

- **Not trustworthy:** every pAUC/low-FP claim made before this fix; any notion that the model has
  latent low-FP signal. It does not — it collapsed.
- **Trustworthy:** the model does not beat the NLI baseline at any FP budget (it is far worse); the
  data/leakage/fingerprint machinery worked; calibration numbers are moot on a constant output.

## 5. Proposed M3.2 (new experiment — owner-gated, ~A$0.35)

Fix the training, then re-measure with the corrected stack:

1. **Precision `fp32`** (the maximally-safe choice; primary hypothesis for the collapse).
2. **Keep** the corrected pAUC checkpoint selection + full epochs + the new `grad_norm`/`train_loss`
   telemetry (to confirm stability: gradient norms should settle, loss should fall monotonically).
3. If fp32 still collapses, escalate: lower LR / longer warmup, or inspect the input encoding.
4. Same corrected reporting (recall at FP budgets + fixed pAUC + paired vs NLI).

**Success is now a low bar to clear first:** M3.2 must simply produce a **non-collapsed** model
(distinct, input-dependent scores; pAUC > 0) before comparison with NLI at separately
validation-selected target-FPR thresholds is even meaningful. Realized internal-test FP/FPR must
be reported independently. The M3.2 config lands right before its (owner-approved) run, so CI's
report drift gate stays green.

## Reproduce / gates

`uv run pytest` (incl. `tests/test_curves.py`) · `polarity train report --check --config <cfg>` for
every committed experiment · `uv run python -m polarity.eval.curves` · $0, no GPU, no network.
