# M2 REPORT — no-training baselines

**Disposition: `CUSTOM_TRAINING_JUSTIFIED`.** No off-the-shelf baseline is deployable at a safe
false-positive budget, the strongest (an NLI cross-encoder) still leaves the **equivalent-negation
slice essentially unsolved**, and a vCache-inspired adaptive threshold does not close the gap. M2
used **$0**, no training, no GPU, no Veritas evaluation. Protocol was frozen before results
(`docs/M2-METHODOLOGY.md`).

## M2 question

Evaluate the preregistered NO-TRAINING baselines on the committed M1 primary data, under the
operational metric — maximize recall while keeping unsafe false-positive cache matches low — and
decide whether a custom trained model is still justified. Positive = `CACHE_EQUIVALENT`; a **false
positive = `NOT_CACHE_EQUIVALENT` predicted `CACHE_EQUIVALENT`**.

## Data (M1 manifests, authoritative)

Selection on `manifests/primary/validation.jsonl` (sha256 `6b31c102…`, 1,644 pairs); final
reporting on `manifests/primary/internal_test.jsonl` (sha256 `2f674ccb…`, 1,608 pairs; 725
positive / 883 negative). **Thresholds/mappings/calibrators were selected on validation and
frozen before internal_test was scored. Veritas was not touched.** Pair text was reconstructed
from `data/raw` and verified against each manifest row's committed hash.

## Baselines & models

Five baselines, each emitting a per-pair score in [0,1] (committed under `predictions/m2/`):
exact, normalized (`textnorm.normalize` — keeps negation/numbers/entities), embedding-cosine,
adaptive (vCache-inspired), and an NLI cross-encoder. Models (pinned, free, CPU, no
`trust_remote_code`; `reports/m2/models.json`):

| Role | Model | Revision | License |
|---|---|---|---|
| Embedding | BAAI/bge-small-en-v1.5 | `5c38ec7c…` | MIT |
| NLI cross-encoder | cross-encoder/nli-deberta-v3-base | `6c749ce3…` | Apache-2.0 (SNLI+MNLI, **no ANLI**) |

MoritzLaurer NLI models were excluded (ANLI = CC-BY-NC training data, same policy as M1). NLI
mapping: `score = min(P_entail(a,b), P_entail(b,a))` (bidirectional; entailment index read from
`config.label2id`).

## Internal-test results (validation-frozen operating points)

Threshold labels below describe **selection on validation**, not guaranteed internal-test error
rates. The threshold with zero observed validation false positives rejects nearly everything; when
applied unchanged to internal_test, NLI admits 2/883 false positives while the other four admit
none. More discrimination appears at thresholds selected to target validation FPR≤1%, but the
realized internal-test rates must be reported separately:

| baseline | recall at validation zero-observed-FP threshold | internal-test FP at that threshold | recall at validation-selected target-FPR≤1% threshold (95% CI) | precision | realized internal-test FP / FPR | equiv-negation recall |
|---|--:|--:|--:|--:|--:|--:|
| exact | 0.000 | 0 | 0.001 | 0.33 | 2 / 883 = 0.23% | 0.00 |
| normalized | 0.000 | 0 | 0.007 | 0.71 | 2 / 883 = 0.23% | 0.00 |
| adaptive (vCache-inspired) | 0.003 | 0 | 0.015 (0.007–0.023) | 0.69 | 5 / 883 = 0.57% | 0.00 |
| embedding-cosine | 0.000 | 0 | 0.043 (0.029–0.058) | 0.74 | 11 / 883 = 1.25% | 0.00 |
| **nli-cross-encoder** | 0.019 | 2 | **0.073 (0.055–0.094)** | **0.77** | **16 / 883 = 1.81%** | **0.052** |

Validation has 863 negatives, so its 1% selection budget permits at most 8 observed false
positives. Internal test has 883 negatives and FPR granularity 1/883 ≈ 0.11%; the table reports its
realized counts rather than transferring the validation budget by name. The full FP-vs-recall / PR
grid and threshold table are in `reports/m2/metrics-*.json`. A separate descriptive sweep of the
internal-test curve constrained to **actual internal-test FPR≤1%** gives NLI recall **0.047**. Best-F1
on the grid: NLI **0.663** (precision
0.68 / recall 0.65) — the only baseline exceeding 0.5 precision at its best operating point; the
deterministic baselines' "best F1" is the trivial accept-all point (precision = base rate 0.45).

## The equivalent-negation slice at validation-selected target-FPR≤1% thresholds (headline)

`equivalent_negation` positives ("X is not safe" ≡ "X is insecure"), n=58: **recall 0.0 for
exact / normalized / embedding / adaptive, and only 0.052 for the NLI cross-encoder.** Meanwhile
**`explicit_negation` negatives** (n=107, genuine "not" flips) are rejected correctly by all
baselines at those validation-selected thresholds (FP-rate ≈ 0.0). Thus the five evaluated
approaches leave the meaning-preserving negation slice largely unresolved while rejecting the
explicit-negation negatives; this is the specific residual slice Polarity targets.

## Adaptive threshold (vCache-inspired) — does it fix the naive-threshold failure?

**No.** The per-neighborhood adaptive threshold is the **best-calibrated** baseline (ECE 0.076 vs
embedding-cosine 0.515; raw cosine is not a probability), but its recall at the
validation-selected target-FPR≤1% threshold (0.015, with 0.57% realized internal-test FPR) is
*lower* than plain cosine (0.043, with 1.25% realized internal-test FPR). Adaptive/vCache-style
thresholding calibrates the score; it does
not recover the recall lost to the frozen-embedding geometry. (Faithful-offline caveats — online
adaptation, stochastic gate, live judge, δ-guarantee validity — are disclosed in
`docs/M2-METHODOLOGY.md`; this is "vCache-inspired", not vCache.)

## Calibration

Temperature scaling (grid-search, validation-fit, applied frozen): NLI ECE 0.287 → 0.102 (T=5.0);
embedding 0.515 → 0.241; adaptive already 0.076 (T=1.0). Calibration improves the raw scores but
does not change the recall/FP trade-off at the operating point.

## Statistical uncertainty

Bootstrap 95% CIs on recall at the validation-selected target-FPR≤1% thresholds overlap slightly:
NLI 0.055–0.094 and embedding 0.029–0.058. McNemar at those separately validation-selected
thresholds gives 63 vs 46 discordant predictions and a continuity-corrected statistic of 2.35
(< 3.84), so this single operating-point comparison is **not decisive**. NLI has the higher point
estimate and stronger descriptive curve, but these uncertainty results do not establish a
statistically clear advantage at the separately selected thresholds.

## Shortcut / error analysis

The original per-source error-rate check was balanced (NLI: CANNOT 0.437 vs PAWS-Wiki 0.419), and
the coarse M1 source-majority heuristic was 0.54. Those checks did not rule out richer
dataset-family or lexical shortcuts: later author-run bag-of-words diagnostics found available
source and label signal. Their procedure/output artifacts are not committed, so they are historical
evidence and do not establish whether the trained neural model used those signals. Error was lower
on pairs carrying a negation cue (NLI 0.284) than without (0.447), so the observed NLI failure was
not a simple reject-on-negation pattern; equivalent negation remained the specific weak slice.

## Latency (`reports/m2/latency.json`)

Local CPU (Darwin arm64), 100-pair batch, 1 warm-up + 5 timed runs: bge-small **0.94 ms/pair**
(~1,063 pairs/s); NLI cross-encoder **10.2 ms/pair** (~98 pairs/s, two forward passes for the
bidirectional score). **Local inference benchmark only; API comparison deferred** — no comparable
historical Haiku latency measurement exists.

## Relation to the historical Veritas motivation (kept separate)

M2 evaluates the **primary internal_test set, not Veritas** (sealed until M6). The M0 finding —
on the frozen Veritas 75-pair set, cosine recovers no paraphrase at 0 observed FP because the top
negative (0.9590) outscores every paraphrase (0.9465), while a Haiku judge lifted precision
0.48→1.00 — is *historical Veritas baseline* provenance (M0, not re-run). M2 independently
reproduces the qualitative pattern on a larger set: frozen cosine fails at 0 FP, an off-the-shelf
cross-encoder helps but leaves equivalent-negation unsolved. These are **separate evaluations**;
the Veritas transfer is M6.

## Decision — is custom training justified?

**Yes (`CUSTOM_TRAINING_JUSTIFIED`).** Evidence, from validation + internal_test only:

1. **No no-training baseline is deployable** — NLI reached 7.3% internal-test recall at its
   validation-selected target-FPR≤1% threshold, but admitted 16/883 false positives (1.81% realized
   internal-test FPR). In a descriptive sweep at actual internal-test FPR≤1%, its recall was 4.7%.
2. **The off-the-shelf NLI cross-encoder is the strongest candidate by point estimate and best-F1**
   (best precision 0.77; only >0.5-precision best-F1), although its selected-threshold comparison
   with embedding is statistically non-decisive. It still leaves a clear residual failure slice —
   equivalent-negation recall 0.052 at its validation-selected threshold — aligned with Polarity's
   negation/scope specialization.
3. **Stronger/adaptive thresholding (vCache-inspired) does not close the gap** (better calibrated,
   not better recall).
4. The task remains operationally useful (real cache-correctness, asymmetric FP cost).

Not chosen because training was on the roadmap — chosen because the baselines leave a specific,
measurable gap on the slice Polarity exists to address.

## Reproduce / gates (keyless)

`uv run pytest` (offline) · `ruff` + strict `mypy src` · `uv run polarity baseline report --check`
(reports reproduce byte-for-byte from committed predictions; **CI never runs model inference**) ·
`uv run polarity audit` (sealed-set isolation, incl. M1 manifests) · `$0`.

## Exact proposed M3 (needs owner go + a cost packet — the first paid milestone)

A single controlled `microsoft/deberta-v3-base` cross-encoder fine-tune on the committed **primary
train** manifest (MNLI excluded), selected on **validation**, reported on **internal_test** under
the same FP-vs-recall protocol, with the **equivalent-negation slice as the primary success
criterion** (must beat the NLI cross-encoder's 0.052 there without inflating FP). Cost packet
(provider, hardware, duration, cap) from `docs/COSTS.md`; the sealed Veritas transfer stays at M6.
