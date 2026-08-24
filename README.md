# Polarity

**A cache-equivalence cross-encoder: telling paraphrases from meaning-changing negation better than a single embedding threshold.**

Semantic caches for LLMs reuse a stored answer when a new query is "similar enough" to an old
one. In the historical 75-pair Veritas evaluation, a *meaning-changing* pair scored **0.9590** on
the frozen embedding while the highest-scoring genuine paraphrase reached **0.9465**, so no
threshold recovered a paraphrase at zero observed false positives on 44 negatives. The embedding
vectors are not included here, so this is historical evidence rather than a metric reproducible
from this repository alone. A historical Veritas LLM-judge result mitigated the failure on that
small set but added a paid call per lookup. Polarity asks whether a base-sized sentence-pair
**cross-encoder**, trained specifically for cache equivalence, can close more of that gap. No
controlled xp-006-versus-judge cost or latency advantage is claimed.

> **Current finding:** Polarity is a bounded case study, not a general semantic-equivalence
> system. It shows strong ranking on its PAWS-Wiki+CANNOT distribution, promising behavior on a
> small author-created cache-query set, and weak transfer to MRPC. The negative result and the
> limits of threshold transfer are part of the contribution. Learned/adaptive semantic caches and
> cache-equivalence cross-encoders already exist; this project is
> **replication-plus-specialization**, not a claim of priority. See `docs/LITERATURE.md` and
> `MODEL_CARD.md`.

## Results at a glance

| evaluation / artifact | bounded result |
|---|---|
| Veritas target-domain transfer at validation-selected `tau=0.994833` | 31/31 equivalent pairs recalled; 2/44 false accepts (**4.55% realized FPR**); 0/22 negation false accepts |
| MRPC external transfer at the same frozen threshold | ROC-AUC **0.640**; 332/578 false accepts (**57.44% realized FPR**) — weak transfer |
| checkpoint | **withheld / private** pending resolution of its licensing posture |

The threshold was selected on validation to target FPR≤1%; the realized FPRs above are measured on
the named evaluation sets and show that the selection budget does not transfer as a guarantee.

## Status

**M0 — research + reproducible foundation. No model, no training, no paid calls.** The research
design (`docs/SPEC.md`, `docs/METHODOLOGY.md`, `docs/LITERATURE.md`, `docs/DATA.md`, threat
model, experiment registry, cost model); reproducible keyless primitives (typed records,
deterministic source-family splitting, a fail-closed leakage audit, calibrated-operating-point
metrics with a byte-for-byte drift gate); and a **sealed 75-pair Veritas holdout** (pair text +
labels only, never trained or tuned on). Full evidence: `docs/M0-REPORT.md`.

**M1 — dataset foundation. Still no training, no GPU, no paid calls.** Deterministic,
script-free builders fetch four license-reviewed, conditionally usable sources at pinned revisions
and normalize them
into a canonical schema; a `primary` recipe (**PAWS-Wiki + CANNOT**) is split leakage-safely by
source-family and proven isolated from the sealed set. Headline results (`docs/M1-REPORT.md`):
primary recipe **`TRAINING_RECIPE_READY`** (15,983 pairs, source→label confound 0.54, 705
equivalent-negation positives); **MNLI EXCLUDE** (negatives-only, raises confound +0.15);
**0 Veritas collisions** and a documented NaN-NLI probe contamination (CANNOT ⊃ NaN-NLI)
remediated. The 0.54 figure is a coarse source-majority heuristic, not proof that shortcut signals
are absent; later author-run lexical/source diagnostics found such signals were available. Those
later diagnostics are historical evidence, not outputs of a public reproduction gate.

**M2 — no-training baselines / falsification. Still no training, no GPU, no paid calls.** Five
preregistered no-training baselines on the primary manifests (exact, normalized, frozen embedding
cosine, a vCache-inspired adaptive threshold, and an off-the-shelf NLI cross-encoder), each
compared under recall-vs-false-positive with thresholds selected on validation and frozen for
internal_test. Disposition (`docs/M2-REPORT.md`): **`CUSTOM_TRAINING_JUSTIFIED`** — no baseline
is deployable at a safe FP budget. At the threshold selected to target validation FPR≤1%, the best
NLI cross-encoder reached **7.3% internal-test recall with 16/883 false accepts (1.81% realized
FPR)**. A separate descriptive internal-test sweep constrained to actual FPR≤1% reached **4.7%
recall**. At each baseline's validation-selected target-FPR≤1% threshold, all five evaluated
baselines had equivalent-negation recall at or below 0.052 (0.0 for
exact/normalized/embedding/adaptive; 0.052 for the NLI cross-encoder), leaving the target slice
largely unresolved. Model inference is the one author-run step; every metric is recomputed
keylessly from committed predictions.

**M3 + M3.1 — two controlled training runs (executed). Both `DATA_OR_OBJECTIVE_PROBLEM`.** The frozen
DeBERTa-v3-base recipe trained twice on an RTX 4090 (bf16; **approximately A$0.68 total,
author-reported**, pods torn down): M3
(xp-003) and the corrected M3.1 (xp-004: pAUC selection + full 4 epochs). Neither is a clean model
verdict — the **low-FP operating-point measurement is broken on this setup**: recall@FPR≤1% on ~863
validation negatives is pinned by ~8 points (tau→1.0 → recall 0), and the `_pauc` selection metric had
a bug that **over-credits degenerate all-or-nothing curves** (so M3.1 selected a *worse* checkpoint).
**M4 — measurement fix + failure diagnosis ($0, no GPU).** Correcting the pAUC bug and re-scoring the
committed predictions overturned the earlier read: with the fixed metric the trained model has
**pAUC 0.0 and 0.0 recall at every FP budget up to 5%** — strictly worse than the NLI baseline. The
score distributions show why: the model **collapsed to a constant output** (M3.1 emits one value,
0.488, for all 1,608 pairs) — it learned the class prior and nothing input-dependent, a mode collapse
consistent with **bf16 training instability** (gradient-norm spikes; the research flagged fp32 for
DeBERTa). Details: `docs/M4-REPORT.md`. The earlier "2× low-FP signal" is retracted.

**M3.2 — the fp32 recovery experiment (executed). `FP32_HYPOTHESIS_NOT_SUFFICIENT`.** A single
controlled paid run (`xp-005`, approximately A$0.37 author-reported) identical to M3.1 except
**precision bf16 → fp32** (the
fingerprint diff is that one field). Result: **fp32 collapsed identically** — constant output,
internal_test pAUC 0.0, recall 0.0 at every FP budget — but with **stable gradients** (max
grad-norm 8.87, no ~70 spikes, no NaN). So **bf16 instability is refuted as the dominant cause**:
the M4 grad-spikes were a symptom, not the driver, and the collapse is precision-independent. A
pre-registered runtime **collapse gate** hard-stopped the doomed run in <1 min. Details:
`docs/M3.2-REPORT.md`.

**M5 — Training Signal Audit (executed, $0, local). Version-sensitive collapse diagnosed.** The
author-run overfit-one-batch diagnostics reported that `distilbert` learned through the same
pipeline and DeBERTa-v3-base learned under Transformers 4.x but collapsed under 5.x. Committed
predictions and reports independently support collapse under 5.15.1 and success under 4.57.6; the
decisive diagnostic runs do not have machine-readable committed logs, so the causal experiment
cannot be replayed from this repository alone. The xp-006 release profile pins Transformers
`4.57.6` and Torch `>=2.6,<2.7`, with a runtime compatibility guard. Details:
`docs/M5-REPORT.md`, `docs/M5-DIAGNOSTIC-PLAN.md`, and `docs/RELEASE-INTEGRITY.md`.

**M3.3 — the corrected re-run (executed). `STRONG_SUCCESS` (in-distribution).** Same fp32 recipe
fingerprint as M3.2, with a distinct execution fingerprint for the Transformers 4.57.6 environment;
RunPod 4090,
**approximately A$0.38 author-reported**, full 4 epochs, collapse gate never fired. On threshold-free
internal-test curves: **custom step-pAUC[0,2%] 0.716 vs NLI 0.048; recall at actual FPR≤1% 0.763 vs
0.047; recall at actual FPR≤5% 0.968 vs 0.232**. Separately, at each model's threshold selected on
validation to target FPR≤1%, xp-006 achieved **0.903 recall with 21/883 false accepts (2.38%
realized FPR)** versus NLI **0.073 recall with 16/883 false accepts (1.81% realized FPR)**; paired
Δrecall was **+0.830** (95% CI [0.80, 0.86]). At xp-006's validation-selected `tau=0.994833`,
equivalent-negation recall was **0.983** versus 0.052 for NLI, and explicit-negation FPR was **1/107
= 0.009**. This is consistent with the M5 version-sensitive diagnosis at full scale. **Honest scope:** these
are **in-distribution** numbers (internal_test = same PAWS-Wiki+CANNOT families as train). Veritas
later adds small author-created target-domain evidence; MRPC later shows weak external transfer.
Details: `docs/M3.3-REPORT.md`.

**M6 — author-created target-domain transfer (executed).** The `xp-006` checkpoint matched the
committed M3.3 validation and internal-test scores exactly, then was evaluated under a protocol
committed beforehand on the 75-pair Veritas cache-query set. This chronology supports a frozen
evaluation, but the repository cannot technically prove that scoring occurred only once. At the
threshold selected on validation to target FPR≤1% (`tau=0.994833`), Veritas recall was **31/31**
with **2/44 false accepts (4.55% realized FPR)** and **0/22 negation false accepts**. A separate
descriptive sweep constrained to actual Veritas FPR≤1% achieved **24/31 recall (0.774)**. Veritas
was author-created and present from M0, so this is protocol-held-out target-domain evidence, not an
independently blinded or general-OOD benchmark. Details: `docs/M6-REPORT.md` and
`docs/M6-TRANSFER-PROTOCOL.md`.

**M6.1 — post-hoc probe + external MRPC transfer ($0 diagnostics).** The small, author-curated
probe concentrates observed false accepts in its measurement, temporal, and numeric categories;
it is not an independent benchmark and cannot identify a cause or population failure rate. MRPC
(1,725 news-domain paraphrase pairs) showed weak transfer: ROC-AUC **0.639870**, repository-defined
step-pAUC **0.031938**, and at validation-selected `tau=0.994833`, **332/578 false accepts
(57.44% realized FPR)**. A separate author-run overlap audit, not reproduced by the public gate,
reported that higher-overlap and low-overlap negatives both fail. The committed artifacts support
score saturation, distribution/calibration shift, semantic overacceptance, and weak ranking
(ROC-AUC 0.639870) without isolating one cause. Details:
`docs/M6.1-DIFFERENT-ANSWER-PROBE.md` and `docs/M6.1-MRPC-TRANSFER.md`.

In Polarity reports, low-FPR **pAUC** means the repository-defined normalized step integral of
maximum achievable recall over the stated FPR range. It is not scikit-learn's standardized partial
AUC.

**Release posture:** code, documentation, and aggregate results are being prepared as a
**source-only** release. The checkpoint, raw third-party data, ignored runs, and complete MRPC gold
label vector are excluded. Weight licensing is unresolved, so the checkpoint is withheld. The
cumulative project spend estimate is **approximately A$1.78, author-reported; no receipts**.
Publication, any Portfolio change, and any future experiment remain separate owner-approval
boundaries. The M7 hypothesis is banked; it is not the next recommended experiment.

## Reproduce it (keyless, $0)

```sh
uv sync --locked
uv run pytest                       # full behavior/regression suite; count intentionally not hard-coded
uv run polarity evaluate --check    # M0 drift gate
uv run polarity data verify         # sealed-snapshot schema + checksum + provenance
uv run polarity audit               # isolation: sealed set held out from fixtures AND manifests
uv run polarity data stats          # M1 build summary + the two-recipe comparison
uv run polarity baseline report --check  # M2 reports reproduce byte-for-byte from predictions
uv run polarity train fingerprint   # M3 refuse-to-start gate: config valid + manifests unchanged
uv run polarity transfer report --check        # M6 metrics from committed scores + protocol
uv run polarity transfer probe-report --check  # authored-probe deterministic report
uv run polarity transfer mrpc-report --check   # structure/hash only; metrics NOT reproduced
```

After the locked dependencies are installed, no GPU, dataset download, model download, or API key
is required for those commands. CI runs the source-free gates on Python 3.12 and 3.13. The
source-free MRPC command checks the score-only schema, sequential `row_index` values and row count,
the synthetic `id = mrpc-test-{row_index}` format, and the internal consistency of committed
report/provenance/binding hashes. It cannot authenticate those rows against official MRPC
identities or labels and deliberately does **not** reproduce label-dependent metrics.

To rematerialize the committed M1 8,000-per-source build, use the separate networked author-run
path. Each fetched artifact is checked against its committed expected SHA-256 before that artifact
is written, converted, or extracted:

```sh
uv run polarity data fetch all
uv run polarity data build --limit-per-source 8000 --exclude-sealed-collisions
```

Full reproduction requires an **already constructed** local JSON file whose bytes, schema, and row
order match the committed contract. The repository records the expected loader/revision/checksums
in `reports/m6.1/mrpc-provenance.json`, but it does not currently provide a deterministic procedure
for converting Microsoft's distribution or the named loader output into that exact JSON. Once such
a checksum-matched local file exists, run:

```sh
uv run polarity transfer mrpc-report --check \
  --source data/raw/mrpc/mrpc-test-golden.json
```

The command verifies the expected checksum, schema, and row order before joining local labels to
committed scores. Author-run steps that use network or compute include dataset acquisition/building,
model inference, and training; CI never installs those extras, downloads data, runs inference, or
trains.

A holder of the private model/tokenizer bundle can verify every required file independently of
model loading:

```sh
uv run polarity artifact verify \
  --manifest reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json \
  --artifact-dir artifacts/xp-006
```

The source-only repository does not supply that bundle; integrity verification does not grant
redistribution rights.

## Design

```
                 committed, versioned artifacts (no GPU / no network to reproduce)
                 ────────────────────────────────────────────────────────────────
 data/fixtures/  ─┐                                              ┌─ reports/metrics.json
 (synthetic)      ├─► split.py ─► manifests/  ─► leakage.py ─►   │   (recall vs false-positive
 data/snapshots/ ─┘   (group-safe)   (immutable)   (fail-closed) │    at a calibrated point)
 veritas-golden/      target-domain holdout ───────► isolated ───┘
 (author-created, never trained on)                            metrics.py + calibration/
```

The positive class is `CACHE_EQUIVALENT`; a **false positive** (`NOT_CACHE_EQUIVALENT`
predicted `CACHE_EQUIVALENT`) is the failure that serves a wrong cached answer, so the primary
curve is recall vs. false-positive rate — never accuracy or AUC alone.

## Data & license

This repository's own code and documentation are MIT (`LICENSE`, `NOTICE`). That license does not
relicense third-party datasets, the Microsoft base model, or the private fine-tuned checkpoint.
Raw public datasets are not bundled. The public MRPC prediction artifact contains scores and stable
row identities only—no text or gold labels. See `docs/DATA.md`, `ATTRIBUTIONS.md`, and
`MODEL_CARD.md`. This is a conservative release posture, not legal advice.

## Acknowledgements

Motivated by the semantic-cache evaluation in the sibling **veritas** project. Prior art that
shapes the honest framing is cited in `docs/LITERATURE.md` (vCache; Redis LangCache /
calibration-gap; NevIR; DeBERTa-v3).
