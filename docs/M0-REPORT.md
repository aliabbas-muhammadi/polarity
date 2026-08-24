# M0 REPORT — Polarity research + reproducible foundation

M0 is complete: research design, sealed evaluation holdout, and reproducible keyless
primitives. **No model was trained, no GPU rented, no paid API called, and the veritas repo was
not modified.** This report is the evidence packet; deeper detail lives in the sibling docs it
links.

## Research question

Can a sentence-pair **cross-encoder** trained for semantic-cache equivalence separate genuine
paraphrases from meaning-changing negation/scope/quantity/entity perturbations **substantially
better than a single frozen-embedding cosine threshold** at a **calibrated, fixed false-positive
operating point**, while avoiding a required paid judge call on every lookup? A partial/negative
result is acceptable. No controlled xp-006-versus-judge cost or latency comparison was performed.
Full statement: `docs/SPEC.md`.

## Historical Veritas motivating evidence

This evidence was independently recomputed (read-only) from the sibling Veritas repository during
M0:
`eval/golden.json` + `data/golden-embeddings.json` (OpenAI `text-embedding-3-small`, dim 1536),
veritas HEAD `bd36420`, golden.json last touched in commit `164dc8e` (2026-06-30). The 75-pair
set is real: **31 CACHE_EQUIVALENT / 44 NOT**; pairTypes exact-repeat 10 / paraphrase 21 /
negation 22 / different-answer 22.

The public Polarity repository includes the 75 pair texts and labels but not those embedding
vectors. The embedding figures below are therefore **historical evidence**, not metrics a public
user can reproduce from this repository alone.

**The illustrative cosines reproduce to 3 dp** — negation `coffee-health-neg` **0.9246**,
`med-pregnancy-neg` **0.9510**; paraphrase `capital-france-para` **0.8989**, `boiling-water-para`
**0.8127**. (Veritas's README rounds to 2 dp, e.g. 0.93 — a presentation choice, not a mismatch.)

### Descriptive: means overlap
Per-class cosine (base↔probe): paraphrase mean **0.846** [min 0.5623, max 0.9465]; negation mean
**0.898** [0.7623, 0.9571]; different-answer mean 0.784 [0.5524, 0.9590]. Mean negation cosine
**exceeds** mean paraphrase cosine — but a mean gap alone does **not** prove non-separability.

### Direct proof: no single cosine threshold separates the classes
On the frozen set, computing cosine over the committed vectors (the recompute recipe is above;
Polarity does **not** import the embeddings — governance G1):

- The **highest-scoring negative** is `temp-convert-flip` at **0.9590**; the **highest-scoring
  paraphrase** is **0.9465**. The top negative outscores *every* paraphrase.
- Therefore any threshold admitting even one paraphrase also admits ≥1 negative:
  **recall at 0 observed FP (on 44 negatives) = 0/21 paraphrases = 0.000.**
- The positive and negative score ranges **overlap across [0.5623, 0.9590]** — essentially the
  entire paraphrase range.
- Best single-threshold operating points (cosine only, all 75 pairs): best-F1 at τ≈0.813 gives
  **precision 0.491 / recall 0.871** (FP 28/44); at the shipped **τ=0.92, precision 0.480 /
  recall 0.387 / FP 13/44 (FPR 0.295)** — reproducing veritas's naive P=0.48/recall 0.39/FP~30%.

Precise claim: *"On the frozen Veritas golden set under embedding config `text-embedding-3-small`,
no tested single cosine threshold recovers any paraphrase at zero observed false positives on 44
negatives, because the top negative (0.9590) outscores every paraphrase (max 0.9465)."* This is a
**finite-sample** statement about this set and this embedding — **not** a claim about embeddings
universally. With 44 negatives the FPR granularity is 1/44 ≈ 2.27%, so sub-2.3% FPR points are
not resolvable here; we report "0 observed FP on 44 negatives".

### Historical Veritas baseline (labelled, not re-run — G6)
The LLM-judge arm lifting precision 0.48→1.00 with 0 observed false positives on 44 negatives
(and guarded recall 0.97 with 0/44 observed false positives at τ=0.72, a 3.0× gap over
judge-free) is a **historical Veritas baseline**: model
`claude-haiku-4-5`, committed verdicts `data/golden-guard.json`, commit `a482d4a` (2026-06-30),
temperature 0. **M0 did not reproduce the LLM call** — it only records the committed provenance.
Historical pricing metadata was recorded in Veritas `data/pricing.json` as of 2026-06, but no
controlled xp-006-versus-judge cost or latency comparison was performed. **Prior-knowledge
limitation (G3):** these historical numbers are
already known, so the sealed set is not epistemically pristine; what governance protects is that
no *Polarity* model, threshold, or calibrator is fit against it.

## Literature findings

Full review + citations: `docs/LITERATURE.md`. Load-bearing conclusions:

- **Prior art exists.** vCache (ICLR 2026, arXiv:2502.03771) learns per-prompt thresholds on a
  *frozen* embedding; Redis/NYU (arXiv:2606.19719, Jun 2026) *already trains cross-encoders for
  cache equivalence* on PAWS/QQP; Redis LangCache ships commercially. Polarity is **not** the
  first learned cache nor the first cache-equivalence cross-encoder.
- **Architecture support.** NevIR (EACL 2024): cross-encoders are the best class on negation,
  most models at/below random — supports the model choice, caps the ceiling ("solves negation" is
  off the table; robustness doesn't transfer, SIGIR 2025).
- **Calibration warning.** A cross-encoder can have great PR-AUC yet collapse at the deployment
  threshold (0.816 → 0.199); any Polarity win must be shown at a calibrated fixed-FP point.

## What is genuinely novel / differentiated

**Replication-plus-specialization, not a new category.** The defensible contribution is a
negation/polarity-**isolated** evaluation + training slice for cache equivalence, plus a
**calibrated low-FP** comparison against exact, embedding, adaptive-threshold, and off-the-shelf
NLI baselines. The LLM-judge result remains historical motivation, not a controlled economic or
latency comparison. Claims explicitly avoided: "first", "solves negation", "beats the judge on
accuracy", any AUC/F1-only headline (`docs/SPEC.md` §Contribution).

## Cache-equivalence label definition

`safe_to_reuse_cached_answer(query_a, query_b) → P(CACHE_EQUIVALENT)`, symmetric and binary,
defined **only inside a controlled cache context** (equal system prompt, session, retrieval
corpus+version, tool state, temporal context; no hidden per-user state unless in the cache key —
governance G4). Positive = CACHE_EQUIVALENT ⇒ **false positive = NOT_CACHE_EQUIVALENT predicted
CACHE_EQUIVALENT** (G5). Full definition: `docs/SPEC.md`.

## Dataset matrix (build spec for M1; nothing downloaded in M0)

`docs/DATA.md`. Licensing gate outcome: **PAWS-Wiki, MNLI, CANNOT, NaN-NLI** =
USABLE_WITH_CONDITIONS for the research build; **ANLI** declares CC BY-NC and is excluded from any
future provenance-clean unrestricted-release recipe as a conservative project rule, without
claiming that its terms necessarily attach to learned parameters; **QQP and PAWS-QQP =
`LICENSE_UNRESOLVED`** (Quora ToS) → **excluded from M1**. This
reshapes the original recipe, which had leaned on QQP positives. Label-mapping caveats
documented (NLI labels ≠ cache labels).

## Leakage-safe split design

Split by **source family before augmentation**; a group's split is a pure function of
`(seed, group)`; immutable manifests carry `pair_sha`. Connected-components check catches hidden
families. Fail-closed leakage audit. `docs/METHODOLOGY.md` §1; `src/polarity/data/split.py`,
`src/polarity/eval/leakage.py`.

## Evaluation sets

(A) in-domain held-out; (B) cross-dataset transfer; (C) the **sealed Veritas 75-pair** holdout,
governed G1/G2 and evaluated once at M6. `docs/METHODOLOGY.md` §2–3.

## Baselines

Exact/normalized; frozen embedding + single threshold; **vCache-style per-prompt adaptive
threshold** (strong baseline); off-the-shelf NLI cross-encoder; historical LLM judge
(provenance-labelled). `configs/eval/operating-points.yaml`.

## Metrics

Primary = recall vs. false-positive rate (FP per G5), + PR curve, confusion, threshold table,
per-category error rates, operating points (0-observed-FP; FPR≤0.1/0.5/1% only where N supports),
bootstrap CIs; ROC/AUC secondary. Implemented + tested in `src/polarity/eval/metrics.py`.

## Category taxonomy

13 pre-registered categories incl. `equivalent_negation` (the central trap) — `docs/SPEC.md`,
`src/polarity/types.py::Category`. No aggregate may hide a per-category failure.

## Model candidates

Working candidate **`microsoft/deberta-v3-base`** (~184M; 86M backbone + 98M embedding; MNLI
90.6/90.7) as a joint-attention cross-encoder — the best architecture class on negation (NevIR).
Alternatives noted (ModernBERT-style, MiniLM cross-encoders) but not adopted for novelty's sake.

## Training plan

Tentative (M3, owner-gated): sequence-pair binary classification, BCE, `max_length=128`,
`lr=2e-5`, `batch=32` fp16, 2–3 epochs, seed 42, early stopping on validation (never the holdout),
safetensors, `trust_remote_code=false`. `configs/train/deberta-v3-base.yaml`.

## Calibration plan

Report ECE/Brier/reliability; fit temperature/isotonic/Platt on validation only. ECE + Brier
implemented and tested now (`src/polarity/calibration/`). `docs/METHODOLOGY.md` §5.

## Statistical plan

Bootstrap CIs on headline metrics; paired bootstrap + McNemar for model-vs-baseline on the same
set. Seeded bootstrap shipped (`bootstrap_recall_ci`).

## Compute estimate

Bottom-up (seq-length → batch/VRAM → steps → wall→cost). A base-size cross-encoder on ~100k short
pairs, 2–3 epochs, is a single-GPU sub-hour job (4090/A10/L4). `docs/COSTS.md`.

## Cost estimate

Tentative ceilings **not yet authorized**: GPU ~A$5–15, LLM augmentation ~A$2–5. Approval packet
required before any spend. M0 spent **$0**.

## Security / licensing findings

Threat model: pinned revisions + checksums, no `trust_remote_code`, safetensors only, no arbitrary
script execution, no committed secrets (gitleaks in CI). Licensing: QQP/PAWS-QQP blocked; ANLI is
project-excluded from an unrestricted provenance-clean recipe; four sources were conditionally
usable for the research build. No conclusion about learned-parameter license inheritance is made.
`docs/THREAT-MODEL.md`,
`docs/DATA.md`.

## Repository structure

`src/polarity/{types,fingerprint,cli}` + `data/{split,registry,loaders}` +
`eval/{leakage,metrics}` + `calibration/`; `configs/`, `data/{fixtures,snapshots}`, `manifests/`,
`predictions/`, `reports/`, `tests/`, `docs/`, `.github/workflows/ci.yml`. Mirrors the judgelab
conventions (uv + lockfile, hatchling src-layout, Typer CLI, strict mypy + pydantic). Only the
M0-necessary CLI commands are implemented (proportionality G12).

## Tests / gates (all green, keyless)

- **The behavior suite passes** (Python 3.13 locally; CI matrix 3.12 + 3.13): deterministic split +
  group-cohesion + leakage-safety, leakage-audit detectors, schema validation, threshold-sweep +
  FP-definition correctness, recall-at-0-FP, calibration (ECE/Brier, cross-checked vs
  scikit-learn), fingerprint stability, sealed-holdout checksum + isolation, drift determinism.
- `ruff check` + `ruff format --check` clean; `mypy src` strict clean (13 files).
- `polarity evaluate --check` reproduces `reports/metrics.json` **byte-for-byte** (keyless drift
  gate); `polarity data verify` + `polarity audit` green (holdout isolated from fixtures).
- No network in tests; no credentials; gitleaks job in CI.

## Provenance (G7 — no self-referential SHA)

This report cannot contain the SHA of the commit that contains it. Content-addressed inputs:
sealed snapshot `golden.json` sha256 **`63b23deafc06a9acee264e88dc97656d8f1430829ec825013b9e86be3f90e578`**
(75 rows), from veritas `eval/golden.json` @ `164dc8e` (veritas HEAD `bd36420`); the metrics
report is reproduced from `predictions/synthetic.jsonl` by the drift gate. **The resulting
Polarity M0 commit SHA is reported in the delivery message**, and a post-commit provenance line
may be added to the repo out-of-band; no throwaway commit is made just to self-reference a SHA.

## Open risks

- **Prior art narrows the contribution** (Redis/NYU cross-encoder; vCache) — the win must be the
  negation slice + calibrated fixed-FP comparison, or there is no result.
- **Data-recipe risk from licensing**: QQP blocked removes the obvious positive source; PAWS-Wiki
  + curated equivalent-negation must carry the positives.
- **Non-transfer of negation robustness** (SIGIR 2025) — a single training slice may not
  generalize; report per-slice.
- **Calibration collapse** at the operating threshold is a documented failure mode; mandatory to
  measure.
- **Small transfer set** (75 pairs) limits FPR resolution; report finite-sample only.
- **Share-alike reach to weights** is legally unsettled. The source-only release withholds the
  checkpoint rather than relying on one interpretation.

## Exact proposed M1

Deterministic, keyless public-data builders. For **each source cleared by the DATA.md gate only**
(PAWS-Wiki, MNLI, CANNOT, NaN-NLI; ANLI project-excluded from the unrestricted-release recipe;
**QQP/PAWS-QQP excluded**): download at a
**pinned revision with checksum**, apply the documented label-mapping, build balanced positive/
hard-negative pairs (ensuring `equivalent_negation` positives are present), generate immutable
**source-family split manifests**, run the leakage audit against the sealed 75-pair holdout
(schema/checksum/leakage only — G2), and commit manifests + `PROVENANCE.json`. **No training, no
GPU, no paid calls.** Needs owner go.
