# M1 REPORT — Polarity dataset foundation

M1 built a scientifically defensible dataset foundation from sources classified as conditionally
usable for the research build. A later release audit found that CANNOT's upstream rights chain and
the effect of share-alike material on learned parameters are not sufficiently resolved for
unrestricted checkpoint distribution. This report's data-build disposition does not clear weights;
the current release is source-only.
**No training, no GPU, no paid API, no model download; the sealed Veritas set was not touched
except for a read-only isolation audit.** Every number below is reproduced by
`uv run polarity data build --limit-per-source 8000 --exclude-sealed-collisions` from the pinned,
integrity-checked sources after `polarity data fetch` matches the expected raw-byte SHA-256 values
committed in `src/polarity/data/sources.py`. Generated output hashes and acquisition metadata are
recorded separately in `data/sources/*/PROVENANCE.json`. The committed build is a **seeded,
representative sample** (`--limit-per-source 8000`, seed 42); full-scale materialization is a
deterministic re-run deferred to M3.

## Disposition

- **Primary recipe (PAWS-Wiki + CANNOT): `TRAINING_RECIPE_READY`.**
- **MNLI: `EXCLUDE`** from the training recipe (adds source→label confounding, no new coverage).
- **Equivalent-negation coverage: sufficient** (705 ≥ 200 floor).
- **Sealed Veritas holdout: isolated** (0 collisions); NaN-NLI probe contamination found and remediated.

## Dataset inventory & licensing

All four ship labelled text directly (no reconstruction), fetched script-free at a pinned
revision (no `trust_remote_code`). Full matrix + citations in `docs/DATA.md`.

| Source | Tier | Pinned revision | License | Status |
|---|---|---|---|---|
| PAWS-Wiki | primary | HF parquet @ `161ece9` | Google notice + Wikipedia CC-BY-SA-3.0 | USABLE_WITH_CONDITIONS |
| CANNOT v1.1 | primary | GitHub @ `3b94fb5` | CC-BY-SA-4.0 | USABLE_WITH_CONDITIONS |
| MNLI | exploratory | HF parquet @ `da70db2` | OANC + CC-BY(-SA)-3.0 | USABLE_WITH_CONDITIONS |
| NaN-NLI | eval probe | GitHub @ `1f84029` | CC-BY-SA-4.0 | USABLE_WITH_CONDITIONS |

Excluded up front (authoritative from M0): **QQP, PAWS-QQP** (`LICENSE_UNRESOLVED`, Quora ToS);
**ANLI** (declared CC BY-NC and project-excluded from any provenance-clean unrestricted-release
recipe). This is a conservative source-selection rule, not a conclusion that ANLI terms would
necessarily attach to learned parameters. The original GCS PAWS tarball now 403s;
the HF parquet mirror is the maintained, content-addressable source (and MNLI moved off the
slow, un-versioned NYU zip onto the SHA-pinned HF parquet).

## Upstream rows → accepted / rejected (`reports/m1/mapping_audit.json`)

Every mapping is a named, reviewed rule; the derived cache-equivalence label is kept distinct
from the retained upstream label. Counts are over the FULL upstream data (before sampling):

| Source | Upstream rows | Accepted | Rejected (reasons) |
|---|---|---|---|
| PAWS-Wiki (train+val+test pooled) | 65,401 | 65,401 | 0 |
| CANNOT | 77,376 | 77,376 | 0 |
| MNLI (train+dev_m+dev_mm pooled) | 412,349 | 137,356 | 274,993 (entailment 137,841 · neutral 137,152) |
| NaN-NLI (eval probe) | 258 | 258 | 0 |

Label mapping: PAWS `1→CACHE_EQUIVALENT`, `0→NOT` (high-overlap hard negatives); CANNOT
`0(non-negated)→CACHE_EQUIVALENT`, `1(negated)→NOT`; **MNLI `contradiction→NOT` only** —
`entailment` (directional) and `neutral` (ambiguous) **rejected as unsuitable**, `-1`
(no-consensus) dropped; NaN-NLI retained unmapped (eval only). MNLI therefore contributes
**negatives only**, by design the confound risk the comparison below quantifies.

## Sampling (`reports/m1/sampling.json`)

Deterministic per-source cap 8,000 (seed 42): PAWS kept 8,000 / dropped 57,401; CANNOT kept
8,000 / dropped 69,376; MNLI kept 8,000 / dropped 129,356; NaN-NLI 258 (uncapped). Drops are
logged, not silent.

## Primary recipe — the numbers (`reports/m1/stats-primary.json`)

- **15,983 examples across 12,871 source-families.** Labels **7,325 equivalent / 8,658 not**
  (0.458 / 0.542) — a natural balance, not forced.
- **Per-source, both classes from each source** (so source does not determine label): CANNOT
  7,983 (P(not)=0.522), PAWS-Wiki 8,000 (P(not)=0.562).
- **Coarse source-majority heuristic `source_majority_accuracy` = 0.542** — barely above the 0.5
  base rate, so this predefined M1 heuristic did not flag a dominant source→label mapping. It did
  not test lexical or richer dataset-family separability: later author-run diagnostics found
  available source and label signals, without proving that the trained neural model relied on
  either. Those later diagnostics are historical evidence, not outputs of the public M1 gate.
- Family sizes p50/p95/max = 1 / 2 / 14. Char length p50/p95 = 173 / 310; token-proxy = 30 / 57.

### Category distribution & equivalent-negation coverage

| Category | n | | Category | n |
|---|--:|---|---|--:|
| paraphrase | 6,239 | | explicit_negation | 1,205 |
| unknown | 7,037 | | **equivalent_negation** | **705** |
| hard_paraphrase | 352 | | lexical_contradiction | 149 |
| quantifier_change | 198 | | numeric_change | 44 |
| exact_repeat | 41 | | scope_change | 13 |

`unknown` (7,037) is reported honestly rather than force-fit — mostly PAWS label-0 hard
negatives whose specific perturbation the deterministic categorizer can't name. **Equivalent-
negation = 705 ≥ 200 → sufficient**; these are real "not X" ≡ "un-X" positives from CANNOT,
the slice that stops the model from learning "contains `not` ⇒ reject".

### Negation-token distribution by label (shortcut check)

Fraction carrying an explicit negation cue: **equivalent 0.094 vs not 0.177** (delta 0.083).
The predefined delta heuristic did not flag this cue. Equivalent-negation positives reduce the
coarse "negation ⇒ negative" association, but this check does not establish that a trained model
could not use negation or richer lexical/source signals.

### Split counts (family-cohesive, 80/10/10 over families, seed 42)

train **12,731** · validation **1,644** · internal_test **1,608**. Ratios chosen after
inspecting family sizes (mostly singletons); family integrity beats stratification, so
example-level ratios differ slightly from the family-level 80/10/10.

## Cross-dataset duplicates (`reports/m1/dedup.json`)

Over the combined set: cross-source shared texts **28**, duplicate normalized pairs **11**,
**reverse pairs 427**, duplicate example-ids **0**. The reverse pairs are harmless: `(a,b)` and
`(b,a)` share both text nodes, so they always land in the **same family and split** (never a
cross-split leak). Duplicate-ids are 0 after fixing a real integrity bug — **MNLI `pairID` is
not unique** (2,337 repeats), so the example-id now includes the line index; the build fails
closed on any duplicate id.

## Sealed-set leakage audit (`reports/m1/leakage.json`)

- **Sealed Veritas 75-pair holdout: 0 collisions.** Fully isolated.
- **NaN-NLI eval probe: 17 collisions** (in the 8k sample; 320 across full CANNOT). Root cause:
  **CANNOT was compiled partly from a NaN-NLI subset**, so training on CANNOT would contaminate
  the NaN-NLI probe. Remediation (documented, opt-in `--exclude-sealed-collisions`, not silent):
  **13 colliding source-families excluded** from training; re-audit confirms clean.
- Keyless re-verification (`polarity audit`) on the committed manifests: `manifests-vs-sealed:
  OK (collisions=0, reverse=0, cross_split_families=0)`.

## The two-recipe comparison → MNLI verdict (`reports/m1/recipe-comparison.json`)

| | Primary (PAWS+CANNOT) | + MNLI (exploratory) |
|---|--:|--:|
| examples | 15,983 | 23,983 |
| source→label confound (majority-acc) | **0.542** | **0.695** |
| equivalent-negation | 705 | 705 |
| new categories vs primary | — | none |

Adding MNLI raises the source→label confound by **+0.153** (past the 0.1 bar; MNLI's
P(not\|mnli)=1.0 — negatives only) while adding **no new semantic categories** and **zero new
equivalent-negation coverage**. **Verdict: EXCLUDE MNLI from the training recipe.** It remains a
documented, separable exploratory tier (`manifests/with-mnli/`), never merged into primary.

## Label-noise & shortcut risks

- **PAWS label noise**: PAWS-Wiki is high-overlap adversarial; a small fraction of label-0
  "hard negatives" may be arguable — weak-supervision caveat, carried to M2.
- **`unknown` category is large** (44% of primary) — coverage of the *fine* taxonomy is
  partial; the deterministic categorizer is conservative by design. Not a training blocker, but
  per-category metrics in later milestones must not over-read sparse categories (scope 13,
  numeric 44).
- The coarse M1 source-majority check did **not flag a dominant source→label mapping** in primary
  (`source_majority_accuracy=0.542`). This supported the planned comparison but did not rule out
  lexical or dataset-family shortcut signals; later author-run diagnostics found such signals were
  available but are historical evidence rather than public-gate outputs, and did not establish
  whether the trained model used them.

## Reproduce / gates (keyless)

`uv run pytest` (offline behavior suite) · `ruff` + `mypy src` clean · `uv run polarity audit`
(manifests + sealed isolation, keyless) · `uv run polarity data stats` · `uv run polarity
evaluate --check` (M0 drift gate still green). CI never re-fetches; the committed manifests +
`golden.json` are enough to re-verify isolation without network or keys.

## Exact proposed M2 (needs owner go; still no training)

Reproduce credible baselines on the committed **primary** manifests (MNLI stays excluded per the
verdict): exact/normalized match; a frozen-embedding single cosine threshold; a **vCache-style
per-prompt adaptive threshold** (the strong baseline); and an off-the-shelf NLI cross-encoder —
each reported at the calibrated fixed-false-positive operating points from
`configs/eval/operating-points.yaml`, on validation + internal_test (the sealed Veritas set
stays untouched until M6). Materialize normalized example text locally via the deterministic
builder; keep the public-repo decision deferred until after M2.
