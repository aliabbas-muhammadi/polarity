# DATA — dataset matrix, licensing gate, label mapping, registry

M0 downloads **nothing** (the sealed Veritas snapshot is the only real data present, and it is
the author's own eval set). This document is the **build spec for M1**, and its licensing gate
decides which sources M1 is allowed to touch.

## The licensing gate

**Accessibility is not permission.** A dataset being downloadable grants no redistribution or
derivative rights. Each source is classified; anything **`LICENSE_UNRESOLVED`** is excluded from
the build until a written grant is obtained. The final checkpoint is withheld: this repository has
not established that every upstream training source permits unrestricted/commercial distribution
of the learned parameters. That conservative posture is not legal advice and does not decide
whether a learned parameter file is a copyright adaptation.

Legend: **Train?** records the project's research-use decision at build time; **Commit text?**
means whether derived pair text could be committed under the stated conditions; **Weights?** is
the current release posture, not a general legal conclusion.

| Dataset | License | Train? | Commit text? | Redistribute? | Weights? | Status |
|---|---|---|---|---|---|---|
| **PAWS-Wiki** | Google notice + Wikipedia CC-BY-SA-3.0 | Yes | Yes* | Yes* | **Withheld** | **USABLE_WITH_CONDITIONS** |
| **PAWS-QQP** | Google notice; raw data QQP-bound, not redistributable | reconstruct only | No | No | No | **LICENSE_UNRESOLVED** |
| **QQP** | Quora ToS — "non-commercial use", no open license | NC only | No | No | No | **LICENSE_UNRESOLVED** |
| **MNLI** | OANC + CC-BY-3.0 / CC-BY-SA-3.0 / US-PD | Yes | Yes* | Yes* | Not evaluated | **USABLE_WITH_CONDITIONS** |
| **ANLI** | CC-BY-**NC**-4.0 | NC only | Yes* (NC) | Yes* (NC) | **Project-excluded** | **USABLE_WITH_CONDITIONS** |
| **CANNOT** | declared CC-BY-SA-4.0; upstream chain requires review | Yes | Yes* | Yes* | **Withheld** | **USABLE_WITH_CONDITIONS** |
| **NaN-NLI** | CC-BY-SA-4.0 (HF card) | Yes | Yes* | Yes* | Not evaluated | **USABLE_WITH_CONDITIONS** |

`Yes*` = yes with a binding condition (attribution and, for CC-BY-SA / Wikipedia-derived text,
**share-alike** on distributed adapted text). The source-only release distributes neither raw
third-party corpora nor the private checkpoint. The top-level MIT license covers original Polarity
code and documentation; it does not automatically grant rights over manifests, third-party-derived
prediction rows, reports, dataset text, base/tokenizer files, or weights. See the artifact-scope
table in `ATTRIBUTIONS.md`.

### Consequences (this reshapes the original recipe)

- **QQP and PAWS-QQP are blocked.** The original plan leaned on "QQP duplicates" as positives.
  Under Quora's ToS (non-commercial, no redistribution grant, and Google itself refuses to
  redistribute PAWS-QQP), both are `LICENSE_UNRESOLVED` and **out of M1**. Positives must come
  from **PAWS-Wiki** (with attribution + share-alike) and later curated equivalent-negation
  cases, not QQP.
- **ANLI declares CC BY-NC 4.0.** It was not part of the final primary recipe and is
  project-excluded from any future provenance-clean unrestricted-release effort. This is a
  conservative source-selection rule, not a conclusion about license inheritance by learned
  parameters.
- **CANNOT** is the workhorse for the negation slice. Its repository declares CC BY-SA 4.0, but it
  compiles NaN-NLI, GLUE Diagnostic, WikiFactCheck-English, Sentiment Labelled Sentences, and
  SemAntoNeg/antonym-substitution material. The current project has not resolved every upstream
  right in that chain; in particular, the release audit identified no explicit redistribution
  grant for GLUE Diagnostic itself. The checkpoint remains private, and the repository makes no
  assertion that this upstream chain permits checkpoint redistribution.
- **NaN-NLI** license rests on a single authoritative source (the author's HF card); re-check
  before the release milestone. Use as an **eval** probe only (too small to train on).

Per-dataset citations and the full reasoning are in the licensing appendix at the end of this
file.

## Label-mapping risk (do not copy labels blindly)

"Cache equivalence" is *would the same generated answer be acceptable for both queries?* — not
paraphrase-identity and **not** an NLI label. Each source needs a documented mapping function:

- **PAWS** label 1 → likely CACHE_EQUIVALENT; label 0 (high-overlap non-paraphrase) → excellent
  **hard negatives**. PAWS targets word-order, **not** negation.
- **QQP** (if ever unblocked) duplicate → CACHE_EQUIVALENT (noisy; weak supervision).
- **MNLI/ANLI** — **do not** equate: entailment is directional (A⊨B ≠ same answer); contradiction
  ≠ automatically cache-non-equivalent. Use for backbone pretraining with a re-labeling function,
  never as raw cache labels.
- **CANNOT** negated/non-negated → maps to the negation slice (meaning changed by negation ⇒
  NOT_CACHE_EQUIVALENT), with the caveat that it is synthetic/rule-augmented.
- **NaN-NLI / NevIR** → hard held-out negation-scope evaluation, not training.

## Registry & provenance format

Every dataset (and the sealed snapshot) carries a `PROVENANCE.json` written by
`src/polarity/data/registry.py::write_provenance` (deterministic: sorted keys, 2-space indent).
Fields: `name, role, source, revision, license, license_status, retrieved, rows, sha256,
changes, sealed, notes`. `license_status ∈ {USABLE, USABLE_WITH_CONDITIONS, LICENSE_UNRESOLVED}`
is the gate M1 reads.

Raw third-party datasets are **not** committed; M1 ships deterministic download+build scripts
(pinned revision + checksum) instead. No mystery CSVs.

## MRPC evaluation provenance

MRPC was used only for the M6.1 external diagnostic. Microsoft's official installer embeds a
Microsoft Research Shared Source license permitting non-commercial redistribution under its
conditions and prohibiting commercial use/distribution; no commercial redistribution grant was
identified. The original GLUE downloader separately states that MRPC cannot be hosted for legal
reasons. The source-only release therefore excludes raw MRPC text and the complete gold-label
vector from the public tree and reachable history.

The public artifact `predictions/m6.1/mrpc-transfer.jsonl` contains score-only rows. Full metric
reproduction requires the user to obtain the source independently and pass a checksum/order-matched
local JSON file to `polarity transfer mrpc-report --check --source PATH`. Exact provenance and
duplicate policy live in `reports/m6.1/mrpc-provenance.json`.

## Split & balance (see METHODOLOGY.md)

Split by **source family before augmentation**; commit immutable manifests. Track class balance
(positive/negative) **and** category balance — the equivalent-negation positives especially,
since they are the rarest and hardest slice and must be present in every split.

---

## Licensing appendix (citations)

- **PAWS** — Google custom permissive notice (github.com/google-research-datasets/paws/LICENSE);
  PAWS-Wiki sentences are adaptations of Wikipedia (CC-BY-SA-3.0), so committed derived text
  carries attribution + share-alike. **PAWS-QQP is not redistributed by Google** "due to the
  license of QQP" (repo README) → reconstruct-only, not committable.
- **QQP** — Quora's release states the data "is subject to our Terms of Service, allowing for
  non-commercial use" (Quora, 2017; corroborated by Google's PAWS-QQP refusal). No open license,
  no redistribution/derivative grant → `LICENSE_UNRESOLVED`.
- **MNLI** — paper/dataset-card licensing: majority under OANC's permissive license, FICTION
  under CC-BY-3.0 / CC-BY-SA-3.0 / US public domain; no NC term. Committed derived text →
  attribution + share-alike (FICTION "Seven Swords" is CC-BY-SA-3.0).
- **ANLI** — the HF card and repository declare CC BY-NC 4.0. Polarity excludes it from any
  provenance-clean unrestricted-release recipe; no claim is made that its terms necessarily attach
  to learned parameters.
- **CANNOT** — the repository and dataset card declare CC-BY-SA-4.0. It compiles NaN-NLI, GLUE
  Diagnostic, WikiFactCheck-English, Sentiment Labelled Sentences, and SemAntoNeg. The aggregate
  declaration does not by itself resolve every upstream right for an unrestricted checkpoint
  release; no explicit GLUE Diagnostic redistribution grant was identified during the release
  audit. See the original-source links and detailed caveats in `ATTRIBUTIONS.md`.
- **NaN-NLI** — CC-BY-SA-4.0 per the author's HF card (github repo has no LICENSE file); premises
  adapted from Pullum & Huddleston (2002) grammar examples (de-minimis). Single-source license →
  re-check before release.
- **MRPC** — the official Microsoft installer embeds a non-commercial shared-source redistribution
  grant under stated conditions and prohibits commercial use/distribution; no commercial
  redistribution grant was identified. GLUE's downloader separately says the project cannot host
  MRPC for legal reasons. Source-only publication carries acquisition/checksum instructions and
  aggregate results, not raw text or gold labels.
