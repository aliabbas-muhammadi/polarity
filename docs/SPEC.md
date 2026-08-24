# SPEC — Research question, label definition, contribution

## Research question

Can a sentence-pair **cross-encoder**, trained specifically for semantic-cache equivalence,
distinguish genuine paraphrases from meaning-changing negation/scope/quantity/entity
perturbations **substantially better than a single frozen-embedding cosine threshold** at a
**validation-selected operating point with realized evaluation-set false-positive rates reported
separately**, not merely by AUC/F1, while avoiding a required paid judge call on every lookup? No
controlled xp-006-versus-judge cost or latency advantage is part of this research question.

This is a hypothesis. A partial or negative result (e.g. "recovers negation but regresses on
ordinary paraphrase", "cannot beat a vCache-style adaptive threshold at equal FP", "off-the-
shelf NLI already closes most of the gap") is a valid, publishable outcome.

## The prediction target

Polarity predicts a single quantity:

```
safe_to_reuse_cached_answer(query_a, query_b) -> P(CACHE_EQUIVALENT)
```

**CACHE_EQUIVALENT** — reusing the *same* generated answer for both requests is semantically
safe: the correct answer to `query_a` would be a fully correct answer to `query_b`.

**NOT_CACHE_EQUIVALENT** — reusing it is unsafe: the answers could differ, be opposite, or be
logically inverted.

This is **not** generic textual entailment (which is directional and 3-way) and **not**
paraphrase-identity. It is a symmetric, binary, cache-reuse safety judgment.

### Operating context (assumptions — governance G4)

The label is only well-defined *inside a controlled cache context*. Polarity models
**textual/query equivalence under that context**, not universal cache safety. Assumed equal
across the two queries: system prompt; user/session context; retrieval corpus and its
version; tool state; temporal context; and no user-specific hidden state unless it is part of
the cache key. Two textually equivalent queries can still be unsafe to co-serve for reasons
outside this scope (e.g. per-user personalization) — those are out of scope and must not be
read as Polarity failures.

### False-positive convention (governance G5)

The **positive class is CACHE_EQUIVALENT.** Therefore:

```
false positive = NOT_CACHE_EQUIVALENT predicted CACHE_EQUIVALENT (score >= tau)
```

A false positive serves a wrong cached answer — the expensive failure. A false negative
merely costs one extra generation. The primary metric is recall vs. false-positive rate; this
definition is printed next to every FP/FPR table in the code and reports. We never report a
population `0% FPR` from a finite set — only "0 observed FP on N negatives", with an interval.

## Pre-registered category taxonomy

No aggregate metric may hide a per-category failure, so every pair is tagged with one
category (`src/polarity/types.py::Category`). The taxonomy is fixed **before** results are
seen; it will be refined only if the data demands it, and never retroactively to flatter a
number.

| Category | Typical label | Example |
|---|---|---|
| `exact_repeat` | equivalent | verbatim repeat |
| `paraphrase` | equivalent | "how tall" ↔ "what is the height" |
| `hard_paraphrase` | equivalent | acronym/definition swap, no lexical overlap |
| `equivalent_negation` | **equivalent** | "not secure" ↔ "insecure"; "not uncommon" ↔ "common" |
| `explicit_negation` | not equivalent | "is X deprecated?" ↔ "is X **not** deprecated?" |
| `lexical_contradiction` | not equivalent | "passing" ↔ "failing"; antonym flip |
| `numeric_change` | not equivalent | "15% of 200" ↔ "15% of 400" |
| `entity_change` | not equivalent | "Canada" ↔ "Brazil" |
| `temporal_change` | not equivalent | "December" ↔ "November" |
| `quantifier_change` | not equivalent | "all" ↔ "some" |
| `modality_change` | not equivalent | "can I" ↔ "must I" |
| `scope_change` | not equivalent | "adults" ↔ "infants" |
| `relation_reversal` | not equivalent | "A depends on B" ↔ "B depends on A" |
| `meaning_change_other` | not equivalent | coarse umbrella for externally-bucketed sets |

### The central trap

The task must **not** collapse to a "contains the token `not`" detector. Two guard rails:

1. **Equivalent-negation positives** — "X is not safe" ≡ "X is insecure", "it is not uncommon"
   ≡ "it is common". These carry a negation cue but are CACHE_EQUIVALENT. They are the single
   hardest slice and are pre-registered as their own category.
2. **Negation cues in benign paraphrases** — "fall asleep *without* medication" ≈ "drug-free
   ways to sleep". A cue is not a verdict.

## Contribution framing (conservative)

After the literature review (`docs/LITERATURE.md`), the honest position is:
**Polarity is replication-plus-specialization, not a new category.** Learned/adaptive semantic
caches already exist (vCache, ICLR 2026), and cross-encoders have already been trained for
cache equivalence (Redis/NYU calibration-gap paper, arXiv:2606.19719).

**Defensible contribution:** a negation/polarity-**isolated** evaluation and training slice for
cache equivalence, plus a **calibrated** comparison using thresholds selected at a
validation false-positive budget and applied unchanged to named evaluation sets. The realized FPR
on each set is part of the result; the selection budget is not a transfer guarantee. Comparisons
include a vCache-style per-prompt adaptive threshold—not only a static cutoff. The LLM judge is a
historical comparison only; no controlled cost or latency comparison was run.

**Claims to avoid** (each falsifiable by cited prior art): "first cache-equivalence
cross-encoder"; "first/learned semantic cache"; "solves negation"; "beats the LLM judge" on
accuracy, cost, or latency; any headline resting on AUC/F1 without a calibrated operating point.

## Non-goals (v1)

Live gateway integration (integration is via a committed eval arm, `armXenc`, at M6); a
retrieval index (the cross-encoder is a second-stage verifier over a small candidate set);
multilingual coverage; and generative rewriting/canonicalization.
