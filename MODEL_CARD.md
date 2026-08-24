---
language:
  - en
library_name: transformers
base_model: microsoft/deberta-v3-base
pipeline_tag: text-classification
tags:
  - sentence-pair-classification
  - semantic-cache
  - cache-equivalence
  - research
---

# Model card for Polarity xp-006

> **No model files are distributed.** This is a repository model card for a source-only research
> release. The private checkpoint is withheld because its licensing posture is not sufficiently
> clear for unrestricted redistribution.

## Model details

### Description

Polarity xp-006 is an English sentence-pair cross-encoder fine-tuned to estimate
`P(CACHE_EQUIVALENT)`: whether the same generated answer could be reused for two queries under an
otherwise identical cache context. It is a binary classifier built on Microsoft DeBERTa-v3-base.
Both texts attend jointly; the output is a raw softmax score, optionally accompanied by a
validation-fit isotonic probability.

This project is a bounded cache-equivalence case study, not a general semantic-equivalence or
paraphrase system.

| field | value |
|---|---|
| experiment | `xp-006-deberta-transformers-fixed` |
| architecture | DeBERTa-v3-base sequence-pair classifier |
| parameters | 184,423,682 |
| language | English |
| base model/tokenizer | `microsoft/deberta-v3-base` |
| exact base/tokenizer revision | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` |
| sequence length | 128 |
| output | raw `P(CACHE_EQUIVALENT)`; calibrated probability reported separately |
| checkpoint availability | unavailable; private pending licensing resolution |

### Repository and checkpoint licenses

Polarity's own source code and documentation are MIT. Microsoft's base-model card and repository
identify DeBERTa-v3-base as MIT. Neither statement assigns a license to Polarity's fine-tuned
checkpoint. PAWS-Wiki/Wikipedia and CANNOT carry separate notices, and CANNOT has a multi-source
upstream chain that has not been fully cleared for unrestricted checkpoint redistribution.

No checkpoint license is asserted. In particular, the private weights are not labelled MIT or
Apache-2.0, and this repository does not grant permission to redistribute them. See `NOTICE`,
`ATTRIBUTIONS.md`, and `docs/DATA.md`. This is a conservative release posture, not legal advice.

## Uses

### Intended use

- Research into second-stage semantic-cache admission after a separate retrieval step.
- Reproducing and auditing the committed aggregate metrics and failure analyses.
- Studying threshold transfer, low-false-positive evaluation, model-collapse diagnosis, artifact
  integrity, and negative-result reporting.
- Local experiments in a controlled cache context where system prompt, user/session context,
  retrieval corpus, tool state, temporal state, and authorization boundary are already part of the
  cache key.

### Out-of-scope use

- Autonomous cache admission in production without a separate fail-closed policy layer.
- Treating a validation-selected FPR budget as a guarantee on a new domain.
- General paraphrase, entailment, duplicate-question, or semantic-equivalence detection.
- Multilingual use.
- Safety-critical medical, legal, financial, security, or personalized-answer caches.
- Reusing answers across users, tool states, retrieval revisions, system prompts, or time-sensitive
  contexts that are not otherwise keyed.
- Using the score as evidence that two claims are true, mutually entailing, or factually current.

## Training details

### Training data

The primary prepared corpus contains 15,983 rows from two capped sources:

| source | rows in prepared corpus | role |
|---|--:|---|
| PAWS-Wiki | 8,000 | paraphrase positives and high-overlap/word-order negatives |
| CANNOT v1.1 | 7,983 | negation-oriented equivalent and non-equivalent pairs |

The group-safe split contains 12,731 training, 1,644 validation, and 1,608 internal-test rows.
Prepared-corpus labels are 7,325 equivalent and 8,658 not equivalent; 705 rows are in the
equivalent-negation category. MNLI was evaluated during data design and excluded from the final
primary recipe. QQP and PAWS-QQP were excluded by the licensing gate.

Raw third-party training text is not distributed. See committed provenance/manifests and
`docs/DATA.md` for acquisition, transformation, label mapping, deduplication, and licensing limits.

These are Polarity's **fine-tuning** sources, not the base model's pretraining corpus. The
[DeBERTa-v3 paper](https://arxiv.org/abs/2111.09543) reports that its main models use the same
roughly 160 GB mixture as RoBERTa; the
[RoBERTa paper](https://arxiv.org/abs/1907.11692) enumerates English Wikipedia, BookCorpus,
CC-News, OpenWebText, and Stories. DeBERTa-v3 also reuses DeBERTa-v2's 128,000-token SentencePiece
vocabulary. Polarity did not assemble, audit, or redistribute those upstream resources; this
lineage is distinct from the PAWS-Wiki+CANNOT recipe above.

### Recipe

| field | value |
|---|---|
| objective | two-class cross-entropy |
| seed | 42 |
| precision | fp32 |
| epochs | 4 |
| effective batch | 32 |
| learning rate | 2e-5 |
| checkpoint selection | repository-defined validation step-pAUC over FPR `[0,0.02]` |
| calibration | isotonic, selected and fitted on validation only; 17 breakpoints |

The repository pAUC is the normalized step integral of maximum achievable recall over the stated
FPR interval. It is not scikit-learn's standardized partial AUC.

### Development history and negative results

The initial DeBERTa runs collapsed to near-constant scores. A controlled fp32 run showed that
precision alone was not sufficient to recover learning. Author-run one-batch controls then found a
strong Transformers-version sensitivity: DeBERTa failed under 5.15.1 and learned under 4.57.6,
while a DistilBERT control learned through the same pipeline. The decisive diagnostic logs are not
committed in machine-readable form, so that causal experiment cannot be independently replayed or
attributed to a specific upstream commit from this repository alone.

M3/M3.1 also exposed an evaluation bug: the original `_pauc` selection metric over-credited a
degenerate all-or-nothing curve, leading M3.1 to select a worse checkpoint. M4 corrected that
metric deterministically and rescored only committed predictions; the corrected result was custom
step-pAUC 0.0 with 0.0 recall through the evaluated low-FPR budgets. The earlier “2× low-FP signal”
was explicitly retracted before the later compatible-environment run.

The corrected 4.57.6 run produced strong same-distribution ranking, but later evaluation also found
threshold drift on Veritas, worsening calibration on that set, and weak MRPC ranking. These
negative results are not superseded by the strong internal-test result.

### Execution environment

Recorded xp-006 telemetry includes Torch `2.6.0+cu124`, Transformers `4.57.6`, and the exact base
and tokenizer revision above. The exact historical Python version, platform/container, and CUDA
runtime independent of the Torch suffix were not recorded. Those fields remain
`unknown:not-recorded` in the execution identity.

The source release profile constrains Transformers to `4.57.6` and Torch to `>=2.6,<2.7`. A
separate broad `models` dependency profile is not the xp-006 training environment. The lockfile's
Torch `2.13.0` resolution belongs only to that separate models profile; it must not be presented as
an xp-006 reproduction environment.

Author telemetry records one RTX 4090 run for xp-006 and an estimated A$0.38 incremental cost.
Across all paid project activity, the reconciled estimate is **approximately A$1.78,
author-reported; no receipts**.

## Evaluation

Positive class = `CACHE_EQUIVALENT`. A false positive accepts a non-equivalent pair and can serve a
wrong cached answer. Thresholds below were selected on validation; realized error rates are always
reported on the named evaluation set.

### Same-distribution internal test (PAWS-Wiki+CANNOT)

- 1,608 rows: 725 positive, 883 negative.
- Threshold-free recall at actual FPR≤1%: **0.763**.
- Repository step-pAUC over FPR `[0,0.02]`: **0.716**.
- At validation-selected `tau=0.994833`: recall **0.903**, 21/883 false positives, and
  **2.38% realized FPR**.
- At that same validation-selected `tau=0.994833`, equivalent-negation recall is **0.983** over 58
  positives, and explicit-negation FPR is **1/107 = 0.009**.

These are source-family-held-out but same-distribution results, not external transfer evidence.

### Veritas target-domain set

Veritas is a 75-pair author-created set that was present from M0: 31 equivalent and 44
non-equivalent. Protocol-before-score chronology is committed, but “scored exactly once” is not
technically enforced. Because the author knew the motivating domain and created the examples, this
is protocol-held-out target-domain evidence—not an independently blinded/general benchmark.

At validation-selected `tau=0.994833`:

- equivalent-pair recall: **31/31 = 1.000**;
- false accepts: **2/44**;
- realized FPR: **4.55%**;
- negation false accepts: **0/22**;
- paraphrases recalled: **21/21**.

At the strict validation zero-observed-FP threshold `0.9996990000000001`, Veritas recall is 22/31
with 0/44 false accepts. A separate descriptive threshold-free sweep at actual Veritas FPR≤1%
achieves 24/31 = **0.774** recall; it is not a replacement operating point.

### External MRPC transfer

MRPC test has 1,725 rows: 1,147 paraphrases and 578 non-paraphrases. Raw text and gold labels are
not distributed. Full reproduction requires a checksum-matched user-obtained local source.

- ROC-AUC: **0.639870**.
- Repository step-pAUC over FPR `[0,0.02]`: **0.031938**.
- At validation-selected `tau=0.994833`: TP=898, FN=249, FP=332, TN=246; recall **0.782912**;
  realized FPR **0.574394**.
- Raw-score medians: paraphrase 0.999194; non-paraphrase 0.997858.

This is weak external transfer. The model is not a general paraphrase detector, and the validation
operating point does not preserve false-positive control on MRPC.

### Post-hoc diagnostic probe

The 44-row authored probe contains six negatives in each of six categories plus eight paraphrase
controls. At `tau=0.994833`, false accepts are entity 0/6, comparative 0/6, operation 0/6, numeric
1/6, temporal 2/6, and measurement 4/6; controls are 8/8 recalled. The probe was written after
Veritas error inspection, partly reuses Veritas material, and cannot establish population rates or
causation.

## Risks, biases, and limitations

- **Threshold transfer:** validation-selected budgets are not deployment guarantees. Realized FPR
  varies sharply across internal test, Veritas, and MRPC.
- **Score saturation:** both MRPC classes score near 1. This is observed alongside semantic
  overacceptance and weak ranking; the artifacts do not establish that saturation caused them.
- **Dataset-family and lexical shortcuts:** author-run bag-of-words diagnostics show available
  source and label signal. Their procedure/output artifacts are not committed, so they are
  historical evidence and do not prove what the neural model relied on.
- **Authorship and sample size:** Veritas and the probe are small, authored by the model developer,
  and not independently blinded.
- **Task mismatch:** MRPC paraphrase labels are related to but not identical with cache equivalence.
- **English/source bias:** training is English and dominated by Wikipedia-derived and
  negation-oriented source families.
- **Calibration shift:** validation-fit isotonic calibration does not improve Veritas ECE.
- **Security:** the model has not been evaluated against adversarial text, prompt injection,
  Unicode/formatting attacks, tenant leakage, or temporal/tool-state drift.
- **Reproducibility:** historical Python/container details and replayable M5 diagnostic logs are
  absent; a byte-for-byte retrain claim is not supported.
- **License chain:** the training-data-to-checkpoint redistribution posture is unresolved, so no
  weights are published.

## Model artifact integrity

The private weight file is 737,719,272 bytes with SHA-256
`8d765a7e55c94bb5a8c7754eb705ddf8dc7e6c74cc72e1b0ce4c9110e7a0f355`. The seven-file
model/tokenizer bundle fingerprint is
`pl1:0dfcb787d1b5ea811b7405606f8a1e1ae95d50802053c0227324d16cb3c33df1`. The recipe
fingerprint is `pl1:04857dd3f1522863cc9e99378569403ab76f72f5f64d7fc59abc98e38a125c17`; the execution
fingerprint is `pl1:bc55b62acce7235c6442b53e2b3cf1147241619cac801899d7993db7ee289cec`.

A holder of the private bundle can verify it with:

```sh
uv run polarity artifact verify \
  --manifest reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json \
  --artifact-dir artifacts/xp-006
```

This verifies integrity only. It does not grant access or redistribution rights.

## Reproducibility

- `polarity transfer report --check` reproduces Veritas aggregate metrics from committed scores.
- `polarity transfer probe-report --check` reproduces the authored diagnostic report.
- `polarity transfer mrpc-report --check` is source-free and checks score schema, sequential
  `row_index` values/count, the synthetic `mrpc-test-{row_index}` ID format, and internal committed
  hashes/bindings. It cannot authenticate those rows against official MRPC rows; its success
  message states that metrics were not reproduced.
- `polarity transfer mrpc-report --check --source PATH` performs full MRPC reproduction only with
  an already constructed local JSON file after byte-checksum/schema/order verification. The
  repository does not currently supply a deterministic conversion procedure for creating it.
- `polarity artifact verify` verifies the private model/tokenizer bundle independently of model
  loading.

See `docs/RELEASE-INTEGRITY.md` for the distinction between metric reproduction, artifact
integrity, and incomplete retraining reproducibility.

## Public claim ceiling

> Built a 184M-parameter DeBERTa-v3 cross-encoder for cache equivalence after diagnosing a
> Transformers-version-sensitive training collapse. At a threshold selected on validation to
> target ≤1% FPR, an author-created held-out 75-pair cache-query set yielded 31/31 equivalent-pair
> recall and 2/44 false accepts (4.5% realized FPR), including 0/22 negation false accepts.
> External MRPC transfer was weak (ROC-AUC 0.640), so the model is not a general paraphrase
> detector.

## Citation

No formal paper citation is assigned. Cite the repository revision and this model card, and retain
the upstream attributions in `ATTRIBUTIONS.md`.
