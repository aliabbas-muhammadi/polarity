# M6 REPORT — Veritas target-domain transfer (bounded evidence)

M6 applied one frozen DeBERTa-v3 cache-equivalence checkpoint and validation-selected decision
rules to an author-created 75-pair cache-query set. The protocol commit predates the committed
Veritas scores, and no later tracked training/configuration change exists. That chronology supports
a protocol-held-out evaluation. It does not make Veritas independently blinded, and Git cannot
technically prove the author's record that the set was scored exactly once.

This report supersedes the earlier broad transfer disposition. The defensible result is a bounded
target-domain observation, followed by a weak external MRPC result in M6.1.

## Question

Does Polarity's in-distribution M3.3 ranking carry into the cache-query setting that motivated the
project without accepting meaning-changing negation?

For comparison, M3.3 internal-test threshold-free recall at actual FPR≤1% is **0.763**. At the
threshold selected on validation to target FPR≤1% (`tau=0.994833`), internal-test recall is
**0.903** and the realized internal-test FPR is **2.38%** (21/883). Selection-set budgets and
evaluation-set realized rates are different quantities.

## Checkpoint continuity and reproducibility limit

The original M3.3 weights were not retained. The author retrained `xp-006` with the recorded recipe
(fp32, seed 42, Torch `2.6.0+cu124`, Transformers `4.57.6`) and observed a maximum score difference
of **0.0** against committed M3.3 predictions over 1,608 internal-test and 1,644 validation rows.
This establishes output agreement on those committed rows; it does not prove a byte-identical
checkpoint or a generally deterministic retrain across hardware.

The historical telemetry did not record the exact Python version, platform/container, or CUDA
runtime independently of the Torch build suffix. The repository therefore separates:

1. metric reproduction from committed scores;
2. integrity verification of the private checkpoint bundle; and
3. retraining reproducibility, which remains incomplete.

See `docs/RELEASE-INTEGRITY.md` for the recipe/execution fingerprint contract.

## Protocol v2 serialization erratum

Protocol v1 was committed before the Veritas score artifact. It selected all raw-score thresholds
on validation and froze an isotonic calibrator for probability reporting. Protocol v1 rounded the
strict zero-observed-FP threshold to `0.999699`, equal to the maximum validation-negative score.
Because decisions use `score >= tau`, that serialization could admit the negative it was intended
to exclude.

Protocol v2 is a post-evaluation serialization correction derived only from committed validation
predictions. It stores the exact `nextafter(max_negative, +infinity)` value
`0.9996990000000001`, records selection and realized metrics separately, and validates the
round-trip invariant. Raw model scores, calibration, and the validation selection rules did not
change; no rescoring or retraining was performed for this correction.

## Artifact identity

The private checkpoint has 184,423,682 parameters. Its weight file SHA-256 is
`8d765a7e55c94bb5a8c7754eb705ddf8dc7e6c74cc72e1b0ce4c9110e7a0f355`. The committed model
manifest covers all seven files required to load the model and tokenizer, with bundle fingerprint
`pl1:0dfcb787d1b5ea811b7405606f8a1e1ae95d50802053c0227324d16cb3c33df1`.

The bundle is private. `polarity artifact verify` can verify a local copy without loading Torch or
model code. This source repository does not distribute the checkpoint.

## Evaluation set

Veritas contains 75 author-created pairs: 31 equivalent and 44 not equivalent. Its slices are 10
exact repeats, 21 paraphrases, 22 negation-must-not-hit pairs, and 22
different-answer-must-not-hit pairs. The examples were present from M0 and the project's motivating
domain was known to the author. Veritas is therefore a **protocol-held-out target-domain
evaluation**, not an independently authored, blinded, or general-purpose benchmark.

No exact Veritas pair overlaps training, validation, or internal test.

## Validation-selected operating points, realized on Veritas

Positive class = `CACHE_EQUIVALENT`; a false positive is an unsafe cache admission.

| validation selection rule | frozen tau | Veritas recall | 95% CI | Veritas FP / 44 | realized Veritas FPR | precision |
|---|--:|--:|:--:|--:|--:|--:|
| zero observed FP | 0.9996990000000001 | 22/31 = 0.710 | [0.534, 0.839] | **0** | 0.000000 | 1.000 |
| **target FPR≤1%** | **0.994833** | **31/31 = 1.000** | [0.890, 1.0] | **2** | **0.045455** | 0.939 |
| target FPR≤5% | 0.035936 | 31/31 = 1.000 | [0.890, 1.0] | 7 | 0.159091 | 0.816 |

At the primary validation-selected threshold, Veritas produced all 31 true positives and 2 false
accepts among 44 negatives. The realized FPR is **4.55%**, not ≤1%. The phrase “target FPR≤1%”
names the validation selection rule only.

A separate descriptive threshold-free sweep constrained to **actual Veritas FPR≤1%** gives
TP=24, FN=7, FP=0, TN=44: recall **24/31 = 0.774194** at `tau=0.999522`. This dataset-specific
sweep did not replace the frozen deployment threshold.

## Slice results at validation-selected `tau=0.994833`

| slice | n | result | note |
|---|--:|---:|---|
| exact-repeat | 10 | 10/10 recalled | all accepted |
| paraphrase-should-hit | 21 | 21/21 recalled | all accepted |
| negation-must-not-hit | 22 | **0/22 false accepts** | Wilson upper bound is not zero |
| different-answer-must-not-hit | 22 | **2/22 false accepts** | both committed for error analysis |

The 0/22 negation result is encouraging on this finite authored slice; it is not proof of a zero
population error rate or of safe autonomous cache admission.

## Baseline context

| on Veritas | recall | false accepts | provenance |
|---|--:|--:|---|
| frozen embedding cosine | 0.387 | 13/44 | historical M0 comparison |
| Polarity at validation-selected `tau=0.994833` | 1.000 | 2/44 | M6; realized FPR 0.045455 |
| Polarity threshold-free at actual FPR≤1% | 0.774 | 0/44 | descriptive M6 sweep |
| Polarity at validation zero-observed-FP threshold | 0.710 | 0/44 | M6 |
| Haiku LLM guard | 0.97 | 0/44 | historical Veritas result; not rerun |

Polarity improved on the frozen embedding comparison on this set. Comparisons with the historical
LLM guard are contextual rather than a controlled head-to-head experiment.

## Calibration and custom pAUC

Validation-fit isotonic calibration changed Veritas ECE from 0.068 raw to 0.087 calibrated, so it
did not improve calibration on this set. Decisions use raw scores.

Where this repository reports low-FPR pAUC, it means a normalized step integral of maximum
achievable recall over the stated FPR interval. It is **not** scikit-learn's standardized partial
AUC.

## Error analysis

The two false accepts at `tau=0.994833` are `account-task-flip` (0.9952) and `rect-measure`
(0.9995), both from the different-answer slice. This is a descriptive observation. The later M6.1
probe concentrated its authored errors in measurement, temporal, and numeric examples, but that
post-hoc probe cannot identify a population failure rate or a causal training-data defect.

## Bounded interpretation

Polarity is a case study in designing, training, debugging, calibrating, and auditing a
task-specific cache-equivalence cross-encoder. It shows strong ranking on its primary distribution
and encouraging negation behavior on a small target-domain set. It also shows that a validation
false-positive budget does not transfer as a guarantee: the primary threshold realized 4.55% FPR
on Veritas and 57.44% FPR on MRPC.

The result does **not** establish general semantic equivalence, broad cross-domain transfer, safe
autonomous cache admission, or a deployable population FPR.

## Limitations

- Veritas is small (N=75), author-created, and present from M0; confidence intervals are wide.
- Protocol-before-score chronology is verifiable, but “scored exactly once” is not technically
  enforced or independently provable.
- Thresholds and calibration were selected on validation; their realized rates differ on each
  evaluation set.
- The exact historical Python/container is unknown, so a fresh clone cannot recreate the original
  training environment exactly.
- One model and one seed were evaluated.
- M6.1 MRPC transfer is weak (ROC-AUC 0.639870), limiting any broader interpretation.

## Public claim ceiling

> Built a 184M-parameter DeBERTa-v3 cross-encoder for cache equivalence after diagnosing a
> Transformers-version-sensitive training collapse. At a threshold selected on validation to
> target ≤1% FPR, an author-created held-out 75-pair cache-query set yielded 31/31 equivalent-pair
> recall and 2/44 false accepts (4.5% realized FPR), including 0/22 negation false accepts.
> External MRPC transfer was weak (ROC-AUC 0.640), so the model is not a general paraphrase
> detector.

This wording is proposed only for a later approved Portfolio update. No Portfolio or resume change
is made here.

## Release posture

The source, documentation, and aggregate results may be released only after the separate release
checklist and approval boundary. The checkpoint remains private because its licensing posture is
not sufficiently clear for unrestricted redistribution. It must not be labelled MIT or Apache-2.0,
and no redistribution license is asserted.

## Reproduce and verify

```sh
uv run polarity transfer report --check

uv run polarity artifact verify \
  --manifest reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json \
  --artifact-dir artifacts/xp-006
```

The first command reproduces metrics from committed Veritas scores and protocol without inference.
The second verifies a separately held private bundle; it cannot run in a source-only clone unless
the user supplies that bundle.
