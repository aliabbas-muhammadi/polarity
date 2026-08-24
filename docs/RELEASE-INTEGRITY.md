# Release integrity

Polarity separates three reproducibility claims that must not be collapsed into one:

1. **Metric reproducibility** recomputes reports from committed prediction artifacts.
2. **Artifact integrity** proves that a local checkpoint bundle is byte-identical to the
   committed manifest.
3. **Training reproducibility** would recreate the model from data, recipe, and execution
   environment. The historical xp-006 run does not meet this strongest claim.

## Checkpoint bundle

`reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json` covers every file required to
load the frozen checkpoint: `model.safetensors`, `config.json`, the tokenizer JSON/config/special
token files, and `spm.model`. Each entry records byte size and SHA-256, and the canonical bundle
declaration has its own `pl1:` content fingerprint.

Verify a private local copy without importing Torch or model code:

```sh
uv run polarity artifact verify \
  --manifest reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json \
  --artifact-dir artifacts/xp-006
```

Verification fails closed for a missing or malformed manifest, missing required declaration,
missing file, size/hash mismatch, symlinked root or traversed path component, unsafe path, or
undeclared extra file. The checkpoint remains private and is not part of the repository. Integrity
coverage is not a distribution license; the bundle is withheld pending licensing resolution.

`polarity transfer veritas` runs that complete bundle verification before loading or scoring the
model. The frozen M6 protocol and its generated report carry the checkpoint, bundle, recipe, and
execution fingerprints. The model manifest also binds the exact protocol bytes by SHA-256 and
repeats the frozen thresholds and calibration method. Scoring is refused before runtime or model
loading if the protocol digest, identity, declared operating points, calibration method, requested
recipe, or pinned base/tokenizer revisions differ; the validated calibrator is rebuilt during this
preflight.

## Recipe and execution identity

`recipe_fingerprint` is the historical fingerprint over the model/data/hyperparameter recipe.
It deliberately remains stable across xp-005 and xp-006 so their controlled library-version
comparison is preserved. `model_fingerprint` is retained as a compatibility alias for that same
value.

`execution_fingerprint` separately covers:

- Python version;
- platform;
- Torch and Transformers versions;
- CUDA version;
- base-model revision; and
- tokenizer revision.

The xp-006 telemetry records Torch `2.6.0+cu124`, Transformers `4.57.6`, and both pinned model
revisions. It did **not** record the exact Python version, operating-system/container identity, or
CUDA runtime independently of the Torch build suffix. Those manifest values are therefore
`unknown:not-recorded`; they are not inferred or reconstructed after the fact. The exact
historical Python/container environment is not recoverable, so xp-006 must not be described as a
fully reproducible retrain.

The supported xp-006 release profile is Transformers exactly `4.57.6` and Torch `>=2.6,<2.7`.
The author-run training path checks installed package metadata before starting. The lockfile's
Torch `2.13.0` resolution belongs only to the unrelated M2 `models` profile with its broader
dependency range; it cannot recreate xp-006 and must not be cited as its Torch version.
