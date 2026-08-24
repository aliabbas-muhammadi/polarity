"""M2 no-training baselines.

Every baseline emits a per-pair score in [0,1] = P(cache-equivalent) as a committed
``BaselinePrediction``. Model inference (embedding / NLI cross-encoder / adaptive) is the one
author-run step that downloads models and uses CPU; everything downstream — threshold selection,
metrics, calibration, shortcut analysis, the decision — is recomputed deterministically and
keylessly from the committed predictions (see ``report.py`` and docs/M2-METHODOLOGY.md).
"""
