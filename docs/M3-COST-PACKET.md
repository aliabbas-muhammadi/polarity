# M3 COST PACKET — the money checkpoint (needs explicit owner approval)

**No spend is authorized yet.** This is the packet required by `docs/COSTS.md` before any paid
action. M0–M2 cost **$0**; this preparation cost **$0**. Below: the bottom-up estimate, the provider
comparison at current prices, the recommendation, and the hard controls.

## Bottom-up estimate (the COSTS.md method, with real numbers)

1. **Sequence length** — measured (`polarity train tokstats`): train joint-pair tokens p99 = **77**,
   max 188; validation p99 = 74. `max_length = 128` truncates 0.016% of train, 0% of validation.
2. **VRAM** — DeBERTa-v3-base (~184M) at max_len 128, micro-batch 16, bf16: peak **well under 8 GB**.
   Every candidate card (24 GB 4090/L4/A10, 40/80 GB A100) has 3–10× the VRAM needed → **no A100**.
3. **Steps** — 12,731 pairs / eff-batch 32 ≈ 398 updates/epoch × 4 epochs ≈ **~1,592 updates**.
4. **Wall → cost** — a base-size cross-encoder on ~12.7k short pairs for 4 epochs is a
   **minutes-long, single-GPU** job: ~5–12 min on a 4090, ~15–30 min on an L4/A10, + ~3–5 min for
   env spin-up + tokenization + a final eval. **Comfortably under one hour**, incl. a short
   `sec_per_step` profiling run.

## Provider comparison (current on-demand prices; FX 1 USD = 1.41 AUD, 2026-08-18)

| Provider | GPU (24 GB unless noted) | US$/hr | Billing | Est. run cost (AUD) | Notes |
|---|---|--:|---|--:|---|
| **RunPod Secure** | **RTX 4090** | 0.74 | per-second | **~A$0.26–0.52** | managed, template env ~1–2 min, hard cap via prepaid balance |
| **RunPod Secure** | **L4** | 0.49 | per-second | **~A$0.35** | cheapest reliable; ~2× slower than 4090 |
| RunPod Community | RTX 4090 | 0.34 | per-second | ~A$0.24 | no-SLA marketplace host (budget option) |
| **Modal** (fallback) | L4 / A10 | 0.80 / 1.10 | per-second, no idle | **~A$0.45 (or $0 w/ US$30/mo free credit)** | env-as-code (most reproducible), `timeout=` hard-kill |
| Vast.ai | RTX 4090 | ~0.44 (median) | per-second | ~A$0.31 | cheapest, variable host reliability |
| Lambda | A10 (no 4090/L4) | 1.29 | per-min VM | ~A$0.91 | manual-terminate; not recommended |
| Paperspace | A4000 (no 24 GB consumer) | 0.76 | **per-hour** | ~A$1.07 | per-hour billing penalizes a short job |

(A100 everywhere = 4–10× the rate for VRAM we never touch — excluded.)

## Recommendation

- **Provider:** RunPod — Secure Cloud
- **GPU:** RTX 4090 (24 GB) — or **L4 (24 GB)** for the lowest reliable rate
- **VRAM:** 24 GB (need < 8 GB) · **Price/hr:** US$0.74 (4090) / US$0.49 (L4)
- **Expected runtime:** ~15–35 min incl. spin-up + a profiling run
- **Expected compute:** ~1,592 optimizer updates, 4 epochs, eff-batch 32, max_len 128, bf16
- **Estimated cost:** **~A$0.30–0.52 per run**
- **Maximum authorized cap requested:** **A$15** (the `docs/COSTS.md` ceiling and the config's
  programmatic `max_estimated_cost_aud` backstop). The **binding** control is the **1.0 h runtime
  cap** ≈ **A$1** — the run stop-losses there long before the cost cap.
- **Why this GPU:** reliable managed availability + reproducible container env + per-second billing +
  24 GB headroom, optimizing reliability over saving a dollar (COSTS.md rule). 4090 for ~2× speed;
  L4 for the lowest reliable price.
- **Fallback:** Modal L4/A10 — tightest spend bound (per-second, no idle charge, code-set `timeout`,
  US$30/mo free credit → effectively A$0), env pinned as code; the trade-off is code-first serverless.

## Hard controls (brief §20/§26)

- **Refuse-to-start:** `polarity train run` calls `assert_manifests_match` (dataset-version guard) and
  `CostGuard.check_estimate` — it will not start if the bottom-up estimate exceeds the cost cap or the
  runtime cap, or if any manifest sha drifted from the approved fingerprint.
- **Stop-loss:** `CostGuard.check_elapsed` at every eval boundary raises before elapsed wall time
  crosses `max_runtime_hours` (1.0 h) — the real ≈A$1 ceiling.
- **What happens on overrun:** the guard raises `CostError` at an eval boundary, after the best
  checkpoint and telemetry have been written. The run is not marked complete and **cannot be
  resumed exactly**: retain it for inspection or restart training in a fresh or cleared run
  directory.
- **How we terminate:** RunPod — leave auto-pay OFF and fund a small fixed prepaid balance (e.g. US$5);
  pods auto-stop at $0, plus a default US$80/hr account cap; stop/destroy the pod when done. Modal —
  per-second billing + a code-set `timeout` force-kills a hung run; no idle charge.
- **Artifacts persisted before termination:** at each eval boundary the runner writes
  `runs/<id>/{best/ (safetensors), logs/metrics.csv}`. After a completed evaluation boundary, those
  artifacts survive an interruption for inspection; a stop before the first boundary may leave
  neither. Training must restart rather than continue from that directory. Only outputs from a
  completed run are candidates for subsequent prediction/reporting.
- **Reporting:** burn-rate at checkpoints; a single run, one seed, no sweep, no unattended re-runs.

## Approval

Paid M3 begins only on your explicit approval of: **provider RunPod Secure, GPU 4090 or L4, expected
~A$0.30–0.52, max authorized cap A$15 (binding runtime cap 1.0 h ≈ A$1)**. Veritas stays sealed (M6).
