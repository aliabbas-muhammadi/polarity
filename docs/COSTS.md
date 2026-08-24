# COSTS — compute estimate, providers, and the approval protocol

**No new spend is authorized.** M0 itself used $0; later owner-approved training runs are recorded
in `docs/EXPERIMENTS.md`. This document remains the packet required before any future paid action.

## Recorded cumulative spend

Across the project's author-run training and checkpoint-recovery activity, the reconciled estimate
is **approximately A$1.78, author-reported; no receipts**. Per-run figures elsewhere are historical
telemetry estimates, not independently verified invoices. Do not present the aggregate or any
individual estimate as an exact audited cost.

## Estimation method (fill in at M2/M3 with real numbers)

The training cost is estimated bottom-up, not guessed:

1. **Sequence-length distribution.** Tokenize the built pair corpus; record p50/p95/max token
   lengths for `[CLS] a [SEP] b [SEP]`. Candidate `max_length = 128` is a hypothesis to confirm
   against p95 — most cache queries are short, but verify.
2. **Batch feasibility → VRAM.** DeBERTa-v3-base (~184M params) at fp16, `max_length` from (1):
   estimate activation + optimizer memory to find the largest stable `batch_size`; use gradient
   accumulation if VRAM-bound.
3. **Step count.** `steps = ceil(n_train / effective_batch) * epochs`. With a target corpus of
   ~60–120k balanced pairs and 2–3 epochs, this is order 10³–10⁴ steps.
4. **Wall time → cost.** `wall = steps * sec_per_step`; `cost = wall * hourly_rate`. `sec_per_step`
   is measured on a short profiling run (still within the approved cap).

## Provider comparison (evaluate at execution time; prices drift)

Compared at M3, optimizing for **reliable availability + enough VRAM + reproducible environment
+ low total cost** — never "cheapest unknown machine":

| Provider | Typical GPUs | Notes |
|---|---|---|
| RunPod | RTX 4090, A10, L4, A100 | community + secure pods; cheap 4090s |
| Vast.ai | 4090, 3090, A100 | cheapest, most variable reliability |
| Lambda | A10, A100, H100 | stable, slightly higher rate |
| Modal | serverless GPU (A10/L4/A100) | pay-per-second; clean reproducible envs |
| Colab | T4/L4/A100 (subscription) | fine for a *small* controlled run |

A base-size cross-encoder on ~100k short pairs for 2–3 epochs is a **single-GPU, well-under-an-
hour** job on a 4090/A10/L4 — the tentative ceiling below reflects that.

## Historical planning ceilings (not a new authorization)

- **GPU training (M3):** ~A$5–15 total.
- **LLM augmentation (M5, optional):** ~A$2–5.

## Approval protocol (mandatory before any paid action)

Before spending, return this packet and **wait for explicit owner approval**:

- provider + exact hardware (or model, for LLM calls)
- estimated duration
- estimated cost **and** the maximum authorized cap
- exact purpose (which experiment id)
- the free alternative considered

Then: report burn-rate at checkpoints, and stop-loss before the cap. No unattended cost
escalation; no paid job running silently across sessions.

## Requirements for any future latency/cost comparison (not performed)

No controlled xp-006-versus-LLM-judge cost or latency comparison exists. Any future comparative
claim would require batch-size-1 and warm latency (p50/p95) on CPU, Apple Silicon, and GPU;
throughput; model size; and estimated cost per 1,000 comparisons against a provenance-matched
judge measurement. Until then, no cheaper/faster claim is supported.
