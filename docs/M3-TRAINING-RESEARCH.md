# M3 TRAINING RESEARCH — evidence for every consequential choice

A focused primary-source pass before freezing the recipe (brief §2). No cargo-culted defaults:
each row is **choice / alternatives / evidence / reason**. The frozen values live in
`configs/train/m3-deberta-v3-base.yaml`; this file is why they are what they are.

Anchor sources: DeBERTaV3 (He et al., ICLR 2023, arXiv:2111.09543) fine-tuning appendix **Table 15**;
the HF model card + `config.json` for `microsoft/deberta-v3-base`; HF Transformers docs; Guo et al.
2017 (temperature scaling, arXiv:1706.04599); scikit-learn calibration guide; PyTorch reproducibility
notes.

## Model provenance (verified)

| Field | Value | Source |
|---|---|---|
| Repo | `microsoft/deberta-v3-base` | HF model card |
| Revision (pinned) | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` (main SHA, HF API 2026-08-21) | HF API |
| License | **MIT** | model card + API |
| Params | ~**184M** = 86M backbone + 98M embedding (128,100 vocab × 768; 12L/768H) | model card + config |
| `trust_remote_code` | **not required** — `model_type: deberta-v2` is a native `DebertaV2ForSequenceClassification` | config.json |
| Weights format | **`pytorch_model.bin` only on main — NO safetensors** | HF file list |

**Provenance note (deviation from "prefer safetensors"):** the pinned official revision ships only
`pytorch_model.bin`. We load it from Microsoft's official pinned SHA (trusted, immutable) with
`use_safetensors=False`, and **save our own checkpoints as safetensors** (`safe_serialization=True`).
This keeps the base load reproducible while our artifacts stay pickle-free (docs/THREAT-MODEL.md).
The CPU smoke confirmed the base loads and our checkpoint writes `model.safetensors`.

## Recipe choices

**1. Learning rate — 2e-5.** Alternatives: 1.5e-5 / 2.5e-5 / 3e-5 (Table 15 grid), 5e-5 (rejected).
Evidence: Table 15's base grid is `{1.5e-5, 2e-5, 2.5e-5, 3e-5}`; the model card's MNLI example uses
2e-5; HF maintainer (nbroad) diagnosed a DeBERTa-v3 "random predictions" failure as pure LR
sensitivity, fixed by lowering LR. Reason: 2e-5 is the low-center of the paper's own grid and the
card's value — the most conservative defensible single point for a no-sweep run.

**2. Optimizer — AdamW, β=(0.9, 0.999), eps=1e-6, wd=0.01** (no decay on bias/LayerNorm). Alt:
torch's default eps=1e-8 (rejected — contradicts the paper). Evidence: Table 15 states Adam
β₁=0.9/β₂=0.999/**eps=1e-6**/weight-decay 0.01. Reason: the paper's exact values; **eps=1e-6 is the
one override from library defaults** and matters for this precision-sensitive model.

**3. Scheduler — linear decay, ~6% warmup (~95 steps).** Alt: cosine (common, marginally better but
off-paper); no warmup (risky). Evidence: Table 15 = "Learning Rate Decay: Linear", warmup grid
`{50,100,500,1000}`; at ~398 updates/epoch × 4 epochs ≈ 1,592 updates, 6% ≈ 95 ≈ grid point. Reason:
faithful to the paper; ~6% is the BERT-family convention and lands on a paper grid value.

**4. Effective batch 32, 4 epochs, early-stop patience 3.** Alt: batch {16,48,64}, up to 10 epochs
(Table 15). Evidence: Table 15 base batch grid `{16,32,48,64}`, "max epochs 10, pick best"; BERT
App. A.3 uses batch {16,32}, 2–4 epochs. Reason: 32 balances gradient noise/step-count for ~12.7k
pairs; 4 epochs + early stopping caps overfitting within the single-recipe constraint. We reach eff-32
via **batch 16 × grad-accum 2** (memory-safe on a 24 GB card; mathematically identical, DeBERTa has no
batch-norm).

**5. Max sequence length — 128 (measured, not assumed).** Evidence: `polarity train tokstats` on the
committed manifests with the pinned tokenizer — train joint-pair tokens p50=37, p95=67, **p99=77**,
max=188; validation p99=74, max=101. Truncation at 128 = **0.016%** of train (2 pairs), **0%** of
validation. Truncation strategy = `longest_first` (symmetric task — neither query privileged; HF
truncation docs). Reason: 128 covers p99 with wide margin at negligible truncation; smaller wastes no
meaningful evidence and O(n²) attention makes a tight cap a real speed win. `max_position_embeddings`
is 512 (hard ceiling).

**6. Precision — bf16 (GPU) / fp32 (CPU smoke). fp16 DISALLOWED.** Evidence: DeBERTa's disentangled
attention overflows fp16 (documented, reproduced HF-forum case: fp16 → NaN loss → constant logits →
F1 0; bf16 fixes it via fp32-range exponent, no loss scaling); HF mixed-precision docs prefer bf16.
Contradiction flagged: one 2026 arXiv preprint claims bf16 *underflow* and demands fp32 — weighted
lower (single contrarian vs a reproduced overflow mechanism). Reason: bf16 is the recommended stable
GPU choice; fp32 is the safe CPU fallback; the config **hard-rejects fp16** to remove the footgun.

**7. Gradient clipping — 1.0.** Evidence: Table 15 "Gradient Clipping: 1.0". Reason: the paper's value.

**8. Tokenizer — SentencePiece Unigram, `use_fast=True`, needs `sentencepiece`+`protobuf`.** Evidence:
`vocab_size: 128100`; HF docs describe v2/v3 as SentencePiece/Unigram; the repo ships `spm.model`
(no `tokenizer.json`), so the fast tokenizer converts on load (needs protobuf); `sentencepiece` is
mandatory. Gotcha: **`type_vocab_size=0`** — DeBERTa-v3 has no segment embeddings, so `token_type_ids`
are ignored; pair separation rests on `[SEP]` + relative position. Reason: we pin transformers,
install `sentencepiece`+`protobuf`, and `tokenizer(a, b)` yields `[CLS] a [SEP] b [SEP]` — no
hand-inserted specials, no segment-id tricks.

**9. Class balance — plain cross-entropy, NO class weights.** Evidence: 0.46 positive rate is
effectively balanced; weighting near-balanced data mainly distorts the loss and calibration; the
supported lever for an asymmetric-cost, low-FPR target is **threshold selection on validation**, which
we do anyway. Reason: operational FP asymmetry is handled entirely at the FPR≤1% operating point, not
in the loss.

**10. Calibration — temperature scaling primary; Platt fallback; isotonic guarded.** Evidence: Guo et
al. 2017 — temperature scaling (1-param) minimizes NLL on held-out data and is **monotonic**, so it
does **not** move the ROC/operating point (calibration and threshold selection are orthogonal);
scikit-learn/Niculescu-Mizil & Caruana — isotonic overfits below ~1000 samples. Reason: fit all
candidates on **validation only**, pick the lowest validation NLL, freeze, apply to internal_test;
temperature is the low-variance default at our validation size (~1.6k), isotonic is guarded by
`isotonic_min_obs`.

**11. Reproducibility.** Knobs: `transformers.set_seed` + `torch.manual_seed`,
`torch.use_deterministic_algorithms(True, warn_only=True)`, cudnn deterministic/benchmark-off (GPU),
`PYTHONHASHSEED` (env, before start), `CUBLAS_WORKSPACE_CONFIG=:4096:8` (GPU). Reality (PyTorch docs):
CPU run-to-run is realistically bit-reproducible on a fixed build; GPU determinism is best-effort on
fixed hardware + library stack and not portable across hardware/versions. Reason: we pin seed 42 +
software/hardware and record the real env in the run dir; the fingerprint pins the recipe.

**12. Checkpoint selection — validation operational metric, early stop.** Evidence: select by the
metric you report (recall@FPR≤1%), not loss/accuracy, which can improve while low-FPR recall degrades;
caveat: the FPR≤1% threshold is pinned by only ~top-1% of validation negatives (~8–9 of 863), so it is
noisy. Reason: our frozen lexicographic rule (satisfy FPR≤1%, then max recall, tie→earliest step) plus
a **pAUC over FPR∈[0,2%] stability cross-check** addresses the thin-operating-point noise.
