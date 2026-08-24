# M5 — Training Signal Audit: diagnostic plan (pre-registered hypotheses)

**Question.** Why does a 184M DeBERTa-v3-base sequence classifier converge to a constant
(input-invariant) predictor across M3/M3.1/M3.2 despite stable gradients? This is a
debugging/falsification milestone, **$0, local CPU/MPS only** — not a tuning exercise.

> **Release-hardening note:** the repository does not include machine-readable logs for the
> decisive diagnostic runs summarized in the resolution column. Committed full-run predictions
> support version-sensitive collapse/success, but the causal controls below cannot be independently
> replayed from committed evidence alone.

**Method.** The load-bearing test is **overfit-one-batch**: a correctly wired classifier memorizes
a tiny labelled set; a broken pipeline cannot. A single well-chosen control (a *different*
architecture through the *identical* pipeline) eliminates most hypotheses at once — if the label,
encoding, optimizer, loss, score-extraction, checkpoint, or sampler code were broken, the control
would fail too.

## Hypotheses

| # | Hypothesis | Test | How it was resolved |
|---|---|---|---|
| H1 | labels inverted/malformed | print enum→int; `label_index` | **ELIMINATED** — `CACHE_EQUIVALENT→1`, `NOT→0`, verified; unit-tested |
| H2 | train/eval label mapping differ | same `label_index` on both paths | **ELIMINATED** — single shared convention (`predict.POSITIVE_INDEX`) |
| H3 | second sequence lost in encoding | inspect `input_ids` | **ELIMINATED** — both sequences present; rows distinct 16/16 |
| H4 | tokenizer doesn't distinguish examples | distinct-encoding invariant | **ELIMINATED** — 16/16 distinct input_ids |
| H5 | attention/token_type handling wrong | encode inspection | **ELIMINATED** — deberta ignores token_type (type_vocab_size=0); mask sane |
| H6 | classification head not updating | Δhead norm after step | **ELIMINATED** — Δhead 3.4 (moves a lot) |
| H7 | backbone unintentionally frozen | trainable-param audit | **ELIMINATED** — Δbackbone-layer0 0.41 (backbone trains) |
| H8 | optimizer excludes parameters | optimizer coverage | **ELIMINATED** — all params in one AdamW group; head included |
| H9 | gradients exist but updates ≈0 | param-update proof | **ELIMINATED** — updates are large, not zero |
| H10 | logits/probability extraction wrong | logits vs softmax vs prediction | **ELIMINATED** — control separates cleanly through the same path |
| H11 | loss wired to wrong tensor/labels | `model(**enc, labels=…)` | **ELIMINATED** — control drives CE→0 through the same wiring |
| H12 | scheduler/warmup suppresses learning | with/without warmup | **ELIMINATED** — fails with and without warmup |
| H13 | learning rate pathological | LR sweep 5e-6/1e-5/2e-5 | **ELIMINATED as sole cause** — collapses at every LR, incl. stable low-LR |
| H14 | batching/sampler corrupts examples | control uses same batching | **ELIMINATED** — control memorizes the same batch |
| H15 | data too noisy to learn | trivial synthetic signal | **ELIMINATED** — cannot even memorize A/A-vs-A/Z |
| H16 | checkpoint load resets state | roundtrip | **N/A** — collapse is pre-checkpoint (in-memory) |
| H17 | eval loads wrong checkpoint | checkpoint identity | **N/A** — collapse visible in the live model |
| H18 | prediction score transform wrong | score-extraction audit | **ELIMINATED** — control's scores are correct through the same transform |
| H19 | truncation destroys signal | trivial pairs are ~11 tokens | **ELIMINATED** — no truncation on the trivial batch, still collapses |
| **H20** | **DeBERTa-specific implementation/library issue** | **model-swap + transformers-version swap** | **AUTHOR DIAGNOSIS** — see evidence limit above |

## The decisive experiments

1. **Model-swap control.** `distilbert-base-uncased` through the *identical* encode/label/loss/score
   pipeline **memorizes 16 trivial pairs in <30 steps** (acc 1.0, loss 0.001). This single result
   eliminates H1–H14 and H18 — the pipeline is healthy.
2. **transformers-version swap** (only variable changed). Same deberta-v3-base @ the same pinned
   revision: **collapses under transformers 5.15.1, learns under 4.46.3 / 4.57.6** (acc 1.0). The
   pooled pre-classifier representation has std ~0.004 across inputs under 5.x (constant) vs ~0.42
   under 4.x (discriminative).

**Author diagnosis = H20:** a Transformers-5.x-sensitive DeBERTa-v2 fine-tuning failure. The
recorded controls motivate the pinned release profile, while the missing committed logs and absent
upstream bisection prevent an independently reproduced library-defect claim. See
`docs/M5-REPORT.md`.
