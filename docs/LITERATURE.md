# LITERATURE — prior work, and what remains defensible

> Scope: work bearing on Polarity, a sentence-pair cross-encoder (candidate
> `microsoft/deberta-v3-base`, ~184M) trained for one job — **semantic-cache equivalence**:
> given `(query_a, query_b)`, predict whether it is safe to reuse the same cached answer.
> Motivating failure: a *frozen* embedding cosine (OpenAI `text-embedding-3-small`) ranks some
> negation/meaning-changing variants **above** genuine paraphrases, so no single cosine
> threshold cleanly separates equivalence from meaning change; an LLM judge can mitigate this on a
> given set but is not a general fix and costs a per-call round trip. Hypothesis under test (not a
> conclusion): a cache-equivalence-specific cross-encoder closes more of that gap than a frozen
> threshold while avoiding a required paid judge call on each lookup. No controlled
> xp-006-versus-judge cost or latency advantage is claimed.
>
> Confidence tags: **[high]** = 2+ independent or a directly-fetched primary source; **[med]**
> = single fetched source or corroborated summaries; **[low]** = search-snippet only.

## TLDR

- The "learned / adaptive semantic cache" space is **already occupied**. **vCache** (ICLR 2026)
  learns a *per-prompt* reliability threshold **on a frozen embedding**; **Redis LangCache** is
  a shipping product (public preview Sept 2025); and a June 2026 Redis/NYU paper **already
  trains cross-encoders for cache equivalence** on PAWS/QQP. So Polarity **cannot** claim to be
  the first learned semantic cache, nor the first cache-equivalence cross-encoder. **[high]**
- The one axis those systems do **not** isolate is **negation / polarity / meaning-change** as
  an explicit target and evaluation. That is Polarity's only defensible differentiator — narrow
  (a focused eval + training slice), not a category invention. **[high]**
- The motivating failure is real and well-documented: modern encoders assign high cosine to
  opposite-meaning pairs (e.g. MPNet 0.81 for "Global warming is a hoax" vs "…is real"), and on
  NevIR most retrieval models score at or below random; **cross-encoders are the best
  architecture class on negation**, supporting Polarity's model choice. **[high]**
- Two hard constraints: (1) a per-prompt learned threshold on a frozen embedding (vCache) is a
  stronger baseline than a static cutoff — Polarity must beat *that*. (2) Cross-encoders trained
  with BCE/MNRL can be badly **miscalibrated at the deployment threshold** despite excellent
  AUC — so any win must be shown at a fixed low false-positive operating point, not via AUC/F1.
  **[high]**
- Dataset labels do **not** map cleanly to "cache equivalence." QQP "duplicate" is the closest
  proxy; PAWS targets word-order (not negation); NLI labels (MNLI/ANLI) are directional and
  3-way and must **not** be equated with cache (non-)equivalence. **[high]**

## 1. Semantic caching / learned & adaptive thresholds

**vCache — Verified Semantic Prompt Caching** *(closest incumbent)*. Schroeder et al.,
arXiv:2502.03771, **ICLR 2026**. An online algorithm estimates an **optimal similarity
threshold per cached prompt** — on a **frozen embedding**, no fine-tuning. Evaluated on three
introduced benchmarks (SemCacheClassification/LMArena/SearchQueries) with GTE-Large and
E5-large. **Relation:** the paper Polarity must position against most carefully — it recovers
much of the "one-threshold-fits-all is wrong" loss *without training a model*. **Limitation:**
for long/complex outputs it still falls back to an **LLM judge** (the historical paid-call
dependency that motivated Polarity; no controlled cost comparison was run),
and it **never discusses negation/antonymy** (confirmed by full-text fetch). **Does it change
the contribution? Yes, substantially** — it pre-empts "first adaptive semantic cache" and is
the correct strong baseline. **But** a scalar per-prompt threshold on a frozen geometry cannot
separate a negation that sits *closer* than the true paraphrase: raising the threshold to
exclude the negation also excludes the paraphrase. That non-separable sub-population is exactly
what a pair-aware cross-encoder can attack. Honest framing: *complements* vCache on the
negation slice, does not *replace* thresholds. **[high]**

**GPTCache** (Bang, NLP-OSS @ EMNLP 2023) — embedding + **single global threshold**; the
static-threshold straw man. A fair but *weak* baseline; beating it alone is not defensible given
vCache. **[high / vendor-claim on speedups]**

**Closing the Calibration Gap in Semantic Caching** *(most dangerous prior work for novelty)*.
Baral et al. (Redis) + Rajamohan, Agarwal (NYU), arXiv:2606.19719, **Jun 2026**. They **train
domain-specific cross-encoder rerankers specifically to judge cache equivalence** (BCE + MNRL)
on a large cache-pairs corpus built from **PAWS, QQP, PIT2015, PARADE, STS-B, SICK, MRPC** +
augmentation. Key finding: standard training optimizes **ranking** (PR-AUC) but yields
**miscalibrated absolute scores** that fail at threshold deployment — a BCE reranker with
PR-AUC 0.816 collapses to a deployment "P-CHR AUC" of 0.199; temperature/Platt scaling only
partially recovers it. **Does it change the contribution? Yes, twice:** (1) it **pre-empts
"first / novel cache-equivalence cross-encoder"**; (2) it is a **calibration warning** — Polarity
*must* report a calibrated fixed-FP operating point or its advantage may evaporate in
deployment. What remains open: it does **not isolate negation/polarity**. **[high]**

**Redis LangCache** — shipping product (public preview Sept 2025) with purpose-built cache
embeddings marketed to cut incorrect hits. Confirms the category is **commercialized**;
reinforces "no category novelty; only the negation slice is open." **[med / vendor]**

Context (not competitors): "From Exact Hits to Close Enough" (Biton & Friedman,
arXiv:2603.03301) — cache *management*/eviction, no FP metric; OLAP semantic caching via LLM
canonicalization (Bindschaedler, arXiv:2602.19811) — a per-call path outside Polarity's evaluated
design (the widely-quoted ">0.95 cosine for revenue-growth-vs-decline" example could **not** be
confirmed in the PDF — treat as unverified). **[low–med]**

## 2. Negation in retrieval / embeddings / NLI

**NevIR** (Weller, Lawrie, Van Durme, EACL 2024, arXiv:2305.07614) — rank two documents
differing only by a negation. **Cross-encoders best**, then late-interaction, then bi-encoders;
**most models at or below random.** Direct support for the failure mode *and* the architecture
choice. Caps the ceiling: even the best class is far from solved → "solves negation" is off the
table. **[high]**

**Reproducing NevIR** (van den Elsen et al., SIGIR 2025 Reproducibility, arXiv:2502.13506) —
replicates the result on newer models and finds **negation robustness does not transfer** across
negation datasets. Warning: a model tuned on one negation slice may not generalize; report
per-slice. **[high]**

**Learning Robust Negation Text Representations** (Truong et al., arXiv:2507.12782) — the
canonical datapoint: MPNet assigns **0.81 cosine** to "Global warming is a hoax" vs "…is real";
baseline MPNet scores **8.10%** on NevIR (chance ≈ 25%). Dedicated fine-tuning lifts NevIR to
40.56% but **plateaus on ExcluIR (73%)** and induces side effects (degraded GSM8K when pushed
into an LLM). **Caps the honest claim** to "reduces the gap on a defined negation slice." **[high]**

**On the Theoretical Limitations of Embedding-Based Retrieval** (Weller et al.,
arXiv:2508.21038) — a fixed-dim embedding cannot represent all top-k relevance combinations
(sign-rank lower bound). A *theoretical* backbone for "no single threshold on a frozen embedding
separates everything," but the mapping to cache-vs-negation is an analogy, not a theorem — cite
as motivation, not proof. A cross-encoder sidesteps the bottleneck by jointly attending over
the pair. **[high / med on the analogy]**

General: text encoders are negation-insensitive by construction; robust systems layer keyword/
rule checks or specialized classifiers on top — the known, *recommended* niche Polarity occupies
(so Polarity executes a known pattern, it does not invent one). **[high]**

## 3. Pair classification / cross-encoders

**Cross-encoder vs bi-encoder** (Sentence-Transformers lineage). A cross-encoder feeds the pair
jointly through the transformer → higher accuracy, no precomputation, does not scale to
large-corpus search. Standard practice: bi-encoder retrieve → cross-encoder re-rank. Matches
Polarity's positioning exactly: the cross-encoder is a **second-stage verifier** over the small
candidate set a cache lookup returns, so its scalability downside is largely irrelevant while
its accuracy/negation upside is the point. **[high]**

**DeBERTa-v3-base** (He, Gao, Chen, arXiv:2111.09543) — **184M params (86M backbone + 98M
embedding, 128K vocab)**; disentangled attention + ELECTRA-style pretraining; strong pair
backbone (**MNLI-m/mm 90.6/90.7**). A well-justified choice; note most of the 184M is the
large-vocab embedding table, not compute-heavy backbone. **[high]**

**Calibration of pair classifiers** — the calibration-gap caching paper (§1) is the most
on-point evidence; pretrained transformers are generally overconfident and temperature scaling
degrades under distribution shift (Desai & Durrett 2020). Calibration is **not optional** for
Polarity. **[high / low on the general-calibration cites]**

## 4. Datasets — facts and cache-equivalence mapping risk

> **Central caveat:** "cache equivalence" = *would the same generated answer be acceptable for
> both queries?* — not identical to paraphrase, and **definitely not** identical to NLI labels.

| Dataset | What it is | Label | Maps to cache-equivalence? |
|---|---|---|---|
| **PAWS** (NAACL 2019) | 108,463 high-overlap pairs via word-swap + back-translation | paraphrase / not | **Partial** — great *hard negatives* (looks similar, different answer); does **not** cover negation. **[high]** |
| **QQP** (~400k) | duplicate-question detection | duplicate / not | **Closest proxy** ("same answer serves both"); labels **noisy** → weak supervision. **[med]** |
| **MNLI** (2018) | multi-genre NLI, 433k | entail/neutral/contradiction | **Do NOT map directly** — entailment is directional; contradiction ≠ cache-non-equivalent. Backbone pretraining only, with a documented re-labeling. **[high]** |
| **ANLI** (ACL 2020) | adversarial NLI | 3-way | same mapping risk; hard adversarial pairs, not equivalence labels. **[high]** |
| **CANNOT** (arXiv:2307.13989) | 77,376 negation pairs, 5 merged sources | non-negated / negated | **Best fit for the negation slice**; synthetic/rule-augmented, so distribution caveat. **[high]** |
| **NaN-NLI** (AACL 2022) | small sub-clausal negation-scope test suite | NLI-style | **Eval probe, not training data** — good hard held-out negation-scope check. **[high]** |

**Safest recipe:** QQP ± PAWS for the general "same-answer / different-answer despite overlap"
signal; **CANNOT (train) + NaN-NLI / NevIR (eval)** for the negation differentiator; **never**
treat NLI contradiction/entailment as cache (non-)equivalence. (Licensing gates which of these
may actually be used — see `docs/DATA.md`. QQP/PAWS-QQP are currently `LICENSE_UNRESOLVED`.)

## 5. Contribution defensibility

**Honest claims Polarity can make:** (1) a negation/polarity-focused **evaluation + training
slice** for cache equivalence — no fetched prior work isolates negation as the target of a
cache-equivalence model; (2) an operating-point study comparing the cross-encoder against a static
threshold and a **vCache-style adaptive threshold** at validation-selected false-positive budgets,
with the LLM judge retained only as historical motivation. Narrow but real.

**Claims to avoid** (each refuted by cited prior art): ❌ "first cache-equivalence cross-encoder"
(Redis/NYU, Jun 2026); ❌ "first/learned semantic cache" or "replaces static thresholds" (vCache;
LangCache); ❌ "solves negation" (dedicated efforts plateau, robustness doesn't transfer); ❌
"beats the LLM judge" on accuracy, cost, or latency (no controlled comparison was run); ⚠️ any
headline on AUC/F1 without a calibrated operating point.

**Net:** replication-plus-specialization in a now-commercialized space; defensible novelty is a
**negation-isolated benchmark + a calibrated comparison** against the *right* reproducible
baselines, with historical judge evidence kept separate.

## Open contradictions / gaps

- **vCache headline numbers**: fetched v4 says "~2× hit / ~6× lower error"; another summary
  quotes "12.5× / 26×". Verify against the ICLR-2026 camera-ready before quoting. **[med]**
- **No negation-focused cache-equivalence cross-encoder** was found — an absence of evidence
  (industrial/unindexed code cannot be ruled out), which is Polarity's opening. **[med]**
- QQP/MNLI/ANLI primary papers were not each independently full-text fetched (facts are
  well-established + corroborated). **[med]**

## Sources (fetched)

vCache arXiv:2502.03771 (+ html v4) · GPTCache aclanthology 2023.nlposs-1.24 · Calibration-gap
arXiv:2606.19719 · "Close Enough" arXiv:2603.03301 · NevIR arXiv:2305.07614 · NevIR reproduction
arXiv:2502.13506 · Robust negation arXiv:2507.12782 · Embedding limits arXiv:2508.21038 · CANNOT
github.com/dmlls/cannot-dataset · DeBERTa-v3 hf microsoft/deberta-v3-base · PAWS arXiv:1904.01130
· OLAP caching arXiv:2602.19811 · Redis LangCache blog (Sept 2025) · NaN-NLI arXiv:2210.03256 ·
Sentence-Transformers cross-encoder docs.
