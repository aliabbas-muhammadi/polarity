# Attributions and third-party notices

Polarity's own code and documentation are MIT. That license does not relicense the materials below
or the private fine-tuned checkpoint. Raw third-party datasets and model files are not included in
the source-only release. This inventory records provenance and a conservative release posture; it
is not legal advice.

## Artifact scope

The repository contains several artifact classes with different rights and provenance. The
top-level MIT license is not a blanket relicense of every tracked file.

| artifact class | shipped? | release/licensing scope |
|---|---|---|
| Original Polarity source code and original documentation | yes | MIT under `LICENSE`; quotations, citations, and third-party marks retain their own status |
| `manifests/**` | yes | IDs, hashes, labels, categories, and split metadata derived from named datasets; no source sentences; applicable upstream copyright/database terms prevail where they attach |
| Third-party-derived `predictions/**` | yes | Polarity-generated scores and stable row identities, without third-party sentence text; applicable upstream terms prevail where they attach |
| `reports/**` and report prose | yes | Original analysis and aggregate metrics are released with the repository, but MIT does not override rights in cited or source-derived material |
| Author-created Veritas snapshot and diagnostic probe | yes | Author-owned pair text/labels under their committed provenance; no embedding or judge outputs are included |
| Raw third-party dataset text | no | Must be obtained independently under the source's terms |
| Microsoft base-model/tokenizer files | no | Governed by Microsoft's published terms if independently obtained; not relicensed by Polarity |
| Fine-tuned Polarity model/tokenizer bundle | no | Private; no public checkpoint license is assigned |

Whether a hash, label-only manifest, score row, or aggregate engages copyright or database rights
depends on the artifact and jurisdiction. This table records the conservative release boundary; it
does not make that legal determination.

## Base model and tokenizer

- **Microsoft DeBERTa-v3-base** — Microsoft, `microsoft/deberta-v3-base`, pinned model and
  tokenizer revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. The
  [model card](https://huggingface.co/microsoft/deberta-v3-base) marks the base model MIT; the
  [Microsoft DeBERTa repository license](https://github.com/microsoft/DeBERTa/blob/master/LICENSE)
  contains the MIT notice. Preserve that notice when using Microsoft's material. This attribution
  does not assign a license to Polarity's fine-tuned parameters.

  The [DeBERTa-v3 paper](https://arxiv.org/abs/2111.09543) reports that its main models use the same
  roughly 160 GB pretraining mixture as RoBERTa; the
  [RoBERTa paper](https://arxiv.org/abs/1907.11692) enumerates English Wikipedia, BookCorpus,
  CC-News, OpenWebText, and Stories. DeBERTa-v3 also reuses DeBERTa-v2's 128,000-token
  SentencePiece vocabulary. That is base-model lineage, not Polarity fine-tuning data; Polarity
  did not assemble, audit, or redistribute those corpora. The Apache-2.0 license for SentencePiece
  software does not independently assign that license to Microsoft's trained `spm.model` artifact.

## Training and evaluation data

- **PAWS-Wiki (`labeled_final`)** — Zhang, Baldridge, and He, *PAWS: Paraphrase Adversaries from
  Word Scrambling*, NAACL 2019; Google Research. See the
  [PAWS repository](https://github.com/google-research-datasets/paws) and
  [PAWS license/notice](https://github.com/google-research-datasets/paws/blob/master/LICENSE).
  PAWS-Wiki sentences are derived from Wikipedia; applicable
  [Wikimedia licensing and attribution terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
  continue to govern that text.

- **CANNOT v1.1** — Anschütz, Lozano, and Groh, *This is not correct! Negation-aware Evaluation of
  Language Generation Systems* (2023). The
  [official repository](https://github.com/dmlls/cannot-dataset) and
  [dataset card](https://huggingface.co/datasets/tum-nlp/cannot-dataset) declare CC BY-SA 4.0.
  CANNOT documents five upstream components, which retain their own authorship and any applicable
  terms:

  - [NaN-NLI](https://github.com/joey234/nan-nli) and the
    [author dataset card](https://huggingface.co/datasets/joey234/nan-nli) — Truong et al.,
    *Not another Negation Benchmark*;
  - [GLUE Diagnostic](https://gluebenchmark.com/diagnostics) — Wang et al. and GLUE contributors;
    the [official GLUE FAQ](https://gluebenchmark.com/faq/) delegates task licensing to original
    sources, and this audit did not identify an explicit redistribution grant for the Diagnostic
    set itself;
  - the original [WikiFactCheck-English](https://github.com/aalok-sathe/wikifactcheck-english)
    repository — Sathe et al., *Automated Fact-Checking of Claims from Wikipedia* — has no
    standalone license file; a maintained
    [organizational fork](https://github.com/wikifactcheck-english/wikifactcheck-english) points its
    [license notice](https://github.com/wikifactcheck-english/wikifactcheck-english/blob/main/LICENSE)
    to applicable Wikipedia terms;
  - [Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
    — Kotzias et al., *From Group to Individual Labels Using Deep Features*, distributed by UCI
    under CC BY 4.0; the source sentences originated in IMDb, Amazon, and Yelp reviews; and
  - [SemAntoNeg](https://github.com/Helsinki-NLP/SemAntoNeg), declared CC BY 4.0 and derived through
    [Opusparcus](https://www.kielipankki.fi/corpora/opusparcus/) from
    [OpenSubtitles](https://opus.nlpl.eu/datasets/OpenSubtitles) — Truong et al., *It Is Not Easy To
    Detect Paraphrases*.

  CANNOT also documents generated/rule-processed material involving
  [Pegasus paraphrase](https://huggingface.co/tuner007/pegasus_paraphrase) and the
  [negate tool](https://github.com/dmlls/negate). Their model/software licenses do not by themselves
  establish a complete rights chain for generated or source dataset content.

  The aggregate CC BY-SA declaration does not by itself resolve every upstream right for an
  unrestricted checkpoint release. Polarity therefore withholds its learned parameters and makes
  no assertion that the upstream chain permits checkpoint redistribution.

- **MultiNLI (MNLI)** — Williams, Nangia, and Bowman, NAACL 2018. Multi-source terms include OANC,
  CC BY 3.0, CC BY-SA 3.0, and US public-domain material. See the
  [dataset card](https://huggingface.co/datasets/nyu-mll/multi_nli). MNLI was excluded from the
  final primary training recipe after the M1 comparison.

- **NaN-NLI** — Truong et al., AACL-IJCNLP 2022. The
  [author dataset card](https://huggingface.co/datasets/joey234/nan-nli) states CC BY-SA 4.0.
  Polarity used it only as an evaluation/leakage probe, and detected overlap with CANNOT.

- **Microsoft Research Paraphrase Corpus (MRPC)** — Dolan and Brockett, Microsoft Research. The
  [official Microsoft download](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
  provides an installer whose embedded Microsoft Research Shared Source license permits
  non-commercial use, copying, reproduction, and redistribution under its conditions, while
  prohibiting commercial use or distribution. The
  [GLUE downloader](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py)
  separately states that GLUE cannot host MRPC for legal reasons. No commercial redistribution
  grant was identified. Polarity distributes neither MRPC text nor its complete gold-label vector;
  the public score rows must be joined to a checksum-matched, independently obtained local source
  for full metric reproduction.

## Learned-parameter posture

Creative Commons' [AI-training legal primer](https://creativecommons.org/2025/05/15/understanding-cc-licenses-and-ai-training-a-legal-primer/)
does not settle whether ordinary learned parameters are adaptations of training material. Polarity
does not infer from that uncertainty that its checkpoint is either freely redistributable or
necessarily subject to one particular dataset license. The checkpoint is simply withheld pending
written clarification/legal review or provenance-clean retraining.
