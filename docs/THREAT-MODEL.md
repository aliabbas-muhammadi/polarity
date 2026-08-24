# THREAT-MODEL — supply chain, execution, secrets

Polarity pulls from model hubs and dataset sources. Model repositories, dataset loader scripts,
pickle files, and arbitrary Python dataset loaders are treated as **potentially untrusted**. The
current release posture is source-only; checkpoint weights are withheld.

## Data & model supply chain

- **Pinned dataset bytes + committed checksums.** Every declared external dataset download has an
  expected raw-byte SHA-256 committed alongside its revision in `src/polarity/data/sources.py`.
  `polarity data fetch` verifies those bytes before writing, converting, or extracting them and
  refuses a mismatch. The generated `PROVENANCE.json` separately records hashes of the resulting
  local files; it is an audit record, not the source of the expected download checksum. No
  "latest".
- **No arbitrary code execution.** No `trust_remote_code=True` unless a specific model requires
  it and it has been reviewed and justified in writing. No running of downloaded dataset build
  scripts that execute arbitrary Python; prefer declarative loaders and known formats.
- **safetensors only** for weights (never pickled `.pt`/`.bin` from untrusted sources).
- **Format discipline.** Prefer well-known dataset formats (JSONL/CSV/parquet) with schema
  validation on load (pydantic models); reject malformed rows loudly.

## Execution

- M0 code runs on committed synthetic fixtures + the sealed snapshot only — **no network in
  tests**, no downloads, no paid calls. CI is keyless.
- Later data builders (M1) run offline after the fetch step has matched the committed expected
  download checksums; the networked fetch step is separated from the build step.

## Secrets

- **No credentials in the repo, ever.** `.gitignore` excludes `.env*` (keeping only
  `.env.example`); CI runs a **gitleaks** secret-scan job on every push.
- No API keys are required to reproduce any committed result — that is the point of committing
  prediction artifacts (`predictions/`, `reports/`) and gating on them.

## Artifact and source-only release

- Weight globs (`*.pt *.ckpt *.safetensors *.bin`) and `artifacts/` are git-ignored so a training
  run cannot accidentally commit a checkpoint.
- The complete private seven-file model/tokenizer bundle is integrity-covered by a committed
  manifest. `polarity artifact verify` hashes it before model loading; the manifest is not a grant
  to redistribute the files.
- Raw third-party data, ignored runs, and the MRPC gold-label vector are excluded. Public MRPC rows
  are score-only and full reproduction requires a checksum-matched local source.
- A public candidate must be built in a disposable clone so prior private-history objects cannot
  leak. Full-history secret/artifact searches are release gates; a clean current tree alone is not
  sufficient.
- Weight licensing remains unresolved. No checkpoint upload or public visibility change is part of
  this release-hardening work.

## Deployment safety

Polarity scores semantic similarity; it does not authorize cache reuse safely on its own. The
validation-selected `tau=0.994833` realized 4.55% FPR on Veritas and 57.44% on MRPC, demonstrating
that a validation budget is not a deployment-domain guarantee. Any consuming system must use a
separate policy layer, monitor false accepts, account for per-user/tool/retrieval context, and fail
closed for sensitive content. Security-, medical-, legal-, or financial-answer caches are outside
the evidence ceiling.

## Out of scope

Adversarial robustness of the deployed classifier (prompt injection into cache keys, Unicode or
formatting attacks, targeted score manipulation, and distribution drift) is not established. The
repository also does not solve cache invalidation, authorization boundaries, tenant isolation, or
temporal/tool-state correctness.
