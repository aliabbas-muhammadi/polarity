# Source-only release checklist

This checklist prepares a **local disposable public-release candidate only**. It does not authorize
remote creation, a push, model upload, public visibility, Portfolio changes, paid compute, or any
other external mutation.

The private Polarity repository and its original audit history must remain intact and local. Never
rewrite or publish its refs.

## 1. Preconditions

- [ ] The release-hardening feature branch has completed scoped and whole-branch review.
- [ ] All source-free gates pass on Python 3.12 and 3.13 from the reviewed commit.
- [ ] The owner has supplied and verified the privacy-safe GitHub author name/email to use for any
      public commit. Do not guess a `users.noreply.github.com` address.
- [ ] No remote or public repository has been created for this work.
- [ ] The original private repository path, commit IDs, and 19-commit audit history are recorded and
      remain recoverable.

## 2. Build a disposable candidate

Create the candidate under a newly allocated temporary directory, never in the private repository
or its linked worktree. The safest default is a fresh-root repository populated from the reviewed
tracked tree. This keeps no private Git object reachable and avoids accidentally publishing the
old MRPC label-bearing artifact or personal-email history.

If preserving selected history is later required, use a separately reviewed full-history rewrite
in another disposable clone. It must remove the old MRPC artifact from every reachable commit and
rewrite every exposed author/committer email to the owner-verified privacy-safe identity. Do not
copy private refs, reflogs, remotes, notes, or tags into either candidate.

- [ ] Candidate contains exactly the reviewed tracked source tree.
- [ ] Candidate has no inherited remote, private ref, reflog, worktree metadata, note, or tag.
- [ ] Candidate commits use only the verified privacy-safe author/committer identity.
- [ ] Original private repository is unchanged.

## 3. Source-only content gate

The candidate may contain code, documentation, author-owned fixtures/snapshot, manifests,
score-only predictions, and aggregate reports. It must not contain:

- model or tokenizer bundle files;
- `artifacts/`, `runs/`, `graphify-out/`, provider telemetry outside reviewed aggregate reports, or
  any ignored local output;
- raw third-party training/evaluation text;
- raw MRPC text or the complete MRPC gold-label vector;
- credentials, personal email, absolute local paths, private refs, or machine-specific files; or
- an assertion that the checkpoint carries MIT/Apache-2.0 terms, has unrestricted redistribution
  rights, or is available.

Run the full reachable-object search in the disposable candidate:

```sh
git rev-list --objects --all |
  rg -i 'data/raw|runs/|artifacts/|graphify-out|\.safetensors$|\.pt$|\.pth$|\.ckpt$|pytorch_model\.bin$'

git log --all -p -- predictions/m6.1 |
  rg '"gold_equivalent"|"pair_type"'
```

Both searches must have no match. A clean current tree is insufficient; every reachable object is
in scope.

Also inspect current tracked files:

```sh
git ls-files |
  rg -i 'data/raw|runs/|artifacts/|graphify-out|\.safetensors$|\.pt$|\.pth$|\.ckpt$|pytorch_model\.bin$'

git grep -n -E '"gold_equivalent"|"pair_type"' -- predictions/m6.1
git grep -n '/[U]sers/|/[h]ome/|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
```

Investigate every match; do not mechanically waive one.

- [ ] Complete and record a reviewed recursive inventory of tracked reports, fixtures, snapshots,
      and documentation for third-party text or text-pair fields. Filename/path scans are only a
      coarse first pass; review structured content and remove any raw third-party text, retaining
      only approved hashes, identifiers, and aggregate counts.

## 4. Secret and history scan

Use a pinned Gitleaks release/action and record its exact version and checksum in the release
evidence. Scan the complete candidate history, not only the working tree:

The tracked `.gitleaks.toml` extends Gitleaks' default rules. Its sole allowlist is a reviewed false-
positive exception for the exact public DeBERTa tokenizer revision on `tokenizer_revision` lines in
the four train configs and M6 model manifest. Any change to that revision, regex, target rule, path
set, condition, or any additional allowlist requires explicit review; broad hash or path exceptions
are forbidden.

```sh
gitleaks git . --redact --log-opts="--all"
```

- [ ] Full-history Gitleaks exits successfully with no unresolved finding.
- [ ] Author and committer identities expose no personal email.
- [ ] No ignored/private ref is reachable.
- [ ] `git fsck --full` reports no corrupt object.

## 5. Claim-integrity gate

Review `README.md`, `MODEL_CARD.md`, `docs/M6-REPORT.md`, both M6.1 reports, `NOTICE`, and
`ATTRIBUTIONS.md` together. Exact counts must remain visible:

- validation-selected `tau=0.994833`;
- Veritas 31/31 recall, 2/44 false accepts, 4.55% realized FPR, 0/22 negation false accepts;
- descriptive Veritas actual-FPR≤1% recall 24/31=0.774;
- M3.3 threshold-free internal recall 0.763 at actual FPR≤1%; at validation-selected threshold,
  recall 0.903 and realized FPR 2.38%; and
- MRPC ROC-AUC 0.639870, custom step-pAUC 0.031938, TP/FN/FP/TN 898/249/332/246, recall 0.782912,
  and realized FPR 0.574394.

The repository pAUC must be described as its normalized step metric, not sklearn standardized
partial AUC. Veritas must be called author-created, protocol-held-out target-domain evidence. The
probe must remain post-hoc, author-curated, partly Veritas-reusing, and non-causal. Spend must be
“approximately A$1.78, author-reported; no receipts.”

Run focused stale-claim searches and investigate any match:

```sh
git grep -n -E \
  'recall 1[.]0 @ FPR.?1%|STRONG_[O]OD_TRANSFER|overwhelmingl[y].*high-[o]verlap|A[$]1[.]34'

git grep -n -Ei \
  'weights [o]ptional|licensing already [v]erified|byte-for-byte.*(GPU|[r]etrain)|general [O]OD'
```

## 6. Reproduction and quality gates

Run the locked source-free block under both Python 3.12 and 3.13:

```sh
uv lock --check
uv sync --locked --python 3.12

uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest

uv run polarity evaluate --check
uv run polarity data verify
uv run polarity audit
uv run polarity baseline report --check
uv run polarity train fingerprint

for config_path in configs/train/*.yaml; do
  uv run polarity train report --check --config "$config_path"
done

uv run polarity transfer report --check
uv run polarity transfer probe-report --check
uv run polarity transfer mrpc-report --check

git diff --check
git status --short
```

Repeat after `uv sync --locked --python 3.13`. The source-free MRPC output must explicitly say that
metrics were not reproduced.

The following private-source checks may be run before candidate export, but their inputs must never
be copied into the candidate:

```sh
uv run polarity transfer mrpc-report --check \
  --source data/raw/mrpc/mrpc-test-golden.json

uv run polarity artifact verify \
  --manifest reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json \
  --artifact-dir artifacts/xp-006
```

## 7. External boundaries

Read-only checks may confirm that sibling projects and paid infrastructure remain untouched:

```sh
polarity_veritas_repo=/absolute/path/to/veritas
polarity_portfolio_repo=/absolute/path/to/Portfolio
git -C "$polarity_veritas_repo" status --short --branch
git -C "$polarity_portfolio_repo" status --short --branch
runpodctl pod list --output json
```

- [ ] Veritas has no Polarity-caused change.
- [ ] Portfolio has no Polarity-caused change.
- [ ] No paid pod is running.
- [ ] No weight, raw dataset, or candidate source has been uploaded anywhere.

## 8. Stop and request approval

After every gate passes, record the candidate path, root commit, tree hash, scan versions, test
environments, and any limitations. Then **stop**.

Creating a remote, changing repository visibility, pushing any ref, publishing a release, uploading
the checkpoint, or modifying Portfolio requires a new, explicit external-mutation approval. A
successful local checklist is evidence for that decision, not authorization to perform it.
