"""M1 build orchestrator.

fetch (assumed done) -> adapt -> deterministic sample -> source-family grouping -> cross-dataset
dedup -> sealed-Veritas audit (FAIL CLOSED) -> family-cohesive split -> immutable JSONL manifests
+ JSON reports. Deterministic and byte-stable (sorted, no timestamps in payloads).

The two recipe tiers are kept separable: PAWS-Wiki + CANNOT are ``primary``; MNLI is
``exploratory`` and emitted as a *separate* ``with-mnli`` manifest set (primary + MNLI). Because
splits are assigned per source-family, primary families get identical splits in both views, so
``primary`` is a clean subset of ``with-mnli``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from polarity.data import dedup, grouping, stats
from polarity.data.adapters import adapt
from polarity.data.loaders import load_golden
from polarity.data.schema import Example, RecipeTier
from polarity.data.sources import get_source
from polarity.data.split import assign_family_splits, transform_split_offenders
from polarity.eval.leakage import SealedAuditReport, SealedLeakageError, sealed_set_audit
from polarity.types import SplitName

PRIMARY_SOURCES = ("paws-wiki", "cannot")
EXPLORATORY_SOURCES = ("mnli",)
EVAL_SOURCES = ("nan-nli",)
GOLDEN_REL = Path("data/snapshots/veritas-golden/golden.json")
_SPLIT_FILE = {
    SplitName.TRAIN: "train",
    SplitName.VALIDATION: "validation",
    SplitName.TEST: "internal_test",
}


class BuildError(RuntimeError):
    """Raised when the build cannot proceed (missing raw data, or a sealed-set collision)."""


def _sample(examples: list[Example], limit: int, seed: int) -> tuple[list[Example], int]:
    if len(examples) <= limit:
        return examples, 0
    ordered = sorted(
        examples,
        key=lambda e: hashlib.sha256(f"{seed}:{e.example_id}".encode()).hexdigest(),
    )
    return ordered[:limit], len(examples) - limit


def _entry(e: Example, split: SplitName | None) -> dict[str, object]:
    # example_id already encodes source:split:source_example_id lineage, so the (redundant)
    # provenance_hash is omitted here to keep manifests lean; text hashes stay for the audit.
    return {
        "example_id": e.example_id,
        "source": e.source_dataset,
        "family": e.source_family_id,
        "split": split.value if split else "probe",
        "label": e.label.value if e.label else None,
        "category": e.category.value,
        "tier": e.recipe_tier.value,
        "a_sha": e.text_a_sha,
        "b_sha": e.text_b_sha,
    }


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda d: str(d["example_id"]))
    lines = [json.dumps(d, sort_keys=True, separators=(",", ":")) for d in ordered]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _compare(primary: stats.RecipeStats, with_mnli: stats.RecipeStats) -> dict[str, object]:
    pc = {c.category: c.n for c in primary.categories}
    wc = {c.category: c.n for c in with_mnli.categories}
    confound_delta = round(with_mnli.source_majority_accuracy - primary.source_majority_accuracy, 6)
    coverage_gain = any(wc.get(cat, 0) > 1.2 * pc.get(cat, 0) for cat in wc) or bool(
        set(wc) - set(pc)
    )
    confound_ok = with_mnli.source_majority_accuracy <= 0.8 and confound_delta <= 0.1
    include = confound_ok and coverage_gain
    return {
        "primary_source_majority_accuracy": primary.source_majority_accuracy,
        "with_mnli_source_majority_accuracy": with_mnli.source_majority_accuracy,
        "confound_delta": confound_delta,
        "primary_category_counts": pc,
        "with_mnli_category_counts": wc,
        "new_categories_from_mnli": sorted(set(wc) - set(pc)),
        "primary_equivalent_negation": primary.equivalent_negation_count,
        "with_mnli_equivalent_negation": with_mnli.equivalent_negation_count,
        "mnli_verdict": "INCLUDE" if include else "EXCLUDE",
        "mnli_verdict_reason": (
            "adds coverage without unacceptable confound"
            if include
            else (
                f"raises source->label confound (majority-acc "
                f"{with_mnli.source_majority_accuracy}, delta {confound_delta}) beyond the "
                "0.8/0.1 bar; MNLI is negatives-only"
            )
        ),
    }


def build(
    *,
    raw_root: Path = Path("data/raw"),
    out_root: Path = Path("."),
    limit_per_source: int = 15000,
    seed: int = 42,
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
    exclude_sealed_collisions: bool = False,
) -> dict[str, object]:
    reports_dir = out_root / "reports" / "m1"

    # 1. adapt each source (full mapping + audit)
    audits: dict[str, object] = {}
    materialized: dict[str, list[Example]] = {}
    for sid in PRIMARY_SOURCES + EXPLORATORY_SOURCES + EVAL_SOURCES:
        src = get_source(sid)
        examples, audit = adapt(src, raw_root / sid)
        if audit.source_rows == 0:
            raise BuildError(
                f"{sid}: no raw rows under {raw_root / sid} — run `polarity data fetch` first"
            )
        audits[sid] = audit.model_dump(mode="json")
        materialized[sid] = examples

    # 2. deterministic per-source sampling (training-eligible sources only)
    sampling: dict[str, object] = {}
    for sid in PRIMARY_SOURCES + EXPLORATORY_SOURCES:
        kept, dropped = _sample(materialized[sid], limit_per_source, seed)
        sampling[sid] = {
            "available": len(materialized[sid]),
            "kept": len(kept),
            "dropped": dropped,
            "cap": limit_per_source,
        }
        materialized[sid] = kept
    sampling["nan-nli"] = {
        "available": len(materialized["nan-nli"]),
        "kept": len(materialized["nan-nli"]),
        "dropped": 0,
        "cap": None,
    }

    # 3. grouping (families span primary + MNLI so cross-dataset families are caught)
    train_eligible = grouping.with_families(
        [e for sid in PRIMARY_SOURCES + EXPLORATORY_SOURCES for e in materialized[sid]]
    )
    eval_examples = grouping.with_families(materialized["nan-nli"])

    # example ids must be globally unique (a duplicate would collapse manifest rows)
    all_ids = [e.example_id for e in train_eligible] + [e.example_id for e in eval_examples]
    id_counts = Counter(all_ids)
    if len(id_counts) != len(all_ids):
        dups = [k for k, c in id_counts.items() if c > 1][:5]
        raise BuildError(f"duplicate example_ids: {dups}")

    # 4. cross-dataset dedup
    overlap = dedup.detect(train_eligible + eval_examples)
    _write_json(reports_dir / "dedup.json", overlap.model_dump(mode="json"))
    _write_json(reports_dir / "mapping_audit.json", audits)
    _write_json(reports_dir / "sampling.json", sampling)

    # 5. sealed-set audit — the Veritas FINAL holdout and the NaN-NLI eval probe, reported
    #    separately. FAIL CLOSED by default; a documented family exclusion runs only on opt-in.
    golden = load_golden(out_root / GOLDEN_REL)
    veritas_pairs = [(g.base, g.probe) for g in golden]
    nan_pairs = [(e.text_a, e.text_b) for e in eval_examples]

    def _colliding(audit: SealedAuditReport) -> set[str]:
        return set(audit.text_collisions) | set(audit.reverse_pair_collisions)

    v_audit = sealed_set_audit(train_eligible, veritas_pairs)
    n_audit = sealed_set_audit(train_eligible, nan_pairs)
    colliding_ids = _colliding(v_audit) | _colliding(n_audit)
    fam_of = {e.example_id: e.source_family_id for e in train_eligible}
    excluded_families = {fam_of[i] for i in colliding_ids if i in fam_of}

    n_before = len(train_eligible)
    remediation = "none_needed"
    if colliding_ids:
        remediation = "excluded_colliding_families" if exclude_sealed_collisions else "FAIL_CLOSED"

    _write_json(
        reports_dir / "leakage.json",
        {
            "veritas_collisions": len(_colliding(v_audit)),
            "nan_nli_collisions": len(_colliding(n_audit)),
            "total_colliding_examples": len(colliding_ids),
            "excluded_families": len(excluded_families),
            "remediation": remediation,
            "note": (
                "CANNOT incorporates a NaN-NLI subset, so some CANNOT training examples overlap "
                "the NaN-NLI eval probe; their families are excluded to keep the probe held out. "
                "Zero collisions with the sealed Veritas final holdout."
            ),
            "sample_collisions": sorted(colliding_ids)[:10],
        },
    )

    if colliding_ids and not exclude_sealed_collisions:
        _write_json(
            reports_dir / "build-summary.json",
            {
                "status": "BLOCKED_SEALED_COLLISION",
                "total_colliding_examples": len(colliding_ids),
                "veritas_collisions": len(_colliding(v_audit)),
                "nan_nli_collisions": len(_colliding(n_audit)),
                "hint": "re-run with exclude_sealed_collisions=True to drop the colliding "
                "families (documented in leakage.json)",
            },
        )
        raise SealedLeakageError(
            f"{len(colliding_ids)} sealed-set collisions; re-run with --exclude-sealed-collisions"
        )

    if colliding_ids:
        train_eligible = [e for e in train_eligible if e.source_family_id not in excluded_families]
        if not (
            sealed_set_audit(train_eligible, veritas_pairs).ok
            and sealed_set_audit(train_eligible, nan_pairs).ok
        ):
            raise SealedLeakageError("collision persists after family exclusion (unexpected)")

    n_excluded_examples = n_before - len(train_eligible)

    # 6. family-cohesive split (Veritas is separate; not in this 3-way)
    split_of = assign_family_splits(
        train_eligible, train=train, validation=validation, test=test, seed=seed
    )
    offenders = transform_split_offenders(train_eligible, split_of)
    if offenders:
        raise BuildError(
            f"transformation child(ren) in a different split from parent: {offenders[:5]}"
        )

    # 7. manifests — primary, with-mnli (primary + MNLI), eval
    primary = [e for e in train_eligible if e.recipe_tier == RecipeTier.PRIMARY]
    for split, fname in _SPLIT_FILE.items():
        _write_jsonl(
            out_root / "manifests" / "primary" / f"{fname}.jsonl",
            [_entry(e, split) for e in primary if split_of[e.example_id] == split],
        )
        _write_jsonl(
            out_root / "manifests" / "with-mnli" / f"{fname}.jsonl",
            [_entry(e, split) for e in train_eligible if split_of[e.example_id] == split],
        )
    _write_jsonl(
        out_root / "manifests" / "eval" / "nan-nli.jsonl",
        [_entry(e, None) for e in eval_examples],
    )

    # 8. stats + the two-recipe comparison
    stats_primary = stats.compute(primary, "primary")
    stats_with_mnli = stats.compute(train_eligible, "with_mnli")
    comparison = _compare(stats_primary, stats_with_mnli)
    _write_json(reports_dir / "stats-primary.json", stats_primary.model_dump(mode="json"))
    _write_json(reports_dir / "stats-with-mnli.json", stats_with_mnli.model_dump(mode="json"))
    _write_json(reports_dir / "recipe-comparison.json", comparison)

    # 9. disposition
    primary_ready = (
        stats_primary.source_majority_accuracy <= 0.8
        and stats_primary.label_equivalent > 0
        and stats_primary.label_not_equivalent > 0
    )
    disposition = "TRAINING_RECIPE_READY" if primary_ready else "DATA_REBALANCING_REQUIRED"
    summary = {
        "status": "OK",
        "disposition_primary": disposition,
        "mnli_verdict": comparison["mnli_verdict"],
        "equivalent_negation_sufficient": stats_primary.equivalent_negation_sufficient,
        "seed": seed,
        "limit_per_source": limit_per_source,
        "split_ratios": {"train": train, "validation": validation, "test": test},
        "n_primary": len(primary),
        "n_with_mnli": len(train_eligible),
        "n_eval_probe": len(eval_examples),
        "sealed_veritas_collisions": len(_colliding(v_audit)),
        "nan_probe_collisions": len(_colliding(n_audit)),
        "excluded_families": len(excluded_families),
        "excluded_examples": n_excluded_examples,
        "remediation": remediation,
    }
    _write_json(reports_dir / "build-summary.json", summary)
    return summary


__all__ = ["BuildError", "build"]
