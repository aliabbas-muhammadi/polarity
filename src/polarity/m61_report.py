"""Pure, keyless M6.1 report and provenance validation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from polarity.artifact import (
    ArtifactIdentity,
    ArtifactVerificationError,
    load_artifact_release_context,
)
from polarity.data.loaders import GoldenPair
from polarity.eval import curves, metrics
from polarity.textnorm import normalized_sha
from polarity.transfer import (
    FrozenProtocol,
    GoldenScore,
    ReproductionError,
    load_protocol,
    load_scores,
)
from polarity.types import Category, Label, Prediction


class M61ReportError(RuntimeError):
    """A normalized release-gate failure suitable for CLI output."""


class ProbeScore(BaseModel):
    """One public authored-probe score row, with source metadata deliberately omitted."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_index: int = Field(ge=0)
    id: str = Field(min_length=1)
    raw_score: float = Field(ge=0.0, le=1.0)
    calibrated: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def scores_are_finite(self) -> ProbeScore:
        if not math.isfinite(self.raw_score) or not math.isfinite(self.calibrated):
            raise ValueError("scores must be finite")
        return self


@dataclass(frozen=True)
class _ProbeSourceSnapshot:
    rows: tuple[GoldenPair, ...]
    sha256: str


@dataclass(frozen=True)
class _ProbeScoreSnapshot:
    rows: tuple[ProbeScore, ...]
    sha256: str


class MrpcScore(BaseModel):
    """One public MRPC score row; deliberately contains neither source text nor gold labels."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_index: int = Field(ge=0)
    id: str
    raw_score: float = Field(ge=0.0, le=1.0)
    calibrated: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def identity_matches_order(self) -> MrpcScore:
        expected = f"mrpc-test-{self.row_index}"
        if self.id != expected:
            raise ValueError(f"id must equal {expected}")
        if not math.isfinite(self.raw_score) or not math.isfinite(self.calibrated):
            raise ValueError("scores must be finite")
        return self


class MrpcSourceContract(BaseModel):
    """The distributable assertions needed to verify a locally obtained MRPC source."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    label_mapping: dict[str, bool]
    filtering: Literal["none_all_rows_retained"]
    duplicate_policy: Literal["retain_all_source_rows"]
    normalized_ordered_pair_duplicate_rows: int = Field(ge=0)
    duplicate_label_conflicts: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_and_mapping_are_consistent(self) -> MrpcSourceContract:
        if self.positive_count + self.negative_count != self.row_count:
            raise ValueError("source class counts must sum to row_count")
        expected_mapping = {"paraphrase": True, "not-paraphrase": False}
        if self.label_mapping != expected_mapping:
            raise ValueError(f"label_mapping must equal {expected_mapping}")
        return self


MRPC_SOURCE_CONTRACT = MrpcSourceContract(
    raw_sha256="7999ee0ba2af1c43e416c35d152bdcaa99acb79ae1791914e748535f78d8a18c",
    row_count=1725,
    positive_count=1147,
    negative_count=578,
    label_mapping={"paraphrase": True, "not-paraphrase": False},
    filtering="none_all_rows_retained",
    duplicate_policy="retain_all_source_rows",
    normalized_ordered_pair_duplicate_rows=1,
    duplicate_label_conflicts=0,
)
MRPC_OFFICIAL_URL = "https://www.microsoft.com/en-us/download/details.aspx?id=52398"
MRPC_LOADER = "nyu-mll/glue"
MRPC_CONFIG = "mrpc"
MRPC_SPLIT = "test"
MRPC_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
MRPC_CACHED_ARROW_SHA256 = "0152c178bc09c083804600f657ae610005dfce932f29435cd4938509f9643d71"
M6_PROTOCOL = Path("reports/m6/protocol-xp-006-deberta-transformers-fixed.json")
M6_MODEL_MANIFEST = Path("reports/m6/model-manifest-xp-006-deberta-transformers-fixed.json")
VERITAS_SOURCE = Path("data/snapshots/veritas-golden/golden.json")
VERITAS_PREDICTIONS = Path("predictions/m6/veritas/xp-006-deberta-transformers-fixed.jsonl")
VERITAS_REPORT = Path("reports/m6/transfer-xp-006-deberta-transformers-fixed.json")
VERITAS_BINDING = Path("reports/m6.1/veritas-binding.json")
PROBE_SOURCE = Path("data/diagnostics/different-answer-probe.json")
PROBE_PREDICTIONS = Path("predictions/m6.1/different-answer-probe.jsonl")
PROBE_REPORT = Path("reports/m6.1/different-answer-probe.json")
PROBE_BINDING = Path("reports/m6.1/different-answer-probe-binding.json")
MRPC_PREDICTIONS = Path("predictions/m6.1/mrpc-transfer.jsonl")
MRPC_REPORT = Path("reports/m6.1/mrpc-transfer.json")
MRPC_PROVENANCE = Path("reports/m6.1/mrpc-provenance.json")
MRPC_BINDING = Path("reports/m6.1/mrpc-binding.json")
MRPC_SOURCE_IDENTIFIER = f"microsoft-mrpc-test:{MRPC_LOADER}@{MRPC_REVISION}"


class VerifiedMrpcRow(BaseModel):
    """Private in-memory join of public scores to a verified, locally supplied gold row."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_index: int
    id: str
    gold_equivalent: bool
    raw_score: float
    calibrated: float


class _MrpcLabelMapping(BaseModel):
    """The two labels permitted in the public MRPC report."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    paraphrase: Literal[True]
    not_paraphrase: Literal[False] = Field(alias="not-paraphrase")


class _MrpcDuplicatePolicy(BaseModel):
    """Retention metadata retained from the checksum-verified source contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    policy: Literal["retain_all_source_rows"]
    normalized_ordered_pair_duplicate_rows: int = Field(ge=0)
    label_conflicts: int = Field(ge=0)


class _MrpcThresholdFreeMetrics(BaseModel):
    """Threshold-free report fields; metrics are range-checked, not re-evaluated publicly."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    roc_auc: float = Field(ge=0.0, le=1.0)
    custom_step_pauc_fpr_0_02: float = Field(alias="custom_step_pauc_fpr_0.02", ge=0.0, le=1.0)
    pauc_definition: Literal[
        "normalized step integral of maximum achievable recall over FPR in [0, 0.02]; "
        "not sklearn standardized partial AUC"
    ]
    recall_at_fpr_budgets: dict[Literal["0.005", "0.01", "0.02", "0.05", "0.1"], float]

    @model_validator(mode="after")
    def recall_budgets_are_complete_and_bounded(self) -> _MrpcThresholdFreeMetrics:
        if set(self.recall_at_fpr_budgets) != {"0.005", "0.01", "0.02", "0.05", "0.1"}:
            raise ValueError("recall_at_fpr_budgets must contain the five committed FPR budgets")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.recall_at_fpr_budgets.values()
        ):
            raise ValueError("recall_at_fpr_budgets values must be finite probabilities")
        return self

    @model_validator(mode="after")
    def metrics_are_finite(self) -> _MrpcThresholdFreeMetrics:
        if not math.isfinite(self.roc_auc) or not math.isfinite(self.custom_step_pauc_fpr_0_02):
            raise ValueError("threshold-free metrics must be finite")
        return self


class _MrpcOperatingPointMetrics(BaseModel):
    """A realized MRPC confusion matrix at one frozen threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tau: float
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    tn: int = Field(ge=0)
    fn: int = Field(ge=0)
    recall: float = Field(ge=0.0, le=1.0)
    fpr: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def counts_and_rates_are_consistent(self) -> _MrpcOperatingPointMetrics:
        if not math.isfinite(self.tau):
            raise ValueError("operating-point threshold must be finite")
        values = (self.recall, self.fpr, self.precision)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("operating-point rates must be finite")
        expected_recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        expected_fpr = self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0
        expected_precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0
        if not math.isclose(self.recall, expected_recall, abs_tol=5e-7):
            raise ValueError("operating-point recall does not match counts")
        if not math.isclose(self.fpr, expected_fpr, abs_tol=5e-7):
            raise ValueError("operating-point FPR does not match counts")
        if not math.isclose(self.precision, expected_precision, abs_tol=5e-7):
            raise ValueError("operating-point precision does not match counts")
        return self


class _MrpcScoreDistribution(BaseModel):
    """A bounded shape summary for one MRPC class's raw scores."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    n: int = Field(gt=0)
    mean: float = Field(ge=0.0, le=1.0)
    median: float = Field(ge=0.0, le=1.0)
    population_std: float = Field(ge=0.0, le=1.0)
    q05: float = Field(ge=0.0, le=1.0)
    q95: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def summary_is_finite_and_ordered(self) -> _MrpcScoreDistribution:
        values = (self.mean, self.median, self.population_std, self.q05, self.q95)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("score distribution values must be finite")
        if not self.q05 <= self.median <= self.q95:
            raise ValueError("score distribution quantiles must be ordered")
        return self


class _MrpcScoreDistributions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    paraphrase: _MrpcScoreDistribution
    not_paraphrase: _MrpcScoreDistribution = Field(alias="not_paraphrase")


class _MrpcReport(BaseModel):
    """Fail-closed source-free schema for the complete committed MRPC report."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["polarity-m61-mrpc-report-1"]
    evaluation_id: Literal["m6.1-mrpc-test"]
    metric_basis: Literal["checksum_verified_official_local_source"]
    protocol_version: Literal[2]
    experiment_id: str
    artifact_identity: ArtifactIdentity
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_pairs: int = Field(gt=0)
    n_paraphrase: int = Field(ge=0)
    n_not_paraphrase: int = Field(ge=0)
    label_mapping: _MrpcLabelMapping
    row_retention: Literal["none_all_rows_retained"]
    duplicate_policy: _MrpcDuplicatePolicy
    threshold_free: _MrpcThresholdFreeMetrics
    at_frozen_m6_operating_points: dict[
        Literal["zero_fp", "fpr_le_0.01", "fpr_le_0.05"], _MrpcOperatingPointMetrics
    ]
    raw_score_distributions: _MrpcScoreDistributions
    disposition: Literal["NO_MATERIAL_OOD_TRANSFER_TO_MRPC"]

    @model_validator(mode="after")
    def report_counts_are_internally_consistent(self) -> _MrpcReport:
        if self.n_paraphrase + self.n_not_paraphrase != self.n_pairs:
            raise ValueError("MRPC class counts must sum to n_pairs")
        expected_points = {"zero_fp", "fpr_le_0.01", "fpr_le_0.05"}
        if set(self.at_frozen_m6_operating_points) != expected_points:
            raise ValueError("MRPC operating points must use the complete committed key set")
        for point in self.at_frozen_m6_operating_points.values():
            if point.tp + point.fn != self.n_paraphrase:
                raise ValueError("operating-point positive counts differ from n_paraphrase")
            if point.fp + point.tn != self.n_not_paraphrase:
                raise ValueError("operating-point negative counts differ from n_not_paraphrase")
        if self.raw_score_distributions.paraphrase.n != self.n_paraphrase:
            raise ValueError("paraphrase score distribution count differs from n_paraphrase")
        if self.raw_score_distributions.not_paraphrase.n != self.n_not_paraphrase:
            raise ValueError(
                "not-paraphrase score distribution count differs from n_not_paraphrase"
            )
        return self


def load_mrpc_scores(path: Path) -> list[MrpcScore]:
    """Load the public, text-free MRPC prediction artifact."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise M61ReportError(f"cannot read MRPC scores {path}: {exc}") from exc
    scores: list[MrpcScore] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            scores.append(MrpcScore.model_validate_json(line))
        except ValidationError as exc:
            raise M61ReportError(f"invalid MRPC score row {line_number}: {exc}") from exc
    return scores


def _load_probe_score_snapshot(path: Path) -> _ProbeScoreSnapshot:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise M61ReportError(f"invalid M6.1 probe scores: cannot read {path}: {exc}") from exc
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise M61ReportError(f"invalid M6.1 probe scores: cannot read {path}: {exc}") from exc
    scores: list[ProbeScore] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            score = ProbeScore.model_validate_json(line)
        except ValidationError as exc:
            raise M61ReportError(
                f"invalid M6.1 probe scores: invalid row at line {line_number}: {exc}"
            ) from exc
        expected_index = len(scores)
        if score.row_index != expected_index:
            raise M61ReportError(
                "invalid M6.1 probe scores: "
                f"row_index {score.row_index} must equal line order {expected_index}"
            )
        scores.append(score)
    return _ProbeScoreSnapshot(
        rows=tuple(scores),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_probe_scores(path: Path) -> list[ProbeScore]:
    """Load the public, source-metadata-free authored-probe prediction artifact."""
    return list(_load_probe_score_snapshot(path).rows)


def _load_golden_scores(path: Path, *, kind: str) -> list[GoldenScore]:
    """Normalize author-owned score artifact failures at the M6.1 release-gate boundary."""
    try:
        return load_scores(path)
    except (OSError, UnicodeError) as exc:
        raise M61ReportError(f"invalid M6.1 {kind} scores: cannot read {path}: {exc}") from exc
    except ValidationError as exc:
        raise M61ReportError(f"invalid M6.1 {kind} scores: {exc}") from exc


def _parse_source_bytes(raw: bytes, *, kind: str, strict: bool = False) -> tuple[GoldenPair, ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise M61ReportError(f"invalid {kind}: cannot read source as UTF-8: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise M61ReportError(
            f"invalid {kind}: malformed JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, list):
        raise M61ReportError(f"invalid {kind}: root must be a JSON array")
    try:
        return tuple(GoldenPair.model_validate(row, strict=strict) for row in payload)
    except ValidationError as exc:
        raise M61ReportError(f"invalid {kind} row: {exc}") from exc


def _read_source(path: Path, *, kind: str = "MRPC source") -> list[GoldenPair]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise M61ReportError(f"invalid {kind}: cannot read {path}: {exc}") from exc
    return list(_parse_source_bytes(raw, kind=kind))


def _load_probe_source_snapshot(path: Path) -> _ProbeSourceSnapshot:
    kind = "M6.1 probe source"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise M61ReportError(f"invalid {kind}: cannot read {path}: {exc}") from exc
    return _ProbeSourceSnapshot(
        rows=_parse_source_bytes(raw, kind=kind, strict=True),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _duplicate_counts(rows: list[GoldenPair]) -> tuple[int, int]:
    labels_by_pair: dict[tuple[str, str], set[bool]] = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (normalized_sha(row.base), normalized_sha(row.probe))
        pair_counts[key] += 1
        labels_by_pair[key].add(row.equivalent)
    duplicate_rows = sum(count - 1 for count in pair_counts.values())
    conflicts = sum(1 for labels in labels_by_pair.values() if len(labels) > 1)
    return duplicate_rows, conflicts


def verify_mrpc_source(
    source_path: Path,
    scores: list[MrpcScore],
    contract: MrpcSourceContract,
) -> list[VerifiedMrpcRow]:
    """Verify source identity, ordering, label mapping, retention, and duplicate policy."""
    try:
        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise M61ReportError(f"cannot read MRPC source {source_path}: {exc}") from exc
    if actual_sha != contract.raw_sha256:
        raise M61ReportError(
            f"MRPC source checksum mismatch: {actual_sha} != {contract.raw_sha256}"
        )
    source = _read_source(source_path)
    source_ids = [row.id for row in source]
    duplicate_ids = sorted(item for item, count in Counter(source_ids).items() if count > 1)
    if duplicate_ids:
        raise M61ReportError(f"duplicate source id: {duplicate_ids[0]}")
    for index, row in enumerate(source):
        expected_id = f"mrpc-test-{index}"
        expected_label = contract.label_mapping.get(row.pairType)
        if expected_label is None or row.equivalent != expected_label:
            raise M61ReportError(f"label mapping mismatch for {row.id}")
        if row.id != expected_id:
            raise M61ReportError(f"source row {index} must have id {expected_id}")

    score_ids = [score.id for score in scores]
    duplicate_score_ids = sorted(item for item, count in Counter(score_ids).items() if count > 1)
    if duplicate_score_ids:
        raise M61ReportError(f"duplicate prediction id: {duplicate_score_ids[0]}")
    missing = sorted(set(source_ids) - set(score_ids))
    extra = sorted(set(score_ids) - set(source_ids))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {missing[0]}")
        if extra:
            parts.append(f"extra {extra[0]}")
        raise M61ReportError(f"prediction ID set differs: {', '.join(parts)}")
    if score_ids != source_ids:
        raise M61ReportError("prediction order differs from the official source order")

    positives = sum(row.equivalent for row in source)
    negatives = len(source) - positives
    if len(source) != contract.row_count:
        raise M61ReportError(
            f"MRPC source row count mismatch: {len(source)} != {contract.row_count}"
        )
    if positives != contract.positive_count or negatives != contract.negative_count:
        raise M61ReportError("MRPC source class counts differ from provenance")
    duplicate_rows, duplicate_conflicts = _duplicate_counts(source)
    if duplicate_rows != contract.normalized_ordered_pair_duplicate_rows:
        raise M61ReportError(
            "normalized ordered-pair duplicate count differs from provenance: "
            f"{duplicate_rows} != {contract.normalized_ordered_pair_duplicate_rows}"
        )
    if duplicate_conflicts != contract.duplicate_label_conflicts:
        raise M61ReportError("normalized ordered-pair duplicate label conflicts differ")

    return [
        VerifiedMrpcRow(
            row_index=index,
            id=row.id,
            gold_equivalent=row.equivalent,
            raw_score=score.raw_score,
            calibrated=score.calibrated,
        )
        for index, (row, score) in enumerate(zip(source, scores, strict=True))
    ]


def _as_predictions(rows: list[VerifiedMrpcRow]) -> list[Prediction]:
    return [
        Prediction(
            pair_id=row.id,
            gold=(Label.CACHE_EQUIVALENT if row.gold_equivalent else Label.NOT_CACHE_EQUIVALENT),
            score=row.raw_score,
            category=Category.PARAPHRASE,
            source="mrpc-test",
        )
        for row in rows
    ]


def _roc_auc(rows: list[VerifiedMrpcRow]) -> float:
    """Binary ROC-AUC via average ranks, including exact tie handling."""
    ranked = sorted((row.raw_score, row.gold_equivalent) for row in rows)
    n_positive = sum(row.gold_equivalent for row in rows)
    n_negative = len(rows) - n_positive
    if not n_positive or not n_negative:
        raise M61ReportError("ROC-AUC requires both MRPC classes")
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _threshold_metrics(predictions: list[Prediction], threshold: float) -> dict[str, int | float]:
    counts = metrics.confusion_at(predictions, threshold)
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
    fpr = counts.fp / (counts.fp + counts.tn) if counts.fp + counts.tn else 0.0
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 1.0
    return {
        "tau": threshold,
        "tp": counts.tp,
        "fp": counts.fp,
        "tn": counts.tn,
        "fn": counts.fn,
        "recall": round(recall, 6),
        "fpr": round(fpr, 6),
        "precision": round(precision, 6),
    }


def _quantile(values: list[float], probability: float) -> float:
    """Nearest-observation quantile, matching the audit's reported score summaries."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    return ordered[round(position)]


def _distribution(values: list[float]) -> dict[str, int | float]:
    if not values:
        raise M61ReportError("score distribution requires at least one row")
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "population_std": round(statistics.pstdev(values), 6),
        "q05": round(_quantile(values, 0.05), 6),
        "q95": round(_quantile(values, 0.95), 6),
    }


def build_mrpc_report(
    rows: list[VerifiedMrpcRow],
    protocol: FrozenProtocol,
    contract: MrpcSourceContract,
) -> dict[str, object]:
    """Recompute the complete MRPC result from checksum-verified local gold plus public scores."""
    if not rows:
        raise M61ReportError("no verified MRPC rows")
    predictions = _as_predictions(rows)
    positives = [row.raw_score for row in rows if row.gold_equivalent]
    negatives = [row.raw_score for row in rows if not row.gold_equivalent]
    budgets = (0.005, 0.01, 0.02, 0.05, 0.1)
    return {
        "schema_version": "polarity-m61-mrpc-report-1",
        "evaluation_id": "m6.1-mrpc-test",
        "metric_basis": "checksum_verified_official_local_source",
        "protocol_version": protocol.protocol_version,
        "experiment_id": protocol.experiment_id,
        "artifact_identity": protocol.artifact_identity.model_dump(mode="json"),
        "source_sha256": contract.raw_sha256,
        "n_pairs": len(rows),
        "n_paraphrase": len(positives),
        "n_not_paraphrase": len(negatives),
        "label_mapping": contract.label_mapping,
        "row_retention": contract.filtering,
        "duplicate_policy": {
            "policy": contract.duplicate_policy,
            "normalized_ordered_pair_duplicate_rows": (
                contract.normalized_ordered_pair_duplicate_rows
            ),
            "label_conflicts": contract.duplicate_label_conflicts,
        },
        "threshold_free": {
            "roc_auc": round(_roc_auc(rows), 7),
            "custom_step_pauc_fpr_0.02": curves.partial_auc(predictions, 0.02),
            "pauc_definition": (
                "normalized step integral of maximum achievable recall over FPR in [0, 0.02]; "
                "not sklearn standardized partial AUC"
            ),
            "recall_at_fpr_budgets": curves.recall_at_fpr_budgets(predictions, budgets),
        },
        "at_frozen_m6_operating_points": {
            key: _threshold_metrics(predictions, point.threshold)
            for key, point in protocol.operating_points.items()
        },
        "raw_score_distributions": {
            "paraphrase": _distribution(positives),
            "not_paraphrase": _distribution(negatives),
        },
        "disposition": "NO_MATERIAL_OOD_TRANSFER_TO_MRPC",
    }


def _probe_group_metrics(scores: list[GoldenScore], threshold: float) -> dict[str, object]:
    by_group: dict[str, object] = {}
    for group in sorted({score.group for score in scores}):
        items = [score for score in scores if score.group == group]
        positives = [score for score in items if score.gold_equivalent]
        negatives = [score for score in items if not score.gold_equivalent]
        by_group[group] = {
            "n": len(items),
            "n_equivalent": len(positives),
            "n_not_equivalent": len(negatives),
            "accepted": sum(score.raw_score >= threshold for score in items),
            "recalled": sum(score.raw_score >= threshold for score in positives),
            "false_accepts": sum(score.raw_score >= threshold for score in negatives),
        }
    return by_group


def _join_probe_source(
    source: Sequence[GoldenPair], scores: Sequence[ProbeScore]
) -> list[GoldenScore]:
    source_ids = [row.id for row in source]
    score_ids = [score.id for score in scores]
    duplicate_source_ids = sorted(item for item, count in Counter(source_ids).items() if count > 1)
    if duplicate_source_ids:
        raise M61ReportError(f"duplicate probe source id: {duplicate_source_ids[0]}")
    duplicate_score_ids = sorted(item for item, count in Counter(score_ids).items() if count > 1)
    if duplicate_score_ids:
        raise M61ReportError(f"duplicate probe prediction id: {duplicate_score_ids[0]}")
    expected_indices = list(range(len(scores)))
    actual_indices = [score.row_index for score in scores]
    if actual_indices != expected_indices:
        raise M61ReportError("probe prediction row_index values do not match artifact order")
    missing = sorted(set(source_ids) - set(score_ids))
    extra = sorted(set(score_ids) - set(source_ids))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {missing[0]}")
        if extra:
            parts.append(f"extra {extra[0]}")
        raise M61ReportError(f"probe prediction ID set differs: {', '.join(parts)}")
    if score_ids != source_ids:
        raise M61ReportError("probe prediction order differs from source")
    return [
        GoldenScore(
            id=row.id,
            group=row.group,
            pair_type=row.pairType,
            gold_equivalent=row.equivalent,
            raw_score=score.raw_score,
            calibrated=score.calibrated,
        )
        for row, score in zip(source, scores, strict=True)
    ]


def _build_probe_report(
    source: _ProbeSourceSnapshot,
    scores: Sequence[ProbeScore],
    protocol: FrozenProtocol,
) -> dict[str, object]:
    aligned = _join_probe_source(source.rows, scores)
    return {
        "schema_version": "polarity-m61-probe-report-1",
        "evaluation_id": "m6.1-different-answer-probe",
        "diagnostic_scope": "author_created_post_hoc_not_a_benchmark",
        "protocol_version": protocol.protocol_version,
        "experiment_id": protocol.experiment_id,
        "artifact_identity": protocol.artifact_identity.model_dump(mode="json"),
        "source_sha256": source.sha256,
        "n_pairs": len(aligned),
        "n_equivalent_controls": sum(score.gold_equivalent for score in aligned),
        "n_meaning_change": sum(not score.gold_equivalent for score in aligned),
        "operating_points": {
            key: {
                "selection": point.model_dump(mode="json"),
                "realized_split": "authored_diagnostic_probe",
                "by_group": _probe_group_metrics(aligned, point.threshold),
            }
            for key, point in protocol.operating_points.items()
        },
    }


def build_probe_report(
    source_path: Path,
    scores: list[ProbeScore],
    protocol: FrozenProtocol,
) -> dict[str, object]:
    """Recompute the author-created M6.1 diagnostic from its source and committed scores."""
    source = _load_probe_source_snapshot(source_path)
    return _build_probe_report(source, scores, protocol)


def dump(payload: object) -> str:
    """Canonical human-diffable serialization for committed M6.1 JSON artifacts."""
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise M61ReportError(f"cannot hash release artifact {path}: {exc}") from exc


def _release_context(
    protocol_path: Path, model_manifest_path: Path
) -> tuple[FrozenProtocol, dict[str, str]]:
    try:
        manifest = load_artifact_release_context(model_manifest_path)
    except ArtifactVerificationError as exc:
        raise M61ReportError(f"invalid model manifest identity: {exc}") from exc
    try:
        protocol = load_protocol(protocol_path, expected_sha256=manifest.protocol_sha256)
    except ReproductionError as exc:
        raise M61ReportError(f"invalid frozen protocol: {exc}") from exc
    if protocol.artifact_identity != manifest.artifact_identity:
        raise M61ReportError("frozen protocol identity differs from model manifest")
    if protocol.thresholds != manifest.frozen_thresholds:
        raise M61ReportError("frozen protocol thresholds differ from model manifest")
    if protocol.calibrator_method != manifest.calibrator_method:
        raise M61ReportError("frozen protocol calibrator differs from model manifest")
    return protocol, manifest.artifact_identity.model_dump(mode="json")


def _build_evaluation_binding_with_prediction_sha256(
    *,
    evaluation_id: str,
    source_identifier: str,
    source_sha256: str,
    source_rows: int,
    prediction_path: Path,
    prediction_rows: int,
    prediction_sha256: str,
    report_path: Path,
    protocol_path: Path,
    model_manifest_path: Path,
) -> dict[str, object]:
    protocol, identity = _release_context(protocol_path, model_manifest_path)
    return {
        "schema_version": "polarity-evaluation-binding-1",
        "evaluation_id": evaluation_id,
        "artifact_identity": identity,
        "model_manifest": {
            "path": model_manifest_path.as_posix(),
            "sha256": _sha256_path(model_manifest_path),
        },
        "protocol": {
            "path": protocol_path.as_posix(),
            "sha256": _sha256_path(protocol_path),
            "protocol_version": protocol.protocol_version,
        },
        "source": {
            "identifier": source_identifier,
            "sha256": source_sha256,
            "rows": source_rows,
        },
        "predictions": {
            "path": prediction_path.as_posix(),
            "sha256": prediction_sha256,
            "rows": prediction_rows,
        },
        "report": {
            "path": report_path.as_posix(),
            "sha256": _sha256_path(report_path),
        },
    }


def build_evaluation_binding(
    *,
    evaluation_id: str,
    source_identifier: str,
    source_sha256: str,
    source_rows: int,
    prediction_path: Path,
    prediction_rows: int,
    report_path: Path,
    protocol_path: Path,
    model_manifest_path: Path,
) -> dict[str, object]:
    """Bind one evaluation's source, predictions, report, protocol, and four-hash identity."""
    return _build_evaluation_binding_with_prediction_sha256(
        evaluation_id=evaluation_id,
        source_identifier=source_identifier,
        source_sha256=source_sha256,
        source_rows=source_rows,
        prediction_path=prediction_path,
        prediction_rows=prediction_rows,
        prediction_sha256=_sha256_path(prediction_path),
        report_path=report_path,
        protocol_path=protocol_path,
        model_manifest_path=model_manifest_path,
    )


def _check_binding_text(binding_path: Path, expected: str) -> None:
    try:
        committed = binding_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise M61ReportError(f"cannot read evaluation binding {binding_path}: {exc}") from exc
    if committed != expected:
        raise M61ReportError(f"evaluation binding drift: {binding_path}")


def check_evaluation_binding(binding_path: Path, **kwargs: object) -> None:
    """Fail when a committed companion no longer matches any bound release artifact."""
    expected = dump(build_evaluation_binding(**kwargs))  # type: ignore[arg-type]
    _check_binding_text(binding_path, expected)


def build_mrpc_provenance(
    *,
    prediction_path: Path,
    report_path: Path,
    protocol_path: Path,
    model_manifest_path: Path,
) -> dict[str, object]:
    """Build exact source and artifact provenance without requiring or redistributing raw text."""
    protocol, identity = _release_context(protocol_path, model_manifest_path)
    contract = MRPC_SOURCE_CONTRACT
    return {
        "schema_version": "polarity-mrpc-provenance-1",
        "evaluation_id": "m6.1-mrpc-test",
        "source": {
            "official_url": MRPC_OFFICIAL_URL,
            "loader": MRPC_LOADER,
            "config": MRPC_CONFIG,
            "split": MRPC_SPLIT,
            "revision": MRPC_REVISION,
            "raw_sha256": contract.raw_sha256,
            "cached_arrow_sha256": MRPC_CACHED_ARROW_SHA256,
            "redistribution": "raw_text_not_committed_obtain_from_official_source",
        },
        "distribution": {
            "rows": contract.row_count,
            "paraphrase": contract.positive_count,
            "not_paraphrase": contract.negative_count,
        },
        "label_mapping": contract.label_mapping,
        "row_retention": contract.filtering,
        "duplicate_policy": {
            "policy": contract.duplicate_policy,
            "normalized_ordered_pair_duplicate_rows": (
                contract.normalized_ordered_pair_duplicate_rows
            ),
            "label_conflicts": contract.duplicate_label_conflicts,
        },
        "artifact_identity": identity,
        "model_manifest": {
            "path": model_manifest_path.as_posix(),
            "sha256": _sha256_path(model_manifest_path),
        },
        "protocol": {
            "path": protocol_path.as_posix(),
            "sha256": _sha256_path(protocol_path),
            "protocol_version": protocol.protocol_version,
        },
        "predictions": {
            "path": prediction_path.as_posix(),
            "sha256": _sha256_path(prediction_path),
            "rows": contract.row_count,
            "schema": "mrpc_score_only_1",
        },
        "report": {
            "path": report_path.as_posix(),
            "sha256": _sha256_path(report_path),
        },
    }


def _read_text(path: Path, *, kind: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise M61ReportError(f"cannot read {kind} {path}: {exc}") from exc


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _validate_mrpc_score_order(scores: list[MrpcScore], contract: MrpcSourceContract) -> None:
    if len(scores) != contract.row_count:
        raise M61ReportError(
            f"MRPC prediction row count mismatch: {len(scores)} != {contract.row_count}"
        )
    for index, score in enumerate(scores):
        if score.row_index != index:
            raise M61ReportError(
                f"MRPC prediction order mismatch at row {index}: found {score.row_index}"
            )


def _load_report_object(path: Path, *, kind: str) -> dict[str, object]:
    text = _read_text(path, kind=kind)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise M61ReportError(
            f"malformed {kind} JSON at line {exc.lineno}, column {exc.colno}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise M61ReportError(f"{kind} root must be a JSON object: {path}")
    return payload


def _check_mrpc_report_structure(
    report: dict[str, object], protocol: FrozenProtocol, contract: MrpcSourceContract
) -> None:
    try:
        parsed = _MrpcReport.model_validate(report)
    except ValidationError as exc:
        raise M61ReportError(f"invalid MRPC report structure: {exc}") from exc

    expected = {
        "schema_version": "polarity-m61-mrpc-report-1",
        "metric_basis": "checksum_verified_official_local_source",
        "source_sha256": contract.raw_sha256,
        "n_pairs": contract.row_count,
        "n_paraphrase": contract.positive_count,
        "n_not_paraphrase": contract.negative_count,
        "artifact_identity": protocol.artifact_identity.model_dump(mode="json"),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise M61ReportError(f"MRPC report structural mismatch: {key}")
    if parsed.experiment_id != protocol.experiment_id:
        raise M61ReportError("invalid MRPC report structure: experiment_id differs from protocol")
    if parsed.duplicate_policy.policy != contract.duplicate_policy:
        raise M61ReportError(
            "invalid MRPC report structure: duplicate policy differs from contract"
        )
    if (
        parsed.duplicate_policy.normalized_ordered_pair_duplicate_rows
        != contract.normalized_ordered_pair_duplicate_rows
    ):
        raise M61ReportError(
            "invalid MRPC report structure: duplicate row count differs from contract"
        )
    if parsed.duplicate_policy.label_conflicts != contract.duplicate_label_conflicts:
        raise M61ReportError(
            "invalid MRPC report structure: duplicate label conflicts differ from contract"
        )
    for key, point in parsed.at_frozen_m6_operating_points.items():
        if point.tau != protocol.operating_points[key].threshold:
            raise M61ReportError(
                f"invalid MRPC report structure: {key} threshold differs from protocol"
            )


def _binding_kwargs(
    *,
    evaluation_id: str,
    source_identifier: str,
    source_sha256: str,
    source_rows: int,
    prediction_path: Path,
    prediction_rows: int,
    report_path: Path,
    protocol_path: Path,
    model_manifest_path: Path,
) -> dict[str, object]:
    return {
        "evaluation_id": evaluation_id,
        "source_identifier": source_identifier,
        "source_sha256": source_sha256,
        "source_rows": source_rows,
        "prediction_path": prediction_path,
        "prediction_rows": prediction_rows,
        "report_path": report_path,
        "protocol_path": protocol_path,
        "model_manifest_path": model_manifest_path,
    }


def generate_probe_artifacts(root: Path = Path(".")) -> tuple[Path, Path, Path]:
    """Write the deterministic probe report plus Veritas/probe companion manifests."""
    protocol_path = root / M6_PROTOCOL
    manifest_path = root / M6_MODEL_MANIFEST
    protocol, _ = _release_context(protocol_path, manifest_path)
    probe_source = root / PROBE_SOURCE
    probe_predictions = root / PROBE_PREDICTIONS
    probe_report = root / PROBE_REPORT
    probe_binding = root / PROBE_BINDING
    veritas_source = root / VERITAS_SOURCE
    veritas_predictions = root / VERITAS_PREDICTIONS
    veritas_report = root / VERITAS_REPORT
    veritas_binding = root / VERITAS_BINDING

    score_snapshot = _load_probe_score_snapshot(probe_predictions)
    source_snapshot = _load_probe_source_snapshot(probe_source)
    _write_text(
        probe_report,
        dump(_build_probe_report(source_snapshot, score_snapshot.rows, protocol)),
    )
    probe_kwargs = _binding_kwargs(
        evaluation_id="m6.1-different-answer-probe",
        source_identifier=PROBE_SOURCE.as_posix(),
        source_sha256=source_snapshot.sha256,
        source_rows=len(source_snapshot.rows),
        prediction_path=probe_predictions,
        prediction_rows=len(score_snapshot.rows),
        report_path=probe_report,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    _write_text(
        probe_binding,
        dump(
            _build_evaluation_binding_with_prediction_sha256(
                **probe_kwargs,  # type: ignore[arg-type]
                prediction_sha256=score_snapshot.sha256,
            )
        ),
    )

    veritas_rows = len(_read_source(veritas_source, kind="M6.1 Veritas source"))
    veritas_scores = _load_golden_scores(veritas_predictions, kind="Veritas")
    veritas_kwargs = _binding_kwargs(
        evaluation_id="m6-veritas",
        source_identifier=VERITAS_SOURCE.as_posix(),
        source_sha256=_sha256_path(veritas_source),
        source_rows=veritas_rows,
        prediction_path=veritas_predictions,
        prediction_rows=len(veritas_scores),
        report_path=veritas_report,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    _write_text(
        veritas_binding,
        dump(build_evaluation_binding(**veritas_kwargs)),  # type: ignore[arg-type]
    )
    return probe_report, probe_binding, veritas_binding


def check_probe_artifacts(root: Path = Path(".")) -> None:
    """Reproduce the probe report and validate both author-owned evaluation bindings."""
    protocol_path = root / M6_PROTOCOL
    manifest_path = root / M6_MODEL_MANIFEST
    protocol, _ = _release_context(protocol_path, manifest_path)
    probe_source = root / PROBE_SOURCE
    probe_predictions = root / PROBE_PREDICTIONS
    probe_report = root / PROBE_REPORT
    score_snapshot = _load_probe_score_snapshot(probe_predictions)
    source_snapshot = _load_probe_source_snapshot(probe_source)
    expected_report = dump(_build_probe_report(source_snapshot, score_snapshot.rows, protocol))
    if _read_text(probe_report, kind="probe report") != expected_report:
        raise M61ReportError(f"probe report drift: {probe_report}")
    probe_kwargs = _binding_kwargs(
        evaluation_id="m6.1-different-answer-probe",
        source_identifier=PROBE_SOURCE.as_posix(),
        source_sha256=source_snapshot.sha256,
        source_rows=len(source_snapshot.rows),
        prediction_path=probe_predictions,
        prediction_rows=len(score_snapshot.rows),
        report_path=probe_report,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    expected_probe_binding = dump(
        _build_evaluation_binding_with_prediction_sha256(
            **probe_kwargs,  # type: ignore[arg-type]
            prediction_sha256=score_snapshot.sha256,
        )
    )
    _check_binding_text(root / PROBE_BINDING, expected_probe_binding)

    veritas_source = root / VERITAS_SOURCE
    veritas_predictions = root / VERITAS_PREDICTIONS
    veritas_scores = _load_golden_scores(veritas_predictions, kind="Veritas")
    veritas_rows = len(_read_source(veritas_source, kind="M6.1 Veritas source"))
    veritas_kwargs = _binding_kwargs(
        evaluation_id="m6-veritas",
        source_identifier=VERITAS_SOURCE.as_posix(),
        source_sha256=_sha256_path(veritas_source),
        source_rows=veritas_rows,
        prediction_path=veritas_predictions,
        prediction_rows=len(veritas_scores),
        report_path=root / VERITAS_REPORT,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    check_evaluation_binding(root / VERITAS_BINDING, **veritas_kwargs)


def generate_mrpc_artifacts(source_path: Path, root: Path = Path(".")) -> tuple[Path, Path, Path]:
    """Write MRPC report, exact provenance, and companion from a verified official source."""
    protocol_path = root / M6_PROTOCOL
    manifest_path = root / M6_MODEL_MANIFEST
    protocol, _ = _release_context(protocol_path, manifest_path)
    prediction_path = root / MRPC_PREDICTIONS
    report_path = root / MRPC_REPORT
    provenance_path = root / MRPC_PROVENANCE
    binding_path = root / MRPC_BINDING
    scores = load_mrpc_scores(prediction_path)
    _validate_mrpc_score_order(scores, MRPC_SOURCE_CONTRACT)
    verified = verify_mrpc_source(source_path, scores, MRPC_SOURCE_CONTRACT)
    _write_text(report_path, dump(build_mrpc_report(verified, protocol, MRPC_SOURCE_CONTRACT)))
    provenance = build_mrpc_provenance(
        prediction_path=prediction_path,
        report_path=report_path,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    _write_text(provenance_path, dump(provenance))
    binding_kwargs = _binding_kwargs(
        evaluation_id="m6.1-mrpc-test",
        source_identifier=MRPC_SOURCE_IDENTIFIER,
        source_sha256=MRPC_SOURCE_CONTRACT.raw_sha256,
        source_rows=MRPC_SOURCE_CONTRACT.row_count,
        prediction_path=prediction_path,
        prediction_rows=len(scores),
        report_path=report_path,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    binding = build_evaluation_binding(**binding_kwargs)  # type: ignore[arg-type]
    binding["source_provenance"] = {
        "path": provenance_path.as_posix(),
        "sha256": _sha256_path(provenance_path),
    }
    _write_text(binding_path, dump(binding))
    return report_path, provenance_path, binding_path


def check_mrpc_artifacts(
    source_path: Path | None = None, root: Path = Path(".")
) -> Literal["structural_only_metrics_not_reproduced", "full_metrics_reproduced"]:
    """Validate public structure, and optionally reproduce metrics with official local source."""
    protocol_path = root / M6_PROTOCOL
    manifest_path = root / M6_MODEL_MANIFEST
    protocol, _ = _release_context(protocol_path, manifest_path)
    prediction_path = root / MRPC_PREDICTIONS
    report_path = root / MRPC_REPORT
    provenance_path = root / MRPC_PROVENANCE
    binding_path = root / MRPC_BINDING
    scores = load_mrpc_scores(prediction_path)
    _validate_mrpc_score_order(scores, MRPC_SOURCE_CONTRACT)
    report = _load_report_object(report_path, kind="MRPC report")
    provenance_object = _load_report_object(provenance_path, kind="MRPC provenance")
    report_binding = provenance_object.get("report")
    if not isinstance(report_binding, dict) or report_binding.get("sha256") != _sha256_path(
        report_path
    ):
        raise M61ReportError(f"MRPC report hash drift: {report_path}")
    _check_mrpc_report_structure(report, protocol, MRPC_SOURCE_CONTRACT)
    expected_provenance = dump(
        build_mrpc_provenance(
            prediction_path=prediction_path,
            report_path=report_path,
            protocol_path=protocol_path,
            model_manifest_path=manifest_path,
        )
    )
    if _read_text(provenance_path, kind="MRPC provenance") != expected_provenance:
        raise M61ReportError(f"MRPC provenance drift: {provenance_path}")
    binding_kwargs = _binding_kwargs(
        evaluation_id="m6.1-mrpc-test",
        source_identifier=MRPC_SOURCE_IDENTIFIER,
        source_sha256=MRPC_SOURCE_CONTRACT.raw_sha256,
        source_rows=MRPC_SOURCE_CONTRACT.row_count,
        prediction_path=prediction_path,
        prediction_rows=len(scores),
        report_path=report_path,
        protocol_path=protocol_path,
        model_manifest_path=manifest_path,
    )
    expected_binding = build_evaluation_binding(**binding_kwargs)  # type: ignore[arg-type]
    expected_binding["source_provenance"] = {
        "path": provenance_path.as_posix(),
        "sha256": _sha256_path(provenance_path),
    }
    if _read_text(binding_path, kind="MRPC binding") != dump(expected_binding):
        raise M61ReportError(f"evaluation binding drift: {binding_path}")

    if source_path is None:
        return "structural_only_metrics_not_reproduced"
    verified = verify_mrpc_source(source_path, scores, MRPC_SOURCE_CONTRACT)
    expected_report = dump(build_mrpc_report(verified, protocol, MRPC_SOURCE_CONTRACT))
    if _read_text(report_path, kind="MRPC report") != expected_report:
        raise M61ReportError(f"MRPC report drift: {report_path}")
    return "full_metrics_reproduced"


__all__ = [
    "MRPC_SOURCE_CONTRACT",
    "M61ReportError",
    "MrpcScore",
    "MrpcSourceContract",
    "ProbeScore",
    "VerifiedMrpcRow",
    "build_evaluation_binding",
    "build_mrpc_provenance",
    "build_mrpc_report",
    "build_probe_report",
    "check_evaluation_binding",
    "check_mrpc_artifacts",
    "check_probe_artifacts",
    "dump",
    "generate_mrpc_artifacts",
    "generate_probe_artifacts",
    "load_mrpc_scores",
    "load_probe_scores",
    "verify_mrpc_source",
]
