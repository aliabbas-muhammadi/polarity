"""Polarity CLI (M0 subset).

Only the commands M0 needs to prove the foundation are implemented: ``data verify``,
``data inspect``, ``audit``, and ``evaluate [--check]``. Future commands (``train``,
``baseline``, ``calibrate``, ``compare``) are documented in the README until their
milestone — not scaffolded as dead stubs. Paths are relative to the current directory, which
is the repo root under ``uv run`` and in CI.
"""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from polarity import artifact as artifact_mod
from polarity import release_integrity
from polarity.baselines import report as m2report
from polarity.baselines import run as m2run
from polarity.data.build import BuildError, build
from polarity.data.fetch import fetch
from polarity.data.loaders import golden_to_pairs, load_golden, load_pairs
from polarity.data.registry import read_provenance
from polarity.data.sources import SOURCES, get_source
from polarity.eval import leakage, metrics
from polarity.eval.leakage import SealedLeakageError
from polarity.fingerprint import sha256_file
from polarity.textnorm import normalized_sha
from polarity.train import config as trainconfig
from polarity.train import report as m3report

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Polarity — cache-equivalence foundation (M0)."
)
data_app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Dataset + snapshot utilities."
)
app.add_typer(data_app, name="data")
baseline_app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="M2 no-training baselines."
)
app.add_typer(baseline_app, name="baseline")
train_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="M3 controlled DeBERTa fine-tune (owner-gated).",
)
app.add_typer(train_app, name="train")
diagnose_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="M5 training-signal diagnostics (author-run; $0, local).",
)
app.add_typer(diagnose_app, name="diagnose")
transfer_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="M6 sealed-Veritas transfer arm (one-way; author-run).",
)
app.add_typer(transfer_app, name="transfer")
artifact_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Torch-free checkpoint bundle integrity checks.",
)
app.add_typer(artifact_app, name="artifact")

M3_CONFIG = Path("configs/train/m3-deberta-v3-base.yaml")
M3_3_CONFIG = Path("configs/train/m3-3-deberta-fixed.yaml")
M6_REPORTS = Path("reports/m6")
M6_PROTOCOL = M6_REPORTS / "protocol-xp-006-deberta-transformers-fixed.json"
M6_MODEL_MANIFEST = M6_REPORTS / "model-manifest-xp-006-deberta-transformers-fixed.json"
M6_VERITAS_PRED = Path("predictions/m6/veritas/xp-006-deberta-transformers-fixed.jsonl")

FIXTURES = Path("data/fixtures/pairs.jsonl")
SNAPSHOT_DIR = Path("data/snapshots/veritas-golden")
SNAPSHOT = SNAPSHOT_DIR / "golden.json"
PROVENANCE = SNAPSHOT_DIR / "PROVENANCE.json"
PREDICTIONS = Path("predictions/synthetic.jsonl")
REPORT = Path("reports/metrics.json")
MANIFESTS = Path("manifests")
REPORTS_M1 = Path("reports/m1")


@artifact_app.command("verify")
def artifact_verify(
    manifest: Annotated[Path, typer.Option(help="committed model manifest")],
    artifact_dir: Annotated[Path, typer.Option(help="local checkpoint bundle directory")],
) -> None:
    """Fail closed unless every required checkpoint file matches its manifest."""
    try:
        result = artifact_mod.verify_artifact_bundle(manifest, artifact_dir)
    except artifact_mod.ArtifactVerificationError as exc:
        typer.echo(f"VERIFY FAILED: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK: verified {result.files_verified} bundle files ({result.bundle_fingerprint})")


@data_app.command("verify")
def data_verify() -> None:
    """Validate fixtures + sealed snapshot schema, checksum, and provenance consistency."""
    pairs = load_pairs(FIXTURES)
    golden = load_golden(SNAPSHOT)
    record = read_provenance(PROVENANCE)
    actual = sha256_file(SNAPSHOT)

    typer.echo(f"fixtures: {len(pairs)} pairs")
    typer.echo(
        f"snapshot: {len(golden)} pairs (provenance rows={record.rows}, sealed={record.sealed})"
    )

    ok = True
    if actual != record.sha256:
        typer.echo(f"CHECKSUM MISMATCH: {actual} != provenance {record.sha256}")
        ok = False
    if len(golden) != record.rows:
        typer.echo(f"ROW MISMATCH: snapshot {len(golden)} != provenance {record.rows}")
        ok = False
    if not ok:
        raise typer.Exit(code=1)
    typer.echo("OK: schema valid, checksum matches, provenance consistent")


@data_app.command("inspect")
def data_inspect(pair_id: str) -> None:
    """Print one pair (from the fixtures or the sealed snapshot)."""
    for p in load_pairs(FIXTURES):
        if p.id == pair_id:
            typer.echo(p.model_dump_json(indent=2))
            return
    for g in load_golden(SNAPSHOT):
        if g.id == pair_id:
            typer.echo(g.model_dump_json(indent=2))
            return
    typer.echo(f"not found: {pair_id}")
    raise typer.Exit(code=1)


def _load_manifest_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    out: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            out.append(json.loads(line))
    return out


@app.command("audit")
def audit() -> None:
    """Isolation audit — keyless. Fixtures vs the sealed snapshot, and (if built) the M1
    manifests vs the sealed Veritas set + the NaN-NLI eval probe, from committed hashes only."""
    ok = True

    m0 = leakage.audit(load_pairs(FIXTURES), golden_to_pairs(load_golden(SNAPSHOT)))
    typer.echo(f"fixtures-vs-snapshot: {'OK' if m0.ok else 'LEAK'}")
    ok = ok and m0.ok

    if MANIFESTS.exists():
        golden = load_golden(SNAPSHOT)
        veritas_shas = {normalized_sha(g.base) for g in golden} | {
            normalized_sha(g.probe) for g in golden
        }
        veritas_pairs = {(normalized_sha(g.base), normalized_sha(g.probe)) for g in golden}
        eval_entries = _load_manifest_jsonl(MANIFESTS / "eval" / "nan-nli.jsonl")
        eval_shas = {e["a_sha"] for e in eval_entries} | {e["b_sha"] for e in eval_entries}
        protected = veritas_shas | eval_shas

        train_entries = [
            e
            for name in ("train.jsonl", "validation.jsonl", "internal_test.jsonl")
            for e in _load_manifest_jsonl(MANIFESTS / "with-mnli" / name)
        ]
        collisions = [
            e["example_id"]
            for e in train_entries
            if e["a_sha"] in protected or e["b_sha"] in protected
        ]
        reverse = [
            e["example_id"] for e in train_entries if (e["b_sha"], e["a_sha"]) in veritas_pairs
        ]
        fam_splits: dict[str, set[str]] = defaultdict(set)
        for e in train_entries:
            fam_splits[e["family"]].add(e["split"])
        cross = sorted(f for f, s in fam_splits.items() if len(s) > 1)

        m1_ok = not collisions and not reverse and not cross
        typer.echo(
            f"manifests-vs-sealed: {'OK' if m1_ok else 'LEAK'} "
            f"(n_train={len(train_entries)}, collisions={len(collisions)}, "
            f"reverse={len(reverse)}, cross_split_families={len(cross)})"
        )
        ok = ok and m1_ok

    if not ok:
        typer.echo("LEAKAGE DETECTED")
        raise typer.Exit(code=1)
    typer.echo("OK: all isolation checks pass")


@data_app.command("fetch")
def data_fetch(
    source: Annotated[str, typer.Argument(help="source id, or 'all'")] = "all",
) -> None:
    """Download the cleared sources at their pinned revisions (network; author-run)."""
    ids = list(SOURCES) if source == "all" else [source]
    for sid in ids:
        typer.echo(f"fetching {sid} ...")
        dest = fetch(get_source(sid))
        typer.echo(f"  -> {dest}")


@data_app.command("build")
def data_build(
    limit_per_source: Annotated[int, typer.Option(help="deterministic per-source cap")] = 15000,
    seed: Annotated[int, typer.Option()] = 42,
    exclude_sealed_collisions: Annotated[
        bool,
        typer.Option(help="drop families colliding with the sealed/probe sets (documented)"),
    ] = False,
) -> None:
    """Build manifests + reports from fetched raw data (deterministic; fails closed on leakage)."""
    try:
        summary = build(
            limit_per_source=limit_per_source,
            seed=seed,
            exclude_sealed_collisions=exclude_sealed_collisions,
        )
    except (BuildError, SealedLeakageError) as exc:
        typer.echo(f"BUILD BLOCKED: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@data_app.command("stats")
def data_stats() -> None:
    """Print the committed M1 build summary + the two-recipe comparison."""
    printed = False
    for name in ("build-summary.json", "recipe-comparison.json"):
        path = REPORTS_M1 / name
        if path.exists():
            typer.echo(f"== {name} ==")
            typer.echo(path.read_text(encoding="utf-8").rstrip())
            printed = True
    if not printed:
        typer.echo("no M1 reports found — run `polarity data build` first")
        raise typer.Exit(code=1)


@baseline_app.command("list")
def baseline_list() -> None:
    """List the M2 baselines."""
    for name in m2run.BASELINES:
        typer.echo(name)


@baseline_app.command("run")
def baseline_run(name: str) -> None:
    """Author-run inference for one baseline (needs the `models` extra + local data/raw)."""
    m2run.run(name)
    typer.echo(f"wrote predictions/m2/{name}/{{validation,internal_test}}.jsonl")


@baseline_app.command("report")
def baseline_report(
    check: Annotated[
        bool, typer.Option("--check", help="verify reports/m2 reproduce byte-for-byte")
    ] = False,
) -> None:
    """Deterministic keyless M2 report from committed predictions (drift-gated)."""
    if check:
        drift = m2report.check_all()
        if drift:
            typer.echo(f"DRIFT: {drift}")
            raise typer.Exit(code=1)
        typer.echo("OK: reports/m2 reproduces byte-for-byte")
        return
    files = m2report.build_all()
    typer.echo(f"wrote {len(files)} reports to reports/m2/")


@train_app.command("fingerprint")
def train_fingerprint(
    config: Annotated[Path, typer.Option(help="training config")] = M3_CONFIG,
) -> None:
    """Print the experiment fingerprint + verify live manifests match the approved shas."""
    cfg = trainconfig.load_config(config)
    typer.echo(f"experiment_id: {cfg.experiment_id}")
    typer.echo(f"recipe_fingerprint: {trainconfig.recipe_fingerprint(cfg)}")
    try:
        live = trainconfig.assert_manifests_match(cfg)
    except trainconfig.FingerprintMismatch as exc:
        typer.echo(f"MANIFEST MISMATCH: {exc}")
        raise typer.Exit(code=1) from exc
    for k, v in live.items():
        typer.echo(f"  {k}: {v} (matches approved)")
    typer.echo("OK: manifests match the approved fingerprint")


@train_app.command("tokstats")
def train_tokstats(
    config: Annotated[Path, typer.Option(help="training config")] = M3_CONFIG,
    split: Annotated[str, typer.Option(help="train|validation|internal_test")] = "train",
) -> None:
    """Measure the joint pair-token length distribution (author-run; needs the `train` extra)."""
    from polarity.baselines.materialize import materialize
    from polarity.train import data, model

    cfg = trainconfig.load_config(config)
    manifest = Path(
        getattr(cfg.data, f"{'internal_test' if split == 'internal_test' else split}_manifest")
    )
    pairs = materialize(manifest)
    tok = model.load_tokenizer(cfg)
    typer.echo(json.dumps(data.length_stats(pairs, tok), indent=2, sort_keys=True))


@train_app.command("run")
def train_run(
    config: Annotated[Path, typer.Option(help="training config")] = M3_CONFIG,
    device: Annotated[str, typer.Option(help="cpu | cuda")] = "cpu",
    run_dir: Annotated[Path, typer.Option()] = Path("runs/xp-003"),
) -> None:
    """PAID training run (author-run, GPU). Refuses on manifest mismatch; owner cap only."""
    from polarity import diagnose
    from polarity.train import runner
    from polarity.train.cost import CostGuard

    cfg = trainconfig.load_config(config)
    trainconfig.assert_manifests_match(cfg)  # refuse-to-start gate
    if cfg.experiment_id == "xp-006-deberta-transformers-fixed":
        release_versions = release_integrity.assert_installed_xp006_release_profile()
        typer.echo(
            "xp-006 release profile: "
            f"transformers {release_versions['transformers']}, torch {release_versions['torch']}"
        )
    tv = diagnose.assert_transformers_supported()  # M5: refuse a broken transformers version
    typer.echo(f"transformers {tv} (DeBERTa-fine-tuning-safe)")
    guard = CostGuard(
        cfg.cost.max_runtime_hours,
        cfg.cost.max_estimated_cost_aud,
        cfg.cost.planning_hourly_rate_aud,
    )
    summary = runner.train(
        cfg,
        train_manifest=Path(cfg.data.train_manifest),
        val_manifest=Path(cfg.data.validation_manifest),
        run_dir=run_dir,
        device=device,
        guard=guard,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@train_app.command("predict")
def train_predict(
    checkpoint: Annotated[Path, typer.Argument(help="frozen checkpoint dir")],
    config: Annotated[Path, typer.Option(help="training config")] = M3_CONFIG,
    device: Annotated[str, typer.Option(help="cpu | cuda (match the training device)")] = "cpu",
) -> None:
    """Score val + internal_test with the frozen checkpoint -> predictions/m3/<id>/ (author-run)."""
    from polarity.train import predict

    cfg = trainconfig.load_config(config)
    out = predict.PRED_ROOT / cfg.experiment_id
    for split, manifest in (
        ("validation", Path(cfg.data.validation_manifest)),
        ("internal_test", Path(cfg.data.internal_test_manifest)),
    ):
        n = predict.predict_split(
            config, checkpoint, manifest, out / f"{split}.jsonl", device=device
        )
        typer.echo(f"wrote {out / f'{split}.jsonl'} ({n} pairs)")


@train_app.command("report")
def train_report(
    config: Annotated[Path, typer.Option(help="training config")] = M3_CONFIG,
    check: Annotated[
        bool, typer.Option("--check", help="verify reports/m3 reproduce byte-for-byte")
    ] = False,
) -> None:
    """Deterministic keyless M3 report from committed predictions (drift-gated)."""
    if check:
        drift = m3report.check_all(config)
        if drift:
            typer.echo(f"DRIFT: {drift}")
            raise typer.Exit(code=1)
        typer.echo("OK: reports/m3 reproduces byte-for-byte")
        return
    files = m3report.build_all(config)
    typer.echo(f"wrote {len(files)} reports to reports/m3/")


@train_app.command("smoke")
def train_smoke(
    run_dir: Annotated[Path, typer.Option()] = Path("runs/smoke-m3"),
    config: Annotated[Path, typer.Option()] = M3_CONFIG,
) -> None:
    """Local CPU smoke: tiny end-to-end run. NOT a result (author-run; needs `train` extra)."""
    from polarity.train.smoke import run_smoke

    evidence = run_smoke(config, run_dir)
    typer.echo(json.dumps(evidence, indent=2, sort_keys=True))


@diagnose_app.command("overfit")
def diagnose_overfit(
    config: Annotated[Path, typer.Option(help="training config")] = Path(
        "configs/train/m3-2-deberta-fp32.yaml"
    ),
    kind: Annotated[str, typer.Option(help="trivial | real")] = "trivial",
    device: Annotated[str, typer.Option(help="cpu | mps | cuda")] = "cpu",
    steps: Annotated[int, typer.Option()] = 150,
    lr: Annotated[float, typer.Option()] = 1e-5,
    n_per_class: Annotated[int, typer.Option()] = 8,
    json_out: Annotated[Path | None, typer.Option(help="write the machine-readable report")] = None,
) -> None:
    """Overfit-one-batch (M5): can the real pipeline memorize a tiny labelled set? (author-run)."""
    from polarity import diagnose

    cfg = trainconfig.load_config(config)
    pairs = (
        diagnose.trivial_pairs(n_per_class)
        if kind == "trivial"
        else diagnose.real_balanced_subset(Path(cfg.data.train_manifest), n_per_class)
    )
    result = diagnose.overfit_one_batch(cfg, pairs, steps=steps, lr=lr, device=device)
    text = json.dumps(result.as_dict(), indent=2, sort_keys=True)
    typer.echo(text)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n", encoding="utf-8")
    if result.verdict != "OVERFIT_PASS":
        raise typer.Exit(code=1)


@transfer_app.command("reproduce")
def transfer_reproduce(
    new_internal_test: Annotated[Path, typer.Argument(help="candidate internal_test predictions")],
    committed: Annotated[str, typer.Option(help="committed M3.3 run id")] = (
        "xp-006-deberta-transformers-fixed"
    ),
) -> None:
    """Compare aligned scores; this does not verify checkpoint identity or retraining."""
    from polarity import transfer as tr

    try:
        candidate = tr.load_preds(new_internal_test, kind="candidate predictions")
        committed_predictions = tr.load_preds(
            Path("predictions/m3") / committed / "internal_test.jsonl",
            kind="committed predictions",
        )
        protocol = tr.load_protocol(M6_PROTOCOL)
        if protocol.experiment_id != committed:
            raise tr.ReproductionError(
                f"frozen protocol experiment_id {protocol.experiment_id} does not match "
                f"committed run {committed}"
            )
        verdict = tr.reproduction_classify(
            candidate,
            committed_predictions,
            protocol=protocol,
        )
    except tr.ReproductionError as exc:
        typer.echo(f"REPRODUCTION FAILED: {exc}")
        raise typer.Exit(code=1) from exc
    M6_REPORTS.mkdir(parents=True, exist_ok=True)
    (M6_REPORTS / f"reproduction-{committed}.json").write_text(tr.dump(verdict), encoding="utf-8")
    typer.echo(tr.dump(verdict))
    if verdict["verdict"] == "MATERIAL_REPRODUCTION_DRIFT":
        typer.echo("MATERIAL DRIFT — do NOT proceed to Veritas")
        raise typer.Exit(code=1)


@transfer_app.command("freeze")
def transfer_freeze(
    checkpoint: Annotated[Path, typer.Argument(help="frozen checkpoint dir")],
    val_preds: Annotated[Path, typer.Argument(help="retrained validation predictions")],
    config: Annotated[Path, typer.Option()] = M3_3_CONFIG,
) -> None:
    """Freeze thresholds + calibrator from validation; write protocol + manifest (pre-Veritas)."""
    from polarity import transfer as tr

    cfg = trainconfig.load_config(config)
    bundle = artifact_mod.build_artifact_bundle(checkpoint)
    sha = str(bundle["files"]["model.safetensors"]["sha256"])
    release_versions = release_integrity.assert_installed_xp006_release_profile()
    import torch

    execution = release_integrity.ExecutionIdentity(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=release_versions["torch"],
        transformers=release_versions["transformers"],
        cuda=str(torch.version.cuda) if torch.version.cuda is not None else "none",
        base_revision=cfg.model.revision,
        tokenizer_revision=cfg.model.tokenizer_revision,
    )
    recipe_fp = trainconfig.recipe_fingerprint(cfg)
    identity = artifact_mod.ArtifactIdentity(
        checkpoint_sha256=sha,
        artifact_bundle_fingerprint=artifact_mod.artifact_bundle_fingerprint(bundle),
        recipe_fingerprint=recipe_fp,
        execution_fingerprint=release_integrity.execution_fingerprint(execution),
    )
    proto = tr.derive_protocol(
        tr.load_preds(val_preds),
        experiment_id=cfg.experiment_id,
        artifact_identity=identity,
        isotonic_min_obs=cfg.calibration.isotonic_min_obs,
    )
    M6_REPORTS.mkdir(parents=True, exist_ok=True)
    protocol_path = M6_REPORTS / f"protocol-{cfg.experiment_id}.json"
    protocol_path.write_text(tr.dump(proto.model_dump(mode="json")), encoding="utf-8")
    manifest = {
        "experiment_id": cfg.experiment_id,
        "model_fingerprint": recipe_fp,
        "recipe_fingerprint": recipe_fp,
        "execution": execution.model_dump(mode="json"),
        "execution_fingerprint": identity.execution_fingerprint,
        "artifact_bundle": bundle,
        "artifact_bundle_fingerprint": identity.artifact_bundle_fingerprint,
        "checkpoint_sha256": sha,
        "checkpoint_dir": str(checkpoint),
        "model_revision": cfg.model.revision,
        "tokenizer_revision": cfg.model.tokenizer_revision,
        "manifest_sha256": cfg.data.approved_sha256,
        "precision": cfg.training.precision,
        "seed": cfg.training.seed,
        "frozen_thresholds": proto.thresholds,
        "calibrator_method": proto.calibrator_method,
        "protocol_sha256": sha256_file(protocol_path),
    }
    (M6_REPORTS / f"model-manifest-{cfg.experiment_id}.json").write_text(
        tr.dump(manifest), encoding="utf-8"
    )
    typer.echo(f"froze protocol + manifest for {cfg.experiment_id} (checkpoint sha256 {sha[:16]}…)")
    typer.echo(tr.dump(proto.thresholds))


@transfer_app.command("veritas")
def transfer_veritas(
    checkpoint: Annotated[Path, typer.Argument(help="frozen checkpoint dir")],
    config: Annotated[Path, typer.Option()] = M3_3_CONFIG,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Score the 75 sealed Veritas pairs ONCE with the frozen checkpoint + protocol (author-run)."""
    from polarity import transfer as tr

    try:
        verified_release = artifact_mod.verify_artifact_release(M6_MODEL_MANIFEST, checkpoint)
        manifest_context = verified_release.release_context
    except artifact_mod.ArtifactVerificationError as exc:
        typer.echo(f"REFUSING: artifact bundle verification failed: {exc}")
        raise typer.Exit(code=1) from exc
    try:
        proto = tr.load_protocol(
            M6_PROTOCOL,
            expected_sha256=manifest_context.protocol_sha256,
        )
    except tr.ReproductionError as exc:
        typer.echo(f"REFUSING: invalid frozen protocol: {exc}")
        raise typer.Exit(code=1) from exc
    try:
        cfg = trainconfig.load_config(config)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        typer.echo(f"REFUSING: invalid scoring config: {exc}")
        raise typer.Exit(code=1) from exc
    try:
        calibrator = tr.validate_veritas_scoring_context(cfg, proto, manifest_context)
    except tr.ScoringPreflightError as exc:
        typer.echo(f"REFUSING: {exc}")
        raise typer.Exit(code=1) from exc
    try:
        release_integrity.assert_installed_xp006_release_profile()
    except release_integrity.ReleaseProfileError as exc:
        typer.echo(f"REFUSING: xp-006 release profile mismatch: {exc}")
        raise typer.Exit(code=1) from exc
    if tr.checkpoint_sha256(checkpoint) != proto.checkpoint_sha256:
        typer.echo("REFUSING: checkpoint sha256 does not match the frozen protocol")
        raise typer.Exit(code=1)
    scores = tr.score_golden(checkpoint, cfg, calibrator, device=device)
    tr.write_scores(scores, M6_VERITAS_PRED)
    typer.echo(f"scored {len(scores)} Veritas pairs -> {M6_VERITAS_PRED}")


@transfer_app.command("report")
def transfer_report_cmd(
    check: Annotated[bool, typer.Option("--check", help="verify reports/m6 reproduces")] = False,
) -> None:
    """Keyless M6 transfer report from committed Veritas scores + frozen protocol (drift-gated)."""
    from polarity import transfer as tr

    proto = tr.FrozenProtocol.model_validate_json(M6_PROTOCOL.read_text(encoding="utf-8"))
    scores = tr.load_scores(M6_VERITAS_PRED)
    if check:
        drift = tr.check_report(scores, proto)
        if drift:
            typer.echo(f"DRIFT: {drift}")
            raise typer.Exit(code=1)
        typer.echo("OK: reports/m6 reproduces byte-for-byte")
        return
    files = tr.build_report_files(scores, proto)
    for fname, text in files.items():
        (M6_REPORTS / fname).parent.mkdir(parents=True, exist_ok=True)
        (M6_REPORTS / fname).write_text(text, encoding="utf-8")
    typer.echo(f"wrote {len(files)} report(s) to reports/m6/")


@transfer_app.command("probe-report")
def transfer_probe_report(
    check: Annotated[
        bool,
        typer.Option("--check", help="verify the M6.1 probe report and bindings"),
    ] = False,
) -> None:
    """Reproduce the author-created M6.1 probe report without model inference."""
    from polarity import m61_report

    try:
        if check:
            m61_report.check_probe_artifacts()
            typer.echo("OK: M6.1 probe report and evaluation bindings reproduce byte-for-byte")
            return
        report, probe_binding, veritas_binding = m61_report.generate_probe_artifacts()
    except m61_report.M61ReportError as exc:
        typer.echo(f"VERIFY FAILED: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {report}, {probe_binding}, and {veritas_binding}")


@transfer_app.command("mrpc-report")
def transfer_mrpc_report(
    check: Annotated[
        bool,
        typer.Option("--check", help="verify structural integrity or reproduce with --source"),
    ] = False,
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            help="locally obtained official MRPC test JSON (raw text remains ignored)",
        ),
    ] = None,
) -> None:
    """Validate MRPC structure keylessly; supply official source to reproduce metrics."""
    from polarity import m61_report

    if not check and source is None:
        typer.echo("REFUSING: --source is required to generate the MRPC report")
        raise typer.Exit(code=1)
    try:
        if check:
            result = m61_report.check_mrpc_artifacts(source)
            if result == "full_metrics_reproduced":
                typer.echo("OK: M6.1 MRPC metrics reproduced from checksum-verified local source")
            else:
                typer.echo(
                    "OK: M6.1 MRPC artifacts structurally verified; metrics NOT reproduced "
                    "(provide --source PATH)"
                )
            return
        assert source is not None
        report, provenance, binding = m61_report.generate_mrpc_artifacts(source)
    except m61_report.M61ReportError as exc:
        typer.echo(f"VERIFY FAILED: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {report}, {provenance}, and {binding}")


@app.command("evaluate")
def evaluate(
    check: Annotated[
        bool,
        typer.Option("--check", help="Verify the committed report reproduces byte-for-byte."),
    ] = False,
) -> None:
    """Compute the metrics report from committed predictions (keyless, no GPU, no network)."""
    preds = metrics.load_predictions(PREDICTIONS)
    text = metrics.serialize_report(metrics.build_report(preds))
    if check:
        if text != REPORT.read_text(encoding="utf-8"):
            typer.echo("DRIFT: recomputed metrics differ from committed reports/metrics.json")
            raise typer.Exit(code=1)
        typer.echo("OK: reports/metrics.json reproduces byte-for-byte")
        return
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {REPORT} ({len(preds)} predictions)")


__all__ = ["app"]
