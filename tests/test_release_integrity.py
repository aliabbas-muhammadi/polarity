import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

import polarity.release_integrity as release_integrity
from polarity.release_integrity import (
    UNKNOWN_EXECUTION_VALUE,
    ExecutionIdentity,
    ReleaseProfileError,
    assert_installed_xp006_release_profile,
    assert_xp006_release_profile,
    execution_fingerprint,
)
from polarity.train.config import load_config

XP006_CONFIG = Path("configs/train/m3-3-deberta-fixed.yaml")
GITLEAKS_CONFIG = Path(".gitleaks.toml")
DEBERTA_PUBLIC_COMMIT = "".join(
    (
        "8ccc9b6f36199bec6961081d44eb72fb",
        "3f7353f3",
    )
)


def _xp006_execution(**changes: str) -> ExecutionIdentity:
    cfg = load_config(XP006_CONFIG)
    values = {
        "python": UNKNOWN_EXECUTION_VALUE,
        "platform": UNKNOWN_EXECUTION_VALUE,
        "torch": "2.6.0+cu124",
        "transformers": "4.57.6",
        "cuda": UNKNOWN_EXECUTION_VALUE,
        "base_revision": cfg.model.revision,
        "tokenizer_revision": cfg.model.tokenizer_revision,
    }
    values.update(changes)
    return ExecutionIdentity.model_validate(values)


def test_xp005_and_xp006_execution_fingerprints_are_deterministic_and_distinct():
    xp006 = _xp006_execution()
    xp005 = _xp006_execution(transformers="5.15.1")

    assert execution_fingerprint(xp006) == execution_fingerprint(_xp006_execution())
    assert execution_fingerprint(xp005) == (
        "pl1:0b96cd113cfc7b70f1b14ba3c693652d5dbccc49b9f174b7a11483d08462ad56"
    )
    assert execution_fingerprint(xp005) != execution_fingerprint(xp006)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python", "3.13.14"),
        ("platform", "Linux-x86_64"),
        ("torch", "2.13.0"),
        ("transformers", "5.15.1"),
        ("cuda", "12.4"),
        ("base_revision", "different-base"),
        ("tokenizer_revision", "different-tokenizer"),
    ],
)
def test_execution_fingerprint_covers_every_required_field(field: str, value: str):
    assert execution_fingerprint(_xp006_execution(**{field: value})) != execution_fingerprint(
        _xp006_execution()
    )


def test_historical_unknowns_are_explicit_and_blank_values_are_rejected():
    assert execution_fingerprint(_xp006_execution()) == (
        "pl1:bc55b62acce7235c6442b53e2b3cf1147241619cac801899d7993db7ee289cec"
    )
    with pytest.raises(ValidationError, match="use 'unknown:not-recorded'"):
        _xp006_execution(python=" ")


def test_xp006_release_profile_accepts_recorded_runtime():
    assert_xp006_release_profile(
        transformers_version="4.57.6",
        torch_version="2.6.0+cu124",
    )


def test_xp006_release_profile_rejects_known_bad_transformers():
    with pytest.raises(
        ReleaseProfileError,
        match=r"xp-006 requires transformers exactly 4.57.6; found 5.15.1",
    ):
        assert_xp006_release_profile(
            transformers_version="5.15.1",
            torch_version="2.6.0+cu124",
        )


def test_xp006_release_profile_rejects_torch_2_13():
    with pytest.raises(
        ReleaseProfileError,
        match=r"xp-006 requires torch >=2\.6,<2\.7; found 2\.13\.0",
    ):
        assert_xp006_release_profile(
            transformers_version="4.57.6",
            torch_version="2.13.0",
        )


def test_train_extra_is_the_exact_xp006_profile_without_narrowing_models_extra():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert "transformers==4.57.6" in extras["train"]
    assert "torch>=2.6,<2.7" in extras["train"]
    assert "transformers>=4.40" in extras["models"]
    assert "torch>=2" in extras["models"]


def test_gitleaks_allowlist_is_exactly_scoped_to_the_public_tokenizer_revision():
    config = tomllib.loads(GITLEAKS_CONFIG.read_text(encoding="utf-8"))

    assert set(config) == {"title", "minVersion", "extend", "allowlists"}
    assert config["minVersion"] == "v8.25.0"
    assert config["extend"] == {"useDefault": True}
    assert len(config["allowlists"]) == 1
    allowlist = config["allowlists"][0]
    assert set(allowlist) == {
        "description",
        "targetRules",
        "condition",
        "regexTarget",
        "regexes",
        "paths",
    }
    assert allowlist["targetRules"] == ["generic-api-key"]
    assert allowlist["condition"] == "AND"
    assert allowlist["regexTarget"] == "line"
    assert allowlist["regexes"] == [
        (r'^\s*"?tokenizer_revision"?\s*:\s*"?' + DEBERTA_PUBLIC_COMMIT + r'"?\s*,?\s*$')
    ]
    assert "[0-9a-f]" not in allowlist["regexes"][0]
    assert len(allowlist["paths"]) == 5
    assert set(allowlist["paths"]) == {
        r"^configs/train/m3-1-deberta-v3-base\.yaml$",
        r"^configs/train/m3-2-deberta-fp32\.yaml$",
        r"^configs/train/m3-3-deberta-fixed\.yaml$",
        r"^configs/train/m3-deberta-v3-base\.yaml$",
        r"^reports/m6/model-manifest-xp-006-deberta-transformers-fixed\.json$",
    }


def test_installed_profile_guard_uses_package_metadata_without_importing_runtimes(monkeypatch):
    versions = {"transformers": "4.57.6", "torch": "2.6.0+cu124"}
    monkeypatch.setattr(release_integrity.metadata, "version", versions.__getitem__)

    assert assert_installed_xp006_release_profile() == versions
