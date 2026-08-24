import hashlib
from pathlib import Path

from polarity.fingerprint import SCHEME, canonical_bytes, content_hash, sha256_file


def test_hash_is_order_independent():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_hash_is_sensitive_to_value():
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_scheme_prefix():
    assert content_hash({"x": 1}).startswith(f"{SCHEME}:")


def test_canonical_bytes_sorted_compact():
    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_sha256_file(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()
