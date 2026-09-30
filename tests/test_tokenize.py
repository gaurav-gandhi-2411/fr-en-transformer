from __future__ import annotations

# Tokenizer + shard tests (spec §4, §12): decode(encode(x)) round-trip and byte-fallback on
# unseen characters use the real, committed `tokenizer/spm.model` -- skipped, with an explicit
# reason (the only skip in this suite, same pattern as tests/test_leakage.py's real-committed-
# data test), when it hasn't been trained yet in this checkout. Everything else (sampling,
# length-bucket histograms, shard writer/reader round-trip against the unmodified
# `nmt.data.loader.ShardDataset`, 256-cap drop counting, manifest sha256 tamper detection) is a
# fast, network-free unit test against synthetic/tiny data.
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sentencepiece as spm

from nmt.data.hub_data import verify_shards_against_manifest
from nmt.data.loader import ShardDataset
from nmt.data.normalize import normalize_text
from nmt.data.tokenize import (
    BOS_ID,
    EOS_ID,
    LENGTH_CAP,
    PAD_ID,
    UNK_ID,
    _length_bucket_label,
    _length_histogram,
    _write_shard_file,
    sample_tokenizer_training_sentences,
    write_split_shards,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPM_MODEL_PATH = REPO_ROOT / "tokenizer" / "spm.model"


@pytest.fixture
def real_sp() -> spm.SentencePieceProcessor:
    if not SPM_MODEL_PATH.is_file():
        pytest.skip(f"{SPM_MODEL_PATH} missing -- run `python -m nmt.data.tokenize` first")
    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_MODEL_PATH))
    return sp


# --------------------------------------------------------------------------------------
# Special ids (contract; no tokenizer needed)
# --------------------------------------------------------------------------------------


def test_special_id_constants_match_plan_contract() -> None:
    assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)


def test_real_tokenizer_special_ids_match_contract(real_sp: spm.SentencePieceProcessor) -> None:
    assert real_sp.pad_id() == PAD_ID
    assert real_sp.unk_id() == UNK_ID
    assert real_sp.bos_id() == BOS_ID
    assert real_sp.eos_id() == EOS_ID


# --------------------------------------------------------------------------------------
# Round-trip on the real, committed tokenizer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "Bonjour le monde, comment ça va ?",
        "L'œuvre de Moucheboœuf a été publiée en 1875.",
        "The quick brown fox jumps over 12 lazy dogs (again!).",
        "Qu'est-ce que c'est ? \"Rien du tout\", dit-elle.",
        "Le prix est de 12,50 € pour 3 kg.",
    ],
)
def test_roundtrip_decode_encode_is_identity(
    real_sp: spm.SentencePieceProcessor, sentence: str
) -> None:
    normalized = normalize_text(sentence)
    ids = real_sp.encode(normalized, out_type=int)
    assert real_sp.decode(ids) == normalized


@pytest.mark.parametrize(
    "unseen_char",
    [
        "😀",  # emoji
        "",  # Private Use Area -- by Unicode design, never appears in real text
        "\U000f0000",  # Supplementary PUA-A -- ditto, guaranteed unseen regardless of corpus
    ],
)
def test_byte_fallback_on_unseen_character_no_unk_and_roundtrips(
    real_sp: spm.SentencePieceProcessor, unseen_char: str
) -> None:
    text = normalize_text(f"test {unseen_char} test")
    ids = real_sp.encode(text, out_type=int)
    assert UNK_ID not in ids  # byte fallback covers it, never <unk>
    assert real_sp.decode(ids) == text
    pieces = real_sp.encode(text, out_type=str)
    assert any(p.startswith("<0x") for p in pieces)


# --------------------------------------------------------------------------------------
# Tokenizer-training sentence sampling (pure, no SentencePiece involved)
# --------------------------------------------------------------------------------------


def test_sample_uses_all_sentences_when_request_exceeds_population() -> None:
    rows = [{"fr": f"fr_{i}", "en": f"en_{i}"} for i in range(10)]
    result = sample_tokenizer_training_sentences(rows, seed=1, per_lang=1000)
    assert sorted(result.fr_sentences) == sorted(r["fr"] for r in rows)
    assert sorted(result.en_sentences) == sorted(r["en"] for r in rows)
    assert result.stats["used_fr_sentences"] == 10
    assert result.stats["used_en_sentences"] == 10
    assert result.stats["requested_sentences_per_lang"] == 1000
    assert "exceeds" in result.stats["deviation_note"].lower()


def test_sample_is_deterministic_and_subsamples_when_population_exceeds_request() -> None:
    rows = [{"fr": f"fr_{i}", "en": f"en_{i}"} for i in range(500)]
    r1 = sample_tokenizer_training_sentences(rows, seed=42, per_lang=100)
    r2 = sample_tokenizer_training_sentences(rows, seed=42, per_lang=100)
    assert r1.fr_sentences == r2.fr_sentences
    assert r1.en_sentences == r2.en_sentences
    assert len(r1.fr_sentences) == 100
    assert len(r1.en_sentences) == 100
    assert len(set(r1.fr_sentences)) == 100  # sample, no repeats
    r3 = sample_tokenizer_training_sentences(rows, seed=7, per_lang=100)
    assert r3.fr_sentences != r1.fr_sentences  # different seed, different sample


# --------------------------------------------------------------------------------------
# Length bucket histograms
# --------------------------------------------------------------------------------------


def test_length_bucket_label_boundaries() -> None:
    assert _length_bucket_label(0) == "0-10"
    assert _length_bucket_label(10) == "0-10"
    assert _length_bucket_label(11) == "11-20"
    assert _length_bucket_label(20) == "11-20"
    assert _length_bucket_label(21) == "21-40"
    assert _length_bucket_label(160) == "81-160"
    assert _length_bucket_label(161) == "161-256"
    assert _length_bucket_label(256) == "161-256"
    assert _length_bucket_label(257) == ">256"
    assert _length_bucket_label(1000) == ">256"


def test_length_histogram_counts_every_bucket() -> None:
    hist = _length_histogram([5, 15, 15, 30, 300])
    assert hist["0-10"] == 1
    assert hist["11-20"] == 2
    assert hist["21-40"] == 1
    assert hist[">256"] == 1
    assert hist["41-80"] == 0
    assert sum(hist.values()) == 5


# --------------------------------------------------------------------------------------
# Shard writer/reader round-trip against the (unmodified) loader.ShardDataset
# --------------------------------------------------------------------------------------


def test_write_shard_file_roundtrips_through_sharddataset(tmp_path: Path) -> None:
    split_dir = tmp_path / "shards" / "toy"
    ids = ["ex_0", "ex_1", "ex_2"]
    src = [
        np.array([10, 11, 12], dtype=np.uint16),
        np.array([20], dtype=np.uint16),
        np.array([30, 31], dtype=np.uint16),
    ]
    tgt = [
        np.array([40, 41], dtype=np.uint16),
        np.array([50, 51, 52], dtype=np.uint16),
        np.array([60], dtype=np.uint16),
    ]
    split_dir.mkdir(parents=True)
    _write_shard_file(split_dir / "shard_00000.npz", ids, src, tgt)

    dataset = ShardDataset(tmp_path / "shards", "toy")
    assert len(dataset) == 3
    assert dataset.has_tgt
    for i in range(3):
        s, t, ex_id = dataset.get(i)
        assert (s == src[i]).all()
        assert t is not None and (t == tgt[i]).all()
        assert ex_id == ids[i]
        assert dataset.src_len(i) == len(src[i])
        assert dataset.tgt_len(i) == len(tgt[i])


def test_write_shard_file_no_target_side_matches_test_split_contract(tmp_path: Path) -> None:
    split_dir = tmp_path / "shards" / "test"
    ids = ["t_0", "t_1"]
    src = [np.array([1, 2, 3], dtype=np.uint16), np.array([4], dtype=np.uint16)]
    split_dir.mkdir(parents=True)
    _write_shard_file(split_dir / "shard_00000.npz", ids, src, None)

    dataset = ShardDataset(tmp_path / "shards", "test")
    assert not dataset.has_tgt
    s, t, ex_id = dataset.get(0)
    assert t is None
    assert ex_id == "t_0"
    with pytest.raises(ValueError, match="no target side"):
        dataset.tgt_len(0)


# --------------------------------------------------------------------------------------
# `write_split_shards`: cap-drop logic (train) vs. exceeds-only counting (eval/dev/test),
# and multi-shard chunking -- against a minimal fake SentencePieceProcessor stand-in so the
# cap-boundary behavior is tested in isolation from real BPE segmentation.
# --------------------------------------------------------------------------------------


class _FakeSP:
    """Minimal stand-in for `spm.SentencePieceProcessor`: `write_split_shards` only calls
    `.encode(texts, out_type=int)`, so that's all this needs to implement."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self._mapping = mapping

    def encode(self, texts: list[str], out_type: type = int) -> list[list[int]]:
        del out_type
        return [self._mapping[t] for t in texts]


def _fake_ids(n: int) -> list[int]:
    return list(range(4, 4 + n))  # content ids start after the 4 special ids


def test_write_split_shards_apply_cap_drops_pairs_over_256_and_counts_per_side(
    tmp_path: Path,
) -> None:
    short = _fake_ids(6)
    long_ = _fake_ids(LENGTH_CAP + 1)
    sp = _FakeSP({"a": short, "b": long_, "c": short, "x": short, "y": short, "z": long_})
    ids = ["p0", "p1", "p2"]
    result = write_split_shards(
        tmp_path / "train",
        sp,  # type: ignore[arg-type]
        ids,
        src_texts=["a", "b", "c"],
        tgt_texts=["x", "y", "z"],
        shard_size=10,
        apply_cap=True,
    )
    # p0: src=short, tgt=short -> kept. p1: src=long -> dropped. p2: tgt=long -> dropped.
    assert result.pair_count == 1
    assert result.dropped_cap_src == 1
    assert result.dropped_cap_tgt == 1
    assert result.dropped_cap_total == 2
    assert result.n_exceeds_cap_src == 1
    assert result.n_exceeds_cap_tgt == 1

    dataset = ShardDataset(tmp_path, "train")
    assert len(dataset) == 1
    _, _, kept_id = dataset.get(0)
    assert kept_id == "p0"


def test_write_split_shards_apply_cap_false_never_drops_but_still_counts(tmp_path: Path) -> None:
    short = _fake_ids(6)
    long_ = _fake_ids(LENGTH_CAP + 1)
    sp = _FakeSP({"a": short, "b": long_, "c": short, "x": short, "y": short, "z": long_})
    result = write_split_shards(
        tmp_path / "e1",
        sp,  # type: ignore[arg-type]
        ["p0", "p1", "p2"],
        src_texts=["a", "b", "c"],
        tgt_texts=["x", "y", "z"],
        shard_size=10,
        apply_cap=False,
    )
    assert result.pair_count == 3  # nothing dropped
    assert result.dropped_cap_total == 0
    assert result.n_exceeds_cap_src == 1
    assert result.n_exceeds_cap_tgt == 1


def test_write_split_shards_exactly_at_cap_is_not_dropped(tmp_path: Path) -> None:
    at_cap = _fake_ids(LENGTH_CAP)  # exactly 256, not > 256 -> must survive
    sp = _FakeSP({"a": at_cap, "x": at_cap})
    result = write_split_shards(
        tmp_path / "train",
        sp,  # type: ignore[arg-type]
        ["p0"],
        src_texts=["a"],
        tgt_texts=["x"],
        shard_size=10,
        apply_cap=True,
    )
    assert result.pair_count == 1
    assert result.dropped_cap_total == 0
    assert result.n_exceeds_cap_src == 0


def test_write_split_shards_chunks_across_multiple_shard_files(tmp_path: Path) -> None:
    n = 25
    short = _fake_ids(3)
    texts = [f"s{i}" for i in range(n)]
    sp = _FakeSP(dict.fromkeys(texts, short))
    result = write_split_shards(
        tmp_path / "train",
        sp,  # type: ignore[arg-type]
        [f"id_{i}" for i in range(n)],
        src_texts=texts,
        tgt_texts=None,
        shard_size=10,
        apply_cap=False,
    )
    assert len(result.shard_files) == 3  # 10 + 10 + 5
    assert result.pair_count == n
    dataset = ShardDataset(tmp_path, "train")
    assert len(dataset) == n
    assert not dataset.has_tgt


# --------------------------------------------------------------------------------------
# Manifest sha256 tamper detection (nmt.data.hub_data.verify_shards_against_manifest)
# --------------------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_fake_pulled_repo(root: Path) -> dict[str, Any]:
    shard_dir = root / "data" / "shards" / "train"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "shard_00000.npz"
    shard_bytes = b"not a real npz file, just bytes for hashing"
    shard_path.write_bytes(shard_bytes)

    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    spm_bytes = b"not a real spm.model file, just bytes for hashing"
    (tokenizer_dir / "spm.model").write_bytes(spm_bytes)

    manifest = {
        "shard_files_sha256": {"train/shard_00000.npz": _sha256_bytes(shard_bytes)},
        "spm_model_sha256": _sha256_bytes(spm_bytes),
    }
    (root / "data" / "shards" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_verify_shards_against_manifest_passes_on_untampered_repo(tmp_path: Path) -> None:
    _build_fake_pulled_repo(tmp_path)
    manifest = verify_shards_against_manifest(tmp_path)
    assert "shard_files_sha256" in manifest


def test_verify_shards_against_manifest_detects_tampered_shard(tmp_path: Path) -> None:
    _build_fake_pulled_repo(tmp_path)
    shard_path = tmp_path / "data" / "shards" / "train" / "shard_00000.npz"
    shard_path.write_bytes(b"CORRUPTED CONTENT, different bytes entirely")

    with pytest.raises(RuntimeError, match="sha256 verification failed"):
        verify_shards_against_manifest(tmp_path)


def test_verify_shards_against_manifest_detects_missing_file(tmp_path: Path) -> None:
    _build_fake_pulled_repo(tmp_path)
    (tmp_path / "data" / "shards" / "train" / "shard_00000.npz").unlink()

    with pytest.raises(RuntimeError, match="MISSING"):
        verify_shards_against_manifest(tmp_path)


def test_verify_shards_against_manifest_detects_tampered_spm_model(tmp_path: Path) -> None:
    _build_fake_pulled_repo(tmp_path)
    (tmp_path / "tokenizer" / "spm.model").write_bytes(b"tampered tokenizer bytes")

    with pytest.raises(RuntimeError, match="spm.model"):
        verify_shards_against_manifest(tmp_path)
