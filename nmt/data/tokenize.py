from __future__ import annotations

# SentencePiece BPE tokenizer training (joint 16k vocab, byte fallback) and pre-tokenized
# uint16 numpy shard generation plus a manifest with sha256 hashes, counts, length histograms
# and a vocab/OOV report.
#
# CLI: `python -m nmt.data.tokenize --processed data/processed --eval data/eval
# --out-shards data/shards --tokenizer-dir tokenizer --seed 1234 [--vocab-size 16000]
# [--shard-size 250000] [--sample-sentences-per-lang 1000000] [--num-threads N]
# [--push-to-hub] [--hub-repo-id OWNER/fr-en-transformer-data]`.
#
# Ordering/design notes:
# - `data/processed/train.jsonl` rows are already normalized (prepare.py); eval/dev/test raw
#   text is normalized here with the same `normalize_text` the translator applies at inference
#   (normalization lives in one function, never re-implemented).
# - Tokenizer-training sentences are sampled with a single `random.Random(seed)` instance, fr
#   first then en (documented call order, mirrors prepare.py's pattern) — with 922,748 train
#   pairs and a default request of 1,000,000 per language, both languages' full population is
#   used every time (`random.Random.sample` with k == len(population) returns a full
#   deterministic permutation, not a subset); the actual counts and this deviation are recorded
#   in the manifest.
# - `shuffle_input_sentence=False` (we already shuffled deterministically) plus a **fixed**
#   input-file and model-prefix path (both embedded verbatim in the trained model's
#   `trainer_spec`, so two runs into two different tempdirs are never byte-identical even when
#   training itself is deterministic) is what makes the sha256 determinism check meaningful:
#   train twice in place to the exact same path and compare.
# - The 256-subword cap (excl. BOS/EOS) is applied to TRAIN only; eval/dev/test sentences over
#   the cap are counted, never dropped.
import argparse
import hashlib
import json
import logging
import os
import platform
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm

from nmt.data.hub_data import push_to_hub
from nmt.data.normalize import normalize_text

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
LENGTH_CAP = 256  # subword tokens per side, excluding BOS/EOS

_BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
_WORD_INITIAL_MARK = "▁"  # "▁"

_HIST_BOUNDS = (10, 20, 40, 80, 160, 256)
_HIST_LABELS = ("0-10", "11-20", "21-40", "41-80", "81-160", "161-256", ">256")


def _length_bucket_label(n: int) -> str:
    """Bucket label for a token length `n`, per the manifest's histogram bins."""
    for bound, label in zip(_HIST_BOUNDS, _HIST_LABELS[:-1], strict=True):
        if n <= bound:
            return label
    return _HIST_LABELS[-1]


def _length_histogram(lengths: Sequence[int]) -> dict[str, int]:
    counts = dict.fromkeys(_HIST_LABELS, 0)
    for n in lengths:
        counts[_length_bucket_label(n)] += 1
    return counts


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------------------
# Tokenizer training
# --------------------------------------------------------------------------------------


@dataclass
class TokenizerSampleResult:
    fr_sentences: list[str]
    en_sentences: list[str]
    stats: dict[str, Any] = field(default_factory=dict)


def sample_tokenizer_training_sentences(
    train_rows: Sequence[dict[str, str]], seed: int, per_lang: int
) -> TokenizerSampleResult:
    """Sample up to `per_lang` fr and `per_lang` en sentences from `train_rows` (already
    normalized) with a single `random.Random(seed)`, fr first then en. When `per_lang` exceeds
    the available population (as it does by default: 922,748 pairs vs. a 1,000,000 request),
    `random.sample` with k == len(population) returns every sentence, in a random order — the
    deviation from "1M sampled" is recorded in `stats`, not silently absorbed.
    """
    rng = random.Random(seed)
    fr_all = [r["fr"] for r in train_rows]
    en_all = [r["en"] for r in train_rows]
    k_fr = min(per_lang, len(fr_all))
    k_en = min(per_lang, len(en_all))
    fr_sample = rng.sample(fr_all, k_fr)  # rng call #1
    en_sample = rng.sample(en_all, k_en)  # rng call #2
    stats = {
        "requested_sentences_per_lang": per_lang,
        "available_fr_sentences": len(fr_all),
        "available_en_sentences": len(en_all),
        "used_fr_sentences": k_fr,
        "used_en_sentences": k_en,
        "deviation_note": (
            "requested_sentences_per_lang exceeds the available population for at least one "
            "language; all available sentences for that language were used (shuffled by seed), "
            "not a subset."
            if k_fr < per_lang or k_en < per_lang
            else "no deviation: population exceeded the request for both languages."
        ),
    }
    return TokenizerSampleResult(fr_sentences=fr_sample, en_sentences=en_sample, stats=stats)


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def _run_spm_train(input_path: Path, model_prefix: Path, vocab_size: int, num_threads: int) -> None:
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        num_threads=num_threads,
        shuffle_input_sentence=False,  # already shuffled deterministically (seeded rng.sample)
        input_sentence_size=0,  # use every line in input_path; no internal subsampling
    )


def train_tokenizer_with_determinism_check(
    input_path: Path, model_prefix: Path, vocab_size: int, num_threads: int
) -> dict[str, Any]:
    """Train the SentencePiece model twice, in place, to the exact same `model_prefix` (so the
    trained model's embedded `trainer_spec.input`/`model_prefix` fields are identical between
    the two runs — otherwise the model files would never hash-match even if training itself
    were deterministic), and compare sha256 hashes. Falls back to `num_threads=1` and repeats
    the same check if the first attempt isn't reproducible. `model_prefix.with_suffix(".model")`
    holds the final trained model on disk when this returns (the last of the two/four runs;
    content is identical to the first whenever `reproducible` is True).
    """
    model_path = model_prefix.parent / f"{model_prefix.name}.model"

    _run_spm_train(input_path, model_prefix, vocab_size, num_threads)
    hash_1 = _sha256_of(model_path)
    _run_spm_train(input_path, model_prefix, vocab_size, num_threads)
    hash_2 = _sha256_of(model_path)
    reproducible = hash_1 == hash_2
    used_threads = num_threads

    if not reproducible and num_threads != 1:
        logger.warning(
            "spm.model not bit-reproducible with num_threads=%d (hash1=%s hash2=%s); "
            "retrying with num_threads=1",
            num_threads,
            hash_1[:16],
            hash_2[:16],
        )
        _run_spm_train(input_path, model_prefix, vocab_size, 1)
        hash_1 = _sha256_of(model_path)
        _run_spm_train(input_path, model_prefix, vocab_size, 1)
        hash_2 = _sha256_of(model_path)
        reproducible = hash_1 == hash_2
        used_threads = 1

    return {
        "hash_1": hash_1,
        "hash_2": hash_2,
        "reproducible": reproducible,
        "num_threads_used": used_threads,
        "num_threads_requested": num_threads,
    }


# --------------------------------------------------------------------------------------
# Encoding + shard writing
# --------------------------------------------------------------------------------------


def _encode_batch(sp: spm.SentencePieceProcessor, texts: Sequence[str]) -> list[np.ndarray]:
    """Batch-encode `texts` (no BOS/EOS — the shard contract excludes them; the loader adds
    them). Empty list short-circuits (SentencePiece's batch encode of `[]` is fine, but this
    avoids relying on that for an edge case no real split hits)."""
    if not texts:
        return []
    ids = sp.encode(list(texts), out_type=int)
    return [np.array(seq, dtype=np.uint16) for seq in ids]


def _write_shard_file(
    path: Path, ids: Sequence[str], src: Sequence[np.ndarray], tgt: Sequence[np.ndarray] | None
) -> None:
    src_off = [0]
    for s in src:
        src_off.append(src_off[-1] + len(s))
    payload: dict[str, np.ndarray] = {
        "src": np.concatenate(src).astype(np.uint16) if src else np.zeros(0, dtype=np.uint16),
        "src_off": np.array(src_off, dtype=np.int64),
        "ids": np.array(list(ids)),
    }
    if tgt is not None:
        tgt_off = [0]
        for t in tgt:
            tgt_off.append(tgt_off[-1] + len(t))
        payload["tgt"] = (
            np.concatenate(tgt).astype(np.uint16) if tgt else np.zeros(0, dtype=np.uint16)
        )
        payload["tgt_off"] = np.array(tgt_off, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


@dataclass
class SplitEncodeResult:
    shard_files: list[Path]
    pair_count: int
    src_token_count: int
    tgt_token_count: int
    src_length_histogram: dict[str, int]
    tgt_length_histogram: dict[str, int]
    dropped_cap_src: int = 0
    dropped_cap_tgt: int = 0
    dropped_cap_total: int = 0
    n_exceeds_cap_src: int = 0
    n_exceeds_cap_tgt: int = 0


def write_split_shards(
    split_dir: Path,
    sp: spm.SentencePieceProcessor,
    ids: Sequence[str],
    src_texts: Sequence[str],
    tgt_texts: Sequence[str] | None,
    shard_size: int,
    apply_cap: bool,
) -> SplitEncodeResult:
    """Encode one split's (already-normalized) text and write `shard_{k:05d}.npz` files under
    `split_dir`. `apply_cap=True` (train only) drops pairs where either side exceeds
    `LENGTH_CAP` subword tokens (excl. BOS/EOS), counting drops per side; `apply_cap=False`
    (eval/dev/test) never drops, but still counts how many sentences exceed the cap per side.
    """
    src_ids = _encode_batch(sp, src_texts)
    tgt_ids = _encode_batch(sp, tgt_texts) if tgt_texts is not None else None

    kept_ids: list[str] = []
    kept_src: list[np.ndarray] = []
    kept_tgt: list[np.ndarray] = [] if tgt_ids is not None else None  # type: ignore[assignment]
    dropped_src = dropped_tgt = dropped_total = 0
    exceeds_src = exceeds_tgt = 0

    for i, s in enumerate(src_ids):
        t = tgt_ids[i] if tgt_ids is not None else None
        src_over = len(s) > LENGTH_CAP
        tgt_over = t is not None and len(t) > LENGTH_CAP
        if src_over:
            exceeds_src += 1
        if tgt_over:
            exceeds_tgt += 1
        if apply_cap and (src_over or tgt_over):
            dropped_total += 1
            if src_over:
                dropped_src += 1
            if tgt_over:
                dropped_tgt += 1
            continue
        kept_ids.append(ids[i])
        kept_src.append(s)
        if kept_tgt is not None:
            kept_tgt.append(t)  # type: ignore[arg-type]

    shard_files: list[Path] = []
    n = len(kept_ids)
    if n == 0:
        # Still write one (empty) shard so ShardDataset's glob finds a file for this split.
        shard_path = split_dir / "shard_00000.npz"
        _write_shard_file(shard_path, [], [], [] if kept_tgt is not None else None)
        shard_files.append(shard_path)
    else:
        for start in range(0, n, shard_size):
            end = min(start + shard_size, n)
            shard_path = split_dir / f"shard_{start // shard_size:05d}.npz"
            chunk_tgt = kept_tgt[start:end] if kept_tgt is not None else None
            _write_shard_file(shard_path, kept_ids[start:end], kept_src[start:end], chunk_tgt)
            shard_files.append(shard_path)

    src_lens = [len(s) for s in kept_src]
    tgt_lens = [len(t) for t in kept_tgt] if kept_tgt is not None else []

    return SplitEncodeResult(
        shard_files=shard_files,
        pair_count=n,
        src_token_count=sum(src_lens),
        tgt_token_count=sum(tgt_lens),
        src_length_histogram=_length_histogram(src_lens),
        tgt_length_histogram=_length_histogram(tgt_lens),
        dropped_cap_src=dropped_src,
        dropped_cap_tgt=dropped_tgt,
        dropped_cap_total=dropped_total,
        n_exceeds_cap_src=exceeds_src,
        n_exceeds_cap_tgt=exceeds_tgt,
    )


def token_frequencies(
    sp: spm.SentencePieceProcessor, shard_files: Sequence[Path], key: str
) -> np.ndarray:
    """int64 token-id frequency array (len = vocab size) over `key` ("src" or "tgt") across the
    given shard files — used for `tokenizer/freq_{src,tgt}.npy` (analysis rarity features)."""
    vocab_size = sp.get_piece_size()
    freq = np.zeros(vocab_size, dtype=np.int64)
    for f in shard_files:
        with np.load(f) as z:
            if key not in z.files:
                continue
            ids = z[key].astype(np.int64)
        freq += np.bincount(ids, minlength=vocab_size)[:vocab_size]
    return freq


# --------------------------------------------------------------------------------------
# Vocab stats / OOV report
# --------------------------------------------------------------------------------------


def _byte_fallback_ids(sp: spm.SentencePieceProcessor) -> set[int]:
    return {i for i in range(sp.get_piece_size()) if _BYTE_FALLBACK_RE.match(sp.id_to_piece(i))}


def side_stats(
    sp: spm.SentencePieceProcessor, texts: Sequence[str], byte_fallback_ids: set[int]
) -> dict[str, Any]:
    """Per-language-side OOV/byte-fallback/length report over `texts` (already normalized)."""
    if not texts:
        return {"n_sentences": 0}
    encoded = sp.encode(list(texts), out_type=int)
    total_tokens = total_words = total_chars = 0
    total_byte_fallback = total_unk = 0
    sentences_with_byte_fallback = 0
    lengths: list[int] = []
    for text, ids in zip(texts, encoded, strict=True):
        lengths.append(len(ids))
        total_tokens += len(ids)
        total_words += len(text.split())
        total_chars += len(text)
        n_byte = sum(1 for i in ids if i in byte_fallback_ids)
        total_byte_fallback += n_byte
        if n_byte > 0:
            sentences_with_byte_fallback += 1
        total_unk += sum(1 for i in ids if i == UNK_ID)
    lengths_arr = np.array(lengths)
    return {
        "n_sentences": len(texts),
        "tokens_per_word": total_tokens / total_words if total_words else 0.0,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        "byte_fallback_token_rate": total_byte_fallback / total_tokens if total_tokens else 0.0,
        "unk_rate": total_unk / total_tokens if total_tokens else 0.0,
        "share_sentences_with_byte_fallback": sentences_with_byte_fallback / len(texts),
        "max_token_length": int(lengths_arr.max()),
        "mean_token_length": float(lengths_arr.mean()),
        "p95_token_length": float(np.percentile(lengths_arr, 95)),
    }


def top_byte_fallback_chars(
    sp: spm.SentencePieceProcessor, texts: Sequence[str], top_n: int = 20
) -> list[dict[str, Any]]:
    """The `top_n` characters that trigger byte-fallback tokenization most often across `texts`,
    with counts. Uses the proto encode API's per-piece `surface` (the original substring a piece
    came from): a byte-fallback piece's surface is non-empty only on the final byte of a
    multi-byte UTF-8 character, which is exactly the reconstructed character responsible.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        if not text:
            continue
        proto = sp.encode(text, return_type="proto")
        for piece in proto.pieces:
            if _BYTE_FALLBACK_RE.match(piece.piece) and piece.surface:
                counter[piece.surface] += 1
    return [{"char": ch, "count": n} for ch, n in counter.most_common(top_n)]


def vocab_summary(sp: spm.SentencePieceProcessor) -> dict[str, Any]:
    vocab_size = sp.get_piece_size()
    byte_fallback_ids = _byte_fallback_ids(sp)
    n_word_initial = sum(
        1 for i in range(vocab_size) if sp.id_to_piece(i).startswith(_WORD_INITIAL_MARK)
    )
    return {
        "vocab_size": vocab_size,
        "n_byte_fallback_pieces": len(byte_fallback_ids),
        "n_word_initial_pieces": n_word_initial,
        "share_word_initial_pieces": n_word_initial / vocab_size,
    }


# --------------------------------------------------------------------------------------
# Top-level pipeline
# --------------------------------------------------------------------------------------


def run_tokenize(
    processed_dir: Path,
    eval_dir: Path,
    out_shards_dir: Path,
    tokenizer_dir: Path,
    seed: int,
    vocab_size: int = 16000,
    shard_size: int = 250000,
    sample_sentences_per_lang: int = 1_000_000,
    num_threads: int | None = None,
    repo_root: Path = REPO_ROOT,
    train_sample_size: int = 50_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the tokenizer, encode every split into shards, and build the manifest + vocab/OOV
    report. Returns `(manifest, tokenizer_stats)`.
    """
    start = time.monotonic()
    num_threads = num_threads if num_threads is not None else (os.cpu_count() or 1)
    out_shards_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_jsonl(processed_dir / "train.jsonl")
    logger.info("loaded %d train pairs", len(train_rows))

    # --- tokenizer training ---
    sample = sample_tokenizer_training_sentences(train_rows, seed, sample_sentences_per_lang)
    train_input_path = out_shards_dir / "_spm_train_input.txt"
    _write_lines(train_input_path, sample.fr_sentences + sample.en_sentences)
    model_prefix = tokenizer_dir / "spm"
    determinism = train_tokenizer_with_determinism_check(
        train_input_path, model_prefix, vocab_size, num_threads
    )
    train_input_path.unlink(missing_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_dir / "spm.model"))
    actual_vocab_size = sp.get_piece_size()
    byte_fallback_ids = _byte_fallback_ids(sp)

    # --- encode + shard every split ---
    splits: dict[str, Any] = {}
    all_shard_files: list[Path] = []

    train_ids = [r["id"] for r in train_rows]
    train_fr = [r["fr"] for r in train_rows]
    train_en = [r["en"] for r in train_rows]
    train_result = write_split_shards(
        out_shards_dir / "train", sp, train_ids, train_fr, train_en, shard_size, apply_cap=True
    )
    splits["train"] = train_result
    all_shard_files += train_result.shard_files

    eval_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("e1", "e2", "e3"):
        rows = _read_jsonl(eval_dir / split / "inputs.jsonl")
        labels = {r["id"]: r["reference"] for r in _read_jsonl(eval_dir / split / "labels.jsonl")}
        ids = [r["id"] for r in rows]
        src = [normalize_text(r["source"]) for r in rows]
        tgt = [normalize_text(labels[r["id"]]) for r in rows]
        result = write_split_shards(
            out_shards_dir / split, sp, ids, src, tgt, shard_size, apply_cap=False
        )
        splits[split] = result
        all_shard_files += result.shard_files
        eval_rows_by_split[split] = [
            {"id": r["id"], "source": src[i], "reference": tgt[i]} for i, r in enumerate(rows)
        ]

    dev_inputs = _read_jsonl(repo_root / "data" / "dev" / "inputs.jsonl")
    dev_labels_rows = _read_jsonl(repo_root / "data" / "dev" / "labels.jsonl")
    dev_labels = {r["id"]: r["reference"] for r in dev_labels_rows}
    dev_ids = [r["id"] for r in dev_inputs]
    dev_src = [normalize_text(r["source"]) for r in dev_inputs]
    dev_tgt = [normalize_text(dev_labels[r["id"]]) for r in dev_inputs]
    dev_result = write_split_shards(
        out_shards_dir / "dev", sp, dev_ids, dev_src, dev_tgt, shard_size, apply_cap=False
    )
    splits["dev"] = dev_result
    all_shard_files += dev_result.shard_files
    dev_rows_full = [
        {"id": r["id"], "source": dev_src[i], "reference": dev_tgt[i], "slice": r["slice"]}
        for i, r in enumerate(dev_inputs)
    ]

    test_inputs = _read_jsonl(repo_root / "data" / "test" / "inputs.jsonl")
    test_ids = [r["id"] for r in test_inputs]
    test_src = [normalize_text(r["source"]) for r in test_inputs]
    test_result = write_split_shards(
        out_shards_dir / "test", sp, test_ids, test_src, None, shard_size, apply_cap=False
    )
    splits["test"] = test_result
    all_shard_files += test_result.shard_files

    # --- token frequencies over final train shards (analysis rarity features) ---
    freq_src = token_frequencies(sp, train_result.shard_files, "src")
    freq_tgt = token_frequencies(sp, train_result.shard_files, "tgt")
    np.save(tokenizer_dir / "freq_src.npy", freq_src)
    np.save(tokenizer_dir / "freq_tgt.npy", freq_tgt)

    # --- manifest ---
    shard_files_sha256 = {
        f.resolve().relative_to(out_shards_dir.resolve()).as_posix(): _sha256_of(f)
        for f in all_shard_files
    }
    spm_model_sha256 = _sha256_of(tokenizer_dir / "spm.model")
    spm_vocab_sha256 = _sha256_of(tokenizer_dir / "spm.vocab")

    manifest: dict[str, Any] = {
        "seed": seed,
        "created_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "sentencepiece_version": spm.__version__,
        "wall_clock_seconds": round(time.monotonic() - start, 3),
        "vocab_size": actual_vocab_size,
        "special_ids": {"pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID},
        "length_cap": LENGTH_CAP,
        "tokenizer_training": sample.stats,
        "determinism_check": determinism,
        "spm_model_sha256": spm_model_sha256,
        "spm_vocab_sha256": spm_vocab_sha256,
        "shard_files_sha256": shard_files_sha256,
        "splits": {
            name: {
                "pair_count": r.pair_count,
                "src_token_count": r.src_token_count,
                "tgt_token_count": r.tgt_token_count,
                "src_length_histogram": r.src_length_histogram,
                "tgt_length_histogram": r.tgt_length_histogram,
                "dropped_cap_src": r.dropped_cap_src,
                "dropped_cap_tgt": r.dropped_cap_tgt,
                "dropped_cap_total": r.dropped_cap_total,
                "n_exceeds_cap_src": r.n_exceeds_cap_src,
                "n_exceeds_cap_tgt": r.n_exceeds_cap_tgt,
                "has_tgt": name != "test",
            }
            for name, r in splits.items()
        },
    }

    # --- vocab / OOV report ---
    rng2 = random.Random(seed)  # independent of the tokenizer-sample rng; pair-level, not
    # sentence-level, sampling — documented separately since it draws from a different
    # population (paired rows, not two independent per-language sentence lists).
    train_sample_rows = rng2.sample(train_rows, min(train_sample_size, len(train_rows)))

    per_set: dict[str, Any] = {
        "train_sample": {
            "fr": [r["fr"] for r in train_sample_rows],
            "en": [r["en"] for r in train_sample_rows],
        },
        "e1": {
            "fr": [r["source"] for r in eval_rows_by_split["e1"]],
            "en": [r["reference"] for r in eval_rows_by_split["e1"]],
        },
        "e2": {
            "fr": [r["source"] for r in eval_rows_by_split["e2"]],
            "en": [r["reference"] for r in eval_rows_by_split["e2"]],
        },
        "e3": {
            "fr": [r["source"] for r in eval_rows_by_split["e3"]],
            "en": [r["reference"] for r in eval_rows_by_split["e3"]],
        },
        "test": {"fr": test_src},
    }
    for slice_name in ("seen", "long", "unseen_domain"):
        rows = [r for r in dev_rows_full if r["slice"] == slice_name]
        per_set[f"dev_{slice_name}"] = {
            "fr": [r["source"] for r in rows],
            "en": [r["reference"] for r in rows],
        }

    per_set_stats = {
        name: {side: side_stats(sp, texts, byte_fallback_ids) for side, texts in sides.items()}
        for name, sides in per_set.items()
    }

    char_analysis_texts = (
        dev_src
        + dev_tgt
        + test_src
        + [r["source"] for r in eval_rows_by_split["e3"]]
        + [r["reference"] for r in eval_rows_by_split["e3"]]
    )
    top_chars = top_byte_fallback_chars(sp, char_analysis_texts, top_n=20)

    tokenizer_stats: dict[str, Any] = {
        "created_utc": manifest["created_utc"],
        "vocab_summary": vocab_summary(sp),
        "per_set": per_set_stats,
        "top_byte_fallback_chars_dev_test_e3": top_chars,
    }

    return manifest, tokenizer_stats


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _print_stats_summary(tokenizer_stats: dict[str, Any]) -> None:
    vs = tokenizer_stats["vocab_summary"]
    print(
        f"vocab_size={vs['vocab_size']} byte_fallback_pieces={vs['n_byte_fallback_pieces']} "
        f"word_initial_share={vs['share_word_initial_pieces']:.4f}"
    )
    for name, sides in tokenizer_stats["per_set"].items():
        for side, s in sides.items():
            if s.get("n_sentences", 0) == 0:
                continue
            print(
                f"{name:16s} {side:2s} n={s['n_sentences']:6d} "
                f"tok/word={s['tokens_per_word']:.3f} tok/char={s['tokens_per_char']:.3f} "
                f"byte_rate={s['byte_fallback_token_rate']:.4f} unk_rate={s['unk_rate']:.5f} "
                f"share_byte_sent={s['share_sentences_with_byte_fallback']:.4f} "
                f"max={s['max_token_length']} mean={s['mean_token_length']:.2f} "
                f"p95={s['p95_token_length']:.1f}"
            )
    print("top byte-fallback chars (dev+test+E3):")
    for row in tokenizer_stats["top_byte_fallback_chars_dev_test_e3"]:
        print(f"  {row['char']!r}: {row['count']}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SentencePiece tokenizer and build pre-tokenized shards."
    )
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval"))
    parser.add_argument("--out-shards", type=Path, default=Path("data/shards"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--shard-size", type=int, default=250_000)
    parser.add_argument("--sample-sentences-per-lang", type=int, default=1_000_000)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default="OWNER/fr-en-transformer-data",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    manifest, tokenizer_stats = run_tokenize(
        processed_dir=args.processed,
        eval_dir=args.eval,
        out_shards_dir=args.out_shards,
        tokenizer_dir=args.tokenizer_dir,
        seed=args.seed,
        vocab_size=args.vocab_size,
        shard_size=args.shard_size,
        sample_sentences_per_lang=args.sample_sentences_per_lang,
        num_threads=args.num_threads,
    )
    _write_json(args.out_shards / "manifest.json", manifest)
    _write_json(REPO_ROOT / "reports" / "tokenizer_stats.json", tokenizer_stats)
    _print_stats_summary(tokenizer_stats)
    logger.info(
        "tokenize complete: vocab_size=%d wall_clock_seconds=%.1f reproducible=%s",
        manifest["vocab_size"],
        manifest["wall_clock_seconds"],
        manifest["determinism_check"]["reproducible"],
    )

    if args.push_to_hub:
        result = push_to_hub(
            repo_id=args.hub_repo_id,
            shards_dir=args.out_shards,
            tokenizer_dir=args.tokenizer_dir,
            eval_dir=args.eval,
            data_manifest_path=REPO_ROOT / "data" / "data_manifest.json",
        )
        logger.info("pushed to hub: %s", result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
