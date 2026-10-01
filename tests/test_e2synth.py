from __future__ import annotations

# Tests for nmt/data/e2synth.py (the synthetic long-input probe) and its evaluate.py wiring:
# seeded determinism, bucket counts and limits, 2-4 constituents, no empty strings, the committed
# files matching their manifest, and run_evaluation carrying the `synthetic` flag on tiny
# monkeypatched data. Selection-isolation tests live in tests/test_selection.py.
import hashlib
import json
from pathlib import Path

import pytest

from nmt.data.e2synth import BUCKET_NAMES, BUCKETS, bucket_of, build_items, write_e2synth
from nmt.evaluate import EvalRunConfig, load_split, run_evaluation
from tests.test_evaluate import REAL_TOKENIZER, _export_tiny_translator

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_DIR = REPO_ROOT / "data" / "eval" / "e2"
E2SYNTH_DIR = REPO_ROOT / "data" / "eval" / "e2synth"


def _tiny_e2(n: int = 40) -> tuple[list[dict], list[dict]]:
    """Deterministic fake E2: French sources of 190-290 chars so 2-4 pairs span every bucket."""
    inputs, labels = [], []
    for i in range(n):
        src = (f"Phrase numero {i} " + "mot " * (45 + (i * 7) % 25)).strip()
        inputs.append({"id": f"e2_{i:05d}", "source": src, "slice": "e2", "length": len(src)})
        labels.append({"id": f"e2_{i:05d}", "reference": f"Sentence {i} " + "word " * 40})
    return inputs, labels


def test_bucket_of_is_half_open() -> None:
    assert bucket_of(399) is None
    assert bucket_of(400) == "e2synth_400_600"
    assert bucket_of(599) == "e2synth_400_600"
    assert bucket_of(600) == "e2synth_600_800"
    assert bucket_of(800) == "e2synth_800_900"
    assert bucket_of(899) == "e2synth_800_900"
    assert bucket_of(900) is None


def test_build_items_small_buckets_limits_and_constituents() -> None:
    inputs, labels = _tiny_e2()
    items = build_items(inputs, labels, seed=7, per_bucket=5)
    assert len(items) == 15
    by_bucket = {n: [i for i in items if i["slice"] == n] for n in BUCKET_NAMES}
    assert all(len(v) == 5 for v in by_bucket.values())
    lo_hi = {name: (lo, hi) for name, lo, hi in BUCKETS}
    src_by_id = {r["id"]: r["source"] for r in inputs}
    ref_by_id = {r["id"]: r["reference"] for r in labels}
    for it in items:
        lo, hi = lo_hi[it["slice"]]
        assert lo <= len(it["source"]) < hi
        assert it["length"] == len(it["source"])
        assert 2 <= len(it["e2_ids"]) <= 4
        assert it["e2_ids"] == sorted(it["e2_ids"])
        assert it["source"] == " ".join(src_by_id[i] for i in it["e2_ids"])
        assert it["reference"] == " ".join(ref_by_id[i] for i in it["e2_ids"])
        assert it["source"].strip() and it["reference"].strip()
    assert len({tuple(i["e2_ids"]) for i in items}) == len(items)  # no duplicate constituent sets
    assert [i["id"] for i in items] == [f"e2synth_{k:05d}" for k in range(15)]


def test_build_items_is_deterministic_and_seed_sensitive() -> None:
    inputs, labels = _tiny_e2()
    a = build_items(inputs, labels, seed=1, per_bucket=5)
    b = build_items(inputs, labels, seed=1, per_bucket=5)
    c = build_items(inputs, labels, seed=2, per_bucket=5)
    assert a == b
    assert a != c


def test_build_items_rejects_empty_e2_text() -> None:
    inputs, labels = _tiny_e2()
    inputs[3]["source"] = "   "
    with pytest.raises(ValueError, match="empty"):
        build_items(inputs, labels, per_bucket=2)


def test_write_e2synth_same_seed_identical_sha256_and_format(tmp_path: Path) -> None:
    e2 = tmp_path / "e2"
    e2.mkdir()
    inputs, labels = _tiny_e2()
    for name, rows in (("inputs.jsonl", inputs), ("labels.jsonl", labels)):
        (e2 / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    m1 = write_e2synth(e2, tmp_path / "o1", seed=3, per_bucket=5)
    m2 = write_e2synth(e2, tmp_path / "o2", seed=3, per_bucket=5)
    assert m1["files_sha256"] == m2["files_sha256"]
    assert (
        m1["files_sha256"]["inputs.jsonl"]
        == hashlib.sha256((tmp_path / "o1" / "inputs.jsonl").read_bytes()).hexdigest()
    )
    rows_in = [json.loads(x) for x in (tmp_path / "o1" / "inputs.jsonl").read_text().splitlines()]
    rows_lab = [json.loads(x) for x in (tmp_path / "o1" / "labels.jsonl").read_text().splitlines()]
    assert all(set(r) == {"id", "source", "slice", "length", "synthetic"} for r in rows_in)
    assert all(set(r) == {"id", "reference", "slice", "synthetic"} for r in rows_lab)
    assert all(r["synthetic"] is True for r in rows_in + rows_lab)
    assert "length generalization" in m1["note"] and m1["seed"] == 3


def test_committed_e2synth_matches_its_manifest_and_spec() -> None:
    manifest = json.loads((E2SYNTH_DIR / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["files_sha256"].items():
        assert hashlib.sha256((E2SYNTH_DIR / name).read_bytes()).hexdigest() == sha
    inputs, labels = load_split("e2synth")
    assert len(inputs) == len(labels) == manifest["n_items"] == 300
    assert manifest["bucket_counts"] == dict.fromkeys(BUCKET_NAMES, 100)
    e2_ids = {json.loads(x)["id"] for x in (E2_DIR / "inputs.jsonl").read_text().splitlines()}
    lo_hi = {name: (lo, hi) for name, lo, hi in BUCKETS}
    for row, lab, item in zip(inputs, labels, manifest["items"], strict=True):
        assert row["id"] == lab["id"] == item["id"] and row["id"].startswith("e2synth_")
        assert row["slice"] == lab["slice"] == item["slice"]
        assert row["synthetic"] is True and lab["synthetic"] is True
        lo, hi = lo_hi[row["slice"]]
        assert lo <= len(row["source"]) < hi and row["length"] == len(row["source"])
        assert 2 <= len(item["e2_ids"]) <= 4 and set(item["e2_ids"]) <= e2_ids
        assert row["source"].strip() and lab["reference"].strip()


@pytest.mark.skipif(not REAL_TOKENIZER.is_file(), reason="tokenizer/spm.model not built yet (P2)")
def test_run_evaluation_flags_e2synth_as_synthetic_and_keeps_it_out_of_length_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_e1 = (
        [{"id": "e1_0", "source": "Au revoir.", "slice": "e1", "length": 10}],
        [{"id": "e1_0", "reference": "Goodbye.", "slice": "e1"}],
    )
    tiny_synth = (
        [
            {
                "id": "e2synth_0",
                "source": "Bonne nuit. A demain.",
                "slice": "e2synth_400_600",
                "length": 21,
                "synthetic": True,
            },
            {
                "id": "e2synth_1",
                "source": "Il etait une fois.",
                "slice": "e2synth_600_800",
                "length": 18,
                "synthetic": True,
            },
        ],
        [
            {
                "id": "e2synth_0",
                "reference": "Good night. See you tomorrow.",
                "slice": "e2synth_400_600",
                "synthetic": True,
            },
            {
                "id": "e2synth_1",
                "reference": "Once upon a time.",
                "slice": "e2synth_600_800",
                "synthetic": True,
            },
        ],
    )
    monkeypatch.setattr(
        "nmt.evaluate.load_split", lambda name: {"e1": tiny_e1, "e2synth": tiny_synth}[name]
    )
    translator = _export_tiny_translator(tmp_path)
    cfg = EvalRunConfig(beam_size=2, alpha=0.6, batch_size=4, n_bootstrap=20, bootstrap_seed=1)
    out_dir = tmp_path / "out"
    result = run_evaluation(translator, "r", "c", cfg, out_dir=out_dir, splits=("e1", "e2synth"))

    entry = result["sets"]["e2synth"]
    assert entry["synthetic"] is True and entry["label"] == "E2-synth (synthetic)"
    assert "synthetic" not in result["sets"]["e1"]
    assert (out_dir / "e2synth_predictions.json").is_file()
    assert set(entry["official_ci_by_slice"]["chrf"]) == {"e2synth_400_600", "e2synth_600_800"}
    assert {"official", "sacrebleu", "official_bleu_ci", "official_chrf_ci"} <= set(entry)
    # E2-synth must not leak into the E1/E2/E3 length-bucket view (only e1's single sentence).
    assert sum(v["n"] for v in result["length_buckets_e1_e2_e3"].values()) == 1
