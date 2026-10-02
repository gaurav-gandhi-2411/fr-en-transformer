from __future__ import annotations

# Tests for scripts/eval_local.py, the score-only local pipeline over the Colab eval_l4 outputs.
# The HF hub is a FAKE (an in-memory repo keyed by path); predictions are derived from the real
# eval sets' references so the real scorers, bootstrap, nmt.compare and nmt.analysis all run on
# real ids. COMET is mocked (the 2.3 GB model is verified manually, like tests/test_comet_*.py).
# The pipeline must never decode or select: Translator/tune are booby-trapped in the e2e tests.
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

import scripts.eval_local as el
from nmt.eval_l4 import MANIFEST_NAME, expected_candidates, required_rels
from nmt.evaluate import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
REV = "c" * 40
REPO = "owner/eval"
NB = 20  # bootstrap resamples: enough to exercise the code, fast enough for CI


def _subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every k-th row (about 25 per split): all three dev slices and all three E2-synth char
    buckets stay represented, and the real scorers run in seconds instead of minutes."""
    return rows[:: max(1, len(rows) // 25)]


@pytest.fixture(scope="module", autouse=True)
def tiny_splits(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Swap every `load_split` the pipeline reaches (and the official --gold file) for a strided
    subset. Pull-time id checks, scoring, bootstrap, nmt.compare and nmt.analysis then all see the
    same small, real-text splits; the full-size run is measured separately (see the report)."""
    gold_dir = tmp_path_factory.mktemp("gold")
    cache: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

    def small(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if name not in cache:
            inputs, labels = load_split(name)
            keep = {r["id"] for r in _subset(inputs)}
            cache[name] = (
                [r for r in inputs if r["id"] in keep],
                [r for r in labels if r["id"] in keep],
            )
        return cache[name]

    def gold(split: str) -> Path:
        path = gold_dir / f"{split}.jsonl"
        if not path.exists():
            lines = [json.dumps(r) for r in small(split)[1]]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(el, "load_split", small)
        mp.setattr(el, "gold_path", gold)
        mp.setattr("nmt.compare.load_split", small)
        mp.setattr("nmt.analysis.load_split", small)
        yield small


def _degrade(ref: str, drop_every: int) -> str:
    """A deterministic 'translation': the reference with every `drop_every`-th word removed
    (0 = the reference itself), so runs have different, known quality."""
    words = ref.split()
    kept = [w for i, w in enumerate(words) if drop_every == 0 or (i + 1) % drop_every != 0]
    return " ".join(kept) or ref


_PRED_CACHE: dict[tuple[str, int], dict[str, str]] = {}  # tiny splits only


def _preds(split: str, drop_every: int) -> dict[str, str]:
    key = (split, drop_every)
    if key not in _PRED_CACHE:
        _, labels = el.load_split(split)
        _PRED_CACHE[key] = {r["id"]: _degrade(r["reference"], drop_every) for r in labels}
    return _PRED_CACHE[key]


def _run_files(run: str, drop_every: int, tuned_drop_every: int | None = None) -> dict[str, bytes]:
    """The runs/<run>/ tree nmt.eval_l4 uploads, as {repo path: bytes}."""
    tuned = drop_every if tuned_drop_every is None else tuned_drop_every
    files: dict[str, bytes] = {}

    def put(rel: str, obj: Any) -> None:
        files[f"runs/{run}/{rel}"] = json.dumps(obj, ensure_ascii=False).encode("utf-8")

    for variant, de in (("seg_off", drop_every), ("seg_tuned", tuned)):
        for split in el.SPLITS:
            put(f"predictions/{variant}/{split}_predictions.json", _preds(split, de))
    put(
        "selection.json",
        {
            "winner": {
                "candidate": "final",
                "objective": 40.0,
                "alpha": 0.8,
                "beam": 5,
                "segment_threshold": 128,
            }
        },
    )
    put("decode_summary.json", {"candidate": "final"})
    put("bench.json", {"n_sentences": 200})
    if run == "main":
        sample = json.loads((REPO_ROOT / "data" / "test" / "sample_submission.json").read_text())
        put("test_predictions.json", {i: "x" for i in sample})
        put("validation.json", {"valid": True})
    put("run_meta.json", {"run": run})
    for cand in expected_candidates(run):
        put(f"tuning/{cand}.json", {"candidate": cand})
    for name in ("model.safetensors", "config.json", "spm.model"):
        files[f"runs/{run}/model/{name}"] = b"weights-" + name.encode()
    return with_manifest(run, files)


def with_manifest(run: str, files: dict[str, bytes]) -> dict[str, bytes]:
    """Add runs/<run>/manifest.json in the schema nmt.eval_l4.build_manifest writes."""
    prefix = f"runs/{run}/"
    manifest = {
        "schema": 1,
        "run": run,
        "candidate_order": list(expected_candidates(run)),
        "winner": "final",
        "files": {
            f.removeprefix(prefix): {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
            for f, b in files.items()
            if f.startswith(prefix)
        },
    }
    return {**files, prefix + MANIFEST_NAME: json.dumps(manifest).encode("utf-8")}


class FakeHub:
    """In-memory HF repo: records every (filename, revision) download and writes real files."""

    def __init__(self, repos: dict[str, dict[str, bytes]] | None = None) -> None:
        self.private: bool | None = True
        self.files: dict[str, bytes] = {}
        for tree in (repos or {}).values():
            self.files.update(tree)
        self.downloads: list[tuple[str, str, str]] = []

    def is_private(self, repo_id: str) -> bool | None:
        return self.private

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        return sorted(self.files)

    def download(self, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
        self.downloads.append((repo_id, filename, revision))
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[filename])
        return dest


def _hub(*runs: tuple[str, int]) -> FakeHub:
    hub = FakeHub()
    for run, drop in runs:
        hub.files.update(_run_files(run, drop))
    return hub


def _evaluate(hub: FakeHub, run: str, out_root: Path, **over: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n_bootstrap": NB,
        "seed": 1234,
        "comet": "off",
        "do_analysis": False,
    }
    kwargs.update(over)
    return el.evaluate_run(hub, REPO, run, REV, out_root, **kwargs)


# --- run specs and pull ---------------------------------------------------------------------------


def test_parse_run_spec_requires_a_full_sha_and_a_known_run() -> None:
    assert el.parse_run_spec("main", REV) == ("main", REV)
    assert el.parse_run_spec("s1_sin_l4@" + "d" * 40, REV) == ("s1_sin_l4", "d" * 40)
    for spec, default in (("main", None), ("main", "main"), ("main", "abc123"), ("nope", REV)):
        with pytest.raises(el.ScoreError):
            el.parse_run_spec(spec, default)
    with pytest.raises(el.ScoreError, match="40-hex"):
        el.parse_run_spec("main@" + "d" * 39, REV)


def test_required_files_include_test_only_for_main() -> None:
    assert "test_predictions.json" in el.required_files("main")
    assert "test_predictions.json" not in el.required_files("s1_sin_l4")
    assert len([f for f in el.required_files("s2_rope_l4") if f.startswith("predictions/")]) == 10


def test_pull_downloads_at_the_pinned_revision_and_skips_the_model(tmp_path: Path) -> None:
    hub = _hub(("main", 0))
    digests = el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    assert {rev for _, _, rev in hub.downloads} == {REV}
    assert "runs/main/model/model.safetensors" not in digests
    assert "runs/main/test_predictions.json" in digests
    path = tmp_path / "main" / "source" / "runs" / "main" / "selection.json"
    assert el._sha256(path) == digests["runs/main/selection.json"]
    withm = el.pull_run(hub, REPO, "main", REV, tmp_path / "m2", with_model=True)
    assert "runs/main/model/model.safetensors" in withm


def test_pull_refuses_when_a_required_file_is_missing(tmp_path: Path) -> None:
    hub = _hub(("main", 0))
    del hub.files["runs/main/predictions/seg_tuned/e3_predictions.json"]
    del hub.files["runs/main/test_predictions.json"]
    with pytest.raises(el.ScoreError) as err:
        el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    assert "e3_predictions.json" in str(err.value) and "test_predictions.json" in str(err.value)
    assert hub.downloads == []  # nothing was downloaded from an incomplete run


def test_pull_ignores_other_runs_in_the_same_repo(tmp_path: Path) -> None:
    hub = _hub(("main", 0), ("s1_sin_l4", 3))
    el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "s1")
    assert all(f.startswith("runs/s1_sin_l4/") for _, f, _ in hub.downloads)


# --- manifest + hash verification of the pull -------------------------------------------------


def _manifest(hub: FakeHub, run: str) -> dict[str, Any]:
    return json.loads(hub.files[f"runs/{run}/{MANIFEST_NAME}"])


def _put_manifest(hub: FakeHub, run: str, manifest: dict[str, Any]) -> None:
    hub.files[f"runs/{run}/{MANIFEST_NAME}"] = json.dumps(manifest).encode("utf-8")


def test_pull_verifies_every_file_and_records_the_result(tmp_path: Path) -> None:
    hub = _hub(("main", 0))
    el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    rec = json.loads((tmp_path / "main" / el.PULL_RECORD_NAME).read_text(encoding="utf-8"))
    manifest = _manifest(hub, "main")
    assert rec["verified"] is True and rec["private"] is True
    assert rec["hf_repo"] == REPO and rec["hf_revision"] == REV
    manifest_bytes = hub.files[f"runs/main/{MANIFEST_NAME}"]
    assert rec["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    # every manifest file except the (skipped) model weights is individually ok
    assert set(rec["files"]) == {r for r in manifest["files"] if not r.startswith("model/")}
    assert all(v["ok"] for v in rec["files"].values()) and "test_predictions.json" in rec["files"]
    assert rec["model_files_checked"] is False
    withm = el.pull_run(hub, REPO, "main", REV, tmp_path / "m2", with_model=True)
    rec2 = json.loads((tmp_path / "m2" / el.PULL_RECORD_NAME).read_text(encoding="utf-8"))
    assert "model/spm.model" in rec2["files"] and "runs/main/model/spm.model" in withm


def test_manifest_covers_everything_required_per_run() -> None:
    for run in ("main", "s1_sin_l4"):
        listed = set(_manifest(_hub((run, 0)), run)["files"])
        assert set(required_rels(run, expected_candidates(run))) <= listed
    assert "test_predictions.json" in required_rels("main", expected_candidates("main"))
    assert "test_predictions.json" not in required_rels("s1_sin_l4", ("final",))


def test_a_tampered_byte_is_a_loud_error_naming_file_hashes_repo_and_revision(
    tmp_path: Path,
) -> None:
    hub = _hub(("main", 0))
    name = "runs/main/predictions/seg_off/dev_predictions.json"
    good = hub.files[name]
    hub.files[name] = good[:-3] + bytes([good[-3] ^ 1]) + good[-2:]  # same size, one bit flipped
    with pytest.raises(el.ManifestVerificationError) as err:
        el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    e = err.value
    assert e.file == name and e.repo == REPO and e.revision == REV
    assert e.expected == hashlib.sha256(good).hexdigest()
    assert e.actual == hashlib.sha256(hub.files[name]).hexdigest()
    assert "sha256 differs" in str(e) and name in str(e) and REV in str(e)
    assert not (tmp_path / "main" / el.PULL_RECORD_NAME).exists()


def test_a_size_mismatch_is_reported_with_both_sizes(tmp_path: Path) -> None:
    hub = _hub(("s1_sin_l4", 0))
    manifest = _manifest(hub, "s1_sin_l4")
    manifest["files"]["bench.json"]["bytes"] += 7
    _put_manifest(hub, "s1_sin_l4", manifest)
    with pytest.raises(el.ManifestVerificationError, match="size differs") as err:
        el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "r")
    assert err.value.file == "runs/s1_sin_l4/bench.json"
    assert err.value.expected == err.value.actual + 7


def test_a_missing_file_fails_whether_required_or_only_listed(tmp_path: Path) -> None:
    hub = _hub(("s1_sin_l4", 0))
    del hub.files["runs/s1_sin_l4/tuning/final.json"]  # listed in the manifest, gone from the hub
    with pytest.raises(el.ManifestVerificationError, match="absent from the hub") as err:
        el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "r")
    assert err.value.file == "runs/s1_sin_l4/tuning/final.json"
    hub2 = _hub(("s1_sin_l4", 0))
    manifest = _manifest(hub2, "s1_sin_l4")
    del manifest["files"]["run_meta.json"]  # the manifest itself omits a pipeline-required file
    _put_manifest(hub2, "s1_sin_l4", manifest)
    with pytest.raises(el.ManifestVerificationError, match="lacks required"):
        el.pull_run(hub2, REPO, "s1_sin_l4", REV, tmp_path / "r2")


def test_an_extra_file_on_the_hub_or_on_disk_is_refused(tmp_path: Path) -> None:
    hub = _hub(("s1_sin_l4", 0))
    hub.files["runs/s1_sin_l4/extra.json"] = b"{}"
    with pytest.raises(el.ManifestVerificationError, match="not in the manifest") as err:
        el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "r")
    assert err.value.file == "runs/s1_sin_l4/extra.json"
    clean = _hub(("s1_sin_l4", 0))
    stray = tmp_path / "r2" / "source" / "runs" / "s1_sin_l4" / "stray.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}", encoding="utf-8")
    with pytest.raises(el.ManifestVerificationError, match="local file is not in"):
        el.pull_run(clean, REPO, "s1_sin_l4", REV, tmp_path / "r2")


def test_files_outside_the_run_prefix_never_refuse_or_download(tmp_path: Path) -> None:
    hub = _hub(("main", 0), ("s1_sin_l4", 0), ("s2_rope_l4", 2))
    hub.files["_write_probe.txt"] = b"probe"
    hub.files[".gitattributes"] = b"*.safetensors filter=lfs"
    hub.files["README.md"] = b"# repo"
    digests = el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "r")
    assert hub.downloads and all(f.startswith("runs/s1_sin_l4/") for _, f, _ in hub.downloads)
    assert all(f.startswith("runs/s1_sin_l4/") for f in digests)
    rec = json.loads((tmp_path / "r" / el.PULL_RECORD_NAME).read_text(encoding="utf-8"))
    assert rec["verified"] is True


def test_verify_pulled_run_filters_the_whole_hub_listing_to_the_run_prefix(
    tmp_path: Path,
) -> None:
    # verify_pulled_run must scope itself (not rely on pull_run's pre-filter): hand it the
    # repo-wide listing, including the write probe and other runs' files.
    hub = _hub(("main", 0), ("s1_sin_l4", 0), ("s2_rope_l4", 2))
    hub.files["_write_probe.txt"] = b"probe"
    hub.files["README.md"] = b"# repo"
    el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "r")
    record = el.verify_pulled_run(
        REPO, "s1_sin_l4", REV, tmp_path / "r" / "source", sorted(hub.files)
    )
    assert record["verified"] is True
    hub.files["runs/s1_sin_l4/extra.json"] = b"{}"
    with pytest.raises(el.ManifestVerificationError, match="not in the manifest"):
        el.verify_pulled_run(REPO, "s1_sin_l4", REV, tmp_path / "r" / "source", sorted(hub.files))


def test_extra_or_tampered_file_under_the_target_run_is_refused_among_other_runs(
    tmp_path: Path,
) -> None:
    hub = _hub(("main", 0), ("s1_sin_l4", 0))
    hub.files["_write_probe.txt"] = b"probe"
    hub.files["runs/s1_sin_l4/extra.json"] = b"{}"
    with pytest.raises(el.ManifestVerificationError, match="not in the manifest"):
        el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "a")
    del hub.files["runs/s1_sin_l4/extra.json"]
    hub.files["runs/s1_sin_l4/bench.json"] += b" "
    with pytest.raises(el.ManifestVerificationError, match="differs"):
        el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "b")


@pytest.mark.parametrize(
    "key",
    ["../outside.json", "/abs.json", "C:\\x.json", "C:/x.json", "a/../../b.json", "a\\b.json"],
)
def test_unsafe_manifest_keys_are_refused_naming_the_key(tmp_path: Path, key: str) -> None:
    hub = _hub(("s1_sin_l4", 0))
    manifest = _manifest(hub, "s1_sin_l4")
    manifest["files"][key] = {"sha256": "0" * 64, "bytes": 1}
    _put_manifest(hub, "s1_sin_l4", manifest)
    with pytest.raises(el.ManifestVerificationError) as err:
        el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "r")
    assert repr(key) in str(err.value)
    assert [f for _, f, _ in hub.downloads] == ["runs/s1_sin_l4/manifest.json"]


def test_a_failed_re_pull_clears_the_stale_pull_record(tmp_path: Path) -> None:
    hub = _hub(("s1_sin_l4", 0))
    run_dir = tmp_path / "r"
    el.pull_run(hub, REPO, "s1_sin_l4", REV, run_dir)
    assert (run_dir / el.PULL_RECORD_NAME).is_file()
    hub.files["runs/s1_sin_l4/bench.json"] += b" "
    with pytest.raises(el.ManifestVerificationError):
        el.pull_run(hub, REPO, "s1_sin_l4", REV, run_dir)
    assert not (run_dir / el.PULL_RECORD_NAME).exists()


def test_a_missing_manifest_is_refused_before_any_download(tmp_path: Path) -> None:
    hub = _hub(("main", 0))
    del hub.files[f"runs/main/{MANIFEST_NAME}"]
    with pytest.raises(el.ManifestVerificationError, match="manifest.json is missing") as err:
        el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    assert err.value.file == f"runs/main/{MANIFEST_NAME}" and hub.downloads == []


def test_an_unreadable_manifest_is_refused(tmp_path: Path) -> None:
    hub = _hub(("main", 0))
    hub.files[f"runs/main/{MANIFEST_NAME}"] = b"{not json"
    with pytest.raises(el.ManifestVerificationError, match="unreadable"):
        el.pull_run(hub, REPO, "main", REV, tmp_path / "main")


def test_a_manifest_for_another_run_is_refused(tmp_path: Path) -> None:
    hub = _hub(("main", 0), ("s1_sin_l4", 0))
    hub.files[f"runs/main/{MANIFEST_NAME}"] = hub.files[f"runs/s1_sin_l4/{MANIFEST_NAME}"]
    with pytest.raises(el.ManifestVerificationError, match="different run") as err:
        el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    assert err.value.expected == "main" and err.value.actual == "s1_sin_l4"


def test_main_requires_test_predictions_and_validation_in_the_manifest(tmp_path: Path) -> None:
    for rel in ("test_predictions.json", "validation.json"):
        hub = _hub(("main", 0))
        manifest = _manifest(hub, "main")
        del manifest["files"][rel]
        _put_manifest(hub, "main", manifest)
        with pytest.raises(el.ManifestVerificationError, match="lacks required") as err:
            el.pull_run(hub, REPO, "main", REV, tmp_path / rel)
        assert rel in err.value.expected
    # an ablation run needs neither
    hub = _hub(("s1_sin_l4", 0))
    el.pull_run(hub, REPO, "s1_sin_l4", REV, tmp_path / "s1")


@pytest.mark.parametrize("private", [False, None])
def test_a_non_private_or_unknown_visibility_repo_is_refused_before_listing(
    tmp_path: Path, private: bool | None
) -> None:
    hub = _hub(("main", 0))
    hub.private = private
    with pytest.raises(el.ManifestVerificationError, match="not private"):
        el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    assert hub.downloads == []


def test_an_unreadable_repo_visibility_fails_closed(tmp_path: Path) -> None:
    hub = _hub(("main", 0))

    def boom(repo_id: str) -> bool:
        raise OSError("401 unauthorized")

    hub.is_private = boom  # type: ignore[method-assign]
    with pytest.raises(el.ManifestVerificationError, match="cannot read the repo's visibility"):
        el.pull_run(hub, REPO, "main", REV, tmp_path / "main")
    assert hub.downloads == []


def test_a_non_sha_revision_is_refused_by_pull_run_itself(tmp_path: Path) -> None:
    hub = _hub(("main", 0))
    with pytest.raises(el.ManifestVerificationError, match="40-hex"):
        el.pull_run(hub, REPO, "main", "main", tmp_path / "main")
    assert hub.downloads == []


def test_a_rerun_re_verifies_the_cache_instead_of_trusting_it(tmp_path: Path) -> None:
    hub = _hub(("s1_sin_l4", 0))
    run_dir = tmp_path / "s1"
    el.pull_run(hub, REPO, "s1_sin_l4", REV, run_dir)
    n_first = len(hub.downloads)
    cached = run_dir / "source" / "runs" / "s1_sin_l4" / "bench.json"
    cached.write_bytes(b'{"n_sentences": 1}')  # tamper with the cached copy on disk
    el.pull_run(hub, REPO, "s1_sin_l4", REV, run_dir)  # re-downloads, the tamper is overwritten
    assert len(hub.downloads) == 2 * n_first
    assert json.loads(cached.read_bytes())["n_sentences"] == 200

    class StaleHub(FakeHub):
        """A hub client that trusts the local cache (returns without writing, like a hit)."""

        def download(self, repo_id: str, filename: str, revision: str, local_dir: Path) -> Path:
            self.downloads.append((repo_id, filename, revision))
            return Path(local_dir) / filename

    stale = StaleHub()
    stale.files = hub.files
    cached.write_bytes(b'{"n_sentences": 1}')
    with pytest.raises(el.ManifestVerificationError, match="size differs|sha256 differs"):
        el.pull_run(stale, REPO, "s1_sin_l4", REV, run_dir)


def test_cli_exits_non_zero_on_a_mismatch_before_any_scoring(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = _hub(("s1_sin_l4", 0))
    name = "runs/s1_sin_l4/predictions/seg_off/e1_predictions.json"
    hub.files[name] = hub.files[name] + b" "
    monkeypatch.setattr(el, "score_variant", lambda *a, **k: pytest.fail("scored"))
    argv = ["--hf-repo", REPO, "--revision", REV, "--run", "s1_sin_l4", "--out-root", str(tmp_path)]
    code = el.main(argv, hub=hub)
    err = capsys.readouterr().err
    assert code == 1 and name in err and "size differs" in err and REV in err
    assert not (tmp_path / "s1_sin_l4" / "seg_off").exists()


# --- predictions validation -----------------------------------------------------------------------


def test_load_predictions_refuses_wrong_ids_and_empty_values(tmp_path: Path) -> None:
    good = _preds("dev", 0)
    path = tmp_path / "p.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert len(el.load_predictions(path, "dev")) == len(good) > 20
    short = dict(list(good.items())[:-1])
    path.write_text(json.dumps(short), encoding="utf-8")
    with pytest.raises(el.ScoreError, match="ids do not match"):
        el.load_predictions(path, "dev")
    blank = {**good, next(iter(good)): "  "}
    path.write_text(json.dumps(blank), encoding="utf-8")
    with pytest.raises(el.ScoreError, match="empty"):
        el.load_predictions(path, "dev")


# --- evaluate_run: scoring, parity, never-decode, index -------------------------------------------


@pytest.fixture(scope="module")
def scored_main(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("eval") / "reports"
    index = _evaluate(_hub(("main", 0)), "main", out)
    return out, index


def test_scored_eval_json_has_every_view_and_a_passing_cli_parity(
    scored_main: tuple[Path, dict[str, Any]],
) -> None:
    out, _ = scored_main
    for variant in el.VARIANTS:
        ev = json.loads((out / "main" / variant / "eval.json").read_text(encoding="utf-8"))
        assert set(ev["sets"]) == set(el.SPLITS)
        assert ev["decoding_config"]["segment_threshold"] == (None if variant == "seg_off" else 128)
        assert ev["decoding_config"]["beam_size"] == 5 and ev["decoding_config"]["alpha"] == 0.8
        assert ev["decoding_config"]["bootstrap_seed"] == 1234
        assert ev["provenance"]["hf_revision"] == REV
        for split, entry in ev["sets"].items():
            assert entry["official_cli"]["parity"] is True, split
            assert {"official_bleu_ci", "official_chrf_ci", "sacrebleu"} <= set(entry)
            assert set(entry["official_ci_by_slice"]) == {"bleu", "chrf"}
            assert entry["official_bleu_ci"]["n_resamples"] == NB
        assert set(ev["sets"]["dev"]["official_ci_by_slice"]["chrf"]) == {
            "seen",
            "long",
            "unseen_domain",
        }
        assert ev["sets"]["dev"]["overall_ci"]["n"] == ev["sets"]["dev"]["n"] > 20
        assert len(ev["sets"]["e2synth"]["official_ci_by_slice"]["chrf"]) == 3  # 3 char buckets
        assert ev["sets"]["e2synth"]["synthetic"] is True
        assert set(ev["length_buckets_e1_e2_e3"]) <= {"<=10", "11-20", "21-40", "41-80", ">80"}
        # a reference-identical system scores perfectly under the official scorer
        assert ev["sets"]["e1"]["official"]["all"]["chrf"] == pytest.approx(100.0)


def test_scored_numbers_equal_the_official_scorer_run_directly(
    scored_main: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    from nmt.evaluate import run_official_scorer_cli

    out, _ = scored_main
    rep = run_official_scorer_cli(
        el.gold_path("dev"),
        out / "main" / "seg_off" / "dev_predictions.json",
        tmp_path / "cli.json",
    )
    ev = json.loads((out / "main" / "seg_off" / "eval.json").read_text(encoding="utf-8"))
    assert ev["sets"]["dev"]["official"]["OVERALL"] == pytest.approx(rep["OVERALL"])


def test_index_lists_every_artifact_with_its_sha_and_source_revision(
    scored_main: tuple[Path, dict[str, Any]],
) -> None:
    out, index = scored_main
    run_dir = out / "main"
    on_disk = {
        p.relative_to(run_dir).as_posix()
        for p in run_dir.rglob("*")
        if p.is_file() and p.name != "index.json"
    }
    listed = {a["path"] for a in index["artifacts"]}
    # everything except the files the pull itself did not download (the model is skipped)
    assert on_disk == listed
    assert index["hf_revision"] == REV and index["hf_repo"] == REPO
    for art in index["artifacts"]:
        assert el._sha256(run_dir / art["path"]) == art["sha256"]
        if art["origin"] == "hf_download":
            assert art["hf_revision"] == REV and art["hf_path"].startswith("runs/main/")
        else:
            assert art["derived_from_hf_revision"] == REV
    kinds = {a["origin"] for a in index["artifacts"]}
    assert kinds == {"hf_download", "derived"}
    assert (run_dir / "index.json").is_file()
    assert any(a["path"] == "seg_off/eval.json" for a in index["artifacts"])
    assert any(a["path"] == "source/runs/main/test_predictions.json" for a in index["artifacts"])


def test_the_pipeline_never_decodes_or_selects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import nmt.translate
    import nmt.tune

    def boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the local pipeline must never decode or tune")

    monkeypatch.setattr(nmt.translate.Translator, "translate", boom)
    monkeypatch.setattr(nmt.translate.Translator, "from_pretrained", boom)
    monkeypatch.setattr(nmt.tune, "run_tune", boom)
    _evaluate(_hub(("main", 0)), "main", tmp_path / "reports")


def test_source_of_the_pipeline_names_no_decoding_or_selection_machinery() -> None:
    source = (REPO_ROOT / "scripts" / "eval_local.py").read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in source.splitlines() if not ln.lstrip().startswith("#")
    )  # the preamble legitimately talks about what it does not do
    for forbidden in (
        "Translator",
        "nmt.tune",
        "nmt.selection",
        "nmt.hub",
        "load_pretrained",
        "run_tune",
        "select_winner",
        "selection_objective",
        ".translate(",
        "beam_search",
    ):
        assert forbidden not in code, forbidden
    imports = re.findall(r"^(?:from|import) (nmt[\w.]*)", source, re.MULTILINE)
    # nmt.eval_l4 is imported only for the manifest schema constants/helpers
    assert set(imports) <= {"nmt.analysis", "nmt.compare", "nmt.eval_l4", "nmt.evaluate"}


def test_an_official_cli_disagreement_is_a_loud_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = el.run_official_scorer_cli

    def tampered(gold: Path, pred: Path, out: Path) -> dict[str, Any]:
        rep = real(gold, pred, out)
        rep["all"] = {**rep["all"], "bleu": rep["all"]["bleu"] + 0.5}
        return rep

    monkeypatch.setattr(el, "run_official_scorer_cli", tampered)
    with pytest.raises(el.ScoreError, match="official CLI report differs"):
        _evaluate(_hub(("s1_sin_l4", 0)), "s1_sin_l4", tmp_path / "r")


# --- COMET ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query_result", "mode", "gpus", "needle"),
    [
        ("7000, 3", "auto", 1, "GPU free"),
        ("2000, 3", "auto", 0, "not free"),  # most of the VRAM is taken
        ("7000, 90", "auto", 0, "not free"),  # another job is computing
        (None, "auto", 0, "nvidia-smi unavailable"),
        ("garbage", "auto", 0, "could not parse"),
        ("2000, 99", "cpu", 0, "forced CPU"),
        (None, "gpu", 1, "forced GPU"),
    ],
)
def test_comet_device_choice(query_result: str | None, mode: str, gpus: int, needle: str) -> None:
    chosen, reason = el.choose_comet_device(lambda: query_result, mode)
    assert chosen == gpus and needle in reason


def _fake_comet(calls: list[int | None], fail_gpu: bool = False) -> Any:
    def run(
        triples: list[dict[str, str]], out: Path, timeout: float, gpus: int | None = None
    ) -> dict[str, Any]:
        calls.append(gpus)
        if fail_gpu and gpus == 1:
            raise RuntimeError("CUDA out of memory")
        result = {
            "model": "Unbabel/wmt22-comet-da",
            "n": len(triples),
            "scores": [0.5] * len(triples),
            "system_score": 0.5,
            "gpus": gpus,
        }
        out.write_text(json.dumps(result), encoding="utf-8")
        return result

    return run


def test_comet_runs_on_cpu_and_records_the_device_and_per_split_means(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int | None] = []
    monkeypatch.setattr(el, "run_comet", _fake_comet(calls))
    _evaluate(
        _hub(("s1_sin_l4", 0)),
        "s1_sin_l4",
        tmp_path / "r",
        comet="auto",
        comet_query=lambda: "1000, 99",
    )
    ev = json.loads((tmp_path / "r" / "s1_sin_l4" / "seg_off" / "eval.json").read_text("utf-8"))
    assert calls == [0, 0]  # one call per variant, both on CPU
    comet = ev["comet"]
    assert comet["device"]["gpus"] == 0 and "not free" in comet["device"]["reason"]
    assert comet["n"] == sum(e["n"] for e in ev["sets"].values())
    assert "scores" not in comet  # per-sentence scores stay in comet.json
    assert set(comet["system_score_by_split"]) == set(el.SPLITS)
    assert (tmp_path / "r" / "s1_sin_l4" / "seg_off" / "comet.json").is_file()


def test_comet_falls_back_to_cpu_when_the_free_gpu_fails_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int | None] = []
    triples = {s: [{"src": "a", "mt": "b", "ref": "c"}] for s in el.SPLITS}
    monkeypatch.setattr(el, "run_comet", _fake_comet(calls, fail_gpu=True))
    out = el.score_comet(triples, tmp_path / "c.json", "auto", lambda: "9000, 0")
    assert calls == [1, 0]
    assert out["device"]["gpus"] == 0 and "retried on CPU" in out["device"]["fallback"]
    with pytest.raises(RuntimeError, match="out of memory"):  # --comet gpu never hides it
        el.score_comet(triples, tmp_path / "c.json", "gpu", lambda: None)


def test_run_comet_passes_gpus_to_the_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess

    from nmt.evaluate import run_comet

    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_k: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        Path(cmd[cmd.index("--out") + 1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_comet([{"src": "a", "mt": "b", "ref": "c"}], tmp_path / "c.json", gpus=0)
    run_comet([{"src": "a", "mt": "b", "ref": "c"}], tmp_path / "d.json")
    assert seen[0][-2:] == ["--gpus", "0"] and "--gpus" not in seen[1]


# --- analysis -------------------------------------------------------------------------------------


def test_analysis_artifacts_are_produced_per_variant(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    _evaluate(_hub(("main", 3)), "main", out, do_analysis=True)
    for variant in el.VARIANTS:
        d = out / "main" / variant
        analysis = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
        assert {"gap_decomposition", "metric_artifact_share", "figures"} <= set(analysis)
        assert (d / "examples.json").is_file() and (
            d / "figures" / "failure_mode_rates.png"
        ).is_file()
        assert (d / "figures" / "chrf_vs_length_bucket.png").is_file()
    index = json.loads((out / "main" / "index.json").read_text(encoding="utf-8"))
    assert any(a["path"].endswith("figures/chrf_vs_rarity_decile.png") for a in index["artifacts"])


# --- paired tests ---------------------------------------------------------------------------------


def _chrf_entry(delta: float, lo: float, hi: float, p: float) -> dict[str, Any]:
    return {"overall": {"chrf": {"delta": delta, "ci_low": lo, "ci_high": hi, "p_value": p}}}


def test_prereg_readout_applies_the_section_3_decision_rules() -> None:
    cmp_ok = {
        "splits": {
            "e2": _chrf_entry(1.0, 0.2, 1.8, 0.01),
            "e2synth": _chrf_entry(2.0, 1.0, 3.0, 0.001),
            "e1": _chrf_entry(-0.1, -0.4, 0.2, 0.6),
        }
    }
    assert el.prereg_readout("H1", cmp_ok)["supported"] is True
    h2 = el.prereg_readout("H2", cmp_ok)
    assert h2["supported"] is True and len(h2["criteria"]) == 3
    bad_e1 = {"splits": {**cmp_ok["splits"], "e1": _chrf_entry(-0.8, -1.2, -0.5, 0.9)}}
    assert el.prereg_readout("H2", bad_e1)["supported"] is False  # non-inferiority fails
    assert el.prereg_readout("H1", bad_e1)["supported"] is True  # H1 has no E1 criterion
    not_sig = {"splits": {**cmp_ok["splits"], "e2": _chrf_entry(1.0, -0.2, 2.0, 0.07)}}
    assert el.prereg_readout("H1", not_sig)["supported"] is False
    negative = {"splits": {**cmp_ok["splits"], "e2synth": _chrf_entry(-1.0, -2.0, -0.1, 0.01)}}
    assert el.prereg_readout("H1", negative)["supported"] is False


def test_paired_tests_h1_and_h2_run_only_when_both_runs_are_available(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    hub = _hub(("s1_sin_l4", 3), ("s2_rope_l4", 0))
    _evaluate(hub, "s1_sin_l4", out)
    only_s1 = el.run_paired_tests(out, NB, 1234, el._complete_runs(out))
    assert only_s1 == []
    _evaluate(hub, "s2_rope_l4", out)
    available = el._complete_runs(out)
    assert set(available) == {"s1_sin_l4", "s2_rope_l4"}
    results = el.run_paired_tests(out, NB, 1234, available)
    assert {(r["hypothesis"], r["variant"]) for r in results} == {
        ("H1", "seg_off"),
        ("H1", "seg_tuned"),
    }  # H2 needs s3 as well
    primary = json.loads(
        (out / "compare" / "H1_s2_rope_l4_vs_s1_sin_l4_seg_off.json").read_text(encoding="utf-8")
    )
    assert primary["role"] == "primary" and primary["delta"] == "s2_rope_l4 - s1_sin_l4"
    assert "selection_objective" in primary and {"e1", "e2", "e3", "dev", "e2synth"} <= set(
        primary["splits"]
    )
    # s2 (reference-identical) is better than s1 (every 3rd word dropped): Delta chrF > 0
    assert primary["splits"]["e2"]["overall"]["chrf"]["delta"] > 0
    assert primary["prereg_readout"]["supported"] is True
    assert primary["source"]["s1_sin_l4"]["hf_revision"] == REV
    secondary = json.loads(
        (out / "compare" / "H1_s2_rope_l4_vs_s1_sin_l4_seg_tuned.json").read_text(encoding="utf-8")
    )
    assert secondary["role"] == "secondary"
    el.write_compare_index(out, results)
    idx = json.loads((out / "compare" / "index.json").read_text(encoding="utf-8"))
    assert len(idx["artifacts"]) == 2 and all(a["sha256"] for a in idx["artifacts"])


def test_h2_runs_with_s3_and_s2_and_uses_the_objective_bootstrap(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    hub = _hub(("s2_rope_l4", 0), ("s3_rope_concat_l4", 4))
    for run in ("s2_rope_l4", "s3_rope_concat_l4"):
        _evaluate(hub, run, out)
    results = el.run_paired_tests(out, NB, 1234, el._complete_runs(out))
    assert {r["hypothesis"] for r in results} == {"H2"}
    cmp = json.loads(
        (out / "compare" / "H2_s3_rope_concat_l4_vs_s2_rope_l4_seg_off.json").read_text("utf-8")
    )
    obj = cmp["selection_objective"]
    assert obj["delta"] < 0 and obj["n_resamples"] == NB  # s3 here is the degraded system
    assert (
        cmp["prereg_readout"]["supported"] is False and len(cmp["prereg_readout"]["criteria"]) == 3
    )


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_refuses_an_unpinned_revision_without_touching_the_hub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = _hub(("main", 0))
    code = el.main(
        ["--hf-repo", REPO, "--revision", "main", "--run", "main", "--out-root", str(tmp_path)],
        hub=hub,
    )
    assert code == 1 and hub.downloads == [] and "40-hex" in capsys.readouterr().err


def test_cli_end_to_end_with_a_per_run_revision_override(tmp_path: Path) -> None:
    hub = _hub(("s1_sin_l4", 3), ("s2_rope_l4", 0))
    other = "e" * 40
    code = el.main(
        [
            "--hf-repo",
            REPO,
            "--revision",
            REV,
            "--run",
            "s1_sin_l4",
            "--run",
            f"s2_rope_l4@{other}",
            "--out-root",
            str(tmp_path),
            "--n-bootstrap",
            str(NB),
            "--comet",
            "off",
            "--no-analysis",
        ],
        hub=hub,
    )
    assert code == 0
    revs = {(f.split("/")[1], rev) for _, f, rev in hub.downloads}
    assert revs == {("s1_sin_l4", REV), ("s2_rope_l4", other)}
    idx = json.loads((tmp_path / "s2_rope_l4" / "index.json").read_text(encoding="utf-8"))
    assert idx["hf_revision"] == other
    assert (tmp_path / "compare" / "H1_s2_rope_l4_vs_s1_sin_l4_seg_off.json").is_file()
    assert (tmp_path / "compare" / "index.json").is_file()


def test_cli_parser_defaults_match_the_preregistration() -> None:
    args = el._parse_args(["--hf-repo", REPO, "--run", "main"])
    assert args.n_bootstrap == 1000 and args.seed == 1234 and args.comet == "auto"
    assert el.PRIMARY_VARIANT == "seg_off"
    assert el.HYPOTHESES == (
        ("H1", "s2_rope_l4", "s1_sin_l4"),
        ("H2", "s3_rope_concat_l4", "s2_rope_l4"),
    )
