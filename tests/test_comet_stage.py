from __future__ import annotations

# Tests for nmt/comet_stage.py + nmt/comet_worker.py: the COMET-22 stage of the final_all Colab
# session. OFFLINE: the COMET scorer is a deterministic stub (a hash of the triple), the worker
# subprocess is replaced by an in-process runner that drives the REAL chunk/resume code, HF is a
# fake. No network, no GPU, no model. What these tests prove is the plumbing (set enumeration,
# dedup + mapping back, per-set resume, chunk resume, non-fatal failure, the second upload's
# guards, the install commands); that the REAL model produces scores through the same worker
# was checked by hand on CPU (see the PR description), not here.
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import nmt.comet_stage as cs
import nmt.comet_worker as cw
import nmt.eval_l4 as ev
from tests.test_eval_l4 import REPO, REVISION, FakeApi
from tests.test_final_all import _eval_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDS = ["main__beam", "main+A__mbr_beam8", "A__beam"]  # winner, runner-up, production


@pytest.fixture(autouse=True)
def tiny_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every k-th row of every split (about 12 per split): the plumbing runs in seconds."""
    real = cs.load_split_data.__wrapped__

    def small(split: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        ids, src, ref = real(split)
        keep = range(0, len(ids), max(1, len(ids) // 12))
        return tuple(ids[i] for i in keep), tuple(src[i] for i in keep), tuple(ref[i] for i in keep)

    cached = {s: small(s) for s in cs.SPLITS}
    monkeypatch.setattr(cs, "load_split_data", lambda split: cached[split])


# --- stub scorer + in-process worker --------------------------------------------------------------


def stub_score(t: dict[str, str]) -> float:
    """Deterministic pseudo COMET score in [0, 1) of one triple."""
    h = hashlib.sha256(f"{t['src']}|{t['mt']}|{t['ref']}".encode()).hexdigest()
    return int(h[:8], 16) % 10000 / 10000


class StubWorker:
    """Replaces the subprocess: parses the worker argv and runs cw.run_chunks with a counting stub
    scorer, then writes the meta file like the real worker. `die_after_chunks` raises once that
    many chunks were scored in a call (a disconnect)."""

    def __init__(self, die_after_chunks: int | None = None, write_meta: bool = True) -> None:
        self.write_meta = write_meta  # False: killed after the last chunk, before the meta write
        self.calls: list[list[str]] = []
        self.scored: list[int] = []  # triples scored per call
        self.die_after_chunks = die_after_chunks

    def __call__(self, cmd: Sequence[str]) -> None:
        self.calls.append(list(cmd))
        arg = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        triples = json.loads(Path(arg["--in"]).read_text(encoding="utf-8"))
        counter = {"chunks": 0, "triples": 0}
        base_meta = {
            "model": cw.COMET_MODEL,
            "model_revision": cw.COMET_MODEL_REVISION,
            "model_class": "RegressionMetric",
            "class_identifier": "regression_metric",
            "device": "stub-gpu",
            "precision": arg["--precision"],
            "batch_size": int(arg["--batch-size"]),
            "libraries": {"unbabel-comet": cs.COMET_VERSION},
        }

        def get_scorer() -> Any:
            def score(part: list[dict[str, str]]) -> list[float]:
                if self.die_after_chunks is not None and counter["chunks"] >= self.die_after_chunks:
                    raise cs.CometStageError("simulated disconnect")
                counter["chunks"] += 1
                counter["triples"] += len(part)
                return [stub_score(t) for t in part]

            return score

        try:
            stats = cw.run_chunks(
                triples,
                Path(arg["--cache-dir"]),
                int(arg["--chunk-size"]),
                get_scorer,
                log=lambda _m: None,
                chunk_meta=lambda: dict(base_meta),
            )
        finally:
            self.scored.append(counter["triples"])
        meta = {
            "schema": 1,
            "model": cw.COMET_MODEL,
            "model_revision": cw.COMET_MODEL_REVISION,
            "model_class": "RegressionMetric",
            "class_identifier": "regression_metric",
            "device": "stub-gpu",
            "precision": arg["--precision"],
            "batch_size": int(arg["--batch-size"]),
            "libraries": {"unbabel-comet": cs.COMET_VERSION},
            **stats,
        }
        if self.write_meta:
            ev._write_json(Path(arg["--meta-out"]), meta)


# --- a fake eval dir with every prediction file ---------------------------------------------------

# groups whose predictions are identical, so the expected dedup is known: main == s1, and a
# system's seg_tuned == seg_off
GROUP = {
    "final_all": "F",
    "final_all_report": "R",
    "main": "A",
    "s1_sin_l4": "A",
    "s2_rope_l4": "B",
    "s3_rope_concat_l4": "C",
}


def build_eval_root(tmp_path: Path, tag: str = "") -> Path:
    root = tmp_path / "eval"
    root.mkdir(parents=True, exist_ok=True)
    ev._write_json(
        root / "report.json",
        {"winner": CANDS[0], "runner_up": CANDS[1], "production": CANDS[2]},
    )
    for spec in cs.enumerate_sets(root, CANDS):
        if spec.system == cs.SYSTEM_BASELINE:
            continue
        ids, sources, _ = cs.load_split_data(spec.split)
        suffix = f" ~{GROUP[spec.system]}{tag}" + (
            f"{spec.variant}" if spec.system == cs.SYSTEM_REPORT else ""
        )
        ev._write_json(spec.pred, {i: s + suffix for i, s in zip(ids, sources, strict=True)})
    return root


def run_score(root: Path, worker: StubWorker, **kw: Any) -> dict[str, Any]:
    kw.setdefault("chunk_size", 40)
    return cs.score_stage(root, venv=root / "venv", runner=worker, log=lambda _m: None, **kw)


# --- the pinned runs and the plan -----------------------------------------------------------------


def test_pinned_revisions_equal_the_pull_records_in_reports_final() -> None:
    assert list(cs.PINNED_RUNS) == list(ev.RUNS)
    for run, rev in cs.PINNED_RUNS.items():
        rec = json.loads(
            (REPO_ROOT / "reports/final" / run / "pull_record.json").read_text("utf-8")
        )
        assert rec["hf_revision"] == rev and rec["private"] is True and len(rev) == 40


def test_the_comet_model_and_encoder_are_pinned_to_full_shas() -> None:
    assert len(cw.COMET_MODEL_REVISION) == 40 and len(cw.XLMR_REVISION) == 40
    assert cw.COMET_MODEL == "Unbabel/wmt22-comet-da"
    assert "apache-2.0" in cs.MODEL_LICENSE


# --- sets enumerated exactly ----------------------------------------------------------------------


def test_sets_are_enumerated_exactly(tmp_path: Path) -> None:
    sets = cs.enumerate_sets(tmp_path, CANDS)
    assert len(sets) == 10 + 6 + 40 + 5 == cs.expected_set_count(3) == 61
    keys = [(s.system, s.variant, s.split) for s in sets]
    assert len(set(keys)) == 61 and len({s.rel for s in sets}) == 61
    by_system: dict[str, int] = {}
    for s in sets:
        by_system[s.system] = by_system.get(s.system, 0) + 1
    assert by_system == {
        "final_all": 10,
        "final_all_report": 6,
        "main": 10,
        "s1_sin_l4": 10,
        "s2_rope_l4": 10,
        "s3_rope_concat_l4": 10,
        "copy_source": 5,
    }
    final_all = [s for s in sets if s.system == "final_all"]
    assert {(s.variant, s.split) for s in final_all} == {
        (v, p) for v in ("seg_off", "seg_tuned") for p in ("dev", "e1", "e2", "e2synth", "e3")
    }
    assert {s.split for s in sets if s.system == "final_all_report"} == {"e1", "e2"}
    assert {s.variant for s in sets if s.system == "final_all_report"} == set(CANDS)
    assert {s.variant for s in sets if s.system == "copy_source"} == {"baseline"}
    assert cs.enumerate_sets(tmp_path, CANDS[:2])[0] == sets[0]
    assert len(cs.enumerate_sets(tmp_path, CANDS[:2])) == cs.expected_set_count(2) == 59
    # file locations: the winner's and the report's from the eval dir, the runs' from the pull
    assert sets[0].pred == tmp_path / "predictions" / "seg_off" / "dev_predictions.json"
    rep = next(s for s in sets if s.system == "final_all_report")
    assert rep.pred == tmp_path / "report/predictions" / CANDS[0] / "e1_predictions.json"
    main = next(s for s in sets if s.system == "main")
    assert main.pred == cs.pulled_predictions_dir(tmp_path, "main") / "seg_off/dev_predictions.json"
    assert cs.pulled_predictions_dir(tmp_path, "main").parts[-4:] == (
        "source",
        "runs",
        "main",
        "predictions",
    )


def test_report_candidates_are_distinct_and_ordered(tmp_path: Path) -> None:
    ev._write_json(tmp_path / "report.json", {"winner": "a", "runner_up": "b", "production": "a"})
    assert cs.report_candidates(tmp_path) == ["a", "b"]
    ev._write_json(tmp_path / "report.json", {"winner": "a", "runner_up": "a", "production": "a"})
    assert cs.report_candidates(tmp_path) == ["a"]
    ev._write_json(tmp_path / "report.json", {"winner": "a"})
    with pytest.raises(cs.CometStageError, match="runner_up"):
        cs.report_candidates(tmp_path)
    with pytest.raises(cs.CometStageError, match="report.json is missing"):
        cs.report_candidates(tmp_path / "nope")


# --- the copy-the-source baseline -----------------------------------------------------------------


def test_the_baseline_is_the_source_text_per_id_and_deterministic(tmp_path: Path) -> None:
    for split in cs.SPLITS:
        ids, sources, _ = cs.load_split_data(split)
        assert cs.copy_source_predictions(split) == dict(zip(ids, sources, strict=True))
    first = [p.read_bytes() for p in cs.write_copy_source_baseline(tmp_path)]
    second = [p.read_bytes() for p in cs.write_copy_source_baseline(tmp_path)]
    assert first == second and len(first) == 5
    spec = cs.ScoreSet("copy_source", "baseline", "dev", cs.baseline_path(tmp_path, "dev"))
    prepared = cs.prepare_set(spec)
    assert all(src == hyp for src, hyp, _ in prepared.triples)


def test_prepare_set_refuses_wrong_ids_and_non_strings(tmp_path: Path) -> None:
    ids, sources, _ = cs.load_split_data("dev")
    path = tmp_path / "p.json"
    spec = cs.ScoreSet("x", "v", "dev", path)
    with pytest.raises(cs.CometStageError, match="missing"):
        cs.prepare_set(spec)
    ev._write_json(path, {i: s for i, s in list(zip(ids, sources, strict=True))[:-1]})
    with pytest.raises(cs.CometStageError, match="ids do not match"):
        cs.prepare_set(spec)
    ev._write_json(path, {**dict(zip(ids, sources, strict=True)), ids[0]: 5})
    with pytest.raises(cs.CometStageError, match="non-string"):
        cs.prepare_set(spec)


# --- dedup + mapping back -------------------------------------------------------------------------


def _prep(triples: list[tuple[str, str, str]], name: str = "s") -> cs.Prepared:
    spec = cs.ScoreSet(name, "v", "dev", Path("x"))
    return cs.Prepared(spec, tuple(f"id{i}" for i in range(len(triples))), tuple(triples), "", 0)


def test_dedup_scores_identical_triples_once_and_maps_back_in_order() -> None:
    a = _prep([("s1", "h1", "r1"), ("s2", "h2", "r2"), ("s1", "h1", "r1")], "a")
    b = _prep([("s2", "h2", "r2"), ("s3", "h3", "r3"), ("s1", "hX", "r1")], "b")
    unique, layout = cs.dedup_triples([a, b])
    assert unique == [
        ("s1", "h1", "r1"),
        ("s2", "h2", "r2"),
        ("s3", "h3", "r3"),
        ("s1", "hX", "r1"),
    ]  # first-occurrence order; same src+ref with another hypothesis stays separate
    assert layout == [[0, 1, 0], [1, 2, 3]]
    scores = [0.1, 0.2, 0.3, 0.4]
    assert [[scores[i] for i in idx] for idx in layout] == [[0.1, 0.2, 0.1], [0.2, 0.3, 0.4]]
    assert cs.dedup_triples([]) == ([], [])


# --- bootstrap ------------------------------------------------------------------------------------


def test_bootstrap_is_deterministic_brackets_the_mean_and_matches_a_manual_resample() -> None:
    scores = [0.1 * (i % 7) + 0.01 * i for i in range(50)]
    a = cs.bootstrap_mean_ci(scores)
    assert a == cs.bootstrap_mean_ci(scores)
    assert a["n_resamples"] == 1000 and a["seed"] == 1234
    assert a["mean"] == pytest.approx(sum(scores) / 50)
    lo, hi = a["ci95"]
    assert lo <= a["mean"] <= hi and lo < hi
    import numpy as np

    idx = np.random.default_rng(1234).integers(0, 50, size=(1000, 50))
    means = np.sort(np.asarray(scores)[idx].mean(axis=1))
    assert (lo, hi) == (pytest.approx(means[25]), pytest.approx(means[975]))
    assert cs.bootstrap_mean_ci(scores, seed=1)["ci95"] != a["ci95"]


def test_bootstrap_degenerate_inputs() -> None:
    one = cs.bootstrap_mean_ci([0.7])
    assert one["ci95"] == [0.7, 0.7] and one["mean"] == 0.7
    const = cs.bootstrap_mean_ci([0.5] * 10)
    assert const["ci95"] == [0.5, 0.5]
    with pytest.raises(cs.CometStageError, match="empty"):
        cs.bootstrap_mean_ci([])


# --- the score step end to end (stub worker) ------------------------------------------------------


def test_score_stage_writes_every_set_with_provenance_and_dedups(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    worker = StubWorker()
    summary = run_score(root, worker, batch_size=16, precision="fp32")
    sets = cs.enumerate_sets(root, CANDS)
    assert summary["n_sets"] == 61 and len(summary["sets"]) == 61
    prepared = [cs.prepare_set(s) for s in sets]
    unique, _ = cs.dedup_triples(prepared)
    assert summary["n_distinct_triples_all_sets"] == len(unique)
    assert worker.scored == [len(unique)]  # every distinct triple scored exactly once
    assert sum(len(p.triples) for p in prepared) > len(unique)  # dedup really removed some
    for p in prepared:
        rec = json.loads(cs.set_output_path(root, p.spec).read_text(encoding="utf-8"))
        assert rec["scores"] == [stub_score({"src": s, "mt": h, "ref": r}) for s, h, r in p.triples]
        assert rec["ids"] == list(p.ids) and rec["n"] == len(p.ids)
        assert rec["input_sha256"] == hashlib.sha256(p.spec.pred.read_bytes()).hexdigest()
        assert rec["mean"] == pytest.approx(sum(rec["scores"]) / rec["n"])
        assert rec["ci95"][0] <= rec["mean"] <= rec["ci95"][1]
        assert (rec["n_resamples"], rec["seed"]) == (1000, 1234)
        assert rec["model"]["name"] == "Unbabel/wmt22-comet-da"
        assert rec["model"]["revision"] == cw.COMET_MODEL_REVISION
        assert rec["device"] == "stub-gpu" and rec["batch_size"] == 16
        assert rec["precision"] == "fp32" and rec["libraries"]["unbabel-comet"] == "2.2.7"
        assert rec["runtime"]["scoring_wall_seconds"] >= 0.0
    # identical predictions share their score, a changed hypothesis does not
    f_off = json.loads((root / "comet/final_all/seg_off/e1.json").read_text("utf-8"))
    f_tuned = json.loads((root / "comet/final_all/seg_tuned/e1.json").read_text("utf-8"))
    assert f_off["scores"] == f_tuned["scores"] and f_off["n_distinct_triples"] == f_off["n"]
    main = json.loads((root / "comet/main/seg_off/e1.json").read_text("utf-8"))
    s2 = json.loads((root / "comet/s2_rope_l4/seg_off/e1.json").read_text("utf-8"))
    assert main["scores"] != s2["scores"]
    # worker argv: module, batch size, precision, chunk size
    cmd = worker.calls[0]
    assert cmd[1:3] == ["-m", "nmt.comet_worker"]
    assert cmd[cmd.index("--batch-size") + 1] == "16" and cmd[cmd.index("--device") + 1] == "cuda"
    # summary: every set listed with the sha256 of its input and of its output
    row = next(r for r in summary["sets"] if r["system"] == "main" and r["split"] == "e1")
    assert row["file"] == "main/seg_off/e1.json" and len(row["output_sha256"]) == 64
    assert summary["model"]["revision"] == cw.COMET_MODEL_REVISION
    assert summary["bootstrap"] == {"n_resamples": 1000, "seed": 1234}


def test_baseline_scores_equal_scoring_the_source_against_the_reference(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    run_score(root, StubWorker())
    rec = json.loads((root / "comet/copy_source/baseline/dev.json").read_text("utf-8"))
    ids, sources, refs = cs.load_split_data("dev")
    assert rec["scores"] == [
        stub_score({"src": s, "mt": s, "ref": r}) for s, r in zip(sources, refs, strict=True)
    ]
    assert rec["input_file"] == "dev_predictions.json"


def test_rerun_with_everything_valid_scores_nothing(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    worker = StubWorker()
    first = run_score(root, worker)
    second = run_score(root, worker)
    assert len(worker.calls) == 1  # resumable per set: the second run never reaches the worker
    assert [r["output_sha256"] for r in second["sets"]] == [
        r["output_sha256"] for r in first["sets"]
    ]


def test_a_missing_or_changed_set_is_the_only_one_rescored(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    worker = StubWorker()
    run_score(root, worker)
    gone = root / "comet/s2_rope_l4/seg_tuned/e2.json"
    gone.unlink()
    changed = cs.pulled_predictions_dir(root, "main") / "seg_off" / "dev_predictions.json"
    pred = json.loads(changed.read_text("utf-8"))
    first = next(iter(pred))
    pred[first] = pred[first] + " CHANGED"
    ev._write_json(changed, pred)
    kept = (root / "comet/s3_rope_concat_l4/seg_off/e1.json").read_bytes()
    run_score(root, worker)
    assert len(worker.calls) == 2
    prepared = [
        cs.prepare_set(s)
        for s in cs.enumerate_sets(root, CANDS)
        if (s.system, s.variant, s.split)
        in {("s2_rope_l4", "seg_tuned", "e2"), ("main", "seg_off", "dev")}
    ]
    assert worker.scored[1] == len(cs.dedup_triples(prepared)[0])
    assert gone.is_file()
    assert (root / "comet/s3_rope_concat_l4/seg_off/e1.json").read_bytes() == kept  # untouched
    new = json.loads((root / "comet/main/seg_off/dev.json").read_text("utf-8"))
    assert new["input_sha256"] == hashlib.sha256(changed.read_bytes()).hexdigest()


def test_a_different_precision_does_not_reuse_scores(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    worker = StubWorker()
    run_score(root, worker, precision="fp32")
    run_score(root, worker, precision="bf16")
    assert len(worker.calls) == 2 and worker.scored[0] == worker.scored[1]


def test_a_disconnect_mid_scoring_resumes_from_the_finished_chunks(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    dying = StubWorker(die_after_chunks=2)
    with pytest.raises(cs.CometStageError, match="simulated disconnect"):
        run_score(root, dying, chunk_size=40)
    assert not (root / "comet" / cs.SUMMARY_NAME).exists()  # no partial summary
    assert not (root / "comet/final_all/seg_off/dev.json").exists()  # no partial set files
    assert len(list((root / "comet/_cache/fp32-2760a223").glob("chunk_*.json"))) == 2
    worker = StubWorker()
    summary = run_score(root, worker, chunk_size=40)
    total = summary["n_distinct_triples_all_sets"]
    assert worker.scored == [total - 80]  # the two finished chunks (40 triples each) are reused
    assert summary["n_sets"] == 61


def test_a_stale_chunk_for_other_triples_is_not_trusted(tmp_path: Path) -> None:
    triples = [{"src": "a", "mt": "b", "ref": "c"}, {"src": "d", "mt": "e", "ref": "f"}]
    cache = tmp_path / "cache"
    seen: list[int] = []

    def scorer() -> Any:
        return lambda part: seen.append(len(part)) or [0.5] * len(part)

    cw.run_chunks(triples, cache, 10, scorer, log=lambda _m: None)
    other = [{"src": "a", "mt": "b", "ref": "c"}, {"src": "X", "mt": "e", "ref": "f"}]
    stats = cw.run_chunks(other, cache, 10, scorer, log=lambda _m: None)
    assert seen == [2, 2] and stats["skipped"] == 0
    stats = cw.run_chunks(other, cache, 10, scorer, log=lambda _m: None)
    assert stats["skipped"] == 1 and seen == [2, 2]  # now it matches: no model needed


def test_the_model_is_not_loaded_when_every_chunk_is_cached(tmp_path: Path) -> None:
    triples = [{"src": "a", "mt": "b", "ref": "c"}]
    cw.run_chunks(triples, tmp_path, 5, lambda: lambda p: [0.25] * len(p), log=lambda _m: None)

    def boom() -> Any:
        raise AssertionError("the model must not be loaded")

    stats = cw.run_chunks(triples, tmp_path, 5, boom, log=lambda _m: None)
    assert stats["skipped"] == 1 and stats["scored_triples"] == 0


@pytest.mark.parametrize(
    "bad", ["notjson", '{"scores": [0.1]}', "chunk_sha_mismatch", "nan", "short"]
)
def test_read_chunk_fails_closed(tmp_path: Path, bad: str) -> None:
    triples = [{"src": "a", "mt": "b", "ref": "c"}, {"src": "d", "mt": "e", "ref": "f"}]
    path = cw.chunk_path(tmp_path, 0)
    good = {"chunk_sha256": cw.chunk_sha256(triples), "scores": [0.1, 0.2]}
    path.parent.mkdir(parents=True, exist_ok=True)
    if bad == "notjson":
        path.write_text("{", encoding="utf-8")
    elif bad == "chunk_sha_mismatch":
        path.write_text(json.dumps({**good, "chunk_sha256": "0" * 64}), encoding="utf-8")
    elif bad == "nan":
        path.write_text(
            '{"chunk_sha256": "' + good["chunk_sha256"] + '", "scores": [0.1, NaN]}', "utf-8"
        )
    elif bad == "short":
        path.write_text(json.dumps({**good, "scores": [0.1]}), encoding="utf-8")
    else:
        path.write_text(bad, encoding="utf-8")
    assert cw.read_chunk(path, triples) is None
    path.write_text(json.dumps(good), encoding="utf-8")
    assert cw.read_chunk(path, triples) == [0.1, 0.2]


# --- non-fatal failure behaviour and exit codes ---------------------------------------------------


def test_cli_exits_non_zero_with_a_failed_line_and_leaves_no_partial_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cs.main(["score", "--eval-root", str(tmp_path), "--venv", str(tmp_path / "v")])
    err = capsys.readouterr().err
    assert code == 1 and "comet_stage score: FAILED" in err and "report.json" in err
    root = build_eval_root(tmp_path / "e")
    (cs.pulled_predictions_dir(root, "main") / "seg_off" / "dev_predictions.json").unlink()
    code = cs.main(["score", "--eval-root", str(root), "--venv", str(tmp_path / "v")])
    assert code == 1 and "prediction file(s) missing" in capsys.readouterr().err
    assert not (root / "comet" / cs.SUMMARY_NAME).exists()


def test_the_worker_never_receives_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"PATH": "p", "HF_TOKEN": "s1", "HF_TOKEN_WRITE": "s2", "HUGGINGFACEHUB_API_TOKEN": "s3"}
    assert cs.worker_env({**env, "GH_TOKEN": "s4", "WANDB_API_KEY": "s5"}) == {"PATH": "p"}
    seen: dict[str, Any] = {}

    class Done:
        returncode = 0

    def fake_run(cmd: Any, **kw: Any) -> Done:
        seen.update(kw)
        return Done()

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    cs.run_worker_subprocess(["x"])
    assert "HF_TOKEN" not in seen["env"] and "HUGGINGFACEHUB_API_TOKEN" not in seen["env"]
    assert seen["cwd"] == ev.REPO_ROOT and seen["env"]["PATH"] == "p"


def test_a_worker_failure_is_a_comet_stage_error_naming_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Done:
        returncode = 3

    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(cs.CometStageError, match="exit 3"):
        cs.run_worker_subprocess(["x"])


# --- pull (manifest-verified, pinned, private) ----------------------------------------------------


@pytest.fixture
def eval_local_fakes() -> Any:
    """tests.test_eval_local's FakeHub / run-file builder. Its module-level prediction cache is
    keyed by split name only and filled from the FULL splits when this module calls it, which
    would poison that module's own tests (they run on tiny splits): cleared afterwards."""
    from tests import test_eval_local as t

    yield t
    t._PRED_CACHE.clear()


def test_pull_is_pinned_private_and_manifest_verified(
    tmp_path: Path, eval_local_fakes: Any
) -> None:
    FakeHub, _run_files = eval_local_fakes.FakeHub, eval_local_fakes._run_files  # noqa: N806

    rev = "d" * 40
    hub = FakeHub()
    hub.files.update(_run_files("main", 0))
    out = cs.pull_runs(tmp_path, "o/r", hub=hub, runs={"main": rev})
    assert out["main"]["verified"] is True and out["main"]["hf_revision"] == rev
    assert {r for _, _, r in hub.downloads} == {rev}
    assert (
        cs.pulled_predictions_dir(tmp_path, "main") / "seg_off" / "dev_predictions.json"
    ).is_file()
    # a public repo is refused before anything is read
    hub.private, hub.downloads = False, []
    with pytest.raises(cs.CometStageError, match="not private"):
        cs.pull_runs(tmp_path / "x", "o/r", hub=hub, runs={"main": rev})
    assert hub.downloads == []
    # a branch name is not a pin
    hub.private = True
    with pytest.raises(cs.CometStageError, match="40-hex"):
        cs.pull_runs(tmp_path / "y", "o/r", hub=hub, runs={"main": "main"})
    # a tampered file is refused by the manifest check
    hub.files["runs/main/predictions/seg_off/e1_predictions.json"] = b"{}"
    with pytest.raises(cs.CometStageError, match="sha256|size"):
        cs.pull_runs(tmp_path / "z", "o/r", hub=hub, runs={"main": rev})


# --- the second upload ----------------------------------------------------------------------------


def comet_ready(tmp_path: Path) -> tuple[Path, dict[str, Path], FakeApi]:
    """A final_all eval dir with its MAIN upload done on a fake repo and COMET scored."""
    (tmp_path / "fa").mkdir()
    root, models = _eval_dir(tmp_path / "fa")
    api = FakeApi()
    ev.upload_run(api, REPO, "final_all", root, models)
    ev._write_json(
        root / "report.json", {"winner": CANDS[0], "runner_up": CANDS[1], "production": CANDS[2]}
    )
    scored = build_eval_root(tmp_path / "scored")
    run_score(scored, StubWorker())
    import shutil

    shutil.copytree(scored / "comet", root / "comet")
    return root, models, api


def test_upload_comet_is_a_second_private_commit_with_its_own_manifest(tmp_path: Path) -> None:
    root, models, api = comet_ready(tmp_path)
    rec = cs.upload_comet(api, REPO, root)
    assert api.titles == [ev.commit_message("final_all"), "eval_l4: final_all COMET"]
    ops = api.commits[1]
    assert all(p.startswith("runs/final_all/comet/") for p in ops)
    assert "runs/final_all/comet/comet_manifest.json" in ops
    assert "runs/final_all/comet/comet_summary.json" in ops
    assert "runs/final_all/comet/main/seg_off/e1.json" in ops
    assert not any("_cache" in p or "_pulled" in p or "_inputs" in p for p in ops)
    assert len([p for p in ops if p.endswith(".json")]) == 61 + 2  # 61 sets + summary + manifest
    manifest = json.loads(api.remote["runs/final_all/comet/comet_manifest.json"])
    assert set(manifest["files"]) == {p.removeprefix("runs/final_all/comet/") for p in ops} - {
        "comet_manifest.json"
    }
    for rel, entry in manifest["files"].items():
        data = api.remote[f"runs/final_all/comet/{rel}"]
        assert entry == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert manifest["model_revision"] == cw.COMET_MODEL_REVISION
    assert rec["private"] is True and rec["revision"] == REVISION and rec["kind"] == "comet"
    saved = json.loads((root / "comet_hf_upload.json").read_text("utf-8"))
    assert saved["revision"] == REVISION and saved["private"] is True


def test_the_main_upload_still_verifies_complete_with_and_without_comet(tmp_path: Path) -> None:
    root, models, api = comet_ready(tmp_path)
    assert ev.verify_run_on_hf(api, REPO, "final_all")["complete"] is True  # before COMET
    cs.upload_comet(api, REPO, root)
    assert any(p.startswith("runs/final_all/comet/") for p in api.remote)
    assert ev.verify_run_on_hf(api, REPO, "final_all")["complete"] is True  # with COMET files
    # the main upload is still skipped on its own evidence
    again = FakeApi()
    again.remote, again.titles = dict(api.remote), list(api.titles)
    ev.upload_run(again, REPO, "final_all", root, models)
    assert again.commits == []


def test_upload_comet_is_skipped_when_the_same_upload_is_verified_on_hf(tmp_path: Path) -> None:
    root, _models, api = comet_ready(tmp_path)
    cs.upload_comet(api, REPO, root)
    assert len(api.commits) == 2
    cs.upload_comet(api, REPO, root)
    assert len(api.commits) == 2  # no third commit
    (root / "comet" / "main/seg_off/e1.json").write_text("{}", encoding="utf-8")
    ev_summary = json.loads((root / "comet" / cs.SUMMARY_NAME).read_text("utf-8"))
    assert ev_summary["sets"]  # a changed local file means a different manifest: uploaded again
    cs.upload_comet(api, REPO, root)
    assert len(api.commits) == 3


def test_upload_comet_refuses_without_the_main_upload(tmp_path: Path) -> None:
    root, _models, _api = comet_ready(tmp_path)
    fresh = FakeApi()
    with pytest.raises(cs.CometStageError, match="main final_all upload is not complete"):
        cs.upload_comet(fresh, REPO, root)
    assert fresh.commits == []


def test_upload_comet_refuses_a_public_repo_and_a_read_only_token(tmp_path: Path) -> None:
    root, _models, api = comet_ready(tmp_path)
    api.private = False
    with pytest.raises(ev.HFNotPrivateError):
        cs.upload_comet(api, REPO, root)
    assert len(api.commits) == 1
    api.private = True
    from tests.test_eval_l4 import READ

    api.who = READ
    with pytest.raises(ev.HFWriteTokenError):
        cs.upload_comet(api, REPO, root)
    assert len(api.commits) == 1


def test_upload_comet_checks_privacy_again_after_the_commit(tmp_path: Path) -> None:
    root, _models, api = comet_ready(tmp_path)
    api.flip = True  # the repo turns public once a commit has happened (the main one already did)
    api.commits = []  # ...so reset: only the COMET commit counts for the flip
    with pytest.raises(ev.HFNotPrivateError):
        cs.upload_comet(api, REPO, root)


def test_upload_comet_needs_a_finished_score_step(tmp_path: Path) -> None:
    root, _models, api = comet_ready(tmp_path)
    (root / "comet" / cs.SUMMARY_NAME).unlink()
    with pytest.raises(cs.CometStageError, match="score step first"):
        cs.upload_comet(api, REPO, root)
    ev._write_json(root / "comet" / cs.SUMMARY_NAME, {"sets": [{"file": "nope/x/y.json"}]})
    with pytest.raises(cs.CometStageError, match="missing file"):
        cs.upload_comet(api, REPO, root)
    assert len(api.commits) == 1


# --- install --------------------------------------------------------------------------------------


def test_install_mode_follows_the_python_version() -> None:
    assert cs.install_mode("auto", (3, 12)) == "resolve"
    assert cs.install_mode("auto", (3, 13)) == "nodeps"
    assert cs.install_mode("auto", (3, 14)) == "nodeps"
    assert cs.install_mode("resolve", (3, 13)) == "resolve"
    with pytest.raises(cs.CometStageError):
        cs.install_mode("pip", (3, 12))


def test_constraints_pin_the_installed_torch_and_numpy_only_without_deps() -> None:
    assert cs.constraint_lines("resolve", "2.9.0+cu126", "2.0.2") == ["torch==2.9.0+cu126"]
    assert cs.constraint_lines("nodeps", "2.9.0+cu126", "2.0.2") == [
        "torch==2.9.0+cu126",
        "numpy==2.0.2",
    ]


def test_install_commands_use_a_system_site_venv_wheels_only_and_the_constraints() -> None:
    venv, cons = Path("/content/v"), Path("/content/c.txt")
    resolve = cs.install_commands("py", venv, cons, "resolve")
    assert resolve[0][1] == [
        "py",
        "-m",
        "venv",
        "--system-site-packages",
        "--without-pip",
        str(venv),
    ]
    pip = resolve[1][1]
    assert pip[:5] == ["py", "-m", "pip", "--python", str(cs.venv_python(venv))]
    assert "--only-binary=:all:" in pip and pip[pip.index("-c") + 1] == str(cons)
    assert "unbabel-comet==2.2.7" in pip and "setuptools<81" in pip and "--no-deps" not in pip
    assert not any("torch" in a for a in pip if not a.startswith("-"))  # torch is never requested
    nodeps = cs.install_commands("py", venv, cons, "nodeps")
    assert [n for n, _ in nodeps] == ["venv", "pip comet --no-deps", "pip dependencies"]
    assert "--no-deps" in nodeps[1][1] and "--no-deps" not in nodeps[2][1]
    assert "jsonargparse==3.13.1" in nodeps[2][1]
    assert not any(a.startswith("numpy") for a in nodeps[2][1])  # numpy stays the installed 2.x


class FakeRun:
    """subprocess.run stand-in: records argv; `probe` is what the venv probe prints."""

    def __init__(self, probe: dict[str, str] | None, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.probe = probe
        self.fail_on = fail_on
        self.pip_done = False

    def __call__(self, argv: list[str], **_kw: Any) -> Any:
        self.calls.append(list(argv))
        out, code = "", 0
        if (
            argv[-2:-1] == ["-c"]
            or "-c" in argv
            and argv[0].endswith("python")
            and "import" in argv[-1]
        ):
            out = json.dumps(self.probe) if (self.probe and self.pip_done) else ""
            code = 0 if (self.probe and self.pip_done) else 1
        elif "pip" in argv:
            self.pip_done = True
            if self.fail_on and self.fail_on in " ".join(argv):
                code = 1
        return type("R", (), {"returncode": code, "stdout": out})()


def test_install_runs_the_steps_and_checks_that_torch_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cs, "installed_version", lambda p: {"torch": "2.9.0+cu126", "numpy": "2.0.2"}[p]
    )
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    run = FakeRun({"unbabel-comet": "2.2.7", "torch": "2.9.0+cu126", "numpy": "1.26.4"})
    rec = cs.install_comet(
        venv, tmp_path / "c.txt", "auto", python="py", run=run, python_version=(3, 12)
    )
    assert rec["mode"] == "resolve" and rec["skipped"] is False
    assert (tmp_path / "c.txt").read_text("utf-8") == "torch==2.9.0+cu126\n"
    assert any("--system-site-packages" in c for c in run.calls)
    # the venv already has it: SKIP, nothing but the probe runs
    again = FakeRun({"unbabel-comet": "2.2.7", "torch": "2.9.0+cu126", "numpy": "1.26.4"})
    again.pip_done = True
    rec = cs.install_comet(
        venv, tmp_path / "c.txt", "auto", python="py", run=again, python_version=(3, 12)
    )
    assert rec["skipped"] is True and len(again.calls) == 1
    assert "SKIP" in capsys.readouterr().out


def test_install_on_313_takes_the_nodeps_path_and_says_it_is_unverified_on_colab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cs, "installed_version", lambda p: {"torch": "2.9.0", "numpy": "2.0.2"}[p])
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    run = FakeRun({"unbabel-comet": "2.2.7", "torch": "2.9.0", "numpy": "2.0.2"})
    cs.install_comet(venv, tmp_path / "c.txt", "auto", python="py", run=run, python_version=(3, 13))
    assert (tmp_path / "c.txt").read_text("utf-8") == "torch==2.9.0\nnumpy==2.0.2\n"
    assert "UNVERIFIED ON COLAB" in capsys.readouterr().out
    assert any("--no-deps" in c for c in run.calls)


def test_install_failure_message_names_the_step_python_and_untouched_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cs, "installed_version", lambda p: {"torch": "2.9.0", "numpy": "2.0.2"}[p])
    run = FakeRun(
        {"unbabel-comet": "2.2.7", "torch": "2.9.0", "numpy": "1"}, fail_on="unbabel-comet"
    )
    with pytest.raises(cs.CometStageError) as err:
        cs.install_comet(
            tmp_path / "v", tmp_path / "c.txt", "auto", python="py", run=run, python_version=(3, 12)
        )
    msg = str(err.value)
    assert "step 'pip'" in msg and "python 3.12" in msg and "torch 2.9.0 was not modified" in msg


def test_install_refuses_when_the_venv_sees_another_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cs, "installed_version", lambda p: {"torch": "2.9.0", "numpy": "2.0.2"}[p])
    (tmp_path / "v" / "bin").mkdir(parents=True)
    (tmp_path / "v" / "bin" / "python").write_text("", encoding="utf-8")
    run = FakeRun({"unbabel-comet": "2.2.7", "torch": "2.14.0+cpu", "numpy": "1.26.4"})
    with pytest.raises(cs.CometStageError, match="changed torch"):
        cs.install_comet(
            tmp_path / "v",
            tmp_path / "c.txt",
            "resolve",
            python="py",
            run=run,
            python_version=(3, 12),
        )


# --- worker helpers -------------------------------------------------------------------------------


def test_pinned_model_dir_changes_only_the_pretrained_model_line(tmp_path: Path) -> None:
    model = tmp_path / "model"
    (model / "checkpoints").mkdir(parents=True)
    (model / "checkpoints" / "model.ckpt").write_bytes(b"ckpt")
    (model / "hparams.yaml").write_text(
        "a: 1\npretrained_model: xlm-roberta-large\nb: 2\n", "utf-8"
    )
    enc = tmp_path / "enc"
    enc.mkdir()
    ckpt = cw.prepare_pinned_model_dir(tmp_path / "work", model, enc)
    assert ckpt.read_bytes() == b"ckpt"
    text = (tmp_path / "work" / "hparams.yaml").read_text("utf-8")
    assert text == f"a: 1\npretrained_model: {enc.as_posix()}\nb: 2\n"
    (model / "hparams.yaml").write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(cw.WorkerError, match="no pretrained_model"):
        cw.prepare_pinned_model_dir(tmp_path / "work2", model, enc)


def test_device_resolution_never_falls_back_silently_from_cuda() -> None:
    assert cw.resolve_device("auto", True) == 1 and cw.resolve_device("auto", False) == 0
    assert cw.resolve_device("cpu", True) == 0 and cw.resolve_device("cuda", True) == 1
    with pytest.raises(cw.WorkerError, match="cuda"):
        cw.resolve_device("cuda", False)
    with pytest.raises(cw.WorkerError):
        cw.resolve_device("tpu", True)


def test_precision_is_autocast_not_a_weight_cast() -> None:
    assert cw.autocast_dtype("fp32") is None  # the reference numerics: no autocast at all
    assert cw.autocast_dtype("bf16") == "bfloat16" and cw.autocast_dtype("fp16") == "float16"
    with pytest.raises(cw.WorkerError, match="int8"):
        cw.autocast_dtype("int8")


def test_chunk_bounds_cover_the_list_in_order() -> None:
    assert cw.chunk_bounds(0, 5) == []
    assert cw.chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(cw.WorkerError):
        cw.chunk_bounds(3, 0)


# --- the dry-run section --------------------------------------------------------------------------


def test_planned_counts_are_exact_for_the_runs_and_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()  # the real splits: these are the numbers the PR reports
    counts = cs.planned_counts(tmp_path, REPO_ROOT / "reports" / "final")
    assert counts["segments"] == {"runs": 35120, "baseline": 4390, "final_all": 17600}
    assert counts["distinct_exact"]["runs"] == 17294  # == reports/final comet_partial_cpu summary
    assert counts["distinct_exact"]["baseline"] == 4385
    assert counts["distinct_exact"]["final_all"] is None and counts["exact"] is False
    assert counts["distinct_upper_bound"] == counts["distinct_known_union"] + 17600
    assert counts["distinct_known_union"] <= 17294 + 4385


def test_planned_counts_become_exact_once_final_all_exists(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    fake_runs = tmp_path / "runs"
    for run in cs.PINNED_RUNS:
        for v in cs.VARIANTS:
            for s in cs.SPLITS:
                ids, src, _ = cs.load_split_data(s)
                ev._write_json(
                    fake_runs / run / v / f"{s}_predictions.json", dict(zip(ids, src, strict=True))
                )
    counts = cs.planned_counts(root, fake_runs)
    assert counts["exact"] is True and counts["n_candidates"] == 3
    assert counts["distinct_upper_bound"] == counts["distinct_known_union"]


def test_describe_stage_lists_sets_counts_model_device_batch_and_the_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()  # the real splits, like the files in reports/final
    lines = "\n".join(
        cs.describe_stage(
            tmp_path, REPO_ROOT / "reports" / "final", batch_size=32, precision="bf16"
        )
    )
    for needle in (
        "COMET stage: after the upload, NON-FATAL",
        "61 sets",
        "Unbabel/wmt22-comet-da @ " + cw.COMET_MODEL_REVISION,
        "apache-2.0",
        "batch size 32",
        "precision bf16",
        "final_all: seg_off, seg_tuned",
        "copy_source: source text copied",
        "@c3d8598252853fcd7df1ef4a00e8b0382b8f4351",
        "@ac92b8da971fdf64974089f40e8f4ddbf9d9638b",
        "distinct (exact",
        "ASSUMED 50 triples/s",
        "ASSUMED 100 triples/s",
        "ASSUMED 150 triples/s",
        "NO L4 COMET rate has been measured",
    ):
        assert needle in lines, needle


def test_estimate_comet_arithmetic_is_labelled_assumed() -> None:
    est = cs.estimate_comet(30000)
    rows = {r["triples_per_second"]: r for r in est["rows"]}
    assert rows[100.0]["scoring_seconds"] == 300.0
    assert rows[50.0]["total_seconds"] == 600.0 + cs.ASSUMED_COMET_FIXED_SECONDS
    assert rows[150.0]["total_hours"] == pytest.approx((200.0 + 600.0) / 3600)
    assert cs.ASSUMED_COMET_FIXED_SECONDS == 600.0


# --- N1: the model dir is local, never under the (Drive) eval dir ---------------------------------


def test_the_model_dir_is_local_by_default_and_refused_under_the_eval_dir(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    assert cs.DEFAULT_MODEL_DIR == "/content/comet_model"
    assert not Path(cs.DEFAULT_MODEL_DIR).is_relative_to(Path("/content/drive"))
    worker = StubWorker()
    for bad in (root / "comet_model", root / "comet" / "_load", root):
        with pytest.raises(cs.CometStageError, match="LOCAL runtime disk"):
            run_score(root, worker, model_dir=bad)
    assert worker.calls == []  # refused before any work
    run_score(root, worker, model_dir=tmp_path / "local_disk" / "comet_model")
    cmd = worker.calls[0]
    assert Path(cmd[cmd.index("--model-dir") + 1]) == tmp_path / "local_disk" / "comet_model"


def test_the_worker_default_model_dir_is_the_local_temp_dir(tmp_path: Path) -> None:
    args = cw._parse_args(["--in", "i", "--cache-dir", str(tmp_path), "--meta-out", "m"])
    assert args.model_dir.parent == Path(__import__("tempfile").gettempdir())
    assert tmp_path not in args.model_dir.parents


def test_the_chunk_cache_never_holds_the_model(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    run_score(root, StubWorker())
    assert (
        not list((root / "comet").rglob("model.ckpt"))
        and not (root / "comet/_cache/_load").exists()
    )


# --- N2: a kill between the last chunk and the meta write cannot wedge re-runs --------------------


def test_a_lost_worker_meta_is_re_derived_from_the_chunk_cache(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    killed = StubWorker(write_meta=False)  # all chunks written, meta never was
    summary = run_score(root, killed)  # the stage re-derives the model record from the chunks
    assert summary["model"]["revision"] == cw.COMET_MODEL_REVISION
    assert summary["device"] == "stub-gpu" and summary["batch_size"] == cw.DEFAULT_BATCH_SIZE
    # and a re-run after deleting the outputs AND the meta (chunks only) still works, no model load
    for f in (root / "comet").rglob("*.json"):
        if "_cache" not in f.parts and "_inputs" not in f.parts and "_pulled" not in f.parts:
            f.unlink()
    worker = StubWorker(write_meta=False)
    again = run_score(root, worker)
    assert worker.scored == [0] and again["n_sets"] == 61


def test_the_meta_is_written_atomically_and_chunks_carry_the_provenance(tmp_path: Path) -> None:
    root = build_eval_root(tmp_path)
    run_score(root, StubWorker(), batch_size=16)
    chunk = json.loads(
        next((root / "comet/_cache/fp32-2760a223").glob("chunk_*.json")).read_text("utf-8")
    )
    meta = chunk["meta"]
    assert meta["batch_size"] == 16 and meta["precision"] == "fp32" and meta["model"]
    assert not list((root / "comet").rglob("*.tmp"))  # atomic writes leave no temp files


def test_batch_size_is_recorded_but_does_not_invalidate_chunks(tmp_path: Path) -> None:
    triples = [{"src": "a", "mt": "b", "ref": "c"}]
    seen: list[int] = []

    def scorer() -> Any:
        return lambda part: seen.append(len(part)) or [0.5] * len(part)

    cw.run_chunks(
        triples, tmp_path, 4, scorer, log=lambda _m: None, chunk_meta=lambda: {"batch_size": 64}
    )
    stats = cw.run_chunks(
        triples, tmp_path, 4, scorer, log=lambda _m: None, chunk_meta=lambda: {"batch_size": 8}
    )
    assert stats["skipped"] == 1 and seen == [1]  # a different batch size reuses the chunk
    stored = json.loads(cw.chunk_path(tmp_path, 0).read_text("utf-8"))
    assert stored["meta"] == {"batch_size": 64}  # provenance of the run that scored it


# --- N3: no silent CPU --------------------------------------------------------------------------


def test_auto_falls_back_to_cpu_only_with_a_loud_message() -> None:
    notice = cw.cpu_fallback_notice("auto", 0) or ""
    assert "NO CUDA" in notice and "a few triples/s" in notice and "ASSUMED 50-150" in notice
    assert "roughly 2 triples/s" not in notice
    assert cw.cpu_fallback_notice("auto", 1) is None
    assert cw.cpu_fallback_notice("cpu", 0) is None  # asked for on purpose: no alarm
    with pytest.raises(cw.WorkerError, match="cuda"):
        cw.resolve_device("cuda", False)  # the default of the stage refuses instead


def test_the_stage_defaults_to_cuda_everywhere() -> None:
    assert cs._parser().parse_args(["score", "--eval-root", "e"]).device == "cuda"
    assert cs._parser().parse_args(["plan", "--eval-root", "e"]).device == "cuda"
    argv = dict(cs.plan_steps(eval_root=Path("/e"), hf_repo="o/r", python="py"))["comet:score"]
    assert argv[argv.index("--device") + 1] == "cuda"
    assert "device: cuda" in "\n".join(cs.describe_stage(Path("/nope"), None))


# --- N4: the "before the COMET upload" privacy check on its own -----------------------------------


def test_upload_comet_refuses_a_public_repo_before_any_commit_even_if_verify_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _models, api = comet_ready(tmp_path)
    before = len(api.commits)
    api.private = False
    monkeypatch.setattr(
        ev, "verify_run_on_hf", lambda *_a, **_k: {"complete": True, "revision": REVISION}
    )
    with pytest.raises(ev.HFNotPrivateError, match="before the COMET upload"):
        cs.upload_comet(api, REPO, root)
    assert len(api.commits) == before  # nothing was committed
    assert not (root / "comet" / cs.COMET_MANIFEST_NAME).exists()
