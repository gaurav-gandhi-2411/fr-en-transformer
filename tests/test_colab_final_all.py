from __future__ import annotations

# CONFIG = "eval_l4" with RUN = "final_all": the notebook wiring of the staged final selection.
# As in tests/test_colab_eval_l4.py every cell is exec'd by id and `run_step` is a recorder, so no
# model is exported, decoded or uploaded. The pure steps (`nmt.final_all plan / estimate / summary`)
# are run for real through the notebook's own argv, so the notebook <-> CLI contract is exercised.
import json
import subprocess
from pathlib import Path
from typing import Any

import nbformat
import pytest

import nmt.comet_stage as cs
import nmt.eval_l4 as ev
import nmt.final_all as fa
from tests.test_colab_ablations_l4 import NOTEBOOK, REPO_ROOT, SUMMARY, _exec
from tests.test_colab_eval_l4 import (
    EVAL_HELPERS,
    EVAL_RUN,
    YES,
    FakeRunStep,
    _bench_file,
    _eval_params,
    _helpers,
    _run_ns,
)

PURE = {"plan", "estimate", "estimate-measured", "comet-plan"}


class Recorder(FakeRunStep):
    """FakeRunStep that runs the three pure `nmt.final_all` subcommands for real (they only read
    files and print) and records every call. `calls` holds only the other steps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pure: list[tuple[str, list[str]]] = []

    def __call__(self, step: str, argv: list[str], **kwargs: Any) -> str:
        if step.startswith("summary["):
            self.pure.append(("summary", list(argv)))
            done = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT, check=True)
            print(done.stdout, end="")
            return done.stdout
        name = step.removeprefix("eval[").removesuffix("]").partition(":")[2]
        if name in PURE:
            self.pure.append((name, list(argv)))
            done = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT, check=True)
            if kwargs.get("stream"):
                print(done.stdout, end="")
            return done.stdout.strip()
        out = super().__call__(step, argv, **kwargs)
        if name == "upload-stage1" and "upload-stage1" not in self.outputs and self.roots:
            (self.roots["final_all"] / "stage1_hf_upload.json").write_text(
                json.dumps({"repo": "o/r", "revision": "e" * 40, "private": True}),
                encoding="utf-8",
            )
        if name == "comet:upload" and "comet:upload" not in self.outputs and self.roots:
            (self.roots["final_all"] / "comet_hf_upload.json").write_text(
                json.dumps({"repo": "o/r", "revision": "d" * 40, "private": True}),
                encoding="utf-8",
            )
        return out


def _ns(
    tmp_path: Path, rec: Recorder, dry: bool = False, with_ckpts: bool = True, **over: Any
) -> dict[str, Any]:
    ns = _run_ns(
        tmp_path, rec, "final_all", dry=dry, runs_base=tmp_path / "runs", run_prefix="", **over
    )
    if with_ckpts:
        for path in ns["final_all_ckpt_paths"](tmp_path / "runs", "").values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    return ns


def _bench(tmp_path: Path) -> dict[str, Any]:
    return {"bench": _bench_file(tmp_path / "eval" / "final_all" / "bench.json")}


def _step_names(rec: FakeRunStep) -> list[str]:
    return [n for n, _ in rec.calls]


# --- parameters and constants ---------------------------------------------------------------------


def test_final_all_is_a_selectable_run_and_all_stays_the_four_runs() -> None:
    ns = _eval_params(RUN='"final_all"')
    assert ns["EVAL"] is True and ns["EVAL_RUN_LIST"] == ("final_all",)
    assert ns["GIT_REF"] == "v0.3.1-colab" and ns["ALLOW_BRANCH"] is False
    all_ns = _eval_params(RUN='"all"')
    assert all_ns["EVAL_RUN_LIST"] == ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
    assert all_ns["EVAL_RUN_CHOICES"][-1] == "all"  # unchanged: final_all is not part of "all"
    assert ns["EVAL_RUNS"] == ev.RUNS
    with pytest.raises(ValueError, match="final_all"):
        _eval_params(RUN='"final-all"')


def test_notebook_final_all_constants_equal_the_module() -> None:
    ns = _helpers("final_all")
    assert ns["FINAL_ALL_CHECKPOINTS"] == fa.FINAL_ALL_CHECKPOINTS
    assert ns["FINAL_ALL_STAGE2_NOTE"] == fa.STAGE2_NOTE
    assert fa.FINAL_ALL_CHECKPOINTS == {
        "main": ("main", 24645, "main"),
        "A": ("ext_branch_a_l4", 37500, "ext_branch_a_l4"),
        "B": ("ext_branch_b_l4", 50000, "ext_branch_b_l4"),
    }
    for _run, _step, cfg in fa.FINAL_ALL_CHECKPOINTS.values():
        assert (REPO_ROOT / "configs" / f"{cfg}.yaml").is_file()


def test_checkpoint_paths_and_the_refusal_names_every_missing_file(tmp_path: Path) -> None:
    ns = _helpers("final_all")
    paths = ns["final_all_ckpt_paths"](tmp_path, "notebook_")
    assert paths == fa.final_all_checkpoint_paths(tmp_path, "notebook_")
    assert paths["main"] == tmp_path / "notebook_main" / "ckpt" / "step_00024645.pt"
    assert paths["A"].parent.parent.name == "notebook_ext_branch_a_l4"
    assert paths["B"].name == "step_00050000.pt"
    paths["A"].parent.mkdir(parents=True)
    paths["A"].write_bytes(b"x")
    with pytest.raises(RuntimeError) as err:
        ns["check_final_all_ckpts"](paths)
    msg = str(err.value)
    assert str(paths["main"]) in msg and str(paths["B"]) in msg and str(paths["A"]) not in msg
    with pytest.raises(ev.EvalStepError) as err2:  # the module's own check says the same
        fa.check_final_all_checkpoints(paths)
    assert str(paths["main"]) in str(err2.value) and str(paths["B"]) in str(err2.value)


def test_parse_final_all_plan_fails_closed() -> None:
    ns = _helpers("final_all")
    good = json.dumps([["hf-verify", ["py", "x"]], ["hf-check", ["py", "y"]]])
    assert ns["parse_final_all_plan"](good) == [
        ("hf-verify", ["py", "x"]),
        ("hf-check", ["py", "y"]),
    ]
    for bad in ("", "not json", "[]", json.dumps([["hf-check", ["a"]]]), json.dumps([1])):
        with pytest.raises(RuntimeError):
            ns["parse_final_all_plan"](bad)


# --- DRY_RUN --------------------------------------------------------------------------------------


def test_dry_run_prints_the_staged_plan_the_checkpoint_check_and_the_estimate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder()
    ns = _ns(tmp_path, rec, dry=True, with_ckpts=False)  # no checkpoints on disk
    _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    assert rec.calls == [] and not (tmp_path / "eval" / "final_all" / "run_meta.json").exists()
    assert not (tmp_path / "eval" / "final_all").exists()  # nothing created in a dry run
    assert "DRY_RUN: nothing launched" in out and "final_all: planned only (DRY_RUN)" in out
    # the plan, in order, with the stage-2 steps marked
    names = [out.index(f"  [{n}]") for n in ("hf-check", "export:main", "bench", "stage1-select")]
    assert names == sorted(names)
    assert out.index("[stage1-select]") < out.index("[tune-stage2:rank1:mbr_beam8]")
    assert out.index("[tune-stage2:rank2:mbr_eps0.02_n16]") < out.index("[select]")
    assert out.index("[select]") < out.index("[report]") < out.index("[decode]")
    assert out.index("[decode]") < out.index("[validate-test]") < out.index("[upload]")
    assert out.count("depends on stage1-select top-2") == 8
    assert out.count("[tune:") == 7
    # the checkpoint check, naming each file and the refusal
    for step in ("step_00024645.pt", "step_00037500.pt", "step_00050000.pt"):
        assert f"{step}:MISSING" in out
    assert "3 checkpoint file(s) missing: a real run would STOP" in out
    # the staged ESTIMATE from the measured L4 rate, cheapest and dearest case
    assert "ESTIMATE (MEASURED L4 beam-5 rate" in out and "not a measurement" in out
    assert "total (cheapest)" in out and "total (dearest)" in out and "beam 4 ASSUMED" in out
    assert "20.6 h" in out
    assert "[hf-verify]" in out and "skips it if so" in out
    # the COMET stage: listed after the upload, with its own section and ESTIMATE
    assert out.index("[upload]") < out.index("[comet:install]") < out.index("[comet:pull:main]")
    assert out.index("[comet:pull:s3_rope_concat_l4]") < out.index("[comet:score]")
    assert out.index("[comet:score]") < out.index("[comet:upload]")
    assert out.count("COMET stage: non-fatal, after the upload") == 1
    assert [n for n, _ in rec.pure] == ["plan", "estimate", "comet-plan"]
    assert "COMET stage: after the upload, NON-FATAL" in out and "61 sets" in out
    assert "Unbabel/wmt22-comet-da @ " in out and "batch size 64" in out
    assert "device: cuda" in out and "precision fp32" in out
    assert "distinct (exact, from the files here)" in out and "17294" in out and "4385" in out
    assert "ASSUMED 50 triples/s" in out and "ASSUMED 150 triples/s" in out
    assert "NO L4 COMET rate has been measured" in out
    assert not (tmp_path / "eval" / "final_all" / "comet").exists()  # a dry run writes nothing


def test_dry_run_with_all_checkpoints_present_reports_them_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = _ns(tmp_path, Recorder(), dry=True)
    _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    assert out.count(":ok") == 3 and "a real run would STOP" not in out


# --- a real session -------------------------------------------------------------------------------

EXPECTED_STEPS = [
    "hf-verify",
    "hf-check",
    "export:main",
    "export:A",
    "export:B",
    "bench",
    *[f"tune:{c}" for c in fa.STAGE1_CANDIDATES],
    "stage1-select",
    "decode-stage1-test",
    "validate-stage1-test",
    "upload-stage1",
    *[f"tune-stage2:rank{r}:{p}" for r in (1, 2) for p in fa.POOL_LABELS],
    "select",
    "report",
    "decode",
    "validate-test",
    "upload",
    "comet:install",
    *[f"comet:pull:{r}" for r in cs.PINNED_RUNS],
    "comet:score",
    "comet:upload",
]
COMET_STEPS = [n for n in EXPECTED_STEPS if n.startswith("comet:")]
MAIN_STEPS_ONLY = [n for n in EXPECTED_STEPS if not n.startswith("comet:")]
COMET_DONE = {"status": "completed", "revision": "d" * 40}


def test_run_cell_executes_the_whole_staged_plan_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(outputs=_bench(tmp_path))
    ns = _ns(tmp_path, rec)
    _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    assert _step_names(rec) == EXPECTED_STEPS
    names = _step_names(rec)
    assert names.index("hf-check") < names.index("export:main") < names.index("bench")
    assert names.index("upload") == len(MAIN_STEPS_ONLY) - 1 and names.index("hf-check") == 1
    assert names[names.index("upload") + 1 :] == COMET_STEPS  # COMET strictly after the upload
    assert ns["eval_results"]["final_all"] == {
        "status": "completed",
        "revision": "b" * 40,
        "comet": COMET_DONE,
    }
    assert "eval overall: all 1 run(s) ok" in out
    assert "final_all: completed; HF revision " + "b" * 40 in out
    assert "COMET: completed; HF COMET revision " + "d" * 40 in out
    assert f"eval: all {len(MAIN_STEPS_ONLY) - 1} steps done" in out  # minus hf-verify and COMET
    assert out.index("EVAL STEP: final_all:upload") < out.index("COMET STAGE")
    assert out.index("COMET STAGE") < out.index("EVAL STEP: final_all:comet:install")
    status = json.loads((tmp_path / "eval" / "final_all" / "comet_status.json").read_text("utf-8"))
    assert status["status"] == "completed" and status["revision"] == "d" * 40
    # both estimates: the static one first, the measured-bench one after the bench
    assert [n for n, _ in rec.pure] == ["plan", "estimate", "estimate-measured"]
    assert out.index("MEASURED L4 beam-5 rate") < out.index("MEASURED rates from")
    meta = json.loads((tmp_path / "eval" / "final_all" / "run_meta.json").read_text("utf-8"))
    assert meta["run"] == "final_all" and meta["git_sha"] == "deadbeef"
    assert set(meta["models"]) == {"main", "A", "B"} and "first_started_utc" in meta
    summary = json.loads((tmp_path / "eval" / "final_all" / "run_summary.json").read_text("utf-8"))
    assert summary["status"] == "completed" and summary["hf_revision"] == "b" * 40
    assert summary["private"] is True and summary["run"] == "final_all"


def test_the_notebook_argvs_are_the_real_cli_and_exports_use_the_existing_machinery(
    tmp_path: Path,
) -> None:
    rec = Recorder(outputs=_bench(tmp_path))
    _exec(EVAL_RUN, _ns(tmp_path, rec))
    argv = dict(rec.calls)
    parsers = {
        "nmt.eval_l4": ev._parser(),
        "nmt.final_all": fa._parser(),
        "nmt.comet_stage": cs._parser(),
    }
    for name, a in rec.calls:
        assert parsers[a[2]].parse_args(a[3:]).cmd == a[3], name
    exports = {n: a for n, a in rec.calls if n.startswith("export:")}
    for name, (run, step, cfg) in fa.FINAL_ALL_CHECKPOINTS.items():
        a = exports[f"export:{name}"]
        assert a[3] == "candidates" and a[a.index("--candidate") + 1] == f"{name}={step}"
        assert a[a.index("--config") + 1].endswith(f"{cfg}.yaml")
        assert Path(a[a.index("--ckpt-dir") + 1]).parent.name == run
        assert a[a.index("--out-dir") + 1].endswith("models")
    assert argv["upload"][argv["upload"].index("--run") + 1] == "final_all"
    assert argv["tune-stage2:rank1:mbr_beam8"][3] == "tune-stage2"
    assert argv["decode"][-1] == "--test"


def test_a_run_complete_on_hf_is_skipped_on_hf_evidence_before_anything_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(verify={"final_all": YES})
    ns = _ns(tmp_path, rec, with_ckpts=False)  # not even the checkpoints are needed
    _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    # the main upload is complete WITHOUT COMET: only the COMET steps run (a re-run executes
    # only COMET); none of the selection / decode / upload steps does
    assert _step_names(rec) == ["hf-verify", *COMET_STEPS]
    assert ns["eval_results"]["final_all"] == {
        "status": "skipped",
        "revision": "c" * 40,
        "comet": COMET_DONE,
    }
    assert "final_all: skipped, already complete on HF (verified)" in out
    assert not (tmp_path / "eval" / "final_all" / "run_meta.json").exists()


def test_missing_checkpoints_refuse_before_any_gpu_step_naming_each(tmp_path: Path) -> None:
    rec = Recorder()
    ns = _ns(tmp_path, rec, with_ckpts=False)
    paths = ns["final_all_ckpt_paths"](tmp_path / "runs", "")
    paths["main"].parent.mkdir(parents=True)
    paths["main"].write_bytes(b"x")  # only main exists
    with pytest.raises(RuntimeError) as err:
        _exec(EVAL_RUN, ns)
    msg = str(err.value)
    assert str(paths["A"]) in msg and str(paths["B"]) in msg and str(paths["main"]) not in msg
    assert _step_names(rec) == ["hf-verify"]  # no hf-check, no export, no bench
    assert ns["eval_results"]["final_all"]["status"] == "FAILED"
    saved = json.loads((tmp_path / "eval" / "final_all" / "run_summary.json").read_text("utf-8"))
    assert saved["status"] == "FAILED" and "step_00037500.pt" in saved["error"]


def test_a_failing_hf_check_stops_the_run_before_any_export_or_gpu_step(tmp_path: Path) -> None:
    rec = Recorder(fail_at="final_all:hf-check")
    with pytest.raises(RuntimeError, match="FAILED for 1 of 1"):
        _exec(EVAL_RUN, _ns(tmp_path, rec))
    assert _step_names(rec) == ["hf-verify", "hf-check"]


@pytest.mark.parametrize("failing", ["tune:A__beam", "tune-stage2:rank2:mbr_beam16", "report"])
def test_the_first_failing_step_stops_the_session_and_later_steps_never_start(
    tmp_path: Path, failing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(outputs=_bench(tmp_path), fail_at=f"final_all:{failing}")
    ns = _ns(tmp_path, rec)
    with pytest.raises(RuntimeError, match="FAILED for 1 of 1"):
        _exec(EVAL_RUN, ns)
    names = _step_names(rec)
    assert names[-1] == failing and names == EXPECTED_STEPS[: len(names)]
    out = capsys.readouterr().out
    assert f"EVAL STOPPED at {failing} of final_all" in out and "final_all: FAILED:" in out
    assert "upload" not in names
    saved = json.loads((tmp_path / "eval" / "final_all" / "run_summary.json").read_text("utf-8"))
    assert saved["status"] == "FAILED"


def test_a_rerun_issues_the_same_steps_so_every_subprocess_can_skip_itself(tmp_path: Path) -> None:
    first = Recorder(outputs=_bench(tmp_path))
    _exec(EVAL_RUN, _ns(tmp_path, first))
    second = Recorder(outputs=_bench(tmp_path))
    ns2 = _ns(tmp_path, second)
    _exec(EVAL_RUN, ns2)
    assert first.calls == second.calls
    meta = json.loads((tmp_path / "eval" / "final_all" / "run_meta.json").read_text("utf-8"))
    assert meta["first_started_utc"] <= meta["last_started_utc"]


# --- the COMET stage: non-fatal, loud, re-runnable ---------------------------------------------


@pytest.mark.parametrize(
    "failing", ["comet:install", "comet:pull:s2_rope_l4", "comet:score", "comet:upload"]
)
def test_a_comet_failure_is_loud_non_fatal_for_the_upload_and_fails_the_cell(
    tmp_path: Path, failing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(outputs=_bench(tmp_path), fail_at=f"final_all:{failing}")
    ns = _ns(tmp_path, rec)
    with pytest.raises(RuntimeError, match="COMET FAILED for final_all") as err:
        _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    assert "COMET FAILED: RuntimeError" in out and failing in out
    assert "selection and upload are complete" in str(err.value)
    # the selection and the main upload are reported as complete, and untouched
    res = ns["eval_results"]["final_all"]
    assert res["status"] == "completed" and res["revision"] == "b" * 40
    assert res["comet"]["status"] == "FAILED" and failing in res["comet"]["error"]
    assert "final_all: completed; HF revision " + "b" * 40 in out
    assert "COMET FAILED for: final_all" in out and "(selection, decode, validation" in out
    summary = json.loads((tmp_path / "eval" / "final_all" / "run_summary.json").read_text("utf-8"))
    assert summary["status"] == "completed"  # written BEFORE the COMET stage
    status = json.loads((tmp_path / "eval" / "final_all" / "comet_status.json").read_text("utf-8"))
    assert status["status"] == "FAILED"
    # the stage stops at the failing step (later COMET steps do not start)
    names = _step_names(rec)
    assert names[-1] == failing and names[: len(MAIN_STEPS_ONLY)] == MAIN_STEPS_ONLY


def test_a_comet_upload_without_a_private_revision_is_a_comet_failure(tmp_path: Path) -> None:
    rec = Recorder(outputs={**_bench(tmp_path), "comet:upload": lambda: None})
    ns = _ns(tmp_path, rec)
    with pytest.raises(RuntimeError, match="COMET FAILED"):
        _exec(EVAL_RUN, ns)
    assert "comet_hf_upload.json" in ns["eval_results"]["final_all"]["comet"]["error"]
    assert ns["eval_results"]["final_all"]["status"] == "completed"


def test_after_a_comet_failure_a_rerun_executes_only_comet(tmp_path: Path) -> None:
    first = Recorder(outputs=_bench(tmp_path), fail_at="final_all:comet:score")
    with pytest.raises(RuntimeError, match="COMET FAILED"):
        _exec(EVAL_RUN, _ns(tmp_path, first))
    # the main upload is on the HF repo now: hf-verify says complete, no selection step runs
    second = Recorder(verify={"final_all": YES})
    ns2 = _ns(tmp_path, second, with_ckpts=False)
    _exec(EVAL_RUN, ns2)
    assert _step_names(second) == ["hf-verify", *COMET_STEPS]
    assert ns2["eval_results"]["final_all"]["comet"] == COMET_DONE


def test_the_plan_splitter_refuses_a_comet_step_before_the_upload() -> None:
    ns = _helpers("final_all")
    plan = [("hf-check", ["x"]), ("comet:install", ["y"]), ("upload", ["z"])]
    with pytest.raises(RuntimeError, match="comet:"):
        ns["split_comet_steps"](plan)
    main, comet = ns["split_comet_steps"]([("upload", ["z"]), ("comet:score", ["y"])])
    assert [n for n, _ in main] == ["upload"] and [n for n, _ in comet] == ["comet:score"]


def test_comet_parameters_are_validated() -> None:
    ns = _eval_params(RUN='"final_all"')
    assert (ns["COMET_BATCH_SIZE"], ns["COMET_PRECISION"]) == (64, "fp32")
    assert ns["COMET_DEVICE"] == "cuda"  # refuses without CUDA unless "cpu" is set on purpose
    with pytest.raises(ValueError, match="COMET_DEVICE"):
        _eval_params(RUN='"final_all"', COMET_DEVICE='"tpu"')
    assert _eval_params(RUN='"final_all"', COMET_DEVICE='"cpu"')["COMET_DEVICE"] == "cpu"
    for edit in (
        {"COMET_BATCH_SIZE": "0"},
        {"COMET_BATCH_SIZE": '"64"'},
        {"COMET_PRECISION": '"int8"'},
    ):
        with pytest.raises(ValueError, match="COMET_"):
            _eval_params(RUN='"final_all"', **edit)


def test_a_run_without_a_private_revision_is_a_failure(tmp_path: Path) -> None:
    rec = Recorder(outputs={**_bench(tmp_path), "upload": lambda: None})
    ns = _ns(tmp_path, rec)
    with pytest.raises(RuntimeError, match="no private hf_upload.json"):
        _exec(EVAL_RUN, ns)


def test_a_bench_without_usable_rates_fails_the_run(tmp_path: Path) -> None:
    rec = Recorder()  # the bench step writes nothing
    with pytest.raises(RuntimeError, match="usable rates"):
        _exec(EVAL_RUN, _ns(tmp_path, rec))


def test_other_runs_are_untouched_by_the_final_all_wiring(tmp_path: Path) -> None:
    """RUN='main' still takes the old path (no plan/estimate subprocess, its own step list)."""
    rec = Recorder(outputs={"bench": _bench_file(tmp_path / "eval" / "main" / "bench.json")})
    ns = _run_ns(tmp_path, rec, "main")
    for step in {s for v in ns["eval_candidate_steps"]("main").values() for s in v}:
        (ns["eval_ckpt_dirs"]["main"] / f"step_{step:08d}.pt").write_bytes(b"x")
    _exec(EVAL_RUN, ns)
    assert rec.pure == [] and _step_names(rec)[:3] == ["hf-verify", "hf-check", "candidates:final"]
    assert "final_all" not in ns["eval_results"]


# --- summary cell ---------------------------------------------------------------------------------


def _fill_final_all_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    def put(rel: str, obj: Any) -> None:
        (root / rel).write_text(json.dumps(obj), encoding="utf-8")

    cand = {"alpha": 1.6, "beam": 5, "segment_threshold": None}
    put(
        "bench.json",
        {"modes": {"beam5": {"sentences_per_second": 21.3, "output_tokens_per_second": 1198.9}}},
    )
    put(
        "stage1.json",
        {
            "candidate_order": ["A__beam", "B__beam"],
            "candidates": {
                "A__beam": {"objective": 50.0, "config": cand},
                "B__beam": {"objective": 49.0, "config": {**cand, "segment_threshold": 128}},
            },
            "top2": ["A__beam", "B__beam"],
        },
    )
    put(
        "selection.json",
        {
            "stage2_candidates": ["A__mbr_beam8"],
            "candidates": {"A__mbr_beam8": {"objective": 51.0}},
            "ranking": ["A__mbr_beam8", "A__beam"],
            "winner": {
                "candidate": "A__mbr_beam8",
                "objective": 51.0,
                "alpha": 1.6,
                "beam": 8,
                "segment_threshold": 192,
                "mbr": {"kind": "beam", "n": 8},
            },
            "runner_up": {"candidate": "A__beam", "objective": 50.0},
            "production": {"candidate": "A__beam"},
            "tie_rule": "earlier wins",
        },
    )
    put(
        "report.json",
        {
            "bootstrap": {
                "winner_vs_runner_up": {
                    "delta": 1.0,
                    "ci95": [0.2, 1.8],
                    "p_value": 0.01,
                    "n_resamples": 1000,
                    "seed": 1234,
                },
                "winner_vs_production": {"note": "same candidate: no comparison", "delta": 0.0},
            },
            "latency": {
                "winner": {
                    "sentences_per_second": 2.5,
                    "output_tokens_per_second": 140.0,
                    "gpu": "NVIDIA L4",
                    "settings": {"batch_size": 32},
                }
            },
        },
    )
    put("validation.json", {"valid": True, "n_ids": 330, "empty_strings": 0})
    put("hf_upload.json", {"repo": "o/r", "revision": "b" * 40, "private": True})


def test_summary_cell_prints_stage_objectives_winner_bootstrap_latency_and_hf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = _helpers("final_all")
    _fill_final_all_dir(tmp_path / "fa")
    ns.update(
        GIT_REF="v-test",
        git_sha="cafe123",
        git_describe="v-test",
        DATA_REVISION="rev",
        CONFIG="eval_l4",
        EVAL=True,
        ABLATION=False,
        preflight={"preflight_gpu": "NVIDIA L4", "preflight_precision": "bf16"},
        eval_roots={"final_all": tmp_path / "fa"},
        eval_results={"final_all": {"status": "completed", "revision": "b" * 40}},
        HF_EVAL_REPO="o/r",
        RUN="final_all",
        repo_dir=REPO_ROOT,
        run_step=Recorder(),
    )
    _exec(SUMMARY, ns)
    out = capsys.readouterr().out
    for needle in (
        "final_all: completed; HF revision " + "b" * 40,
        "git commit SHA: cafe123",
        "GPU: NVIDIA L4",
        "stage 1 A__beam: objective 50.0000",
        "T=128",
        "stage 1 TOP 2: A__beam, B__beam",
        "stage 2 A__mbr_beam8: objective 51.0000",
        "WINNER: A__mbr_beam8 objective 51.0000",
        "runner-up: A__beam objective 50.0000",
        "bootstrap winner_vs_runner_up: delta +1.000 [+0.200, +1.800] p=0.01 (n=1000, seed 1234)",
        "bootstrap winner_vs_production: same candidate",
        "latency winner: 2.5 sent/s, 140.0 out-tok/s (NVIDIA L4, batch 32",
        "test validation: OK (330 ids, 0 empty)",
        "HF repo: o/r revision: " + "b" * 40 + " private: True",
        "HF_EVAL_REVISION=" + "b" * 40,
    ):
        assert needle in out, needle


# --- hygiene, CI ----------------------------------------------------------------------------


def test_final_all_cells_import_only_the_stdlib_and_the_notebook_stays_output_free() -> None:
    import ast
    import sys

    for cell_id in (EVAL_HELPERS, EVAL_RUN):
        src = next(c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.id == cell_id)
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import | ast.ImportFrom):
                mods = (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                )
                assert all((m or "").split(".")[0] in sys.stdlib_module_names for m in mods), mods
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert all(
        not c.outputs and c.execution_count is None for c in nb.cells if c.cell_type == "code"
    )


def test_ci_dry_runs_the_final_all_wiring_and_the_notebook_tag_is_the_final_eval_tag() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--set RUN='\"final_all\"'" in ci and "--set RUN='\"all\"'" in ci
    nb = nbformat.read(NOTEBOOK, as_version=4)
    params = next(c.source for c in nb.cells if c.id == "c5459372")
    assert 'GIT_REF = "v0.3.1-colab"' in params and "ALLOW_BRANCH = False" in params
    runbook = (REPO_ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    assert '`RUN = "final_all"`' in runbook and "<FINAL_TAG>" in runbook
    assert "v0.3.1-colab" in runbook


# --- the stage-1 interim result: a safety net BEFORE stage 2 --------------------------------------


def test_the_interim_steps_run_before_any_stage2_step_and_the_revision_is_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(outputs=_bench(tmp_path))
    ns = _ns(tmp_path, rec)
    _exec(EVAL_RUN, ns)
    names = _step_names(rec)
    first_stage2 = next(i for i, n in enumerate(names) if n.startswith("tune-stage2"))
    interim = ["decode-stage1-test", "validate-stage1-test", "upload-stage1"]
    assert names[names.index("stage1-select") + 1 : first_stage2] == interim
    out = capsys.readouterr().out
    assert f"STAGE-1 INTERIM SAVED: HF_STAGE1_REVISION={'e' * 40} (private)" in out
    assert out.index("STAGE-1 INTERIM SAVED") < out.index("EVAL STEP: final_all:tune-stage2")
    root = tmp_path / "eval" / "final_all"
    final = json.loads((root / "run_summary.json").read_text("utf-8"))
    assert final["status"] == "completed" and final["hf_stage1_revision"] == "e" * 40


@pytest.mark.parametrize("failing", ["decode-stage1-test", "validate-stage1-test", "upload-stage1"])
def test_an_interim_failure_stops_the_session_loudly_before_stage_2(
    tmp_path: Path, failing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(outputs=_bench(tmp_path), fail_at=f"final_all:{failing}")
    with pytest.raises(RuntimeError, match="FAILED for 1 of 1"):
        _exec(EVAL_RUN, _ns(tmp_path, rec))
    names = _step_names(rec)
    assert names[-1] == failing and not any(n.startswith("tune-stage2") for n in names)
    out = capsys.readouterr().out
    assert f"STAGE-1 INTERIM FAILED at {failing}: STAGE 2 WAS NOT STARTED" in out
    assert "comet:install" not in names  # nothing after it ran


def test_an_interim_upload_without_a_private_revision_stops_before_stage_2(
    tmp_path: Path,
) -> None:
    rec = Recorder(outputs={**_bench(tmp_path), "upload-stage1": lambda: None})
    with pytest.raises(RuntimeError, match="stage 2 was NOT started"):
        _exec(EVAL_RUN, _ns(tmp_path, rec))
    assert not any(n.startswith("tune-stage2") for n in _step_names(rec))


def test_the_dry_run_shows_the_three_interim_steps_before_stage_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _exec(EVAL_RUN, _ns(tmp_path, Recorder(), dry=True))
    out = capsys.readouterr().out
    pos = [out.index(f"[{n}]") for n in ("stage1-select", "decode-stage1-test")]
    assert pos == sorted(pos)
    assert out.index("[upload-stage1]") < out.index("[tune-stage2:rank1:mbr_beam8]")
    assert out.count("stage-1 interim safety net: BEFORE stage 2") == 3
