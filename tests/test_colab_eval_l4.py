from __future__ import annotations

# CONFIG = "eval_l4": the notebook evaluates one finished run in a single Colab session. As in
# tests/test_colab_ablations_l4.py, every cell is exec'd by id and `run_step` is a recorder, so
# nothing is decoded, downloaded or uploaded. What is tested here is the notebook's own logic:
# parameter refusals, the hard-coded PREREG candidate lists, the missing-checkpoint failure, the
# step order and argv (checked against the real nmt.eval_l4 parser), the stop-at-first-failure
# behaviour inside a run, RUN="all" (order, HF-evidence skipping, continue-and-fail-at-the-end
# policy), the ESTIMATE arithmetic, the secrets/Drive/summary cells. The behaviour behind each
# subprocess (resume, validation, HF guards) is tested in tests/test_eval_l4.py.
import json
import sys
import types
from pathlib import Path
from typing import Any

import nbformat
import pytest

import nmt.eval_l4 as ev
from tests.test_colab_ablations_l4 import (
    DRIVE,
    NOTEBOOK,
    REPO_ROOT,
    SUMMARY,
    _exec,
    _load_execute_notebook,
    _params,
    _src,
)

SECRETS, DATA, WANDB, TRAIN = "70d78ad5", "4bb2a1e6", "ebfe4071", "4e5fb4a0"
EVAL_HELPERS, EVAL_RUN = "e7a1c4d1", "e7a1c4d3"
MAIN_STEPS = {
    "final": (24645,),
    "avg_last5": (19000, 20757, 22500, 24269, 24645),
    "avg_decay": (20757, 22500, 24269, 24645),
}
WORKLOAD = json.loads((REPO_ROOT / "colab" / "eval_workload.json").read_text(encoding="utf-8"))


def _eval_params(**edits: str) -> dict[str, Any]:
    return _params(**{"CONFIG": '"eval_l4"', **edits})


def _helpers(run: str = "main") -> dict[str, Any]:
    ns = _eval_params(RUN=f'"{run}"')
    _exec(EVAL_HELPERS, ns)
    return ns


NO = "HF_RUN_COMPLETE=false\nhf-verify: nothing under runs/"
YES = "HF_RUN_COMPLETE=true\nHF_EVAL_REVISION=" + "c" * 40


class FakeRunStep:
    """Records every step as (name, argv) and its run in `runs`. Step names arrive as
    `eval[<run>:<step>]`. `outputs` maps "<run>:<step>" (or just "<step>" for every run) to a
    callback that creates the files the real subprocess would write; `fail_at` is a set of
    "<run>:<step>" that raise; `verify` maps a run to the hf-verify stdout (default: not
    complete). A successful upload writes hf_upload.json under `roots[run]` like the real step."""

    def __init__(
        self,
        outputs: dict[str, Any] | None = None,
        fail_at: str | set[str] | None = None,
        verify: dict[str, str] | None = None,
        roots: dict[str, Path] | None = None,
    ) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.runs: list[str] = []
        self.outputs = outputs or {}
        self.fail_at = {fail_at} if isinstance(fail_at, str) else set(fail_at or ())
        self.verify = verify or {}
        self.roots = roots

    def __call__(self, step: str, argv: list[str], **kwargs: Any) -> str:
        run, _, name = step.removeprefix("eval[").removesuffix("]").partition(":")
        self.calls.append((name, list(argv)))
        self.runs.append(run)
        if f"{run}:{name}" in self.fail_at or name in self.fail_at:
            raise RuntimeError(f"{step} failed (exit 1)")
        if name == "hf-verify":
            return self.verify.get(run, NO)
        for key in (f"{run}:{name}", name):
            if key in self.outputs:
                self.outputs[key]()
        if name == "upload" and self.roots is not None and "upload" not in self.outputs:
            (self.roots[run] / "hf_upload.json").write_text(
                json.dumps({"repo": "o/r", "run": run, "revision": "b" * 40, "private": True}),
                encoding="utf-8",
            )
        return ""


# --- parameters -----------------------------------------------------------------------------------


def test_defaults_stay_the_main_run_and_eval_is_selectable() -> None:
    main = _params(CONFIG='"main"')
    assert main["EVAL"] is False and main["RUN"] == "main" and main["PLANNED_STEPS"] == 24645
    assert main["HF_EVAL_REPO"] == "OWNER/fr-en-transformer-eval"
    ns = _eval_params()
    assert ns["EVAL"] is True and ns["ABLATION"] is False
    assert ns["EVAL_RUNS"] == ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
    assert ns["EVAL_RUN_LIST"] == ("main",)


def test_run_all_means_the_four_runs_main_first(capsys: pytest.CaptureFixture[str]) -> None:
    ns = _eval_params(RUN='"all"')
    assert ns["EVAL_RUN_LIST"] == ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
    assert ns["EVAL_RUN_LIST"][0] == "main" and ns["EVAL_RUN_CHOICES"][-1] == "all"
    assert "main -> s1_sin_l4 -> s2_rope_l4 -> s3_rope_concat_l4" in capsys.readouterr().out
    with pytest.raises(ValueError, match="RUN='ALL'"):
        _eval_params(RUN='"ALL"')  # exact spelling only


@pytest.mark.parametrize("config", ["main", "eval_l4"])
def test_an_unknown_run_is_refused_loudly(config: str) -> None:
    with pytest.raises(ValueError, match="RUN='s9_nope'"):
        _params(CONFIG=f'"{config}"', RUN='"s9_nope"')


@pytest.mark.parametrize("flag", ["RESUME_TEST", "COOLDOWN_NOW", "RUN_EVAL"])
def test_training_flags_are_refused_with_eval(flag: str) -> None:
    with pytest.raises(ValueError, match=flag):
        _eval_params(**{flag: "True"})


@pytest.mark.parametrize("repo", ["", "noslash", "a/b/c", "/x", "a/"])
def test_a_malformed_hf_repo_is_refused(repo: str) -> None:
    with pytest.raises(ValueError, match="HF_EVAL_REPO"):
        _eval_params(HF_EVAL_REPO=f'"{repo}"')


def test_planned_steps_is_ignored_loudly_and_dry_run_is_allowed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ns = _eval_params(DRY_RUN="True")
    out = capsys.readouterr().out
    assert ns["PLANNED_STEPS"] is None and "EVAL MODE: PLANNED_STEPS=24645 is IGNORED" in out
    assert ns["DRY_RUN"] is True
    with pytest.raises(ValueError, match="DRY_RUN"):
        _params(CONFIG='"main"', DRY_RUN="True")  # still refused outside ablations/eval


# --- hard-coded PREREG lists ----------------------------------------------------------------------


def test_main_candidates_are_exactly_the_preregistered_file_lists() -> None:
    ns = _helpers("main")
    assert ns["eval_candidate_steps"]("main") == MAIN_STEPS
    assert list(ns["eval_candidate_steps"]("main")) == ["final", "avg_last5", "avg_decay"]


@pytest.mark.parametrize("run", ["s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4"])
def test_ablations_use_their_final_checkpoint_alone(run: str) -> None:
    assert _helpers(run)["eval_candidate_steps"](run) == {"final": (4107,)}


def test_prereg_amendment_lists_match_the_committed_prereg_text() -> None:
    text = (REPO_ROOT / "PREREG.md").read_text(encoding="utf-8")
    for step in (19000, 20757, 22500, 24269, 24645):
        assert f"{step:,}" in text or f"{step:08d}" in text.replace("step_", "")
    assert "avg_decay" in text and "4107" in text


def test_notebook_constants_equal_the_module_constants() -> None:
    ns = _helpers()
    assert ns["EVAL_BATCH_SIZE"] == ev.EVAL_BATCH_SIZE
    assert ns["EVAL_RUNS"] == ev.RUNS
    assert ns["EVAL_DECODE_SPLITS"] == ev.DECODE_SPLITS
    assert ns["EVAL_VARIANTS"] == ev.VARIANTS
    assert tuple(ns["eval_candidate_steps"]("main")) == ev.MAIN_CANDIDATES
    assert tuple(ns["eval_candidate_steps"]("s2_rope_l4")) == ev.ABLATION_CANDIDATES


def test_missing_checkpoint_files_fail_before_anything_runs_naming_each(tmp_path: Path) -> None:
    ns = _helpers()
    for step in (19000, 20757, 24645):  # 22500 and 24269 are missing
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    missing = ns["missing_eval_checkpoints"](tmp_path, MAIN_STEPS)
    assert [p.name for p in missing] == ["step_00022500.pt", "step_00024269.pt"]
    with pytest.raises(RuntimeError) as err:
        ns["check_eval_checkpoints"](tmp_path, MAIN_STEPS)
    assert "step_00022500.pt" in str(err.value) and "step_00024269.pt" in str(err.value)
    assert "2 required checkpoint file(s)" in str(err.value)
    for step in (22500, 24269, 20757):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    ns["check_eval_checkpoints"](tmp_path, MAIN_STEPS)


# --- estimate -------------------------------------------------------------------------------------


def test_estimate_arithmetic_matches_a_hand_computation() -> None:
    ns = _helpers("main")
    rates = {"greedy": 2000.0, "beam": 500.0}
    sp = WORKLOAD["splits"]
    e12 = sp["e1"]["out_tokens"] + sp["e2"]["out_tokens"]
    per_cand = e12 / 2000 + 8 * e12 / 500 + 5 * sp["e2"]["out_tokens"] / 500
    final = sum(sp[s]["out_tokens"] for s in ("dev", "e1", "e2", "e2synth", "e3"))
    expected = (
        3 * per_cand
        + 2 * final / 500
        + sp["test"]["out_tokens"] / 500
        + WORKLOAD["bench_e2_first200_out_tokens"] * (1 / 2000 + 1 / 500)
        + ns["EVAL_ASSUMED_FIXED_SECONDS"]
    )
    est = ns["estimate_eval"](WORKLOAD, "main", rates, "test basis")
    assert est["seconds"] == pytest.approx(expected)
    assert est["cu"] == pytest.approx(expected / 3600 * 1.54)
    assert est["n_candidates"] == 3


def test_an_ablation_has_one_candidate_and_no_test_decode() -> None:
    ns = _helpers("s1_sin_l4")
    rates = {"greedy": 2000.0, "beam": 500.0}
    abl = ns["estimate_eval"](WORKLOAD, "s1_sin_l4", rates, "b")
    main = ns["estimate_eval"](WORKLOAD, "main", rates, "b")
    assert abl["n_candidates"] == 1 and abl["parts_seconds"]["test_decode"] == 0.0
    assert abl["parts_seconds"]["tuning"] == pytest.approx(main["parts_seconds"]["tuning"] / 3)


def test_the_estimate_is_labelled_and_states_its_basis() -> None:
    ns = _helpers()
    static = ns["estimate_eval"](
        WORKLOAD, "main", ns["EVAL_ASSUMED_RATES"], "STATIC: ASSUMED decode rates"
    )
    text = "\n".join(ns["format_estimate"](static))
    assert text.startswith("ESTIMATE (STATIC: ASSUMED decode rates)") and "CU at 1.54 CU/h" in text
    assert "not a measurement" in text and "min =" in text


def test_bench_rates_reads_a_bench_json_and_rejects_unusable_ones() -> None:
    ns = _helpers()
    good = {
        "modes": {
            "greedy": {"output_tokens_per_second": 1234.5},
            "beam5": {"output_tokens_per_second": 456.0},
        }
    }
    assert ns["bench_rates"](good) == {"greedy": 1234.5, "beam": 456.0}
    assert ns["bench_rates"]({}) is None
    assert ns["bench_rates"]({"modes": {"greedy": {}, "beam5": {}}}) is None
    zero = {"modes": {m: {"output_tokens_per_second": 0} for m in ("greedy", "beam5")}}
    assert ns["bench_rates"](zero) is None


# --- plan -----------------------------------------------------------------------------------------


def _plan(ns: dict[str, Any], run: str, tmp_path: Path) -> list[tuple[str, list[str]]]:
    return ns["eval_plan"](
        run,
        ckpt_dir=tmp_path / "ckpt",
        eval_root=tmp_path / "eval",
        hf_repo="o/r",
        repo_dir=REPO_ROOT,
    )


def test_main_plan_order_and_every_argv_parses_with_the_real_cli(tmp_path: Path) -> None:
    plan = _plan(_helpers("main"), "main", tmp_path)
    assert [name for name, _ in plan] == [
        "hf-check",
        "candidates:final",
        "bench",
        "candidates",
        "tune:final",
        "tune:avg_last5",
        "tune:avg_decay",
        "select",
        "decode",
        "validate-test",
        "upload",
    ]
    names = [name for name, _ in plan]
    # (e) throughput bench first (it needs only the final export), then tune + select (E1 + E2
    # inside nmt.tune), only then any decode; test predictions + validation for main only.
    assert names.index("bench") < min(i for i, n in enumerate(names) if n.startswith("tune:"))
    assert max(i for i, n in enumerate(names) if n.startswith("tune:")) < names.index("select")
    assert names.index("select") < names.index("decode") < names.index("validate-test")
    assert names.index("validate-test") < names.index("upload") == len(names) - 1
    parser = ev._parser()
    for _, argv in plan:
        assert argv[:3] == [sys.executable, "-m", "nmt.eval_l4"]
        parser.parse_args(argv[3:])  # every notebook argv is accepted by the real CLI
    by_name = dict(plan)
    assert "--only" in by_name["candidates:final"] and "--only" not in by_name["candidates"]
    assert "--test" in by_name["decode"] and by_name["decode"].count("--batch-size") == 1
    cands = [a for a in by_name["candidates"] if "=" in a]
    assert cands == [
        "final=24645",
        "avg_last5=19000,20757,22500,24269,24645",
        "avg_decay=20757,22500,24269,24645",
    ]
    assert by_name["select"][by_name["select"].index("--candidates") + 1 :][:3] == [
        "final",
        "avg_last5",
        "avg_decay",
    ]
    assert str(tmp_path / "eval" / "candidates" / "final") in by_name["bench"]


def test_ablation_plan_has_no_test_decode_no_validation_and_one_candidate(tmp_path: Path) -> None:
    plan = _plan(_helpers("s2_rope_l4"), "s2_rope_l4", tmp_path)
    names = [n for n, _ in plan]
    assert names == [
        "hf-check",
        "candidates:final",
        "bench",
        "candidates",
        "tune:final",
        "select",
        "decode",
        "upload",
    ]
    assert "--test" not in dict(plan)["decode"]
    assert "final=4107" in dict(plan)["candidates"]
    assert dict(plan)["candidates"][dict(plan)["candidates"].index("--config") + 1].endswith(
        "s2_rope_l4.yaml"
    )


# --- the run cell ---------------------------------------------------------------------------------


def _run_ns(
    tmp_path: Path, run_step: Any, run: str = "main", dry: bool = False, **over: Any
) -> dict[str, Any]:
    """A namespace for the run cell. `run` is the RUN parameter ('all' = the four runs); each
    run's checkpoints live in tmp/ckpt/<run> and its outputs in tmp/eval/<run>."""
    ns = _helpers(run)
    runs = ns["EVAL_RUN_LIST"]
    ns.update(
        run_step=run_step,
        repo_dir=REPO_ROOT,
        eval_ckpt_dirs={r: tmp_path / "ckpt" / r for r in runs},
        eval_roots={r: tmp_path / "eval" / r for r in runs},
        git_sha="deadbeef",
        git_describe="v-test",
        preflight={"preflight_gpu": "NVIDIA L4", "preflight_precision": "bf16"},
        EVAL=True,
        DRY_RUN=dry,
    )
    ns.update(over)
    for r in runs:
        ns["eval_ckpt_dirs"][r].mkdir(parents=True, exist_ok=True)
        if not dry:
            ns["eval_roots"][r].mkdir(parents=True, exist_ok=True)
    if isinstance(run_step, FakeRunStep) and run_step.roots is None:
        run_step.roots = ns["eval_roots"]
    return ns


def _write_ckpts(ckpt_dir: Path, run: str = "main") -> None:
    steps = {s for v in _helpers(run)["eval_candidate_steps"](run).values() for s in v}
    for step in steps:
        (ckpt_dir / f"step_{step:08d}.pt").write_bytes(b"x")


def _bench_file(path: Path) -> Any:
    def write() -> None:
        modes = {
            "greedy": {"output_tokens_per_second": 3000.0},
            "beam5": {"output_tokens_per_second": 1500.0},
        }
        path.write_text(json.dumps({"n_sentences": 200, "modes": modes}), encoding="utf-8")

    return write


def _single(tmp_path: Path, rec: FakeRunStep, run: str = "main", **over: Any) -> dict[str, Any]:
    ns = _run_ns(tmp_path, rec, run, **over)
    _write_ckpts(ns["eval_ckpt_dirs"][run], run)
    return ns


def _bench(tmp_path: Path, run: str = "main") -> dict[str, Any]:
    return {"bench": _bench_file(tmp_path / "eval" / run / "bench.json")}


def test_run_cell_executes_every_step_in_order_and_prints_both_estimates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = FakeRunStep(outputs=_bench(tmp_path))
    ns = _single(tmp_path, rec)
    _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    expected = [n for n, _ in _plan(ns, "main", tmp_path)]
    assert [n for n, _ in rec.calls] == ["hf-verify", *expected]
    assert out.index("STATIC: ASSUMED") < out.index("from bench.json (measured rates)")
    assert "greedy 3000 out-tok/s, beam 1500 out-tok/s" in out
    assert "eval: all 11 steps done" in out and "completed; HF revision " + "b" * 40 in out
    meta = json.loads((tmp_path / "eval" / "main" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["git_sha"] == "deadbeef" and meta["run"] == "main"
    assert meta["candidates"]["avg_decay"] == [20757, 22500, 24269, 24645]
    assert "first_started_utc" in meta
    assert ns["eval_results"]["main"] == {"status": "completed", "revision": "b" * 40}


def test_the_run_summary_records_the_hf_revision_and_the_private_flag(tmp_path: Path) -> None:
    rec = FakeRunStep(outputs=_bench(tmp_path))
    ns = _single(tmp_path, rec)
    _exec(EVAL_RUN, ns)
    summary = json.loads(
        (tmp_path / "eval" / "main" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "completed" and summary["hf_revision"] == "b" * 40
    assert summary["private"] is True and summary["git_sha"] == "deadbeef"
    assert (
        summary["run"] == "main" and summary["hf_repo"] == "OWNER/fr-en-transformer-eval"
    )


def test_a_run_without_a_recorded_private_revision_is_a_failure(tmp_path: Path) -> None:
    rec = FakeRunStep(outputs={**_bench(tmp_path), "upload": lambda: None})  # writes no file
    ns = _single(tmp_path, rec)
    with pytest.raises(RuntimeError, match="no private hf_upload.json revision"):
        _exec(EVAL_RUN, ns)


def test_run_cell_stops_a_run_at_its_first_failing_step_and_starts_nothing_later_in_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = FakeRunStep(outputs=_bench(tmp_path), fail_at="tune:avg_last5")
    ns = _single(tmp_path, rec)
    with pytest.raises(RuntimeError, match="tune:avg_last5"):
        _exec(EVAL_RUN, ns)
    done = [n for n, _ in rec.calls]
    assert done[-1] == "tune:avg_last5" and "tune:avg_decay" not in done and "upload" not in done
    out = capsys.readouterr().out
    assert "EVAL STOPPED at tune:avg_last5 of main" in out and "main: FAILED" in out
    failed = json.loads((tmp_path / "eval" / "main" / "run_summary.json").read_text("utf-8"))
    assert failed["status"] == "FAILED" and "tune:avg_last5" in failed["error"]


def test_a_failing_hf_check_stops_the_run_before_any_gpu_step(tmp_path: Path) -> None:
    """hf-check (the real write probe) is the first plan step; its failure runs nothing later."""
    rec = FakeRunStep(outputs=_bench(tmp_path), fail_at="hf-check")
    ns = _single(tmp_path, rec)
    with pytest.raises(RuntimeError, match="hf-check"):
        _exec(EVAL_RUN, ns)
    assert [n for n, _ in rec.calls] == ["hf-verify", "hf-check"]  # verify is read-only


def test_run_cell_rerun_issues_the_same_steps_so_the_subprocesses_can_skip(
    tmp_path: Path,
) -> None:
    first = FakeRunStep(outputs=_bench(tmp_path))
    ns = _single(tmp_path, first)
    _exec(EVAL_RUN, ns)
    second = FakeRunStep()
    ns2 = _run_ns(tmp_path, second)
    _exec(EVAL_RUN, ns2)
    assert second.calls == first.calls  # resumption is each step's own skip-if-valid
    meta = json.loads((tmp_path / "eval" / "main" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["first_started_utc"] <= meta["last_started_utc"]


def test_run_cell_refuses_before_any_step_when_a_checkpoint_is_missing(tmp_path: Path) -> None:
    rec = FakeRunStep()
    ns = _single(tmp_path, rec)
    (ns["eval_ckpt_dirs"]["main"] / "step_00020757.pt").unlink()
    with pytest.raises(RuntimeError, match="step_00020757.pt"):
        _exec(EVAL_RUN, ns)
    assert [n for n, _ in rec.calls] == ["hf-verify"]  # only the read-only HF question was asked


def test_run_cell_fails_if_the_bench_leaves_no_usable_rates(tmp_path: Path) -> None:
    rec = FakeRunStep()  # the bench step writes nothing
    ns = _single(tmp_path, rec)
    with pytest.raises(RuntimeError, match="bench.json"):
        _exec(EVAL_RUN, ns)
    assert [n for n, _ in rec.calls][-1] == "bench"


def test_run_cell_dry_run_prints_plan_and_estimate_and_launches_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = FakeRunStep()
    ns = _run_ns(tmp_path, rec, dry=True)  # no checkpoints on disk
    _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    assert rec.calls == [] and not (tmp_path / "eval" / "main" / "run_meta.json").exists()
    assert "DRY_RUN: nothing launched" in out and "[upload]" in out and "[hf-check]" in out
    assert "[hf-verify]" in out and "skips it if so" in out
    assert "step_00019000.pt:MISSING" in out and "a real run would STOP" in out
    assert "ESTIMATE (STATIC: ASSUMED" in out and "main: planned only (DRY_RUN)" in out


# --- RUN = "all" ----------------------------------------------------------------------------------

ALL = ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")


def _all_ns(
    tmp_path: Path, rec: FakeRunStep, with_ckpts: bool = True, **over: Any
) -> dict[str, Any]:
    ns = _run_ns(tmp_path, rec, "all", **over)
    if with_ckpts:
        for r in ALL:
            _write_ckpts(ns["eval_ckpt_dirs"][r], r)
    return ns


def _all_benches(tmp_path: Path) -> dict[str, Any]:
    return {f"{r}:bench": _bench_file(tmp_path / "eval" / r / "bench.json") for r in ALL}


def test_all_runs_evaluate_main_first_then_the_ablations_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = FakeRunStep(outputs=_all_benches(tmp_path))
    ns = _all_ns(tmp_path, rec)
    _exec(EVAL_RUN, ns)
    order = list(dict.fromkeys(rec.runs))
    assert order == list(ALL)
    # each run is one contiguous block: hf-verify first, then its own plan, then the next run
    assert rec.runs == sorted(rec.runs, key=ALL.index)
    for run in ALL:
        names = [n for n, r in zip((c[0] for c in rec.calls), rec.runs, strict=True) if r == run]
        assert names == ["hf-verify", *(n for n, _ in _plan(ns, run, tmp_path))]
    out = capsys.readouterr().out
    assert "eval overall: all 4 run(s) ok" in out
    assert all(ns["eval_results"][r]["status"] == "completed" for r in ALL)
    for r in ALL:  # every run has its own summary with the HF revision
        saved = json.loads((tmp_path / "eval" / r / "run_summary.json").read_text("utf-8"))
        assert saved["run"] == r and saved["hf_revision"] == "b" * 40


def test_all_dry_run_shows_the_plan_of_every_run_and_launches_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = FakeRunStep()
    ns = _run_ns(tmp_path, rec, "all", dry=True)
    _exec(EVAL_RUN, ns)
    out = capsys.readouterr().out
    assert rec.calls == []
    positions = [out.index(f"EVAL RUN {r}:") for r in ALL]
    assert positions == sorted(positions)  # main first
    for r in ALL:
        assert f"--run {r}" in out and f"{r}.yaml" in out  # upload / candidates argv of that run
    assert out.count("[hf-verify]") == 4 and out.count("[upload]") == 4
    assert out.count("[validate-test]") == 1 and out.count(" --test") == 1  # main only
    assert out.count("planned only (DRY_RUN)") == 4 and "eval overall: all 4 run(s) ok" in out
    assert not (tmp_path / "eval").exists() or not any((tmp_path / "eval").iterdir())


def test_a_run_complete_on_hf_is_skipped_on_hf_evidence_not_on_drive_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = FakeRunStep(outputs=_all_benches(tmp_path), verify={"main": YES, "s2_rope_l4": YES})
    ns = _all_ns(tmp_path, rec, with_ckpts=False)  # not even checkpoints are needed for a skip
    for r in ("s1_sin_l4", "s3_rope_concat_l4"):
        _write_ckpts(ns["eval_ckpt_dirs"][r], r)
    # Drive claims s1 is uploaded; the HF repo does not hold it: it must be evaluated anyway.
    (tmp_path / "eval" / "s1_sin_l4" / "hf_upload.json").write_text(
        json.dumps({"repo": "o/r", "revision": "d" * 40, "private": True}), encoding="utf-8"
    )
    _exec(EVAL_RUN, ns)
    done = {r for (n, _), r in zip(rec.calls, rec.runs, strict=True) if n != "hf-verify"}
    assert done == {"s1_sin_l4", "s3_rope_concat_l4"}
    assert ns["eval_results"]["main"] == {"status": "skipped", "revision": "c" * 40}
    assert ns["eval_results"]["s1_sin_l4"]["status"] == "completed"  # overwrote the Drive claim
    out = capsys.readouterr().out
    assert "main: skipped, already complete on HF (verified); HF revision " + "c" * 40 in out
    assert "SKIPPED: complete on the HF repo (verified)" in out


def test_a_failing_run_does_not_skip_the_later_ones_and_the_cell_fails_at_the_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = FakeRunStep(outputs=_all_benches(tmp_path), fail_at={"main:tune:avg_last5"})
    ns = _all_ns(tmp_path, rec)
    with pytest.raises(RuntimeError) as err:
        _exec(EVAL_RUN, ns)
    assert "eval FAILED for 1 of 4 run(s)" in str(err.value) and "main:" in str(err.value)
    later = {r for r in rec.runs}
    assert later == set(ALL)  # the three ablations still ran after main failed
    assert "upload" not in [n for (n, _), r in zip(rec.calls, rec.runs, strict=True) if r == "main"]
    for r in ALL[1:]:
        assert ns["eval_results"][r]["status"] == "completed"
    assert ns["eval_results"]["main"]["status"] == "FAILED"
    out = capsys.readouterr().out
    assert "continuing with: s1_sin_l4, s2_rope_l4, s3_rope_concat_l4" in out
    assert "eval overall: 1 of 4 run(s) FAILED: main" in out  # printed BEFORE the cell raises
    assert "main: FAILED:" in out and "s3_rope_concat_l4: completed" in out


def test_a_missing_checkpoint_fails_only_that_run(tmp_path: Path) -> None:
    rec = FakeRunStep(outputs=_all_benches(tmp_path))
    ns = _all_ns(tmp_path, rec)
    (ns["eval_ckpt_dirs"]["s2_rope_l4"] / "step_00004107.pt").unlink()
    with pytest.raises(RuntimeError, match="(?s)s2_rope_l4: .*step_00004107.pt"):
        _exec(EVAL_RUN, ns)
    assert [r for r in ALL if ns["eval_results"][r]["status"] == "FAILED"] == ["s2_rope_l4"]
    assert "s3_rope_concat_l4" in rec.runs  # the run after the failed one still executed


@pytest.mark.parametrize("answer", ["", "garbage", "HF_RUN_COMPLETE=true", "HF_RUN_COMPLETE=maybe"])
def test_an_unreadable_hf_verify_answer_fails_that_run_closed_and_never_skips_it(
    tmp_path: Path, answer: str
) -> None:
    rec = FakeRunStep(outputs=_all_benches(tmp_path), verify={"main": answer})
    ns = _all_ns(tmp_path, rec)
    with pytest.raises(RuntimeError, match="main: .*hf-verify"):
        _exec(EVAL_RUN, ns)
    assert ns["eval_results"]["main"]["status"] == "FAILED"
    assert "main" not in [
        r for (n, _), r in zip(rec.calls, rec.runs, strict=True) if n != "hf-verify"
    ]
    assert ns["eval_results"]["s1_sin_l4"]["status"] == "completed"


def test_a_failing_hf_verify_subprocess_fails_that_run_and_the_rest_continue(
    tmp_path: Path,
) -> None:
    rec = FakeRunStep(outputs=_all_benches(tmp_path), fail_at={"s1_sin_l4:hf-verify"})
    ns = _all_ns(tmp_path, rec)
    with pytest.raises(RuntimeError, match="eval FAILED for 1 of 4"):
        _exec(EVAL_RUN, ns)
    assert ns["eval_results"]["s2_rope_l4"]["status"] == "completed"


def test_parse_hf_verify_and_format_eval_results() -> None:
    ns = _helpers("all")
    assert ns["parse_hf_verify"](YES) == (True, "c" * 40)
    assert ns["parse_hf_verify"](NO) == (False, None)
    for bad in ("", "HF_RUN_COMPLETE=true", "HF_RUN_COMPLETE=true\nHF_EVAL_REVISION=abc"):
        with pytest.raises(RuntimeError, match="hf-verify"):
            ns["parse_hf_verify"](bad)
    lines = ns["format_eval_results"](
        {
            "main": {"status": "completed", "revision": "b" * 40},
            "s1_sin_l4": {"status": "skipped", "revision": "c" * 40},
            "s2_rope_l4": {"status": "FAILED", "revision": None, "error": "boom"},
        }
    )
    text = "\n".join(lines)
    assert (
        "s2_rope_l4: FAILED: boom" in text
        and "eval overall: 1 of 3 run(s) FAILED: s2_rope_l4" in text
    )
    argv = ns["eval_verify_argv"]("main", hf_repo="o/r")
    assert argv[3:] == ["hf-verify", "--repo", "o/r", "--run", "main"]


def test_every_notebook_verify_argv_parses_with_the_real_cli() -> None:
    ns = _helpers("all")
    argv = ns["eval_verify_argv"]("s2_rope_l4", hf_repo="o/r")
    assert argv[:3] == [sys.executable, "-m", "nmt.eval_l4"]
    args = ev._parser().parse_args(argv[3:])
    assert (args.cmd, args.repo, args.run) == ("hf-verify", "o/r", "s2_rope_l4")


def test_run_cell_is_a_noop_outside_eval_mode(capsys: pytest.CaptureFixture[str]) -> None:
    rec = FakeRunStep()
    _exec(EVAL_RUN, {"EVAL": False, "CONFIG": "main", "run_step": rec})
    assert rec.calls == [] and "not 'eval_l4'" in capsys.readouterr().out


# --- the other cells ------------------------------------------------------------------------------


def _fake_colab(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    def get(name: str) -> str:
        if name not in values:
            raise KeyError(name)
        return values[name]

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=get)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Swap os.environ for a scratch dict so the cell's secret exports never leak between tests."""
    import os

    monkeypatch.setattr(os, "environ", {})
    return os


def test_secrets_cell_in_eval_mode_needs_the_write_token_and_gh_but_not_wandb(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    os = _isolated_env(monkeypatch)
    _fake_colab(monkeypatch, {"HF_TOKEN_WRITE": "hf-write-secret", "GH_TOKEN": "gh-secret"})
    _exec(SECRETS, {"EVAL": True, "MODE": "colab", "os": os})
    assert os.environ["HF_TOKEN"] == "hf-write-secret"  # the name huggingface_hub reads
    assert "HF_TOKEN_WRITE" not in os.environ and "WANDB_API_KEY" not in os.environ
    out = capsys.readouterr().out
    assert "hf-write-secret" not in out and "gh-secret" not in out
    assert "HF_TOKEN_WRITE" in out  # the secret NAME is what gets printed, never a value


def test_secrets_cell_names_the_missing_eval_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    os = _isolated_env(monkeypatch)
    _fake_colab(monkeypatch, {"HF_TOKEN": "read-only", "GH_TOKEN": "gh"})  # no HF_TOKEN_WRITE
    with pytest.raises(RuntimeError, match="HF_TOKEN_WRITE"):
        _exec(SECRETS, {"EVAL": True, "MODE": "colab", "os": os})


def test_secrets_cell_default_mode_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    os = _isolated_env(monkeypatch)
    _fake_colab(monkeypatch, {"HF_TOKEN": "h", "WANDB_API_KEY": "w", "GH_TOKEN": "g"})
    ns = {"EVAL": False, "MODE": "colab", "os": os}
    _exec(SECRETS, ns)
    assert ns["REQUIRED_SECRETS"] == ("HF_TOKEN", "WANDB_API_KEY", "GH_TOKEN")
    assert set(os.environ) == {"HF_TOKEN", "WANDB_API_KEY", "GH_TOKEN"}


def _drive_ns(tmp_path: Path, **over: Any) -> dict[str, Any]:
    ns: dict[str, Any] = {
        "Path": Path,
        "MODE": "local",
        "repo_dir": tmp_path,
        "ABLATION": False,
        "EVAL": True,
        "DRY_RUN": False,
        "CONFIG": "eval_l4",
        "RUN": "main",
    }
    ns.update(over)
    run = ns["RUN"]
    ns.setdefault("EVAL_RUN_LIST", ALL if run == "all" else (run,))
    return ns


def test_drive_cell_eval_local_reads_the_run_ckpt_without_creating_it(tmp_path: Path) -> None:
    ns = _drive_ns(tmp_path)
    _exec(DRIVE, ns)
    assert ns["ckpt_dir"] == tmp_path / "runs" / "notebook_main" / "ckpt"
    assert not ns["ckpt_dir"].exists()  # a missing checkpoint dir must fail in the eval cell
    assert ns["eval_root"] == tmp_path / "runs" / "notebook_eval_main" and ns["eval_root"].is_dir()


def test_drive_cell_all_has_one_ckpt_and_eval_dir_per_run_and_creates_none(
    tmp_path: Path,
) -> None:
    ns = _drive_ns(tmp_path, RUN="all")
    _exec(DRIVE, ns)
    assert list(ns["eval_ckpt_dirs"]) == list(ALL)  # main first
    for r in ALL:
        assert ns["eval_ckpt_dirs"][r] == tmp_path / "runs" / f"notebook_{r}" / "ckpt"
        assert ns["eval_roots"][r] == tmp_path / "runs" / f"notebook_eval_{r}"
        assert not ns["eval_roots"][r].exists()  # created by the eval cell only when it evaluates
    assert ns["ckpt_dir"] is None and ns["eval_root"] is None


def test_drive_cell_eval_colab_paths_follow_the_documented_drive_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mounted: list[str] = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mounted.append)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ns = _drive_ns(tmp_path, MODE="colab", RUN="s2_rope_l4", DRY_RUN=True)
    _exec(DRIVE, ns)
    base = Path("/content/drive/MyDrive/fr-en-transformer")
    assert ns["ckpt_dir"] == base / "runs" / "s2_rope_l4" / "ckpt"
    assert ns["eval_root"] == base / "eval" / "s2_rope_l4"
    assert ns["eval_roots"] == {"s2_rope_l4": base / "eval" / "s2_rope_l4"}


def test_train_data_and_wandb_cells_do_nothing_in_eval_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*_a: Any, **_k: Any) -> str:
        raise AssertionError("no subprocess may run in these cells in eval mode")

    ns: dict[str, Any] = {"EVAL": True, "run_step": boom, "MODE": "colab", "WANDB_ENTITY": "e"}
    ns.update(WANDB_PROJECT="p", os=__import__("os"))
    _exec(DATA, ns)
    assert ns["data_dir"] is None
    _exec(WANDB, ns)
    assert ns["wandb_mode"] == "offline"
    _exec(TRAIN, ns)
    assert ns["wandb_run_url"] is None and ns["stopped_after_first_ckpt"] is False
    out = capsys.readouterr().out
    assert "data: skipped" in out and "wandb: skipped" in out and "train: skipped" in out


# --- summary --------------------------------------------------------------------------------------


def _fill_eval_dir(root: Path, run: str = "main", threshold: int | None = 128) -> None:
    def put(rel: str, obj: Any) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")

    modes = {
        "greedy": {"sentences_per_second": 41.5, "output_tokens_per_second": 1200.0},
        "beam5": {"sentences_per_second": 9.25, "output_tokens_per_second": 275.0},
    }
    put(
        "bench.json",
        {
            "modes": modes,
            "n_sentences": 200,
            "batch_size": 32,
            "gpu": "NVIDIA L4",
            "segmentation": "off",
        },
    )

    def cand(obj: float, t: int | None) -> dict[str, Any]:
        return {
            "objective": obj,
            "bleu_union": 30.1,
            "chrf_union": 55.2,
            "chrf_e1": 60.3,
            "config": {"alpha": 0.8, "beam": 5, "segment_threshold": t},
        }

    put(
        "selection.json",
        {
            "candidates": {"final": cand(45.0, None), "avg_decay": cand(46.5, threshold)},
            "winner": {
                "candidate": "avg_decay",
                "objective": 46.5,
                "alpha": 0.8,
                "beam": 5,
                "segment_threshold": threshold,
            },
        },
    )
    put("decode_summary.json", {})
    for variant in ("seg_off", "seg_tuned"):
        for split in ("dev", "e1", "e2", "e2synth", "e3"):
            put(f"predictions/{variant}/{split}_predictions.json", {})
    if run == "main":
        put("test_predictions.json", {})
        put("validation.json", {"valid": True, "n_ids": 330, "empty_strings": 0})
    put("hf_upload.json", {"repo": "o/r", "revision": "b" * 40, "private": True})


def test_summary_lists_sha_gpu_bench_candidates_winner_files_validation_and_hf(
    tmp_path: Path,
) -> None:
    ns = _helpers("main")
    _fill_eval_dir(tmp_path)
    text = "\n".join(
        ns["format_eval_summary"](
            tmp_path, "main", "cafe123", {"preflight_gpu": "NVIDIA L4"}, "o/r"
        )
    )
    for needle in (
        "git commit SHA: cafe123",
        "GPU: NVIDIA L4",
        "bench greedy: 41.5 sent/s, 1200.0 out-tok/s",
        "bench beam5: 9.25 sent/s, 275.0 out-tok/s",
        "candidate final: objective 45.0000",
        "alpha=0.8 beam=5 T=off",
        "candidate avg_decay: objective 46.5000",
        "T=128",
        "WINNER: avg_decay objective 46.5000",
        "segment_threshold=128",
        "predictions/seg_off: dev_predictions.json, e1_predictions.json, e2_predictions.json",
        "predictions/seg_tuned: dev_predictions.json",
        "test predictions: test_predictions.json",
        "test validation: OK (330 ids, 0 empty)",
        "HF repo: o/r revision: " + "b" * 40 + " private: True",
        "HF_EVAL_REVISION=" + "b" * 40,
    ):
        assert needle in text, needle


def test_summary_says_when_the_tuned_threshold_is_off_and_for_unfinished_runs(
    tmp_path: Path,
) -> None:
    ns = _helpers("main")
    _fill_eval_dir(tmp_path, threshold=None)
    text = "\n".join(ns["format_eval_summary"](tmp_path, "main", "x", {}, "o/r"))
    assert "tuned T is 'off': the seg_tuned files are byte copies of seg_off" in text
    empty = "\n".join(ns["format_eval_summary"](tmp_path / "nothing", "main", "x", {}, "o/r"))
    for needle in ("bench: NOT DONE", "selection: NOT DONE", "test validation: NOT DONE"):
        assert needle in empty
    assert "HF upload to o/r: NOT DONE" in empty and "predictions/seg_off: NONE" in empty


def test_summary_reports_a_failed_test_validation_and_ablations_have_no_test(
    tmp_path: Path,
) -> None:
    ns = _helpers("main")
    _fill_eval_dir(tmp_path)
    (tmp_path / "validation.json").write_text(
        json.dumps({"valid": False, "error": "EvalStepError: 3 empty"}), encoding="utf-8"
    )
    text = "\n".join(ns["format_eval_summary"](tmp_path, "main", "x", {}, "o/r"))
    assert "test validation: FAILED: EvalStepError: 3 empty" in text
    abl = _helpers("s1_sin_l4")
    root = tmp_path / "abl"
    _fill_eval_dir(root, run="s1_sin_l4")
    out = "\n".join(abl["format_eval_summary"](root, "s1_sin_l4", "x", {}, "o/r"))
    assert "n/a (ablation runs are not decoded on the test set)" in out


def test_summary_cell_in_eval_mode_prints_the_eval_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = _helpers("main")
    _fill_eval_dir(tmp_path)
    ns.update(
        GIT_REF="v-test",
        git_sha="cafe123",
        git_describe="v-test",
        DATA_REVISION="rev",
        CONFIG="eval_l4",
        EVAL=True,
        ABLATION=False,
        preflight={"preflight_gpu": "NVIDIA L4", "preflight_precision": "bf16"},
        eval_roots={"main": tmp_path},
        eval_results={"main": {"status": "completed", "revision": "b" * 40}},
        HF_EVAL_REPO="o/r",
        RUN="main",
    )
    _exec(SUMMARY, ns)
    out = capsys.readouterr().out
    assert "WINNER: avg_decay" in out and "HF_EVAL_REVISION=" in out and "cafe123" in out
    assert "main: completed; HF revision " + "b" * 40 in out


def test_summary_cell_for_all_lists_every_run_and_does_not_misreport_a_skipped_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ns = _helpers("all")
    _fill_eval_dir(tmp_path / "s1_sin_l4", run="s1_sin_l4")
    results = {
        "main": {"status": "skipped", "revision": "c" * 40},
        "s1_sin_l4": {"status": "completed", "revision": "b" * 40},
        "s2_rope_l4": {"status": "FAILED", "revision": None, "error": "boom"},
        "s3_rope_concat_l4": {"status": "planned", "revision": None},
    }
    ns.update(
        GIT_REF="v-test",
        git_sha="cafe123",
        git_describe="v-test",
        DATA_REVISION="rev",
        CONFIG="eval_l4",
        EVAL=True,
        ABLATION=False,
        preflight={"preflight_gpu": "NVIDIA L4", "preflight_precision": "bf16"},
        eval_roots={r: tmp_path / r for r in ALL},
        eval_results=results,
        HF_EVAL_REPO="o/r",
    )
    _exec(SUMMARY, ns)
    out = capsys.readouterr().out
    assert "eval runs: main, s1_sin_l4, s2_rope_l4, s3_rope_concat_l4" in out
    assert "main: skipped, already complete on HF (verified)" in out
    assert "not re-evaluated: already complete on the HF repo" in out
    assert "s2_rope_l4: FAILED: boom" in out and "eval overall: 1 of 4 run(s) FAILED" in out
    assert "WINNER:" in out  # s1's local summary is printed


# --- notebook hygiene, execute_notebook, CI -------------------------------------------------------


def test_eval_cells_import_only_the_stdlib() -> None:
    import ast

    for cell_id in (EVAL_HELPERS, EVAL_RUN):
        for node in ast.walk(ast.parse(_src(cell_id))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(n.split(".")[0] in sys.stdlib_module_names for n in names), (cell_id, names)


def test_notebook_stays_output_free_with_the_eval_cells_present() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert [c.id for c in nb.cells].count(EVAL_RUN) == 1


def test_execute_notebook_allows_eval_only_with_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_execute_notebook()

    class Noop:
        executed = 0

        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def execute(self) -> None:
            type(self).executed += 1

    monkeypatch.setattr(mod, "NotebookClient", Noop)
    ev_args = ["--set", 'CONFIG="eval_l4"']
    assert mod.main([str(NOTEBOOK), *ev_args]) == 2
    assert "without DRY_RUN=True" in capsys.readouterr().err
    assert mod.main([str(NOTEBOOK), *ev_args, "--allow-non-smoke"]) == 2
    assert Noop.executed == 0
    assert mod.main([str(NOTEBOOK), *ev_args, "--set", "DRY_RUN=True"]) == 0 and Noop.executed == 1


def test_notebook_default_still_executes_as_smoke_in_ci_and_ci_runs_the_eval_dry_run() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--set CONFIG='\"smoke\"' --set PLANNED_STEPS=None --set RESUME_TEST=False" in ci
    assert "--set CONFIG='\"eval_l4\"' --set DRY_RUN=True" in ci
    assert "--set RUN='\"all\"'" in ci  # the CI dry run covers RUN="all"
    mod = _load_execute_notebook()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert mod.effective_config(nb) == "main"  # the committed default is the real main run
    assert mod.effective_param(nb, "RUN") == '"main"'
