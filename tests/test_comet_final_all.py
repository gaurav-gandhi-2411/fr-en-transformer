from __future__ import annotations

# The COMET stage inside the final_all plan, estimate and summary (nmt/final_all.py): the COMET
# steps come after the private upload, the estimate and the summary include the stage. The stage
# itself is tested in tests/test_comet_stage.py.
import json
from pathlib import Path

import pytest

import nmt.comet_stage as cs
import nmt.eval_l4 as ev
import nmt.final_all as fa
from tests.test_comet_stage import (
    CANDS,
    REPO_ROOT,
    REVISION,
    StubWorker,
    build_eval_root,
    run_score,
    tiny_splits,  # noqa: F401  (an autouse fixture: the summary test scores a tiny fake eval dir)
)


def test_plan_has_the_comet_steps_after_the_upload_in_order() -> None:
    plan = fa.plan_final_all(
        eval_root=Path("/e/final_all"), hf_repo="o/r", models=fa_models(), python="py"
    )
    names = [n for n, _ in plan]
    assert names[-8:] == [
        "upload",
        "comet:install",
        "comet:pull:main",
        "comet:pull:s1_sin_l4",
        "comet:pull:s2_rope_l4",
        "comet:pull:s3_rope_concat_l4",
        "comet:score",
        "comet:upload",
    ]
    assert len(names) == 24 + 7  # the 24 steps up to the upload + 7 COMET steps
    assert names.index("upload") < min(i for i, n in enumerate(names) if n.startswith("comet:"))
    assert all(n.startswith("comet:") for n in names[names.index("upload") + 1 :])
    argv = dict(plan)
    assert argv["comet:score"][argv["comet:score"].index("--batch-size") + 1] == "64"
    assert argv["comet:score"][argv["comet:score"].index("--precision") + 1] == "fp32"
    assert argv["comet:pull:s2_rope_l4"][-2:] == ["--run", "s2_rope_l4"]
    assert argv["comet:upload"][-2:] == ["--repo", "o/r"]


def fa_models() -> dict[str, Path]:
    return {n: Path(f"/m/{n}") for n in fa.MODEL_NAMES}


def test_plan_argvs_are_real_subcommands_and_batch_precision_pass_through() -> None:
    plan = fa.plan_final_all(
        eval_root=Path("/e"),
        hf_repo="o/r",
        models=fa_models(),
        python="py",
        comet_batch_size=16,
        comet_precision="bf16",
    )
    for name, argv in plan:
        if name.startswith("comet:"):
            args = cs._parser().parse_args(argv[3:])
            assert args.cmd == name.split(":")[1]
    score = cs._parser().parse_args(dict(plan)["comet:score"][3:])
    assert (score.batch_size, score.precision) == (16, "bf16")
    with pytest.raises(SystemExit):
        cs._parser().parse_args(["score", "--eval-root", "e", "--precision", "int4"])


def test_the_main_upload_does_not_depend_on_the_comet_steps() -> None:
    plan = fa.plan_final_all(eval_root=Path("/e"), hf_repo="o/r", models=fa_models(), python="py")
    before = [n for n, _ in plan if not n.startswith("comet:")]
    assert before[-1] == "upload" and before[0] == "hf-verify"
    # the required files of the main upload never mention the comet directory
    assert not any(
        "comet" in r for r in ev.required_rels("final_all", ev.expected_candidates("final_all"))
    )


def test_final_all_estimate_includes_the_comet_stage_and_the_second_upload() -> None:
    workload = json.loads((REPO_ROOT / "colab" / "eval_workload.json").read_text("utf-8"))
    est = fa.estimate_final_all(workload)
    for label, sc in est["scenarios"].items():
        parts = sc["parts_seconds"]
        assert parts["comet_gpu"] == pytest.approx(est["comet"]["n_triples"] / 100.0)
        assert parts["comet_setup"] == 480.0 and parts["comet_second_upload"] == 120.0
        assert sc["seconds"] == pytest.approx(sum(parts.values())), label
    explicit = fa.estimate_final_all(workload, comet_triples=1000)
    assert explicit["comet"]["n_triples"] == 1000
    text = "\n".join(fa.format_final_estimate(est))
    assert "COMET stage" in text and "50/s" in text and "150/s" in text and "ASSUMED" in text


def test_summary_lines_report_comet_or_say_not_done(tmp_path: Path) -> None:
    pre = {"preflight_gpu": "L4", "preflight_precision": "bf16"}
    lines = fa.format_final_all_summary(tmp_path, "sha", pre, "o/r")
    assert any(line.startswith("COMET-22: NOT DONE") for line in lines)
    scored = build_eval_root(tmp_path / "e")
    run_score(scored, StubWorker())
    root = (
        tmp_path / "summary"
    )  # the comet files only: report.json etc. are not this test's subject
    import shutil

    shutil.copytree(scored / "comet", root / "comet")
    up = {"repo": "o/r", "revision": REVISION, "private": True}
    ev._write_json(root / "comet_hf_upload.json", up)
    lines = fa.format_final_all_summary(root, "sha", pre, "o/r")
    text = "\n".join(lines)
    assert "COMET-22: 61 sets" in text and "COMET main/seg_off: dev " in text
    assert "COMET copy_source/baseline:" in text and f"HF_COMET_REVISION={REVISION}" in text
    assert f"COMET final_all_report/{CANDS[0]}: e1 " in text
