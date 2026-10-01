from __future__ import annotations

# Tests for nmt/pipeline.py: stage wiring (train -> export -> evaluate -> predict -> analyze)
# with the --run-dir override, using configs/smoke.yaml + synthetic training data so no real
# prepare/tokenize network call or real (slow) decode is required. Spec §2, §12.
import json
from pathlib import Path

import pytest

from nmt.pipeline import (
    stage_analyze,
    stage_evaluate,
    stage_export,
    stage_predict,
    stage_train,
    stage_tune,
)
from nmt.submission import validate_submission

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TOKENIZER = REPO_ROOT / "tokenizer" / "spm.model"
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"

pytestmark = pytest.mark.skipif(
    not REAL_TOKENIZER.is_file(), reason="tokenizer/spm.model not built yet (P2)"
)


def test_stage_train_writes_checkpoint_and_metrics_under_run_dir_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    # max_steps=2 stays well below smoke.yaml's eval_every=100, so this exercises the run_dir
    # override and checkpointing without also paying for a real (slower) eval-hook decode pass --
    # the eval hook itself is covered by tests/test_train_eval_hook.py.
    stage_train(
        SMOKE_CONFIG, seed=1, max_steps=2, synthetic=True, wandb_mode="disabled", run_dir=run_dir
    )
    assert (run_dir / "metrics.jsonl").is_file()
    assert list((run_dir / "ckpt").glob("step_*.pt"))


def test_stage_export_builds_a_loadable_model_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_train(
        SMOKE_CONFIG, seed=1, max_steps=2, synthetic=True, wandb_mode="disabled", run_dir=run_dir
    )
    export_dir = stage_export(SMOKE_CONFIG, out_dir=tmp_path / "export", run_dir=run_dir)
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "spm.model").is_file()


_TINY_SPLITS = {
    "e1": (
        [{"id": "e1_0", "source": "Bonjour.", "slice": "e1", "length": 8}],
        [{"id": "e1_0", "reference": "Hello.", "slice": "e1"}],
    ),
    "e2": (
        [{"id": "e2_0", "source": "Au revoir.", "slice": "e2", "length": 10}],
        [{"id": "e2_0", "reference": "Goodbye.", "slice": "e2"}],
    ),
    "e3": (
        [{"id": "e3_0", "source": "Il etait une fois.", "slice": "e3", "length": 18}],
        [{"id": "e3_0", "reference": "Once upon a time.", "slice": "e3"}],
    ),
    "dev": (
        [{"id": "dev_0", "source": "Merci.", "slice": "seen", "length": 6}],
        [{"id": "dev_0", "reference": "Thanks.", "slice": "seen"}],
    ),
}


def _export_tiny(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    stage_train(
        SMOKE_CONFIG, seed=1, max_steps=2, synthetic=True, wandb_mode="disabled", run_dir=run_dir
    )
    return stage_export(SMOKE_CONFIG, out_dir=tmp_path / "export", run_dir=run_dir)


def test_stage_evaluate_writes_eval_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmt.evaluate.load_split", lambda name: _TINY_SPLITS[name])
    export_dir = _export_tiny(tmp_path)
    out_dir = tmp_path / "eval_out"
    result = stage_evaluate(
        export_dir,
        run_name="smoke",
        ckpt_name="export",
        seed=1,
        beam=2,
        batch_size=4,
        n_bootstrap=10,
        splits=("dev", "e1"),
        out_dir=out_dir,
    )
    assert (out_dir / "eval.json").is_file()
    assert set(result["sets"].keys()) == {"dev", "e1"}


def test_stage_tune_writes_selection_grid_with_limits(tmp_path: Path) -> None:
    """CLI/stage smoke test on a tiny model with --limit-e1/--limit-e2 so the real (full-size)
    E1/E2 files stay fast to decode -- exercises the alpha x beam grid, the greedy (beam=1) dedup,
    and the post-hoc segmentation-threshold search, all scored via `nmt.selection.select`."""
    export_dir = _export_tiny(tmp_path)
    out_path = tmp_path / "selection_grid.json"
    result = stage_tune(
        export_dir,
        out_path,
        alphas=(0.6, 1.0),
        beams=(1, 4),
        seg_thresholds=(16,),
        limit_e1=2,
        limit_e2=2,
        batch_size=4,
    )
    assert out_path.is_file()
    assert result["n_e1"] == 2
    assert result["n_e2"] == 2
    assert result["limit_e1"] == 2
    assert result["limit_e2"] == 2
    assert set(result["winner"]) == {"alpha", "beam", "segment_threshold"}
    assert len(result["alpha_beam"]["grid"]) == 4  # 2 alphas x 2 beams
    # beam=1 ignores alpha: decoded once and shared across both nominal alpha=*_beam=1 points.
    assert result["alpha_beam"]["deduped_as"]["alpha=1.0_beam=1"] == "beam=1"
    assert "alpha=0.6_beam=4" not in result["alpha_beam"]["deduped_as"]
    assert set(result["segmentation"]["scores"]) == {"no_segmentation", "T=16"}
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result


def test_stage_tune_greedy_winner_reuses_its_shared_decode(tmp_path: Path) -> None:
    """Regression: when greedy wins, its predictions are stored under the shared decode key
    ("beam=1"), not the grid key ("alpha=0.6_beam=1") -- the segmentation stage must still find
    them. A beam=1-only grid forces a greedy winner."""
    export_dir = _export_tiny(tmp_path)
    result = stage_tune(
        export_dir,
        tmp_path / "selection_grid.json",
        alphas=(0.6, 1.0),
        beams=(1,),
        seg_thresholds=(16,),
        limit_e1=2,
        limit_e2=2,
        batch_size=4,
    )
    assert result["winner"]["beam"] == 1
    assert result["alpha_beam"]["deduped_as"][result["alpha_beam"]["best"]] == "beam=1"


def test_stage_predict_writes_and_validates(tmp_path: Path) -> None:
    export_dir = _export_tiny(tmp_path)
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text(
        json.dumps({"id": "x1", "source": "Bonjour."})
        + "\n"
        + json.dumps({"id": "x2", "source": "Merci."})
        + "\n",
        encoding="utf-8",
    )
    fake_sample = tmp_path / "sample_submission.json"
    fake_sample.write_text(json.dumps({"x1": "a", "x2": "b"}), encoding="utf-8")

    output_path = tmp_path / "preds.json"
    info = stage_predict(export_dir, input_path, output_path, beam=1, batch_size=2, validate=False)
    assert output_path.is_file()
    assert info["n_ids"] == 2
    # separately confirm the written file passes the real validator against a matching sample.
    validate_submission(output_path, sample_path=fake_sample)


def test_stage_analyze_reads_prediction_files_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nmt.analysis.load_split", lambda name: _TINY_SPLITS[name])
    eval_dir = tmp_path / "eval_out"
    eval_dir.mkdir(parents=True)
    (eval_dir / "e1_predictions.json").write_text(json.dumps({"e1_0": "Hello."}), encoding="utf-8")
    (eval_dir / "e3_predictions.json").write_text(
        json.dumps({"e3_0": "Once upon a time."}), encoding="utf-8"
    )
    (eval_dir / "dev_predictions.json").write_text(
        json.dumps({"dev_0": "Thanks."}), encoding="utf-8"
    )

    result = stage_analyze(eval_dir)
    assert (eval_dir / "analysis.json").is_file()
    assert (eval_dir / "examples.json").is_file()
    assert result["n_e1"] == 1
    assert result["n_e3"] == 1
    assert result["n_dev"] == 1
