from __future__ import annotations

# nmt/final_all.py: the STAGED final selection (PREREG 2026-10-03): stage 1 (7 beam model sets, top
# 2), stage 2 (4 MBR pools on those 2 only), the 10-candidate final pick, the report step, the
# plan, the estimate, and the final_all upload / hf-verify manifest extension in nmt/eval_l4.py
# (fake HfApi). Stub translators and fake tuning files: no model, no GPU, no real decoding.
import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

import pytest

import nmt.comet_stage as cs
import nmt.eval_l4 as ev
import nmt.final_all as fa
from tests.test_eval_l4 import REPO, REVISION, FakeApi

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZES = (1940, 1000)
S1 = fa.STAGE1_CANDIDATES  # main, A, B, main+A, main+B, A+B, main+A+B  (all "__beam")


@pytest.fixture(autouse=True)
def _canonical_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "_full_selection_sizes", lambda: SIZES)


# --- the candidate space and the constants ------------------------------------------------------


def test_candidate_space_stage1_names_and_final_candidates() -> None:
    space = fa.candidate_space()
    assert len(space) == 35 == len({c.name for c in space})
    assert S1 == (
        "main__beam",
        "A__beam",
        "B__beam",
        "main+A__beam",
        "main+B__beam",
        "A+B__beam",
        "main+A+B__beam",
    )
    assert fa.POOL_LABELS == ("mbr_beam8", "mbr_beam16", "mbr_eps0.02_n8", "mbr_eps0.02_n16")
    assert fa.candidate_by_name("A+B__mbr_eps0.02_n8").mbr == {
        "kind": "sample",
        "n": 8,
        "epsilon": 0.02,
        "seed": 1234,
    }
    # the final set: top-2 sets in MODEL-SET order (not rank order), beam before MBR, 5 each
    stage1 = {"top2": ["main+A__beam", "A__beam"]}
    final = fa.final_candidates(stage1)
    assert len(final) == 10
    assert [c.name for c in final] == [
        "A__beam",
        "A__mbr_beam8",
        "A__mbr_beam16",
        "A__mbr_eps0.02_n8",
        "A__mbr_eps0.02_n16",
        "main+A__beam",
        "main+A__mbr_beam8",
        "main+A__mbr_beam16",
        "main+A__mbr_eps0.02_n8",
        "main+A__mbr_eps0.02_n16",
    ]


def test_constants_match_the_staged_preregistration_and_mbr_defaults() -> None:
    from nmt import mbr
    from nmt.tune import DEFAULT_ALPHAS, DEFAULT_BEAMS

    assert fa.FINAL_ALL_ALPHAS == (1.2, 1.4, 1.6, 1.8, 2.0)
    assert fa.FINAL_ALL_BEAMS == (4, 5)  # beam 1 dropped from the extended grid
    assert fa.STAGE1_TOP_N == 2
    assert fa.SAMPLING_EPSILON == mbr.DEFAULT_EPSILON and fa.SAMPLING_SEED == 1234
    assert fa.SAMPLING_SEED == mbr.DEFAULT_SAMPLING_SEED
    assert [mbr.MBRConfig(k, n).label for k, n in fa.POOL_SPECS] == list(fa.POOL_LABELS)
    assert DEFAULT_ALPHAS == (0.6, 0.8, 1.0, 1.2) and DEFAULT_BEAMS == (1, 4, 5)  # untouched


def test_existing_runs_are_unchanged_and_final_all_manifest_requirements() -> None:
    assert ev.RUNS == ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
    assert ev.ALL_RUNS[-1] == "final_all" and ev.ALL_RUNS[:-1] == ev.RUNS
    assert ev.expected_candidates("final_all") == S1
    assert ev.expected_candidates("main") == ev.MAIN_CANDIDATES
    rels = ev.required_rels("final_all", S1)
    assert {"stage1.json", "report.json", "validation.json", "tuning/A__beam.json"} <= set(rels)
    assert "model/model.safetensors" not in rels
    assert ev.final_all_model_rels(["A", "B"])[0] == "models/A/model.safetensors"
    assert ev.final_all_stage2_rels(["A__mbr_beam8"]) == ["tuning/A__mbr_beam8.json"]


def test_grid_matches_only_the_staged_grid() -> None:
    grid = [{"alpha": a, "beam": b} for a in fa.FINAL_ALL_ALPHAS for b in (4, 5)]
    assert fa.grid_matches({"alpha_beam": {"grid": grid}})
    with_beam1 = grid + [{"alpha": a, "beam": 1} for a in fa.FINAL_ALL_ALPHAS]
    assert not fa.grid_matches({"alpha_beam": {"grid": with_beam1}})  # the pre-staging grid
    old = [{"alpha": a, "beam": b} for a in (0.6, 0.8, 1.0, 1.2) for b in (4, 5)]
    assert not fa.grid_matches({"alpha_beam": {"grid": old}})
    assert not fa.grid_matches({}) and not fa.grid_matches(None)


def test_parse_model_args_refuses_bad_input() -> None:
    assert fa.parse_model_args(["main=/a", "A=/b"]) == {"main": Path("/a"), "A": Path("/b")}
    for bad in (["main"], ["C=/x"], ["main=/a", "main=/b"], ["=x"]):
        with pytest.raises(ev.EvalStepError):
            fa.parse_model_args(bad)


# --- fixtures: fake models, fake tuning files -----------------------------------------------------

MODELS = {"main": Path("/m/main"), "A": Path("/m/A"), "B": Path("/m/B")}


def _fake_dirs(tmp_path: Path) -> dict[str, Path]:
    models = {}
    for m in fa.MODEL_NAMES:
        d = tmp_path / "models" / m
        d.mkdir(parents=True)
        for f in ("model.safetensors", "config.json", "spm.model"):
            (d / f).write_bytes(f"{m}-{f}".encode())
        models[m] = d
    return models


def _grid() -> list[dict[str, Any]]:
    return [{"alpha": a, "beam": b} for a in fa.FINAL_ALL_ALPHAS for b in fa.FINAL_ALL_BEAMS]


def _tuning(objective: float, **over: Any) -> dict[str, Any]:
    t: dict[str, Any] = {
        "limit_e1": None,
        "limit_e2": None,
        "n_e1": SIZES[0],
        "n_e2": SIZES[1],
        "alpha_beam": {"grid": _grid()},
        "winner": {"alpha": 1.6, "beam": 5, "segment_threshold": None},
        "segmentation": {
            "best": "no_segmentation",
            "scores": {
                "no_segmentation": {
                    "objective": objective,
                    "bleu_union": 1.0,
                    "chrf_union": 2.0,
                    "chrf_e1": 3.0,
                }
            },
        },
    }
    t.update(over)
    return t


def _hashes(models: dict[str, Path] | None, members: tuple[str, ...]) -> dict[str, str]:
    if models is None:
        return {m: f"h_{m}" for m in members}
    return fa._member_sha256(models, members)


def _write_stage1(
    d: Path,
    objectives: dict[str, float] | None = None,
    models: dict[str, Path] | None = None,
    alphas: dict[str, float] | None = None,
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for c in fa.stage1_candidates():
        t = _tuning(
            (objectives or {}).get(c.name, 40.0),
            candidate=c.name,
            members=list(c.members),
            member_sha256=_hashes(models, c.members),
        )
        t["winner"]["alpha"] = (alphas or {}).get(c.name, 1.6)
        (d / f"{c.name}.json").write_text(json.dumps(t), encoding="utf-8")


def _write_pool(
    d: Path,
    cand_name: str,
    objective: float,
    models: dict[str, Path] | None = None,
    alpha: float = 1.6,
) -> None:
    c = fa.candidate_by_name(cand_name)
    t = _tuning(
        objective,
        candidate=c.name,
        members=list(c.members),
        member_sha256=_hashes(models, c.members),
        alpha_beam={"note": "fixed", "alpha": alpha},
    )
    t["winner"] = {**t["winner"], "mbr": c.mbr, "alpha": alpha}
    (d / f"{c.name}.json").write_text(json.dumps(t), encoding="utf-8")


def _write_stage2(
    d: Path,
    sets: list[str],
    objectives: dict[str, float] | None = None,
    models: dict[str, Path] | None = None,
) -> None:
    for s in sets:
        for pool in fa.POOL_LABELS:
            name = f"{s}__{pool}"
            _write_pool(d, name, (objectives or {}).get(name, 40.0), models)


def _touch_later(path: Path, seconds: float = 10.0) -> None:
    t = path.stat().st_mtime + seconds
    os.utime(path, (t, t))


# --- stage 1: top 2 ------------------------------------------------------------------------------


def test_stage1_takes_exactly_the_top_two_by_objective(tmp_path: Path) -> None:
    d = tmp_path / "t"
    _write_stage1(d, {"B__beam": 50.0, "main+B__beam": 45.0, "main__beam": 44.0, "A+B__beam": 43.0})
    data = fa.run_stage1_select(d, tmp_path / "stage1.json")
    assert data["top2"] == ["B__beam", "main+B__beam"]  # two, not three
    assert data["ranking"][:4] == ["B__beam", "main+B__beam", "main__beam", "A+B__beam"]
    assert len(data["ranking"]) == 7 and data["candidate_order"] == list(S1)
    assert data["candidates"]["B__beam"]["objective"] == 50.0
    assert data["candidates"]["main+B__beam"]["members"] == ["main", "B"]
    assert data["grid"] == {"alphas": [1.2, 1.4, 1.6, 1.8, 2.0], "beams": [4, 5]}
    assert json.loads((tmp_path / "stage1.json").read_text("utf-8")) == data


def test_stage1_ties_go_to_the_earlier_listed_model_set(tmp_path: Path) -> None:
    # best is unique; the 2nd place is a three-way tie between sets 2, 4 and 6 -> the earliest wins
    d = tmp_path / "t"
    _write_stage1(
        d,
        {"A+B__beam": 60.0, "B__beam": 50.0, "main+B__beam": 50.0, "main+A+B__beam": 50.0},
    )
    data = fa.run_stage1_select(d, tmp_path / "s1.json")
    assert data["top2"] == ["A+B__beam", "B__beam"]
    assert data["ranking"][:4] == ["A+B__beam", "B__beam", "main+B__beam", "main+A+B__beam"]
    # a tie for first place: the earlier-listed one is ranked first, the other second
    _write_stage1(d, {"A__beam": 55.0, "main+A__beam": 55.0 + 1e-12})
    assert fa.run_stage1_select(d, tmp_path / "s1.json")["top2"] == ["A__beam", "main+A__beam"]
    _write_stage1(d, {"A__beam": 55.0 - 1e-12, "main+A__beam": 55.0})
    assert fa.run_stage1_select(d, tmp_path / "s1.json")["top2"] == ["A__beam", "main+A__beam"]
    # a difference above the tie epsilon is NOT a tie
    _write_stage1(d, {"A__beam": 55.0, "main+A__beam": 55.001})
    assert fa.run_stage1_select(d, tmp_path / "s1.json")["top2"] == ["main+A__beam", "A__beam"]


@pytest.mark.parametrize("damage", ["missing", "smoke", "beam1_grid", "mixed_weights"])
def test_stage1_refuses_unfit_tunings(tmp_path: Path, damage: str) -> None:
    d = tmp_path / "t"
    _write_stage1(d)
    victim = d / "A__beam.json"
    t = json.loads(victim.read_text("utf-8"))
    if damage == "missing":
        victim.unlink()
    elif damage == "smoke":
        t["limit_e1"] = 10
    elif damage == "beam1_grid":
        t["alpha_beam"]["grid"] = _grid() + [{"alpha": 1.2, "beam": 1}]
    else:
        t["member_sha256"] = {"A": "h_other"}  # A__beam and main+A__beam now disagree on A
    if damage != "missing":
        victim.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(ev.EvalStepError):
        fa.run_stage1_select(d, tmp_path / "s1.json")
    assert not (tmp_path / "s1.json").exists()


def test_stage1_file_validity_content_and_mtime(tmp_path: Path) -> None:
    d, out = tmp_path / "t", tmp_path / "s1.json"
    _write_stage1(d, {"A__beam": 50.0, "B__beam": 49.0})
    assert not fa.stage1_is_valid(out, d)  # not written yet
    fa.run_stage1_select(d, out)
    assert fa.stage1_is_valid(out, d)
    _touch_later(d / "main__beam.json")  # a tuning newer than stage1.json
    assert not fa.stage1_is_valid(out, d)
    fa.run_stage1_select(d, out)
    _touch_later(out, 30.0)  # the tuning above was touched into the future
    assert fa.stage1_is_valid(out, d)
    # content change with older mtime: B overtakes A -> the stored top 2 no longer matches
    t = json.loads((d / "B__beam.json").read_text("utf-8"))
    t["segmentation"]["scores"]["no_segmentation"]["objective"] = 90.0
    (d / "B__beam.json").write_text(json.dumps(t), encoding="utf-8")
    os.utime(d / "B__beam.json", (out.stat().st_mtime - 5, out.stat().st_mtime - 5))
    assert not fa.stage1_is_valid(out, d)


# --- stage 2: only the top-2 sets ----------------------------------------------------------------


class _Stub:
    """Translator stand-in: upper-cases sources (plus a tag), records its calls."""

    device = types.SimpleNamespace(type="cpu")

    def __init__(self, tag: str = "") -> None:
        self.tag = tag
        self.calls: list[dict[str, Any]] = []
        self.stats = types.SimpleNamespace(n_beam=0, n_greedy_fallback=0, n_copy_fallback=0)
        self.sp = types.SimpleNamespace(encode=lambda text, out_type=int: text.split())

    def translate(self, texts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append({"n": len(texts), **kwargs})
        self.stats.n_beam += len(texts)
        return [(t.upper() + self.tag) or "x" for t in texts]


def _stage_env(
    tmp_path: Path, objectives: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], Path, Path, list[str]]:
    """Fake model dirs, stage-1 tunings made by those weights, stage1.json; stage-2 tunes are
    faked (they record the call) so no decoding happens. Returns (models, tdir, stage1, calls)."""
    models = _fake_dirs(tmp_path)
    tdir, stage1 = tmp_path / "tuning", tmp_path / "stage1.json"
    _write_stage1(tdir, objectives, models)
    fa.run_stage1_select(tdir, stage1)
    calls: list[str] = []

    def fake_pool_tune(
        translator: Any,
        cand: fa.FinalAllCandidate,
        base: dict[str, Any],
        out_path: Path,
        batch_size: int,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(cand.name)
        t = _tuning(
            40.0,
            alpha_beam={"note": "fixed", "alpha": base["winner"]["alpha"]},
            **extra,
        )
        t["winner"] = {**t["winner"], "mbr": cand.mbr}
        Path(out_path).write_text(json.dumps(t), encoding="utf-8")
        return t

    monkeypatch.setattr(fa, "run_pool_tune", fake_pool_tune)
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: _Stub())
    return models, tdir, stage1, calls


def test_stage2_runs_only_on_the_two_top_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, tdir, stage1, calls = _stage_env(
        tmp_path, {"main+A__beam": 50.0, "B__beam": 49.0, "main__beam": 48.0}, monkeypatch
    )
    names = {
        fa.tune_stage2(pool, rank, models, tdir, stage1)[0]
        for rank in (1, 2)
        for pool in fa.POOL_LABELS
    }
    assert names == {f"{s}__{p}" for s in ("main+A", "B") for p in fa.POOL_LABELS}
    assert len(names) == 8 and set(calls) == names
    # nothing was tuned for any other model set; only stage-1 and these 8 files exist
    files = {p.stem for p in tdir.iterdir()}
    assert files == set(S1) | names
    with pytest.raises(ev.EvalStepError, match="--rank"):
        fa.tune_stage2("mbr_beam8", 3, models, tdir, stage1)  # no third set
    with pytest.raises(ev.EvalStepError, match="unknown pool"):
        fa.tune_stage2("mbr_beam4", 1, models, tdir, stage1)
    # the second call is a skip (valid file), not a re-run
    calls.clear()
    assert fa.tune_stage2("mbr_beam8", 1, models, tdir, stage1) == ("main+A__mbr_beam8", False)
    assert calls == []


def test_stage2_follows_the_top_two_at_run_time_not_a_fixed_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, tdir, stage1, _ = _stage_env(
        tmp_path, {"A+B__beam": 70.0, "A__beam": 60.0}, monkeypatch
    )
    assert fa.tune_stage2("mbr_beam16", 1, models, tdir, stage1)[0] == "A+B__mbr_beam16"
    assert fa.tune_stage2("mbr_beam16", 2, models, tdir, stage1)[0] == "A__mbr_beam16"


def test_a_stale_stage1_file_invalidates_stage2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, tdir, stage1, calls = _stage_env(
        tmp_path, {"A__beam": 50.0, "B__beam": 49.0}, monkeypatch
    )
    assert fa.tune_stage2("mbr_beam8", 1, models, tdir, stage1)[1] is True
    # a stage-1 tuning is redone (newer): stage1.json is stale -> no stage-2 step may run
    _touch_later(tdir / "B__beam.json")
    calls.clear()
    with pytest.raises(ev.EvalStepError, match="stale"):
        fa.tune_stage2("mbr_beam8", 2, models, tdir, stage1)
    assert calls == []
    # ... and the final selection refuses too
    with pytest.raises(ev.EvalStepError, match="stale"):
        fa.run_final_select(tdir, stage1, tmp_path / "sel.json")
    fa.run_stage1_select(tdir, stage1)  # refreshed: stage 2 may run again
    _touch_later(stage1, 30.0)  # (the tuning was touched into the future)
    assert fa.tune_stage2("mbr_beam8", 2, models, tdir, stage1)[1] is True
    # a missing stage1.json refuses as well
    stage1.unlink()
    with pytest.raises(ev.EvalStepError, match="missing or stale"):
        fa.tune_stage2("mbr_beam8", 1, models, tdir, stage1)


def test_stage2_file_for_changed_weights_or_moved_alpha_is_redone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, tdir, stage1, calls = _stage_env(
        tmp_path, {"A__beam": 50.0, "B__beam": 49.0}, monkeypatch
    )
    assert fa.tune_stage2("mbr_beam8", 1, models, tdir, stage1)[1] is True
    assert fa.tune_stage2("mbr_beam8", 1, models, tdir, stage1)[1] is False
    # the stage-1 winner alpha of A moves (stage-1 redone): the pool file used the old alpha
    _write_stage1(tdir, {"A__beam": 50.0, "B__beam": 49.0}, models, alphas={"A__beam": 1.2})
    fa.run_stage1_select(tdir, stage1)
    assert fa.tune_stage2("mbr_beam8", 1, models, tdir, stage1)[1] is True
    assert json.loads((tdir / "A__mbr_beam8.json").read_text("utf-8"))["alpha_beam"]["alpha"] == 1.2
    # a stage-1 beam tuning made by other weights than the current ones is refused
    (models["A"] / "model.safetensors").write_bytes(b"retrained A")
    with pytest.raises(ev.EvalStepError, match="current model weights"):
        fa.tune_stage2("mbr_beam8", 1, models, tdir, stage1)


# --- final selection -----------------------------------------------------------------------------


def _final_env(
    tmp_path: Path,
    s1: dict[str, float],
    s2: dict[str, float],
    top2: list[str],
) -> tuple[Path, Path, Path]:
    tdir, stage1 = tmp_path / "tuning", tmp_path / "stage1.json"
    _write_stage1(tdir, s1)
    fa.run_stage1_select(tdir, stage1)
    assert [n.removesuffix("__beam") for n in ev._read_json(stage1)["top2"]] == top2
    _write_stage2(tdir, top2, s2)
    os.utime(stage1, (stage1.stat().st_mtime - 100, stage1.stat().st_mtime - 100))
    for p in tdir.iterdir():
        os.utime(p, (stage1.stat().st_mtime - 50,) * 2)
    os.utime(stage1, (stage1.stat().st_mtime + 1,) * 2)
    return tdir, stage1, tmp_path / "selection.json"


LOW = {c: 10.0 for c in S1}  # every other model set clearly out of the top 2


def test_final_selection_has_ten_candidates_and_records_everything(tmp_path: Path) -> None:
    tdir, stage1, out = _final_env(
        tmp_path,
        {"A__beam": 50.0, "main+A__beam": 49.0, "main__beam": 45.0},
        {"main+A__mbr_eps0.02_n16": 51.0, "A__mbr_beam8": 50.5},
        ["A", "main+A"],
    )
    sel = fa.run_final_select(tdir, stage1, out)
    assert json.loads(out.read_text("utf-8")) == sel
    assert len(sel["candidate_order"]) == 10 == len(sel["candidates"]) == len(sel["ranking"])
    assert sel["candidate_order"][0] == "A__beam" and sel["candidate_order"][5] == "main+A__beam"
    assert sel["top2"] == ["A__beam", "main+A__beam"]
    assert sel["stage1"]["candidates"]["main__beam"]["objective"] == 45.0
    assert len(sel["stage2_candidates"]) == 8 and all(
        "__mbr_" in n for n in sel["stage2_candidates"]
    )
    win, runner = sel["winner"], sel["runner_up"]
    assert win["candidate"] == "main+A__mbr_eps0.02_n16" and win["members"] == ["main", "A"]
    assert win["mbr"] == {"kind": "sample", "n": 16, "epsilon": 0.02, "seed": 1234}
    assert win["objective"] == 51.0 and win["alpha"] == 1.6
    assert runner["candidate"] == "A__mbr_beam8" and runner["objective"] == 50.5
    assert runner["members"] == ["A"]
    assert sel["ranking"][:3] == ["main+A__mbr_eps0.02_n16", "A__mbr_beam8", "A__beam"]
    assert sel["tie_rule"] == fa.FINAL_TIE_RULE and "beam before MBR" in sel["tie_rule"]
    assert sel["production"]["candidate"] == "A__beam"
    assert fa.final_selection_is_valid(out, tdir, stage1)


def test_final_ties_go_to_model_set_order_then_beam_before_mbr(tmp_path: Path) -> None:
    # everything equal: the winner is the first of the 10 (model-set A, its beam candidate)
    tdir, stage1, out = _final_env(
        tmp_path, {"A__beam": 50.0, "main+A__beam": 49.0}, {"A__mbr_beam8": 50.0}, ["A", "main+A"]
    )
    sel = fa.run_final_select(tdir, stage1, out)
    assert (
        sel["winner"]["candidate"] == "A__beam" and sel["runner_up"]["candidate"] == "A__mbr_beam8"
    )
    # a tie between an MBR config of the EARLIER set and the later set's beam: earlier set wins
    tdir2, stage12, out2 = _final_env(
        tmp_path / "two",
        {**LOW, "A__beam": 40.0, "main+A__beam": 40.0},
        {"A__mbr_beam16": 60.0, "main+A__mbr_beam8": 60.0},
        ["A", "main+A"],
    )
    sel2 = fa.run_final_select(tdir2, stage12, out2)
    assert sel2["winner"]["candidate"] == "A__mbr_beam16"
    assert sel2["tied_candidates"] == ["A__mbr_beam16", "main+A__mbr_beam8"]
    assert sel2["runner_up"]["candidate"] == "main+A__mbr_beam8"
    # a tie between MBR and beam of the same set: beam first
    tdir3, stage13, out3 = _final_env(
        tmp_path / "three",
        {**LOW, "A__beam": 70.0, "main+A__beam": 40.0},
        {"A__mbr_beam8": 70.0},
        ["A", "main+A"],
    )
    sel3 = fa.run_final_select(tdir3, stage13, out3)
    assert sel3["winner"]["candidate"] == "A__beam"
    assert sel3["runner_up"]["candidate"] == "A__mbr_beam8"


def test_runner_up_is_the_second_best_even_when_the_stage1_beam_loses(tmp_path: Path) -> None:
    tdir, stage1, out = _final_env(
        tmp_path,
        {"main__beam": 50.0, "B__beam": 49.0},
        {"main__mbr_beam8": 55.0, "B__mbr_eps0.02_n8": 53.0},
        ["main", "B"],
    )
    sel = fa.run_final_select(tdir, stage1, out)
    assert sel["winner"]["candidate"] == "main__mbr_beam8"
    assert sel["runner_up"]["candidate"] == "B__mbr_eps0.02_n8"
    assert sel["production"]["candidate"] == "main__beam"  # best single by STAGE-1 objective


def test_final_ignores_stage2_files_of_sets_outside_the_top_two(tmp_path: Path) -> None:
    tdir, stage1, out = _final_env(
        tmp_path, {"A__beam": 50.0, "B__beam": 49.0}, {"A__mbr_beam8": 51.0}, ["A", "B"]
    )
    _write_stage2(tdir, ["main"], {"main__mbr_beam8": 99.0})  # a leftover from another stage 1
    os.utime(tdir / "main__mbr_beam8.json", (stage1.stat().st_mtime - 5,) * 2)
    sel = fa.run_final_select(tdir, stage1, out)
    assert sel["winner"]["candidate"] == "A__mbr_beam8"
    assert not any(n.startswith("main__") for n in sel["candidate_order"])


@pytest.mark.parametrize("damage", ["missing", "smoke", "alpha"])
def test_final_refuses_missing_or_unfit_stage2_tunings(tmp_path: Path, damage: str) -> None:
    tdir, stage1, out = _final_env(tmp_path, {"A__beam": 50.0, "B__beam": 49.0}, {}, ["A", "B"])
    f = tdir / "B__mbr_eps0.02_n16.json"
    t = json.loads(f.read_text("utf-8"))
    if damage == "missing":
        f.unlink()
    elif damage == "smoke":
        t["limit_e2"] = 5
    else:
        t["alpha_beam"]["alpha"] = 1.2  # B's stage-1 winner alpha is 1.6
    if damage != "missing":
        f.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(ev.EvalStepError, match="select"):
        fa.run_final_select(tdir, stage1, out)
    assert not out.exists()


def test_final_selection_validity_tracks_inputs(tmp_path: Path) -> None:
    tdir, stage1, out = _final_env(tmp_path, {"A__beam": 50.0, "B__beam": 49.0}, {}, ["A", "B"])
    assert not fa.final_selection_is_valid(out, tdir, stage1)
    fa.run_final_select(tdir, stage1, out)
    assert fa.final_selection_is_valid(out, tdir, stage1)
    newer = out.stat().st_mtime + 60  # a stage-2 tuning newer than selection.json
    os.utime(tdir / "A__mbr_beam8.json", (newer, newer))
    assert not fa.final_selection_is_valid(out, tdir, stage1)


def test_production_config_is_the_best_single_model_beam_only() -> None:
    def entry(name: str, obj: float, **cfg: Any) -> dict[str, Any]:
        return {
            "objective": obj,
            "members": list(fa.candidate_by_name(name).members),
            "config": {"alpha": 1.4, "beam": 4, "segment_threshold": 192, **cfg},
        }

    stage1 = {
        "candidates": {
            "main__beam": entry("main__beam", 50.0),
            "A__beam": entry("A__beam", 52.0),
            "B__beam": entry("B__beam", 52.0),  # tie with A -> the earlier-listed (A)
            "main+A+B__beam": entry("main+A+B__beam", 60.0),  # an ensemble never qualifies
        }
    }
    prod = fa.production_config(stage1)
    assert prod["candidate"] == "A__beam" and prod["mbr"] is None and prod["members"] == ["A"]
    assert (prod["alpha"], prod["beam"], prod["segment_threshold"]) == (1.4, 4, 192)


# --- plan ----------------------------------------------------------------------------------------


def _plan() -> list[tuple[str, list[str]]]:
    return fa.plan_final_all(
        eval_root=Path("/eval/final_all"), hf_repo="o/r", models=MODELS, python="py"
    )


def test_plan_is_staged_gates_first_and_private_upload_last() -> None:
    names = [n for n, _ in _plan()]
    assert names[:3] == ["hf-verify", "hf-check", "bench"]  # hf-check before any GPU work
    assert names[3:10] == [f"tune:{c}" for c in S1]  # stage 1: exactly the 7 beam candidates
    assert names[10] == "stage1-select"
    stage2 = names[11:19]
    assert stage2 == [f"tune-stage2:rank{r}:{p}" for r in (1, 2) for p in fa.POOL_LABELS]
    assert names[19:24] == ["select", "report", "decode", "validate-test", "upload"]
    assert names[24:] == [  # the non-fatal COMET stage, strictly after the private upload
        "comet:install",
        *[f"comet:pull:{r}" for r in ev.RUNS],
        "comet:score",
        "comet:upload",
    ]
    assert len(names) == 31


def test_plan_stage2_steps_read_the_top_two_at_run_time() -> None:
    plan = dict(_plan())
    for name, argv in plan.items():
        if name.startswith("tune-stage2"):
            assert "--rank" in argv and "--pool" in argv and "--stage1" in argv
            # no model-set name is baked into a stage-2 step
            assert not any(c in " ".join(argv) for c in fa.STAGE1_CANDIDATES)
    assert plan["stage1-select"][3] == "stage1-select"
    assert plan["decode"][-1] == "--test" and plan["hf-verify"][-2:] == ["--run", "final_all"]


def test_plan_argvs_are_real_subcommands() -> None:
    for name, argv in _plan():
        parsers = {"nmt.final_all": fa._parser(), "nmt.comet_stage": cs._parser()}
        parser = parsers.get(argv[2], ev._parser())
        args = parser.parse_args(argv[3:])
        assert args.cmd == argv[3], name
    assert fa._parser().parse_args(dict(_plan())["tune-stage2:rank2:mbr_beam16"][3:]).rank == 2
    with pytest.raises(SystemExit):  # stage 1 refuses an MBR candidate name
        fa._parser().parse_args(
            ["tune", "--name", "A__mbr_beam8", "--model", "A=x", "--tuning-dir", "t"]
        )
    with pytest.raises(SystemExit):  # and stage 2 only knows ranks 1 and 2
        fa._parser().parse_args(
            ["tune-stage2", "--pool", "mbr_beam8", "--rank", "3", "--model", "A=x"]
            + ["--tuning-dir", "t", "--stage1", "s"]
        )


def test_plan_module_imports_only_the_stdlib_at_top_level() -> None:
    src = (REPO_ROOT / "nmt" / "final_all.py").read_text(encoding="utf-8")
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("torch" in ln or "nmt.mbr" in ln or "nmt.ensemble" in ln for ln in top)


# --- tune steps (stage 1 beam, stage 2 pool), translators -----------------------------------------


def test_pool_translator_forwards_mbr_and_drops_beam() -> None:
    inner = _Stub()
    pt = fa.PoolTranslator(inner, "CFG")
    out = pt.translate(["a b"], batch_size=4, beam=7, alpha=1.4, segment_threshold=None)
    assert out == ["A B"] and pt.stats is inner.stats
    assert inner.calls[0]["mbr"] == "CFG" and "beam" not in inner.calls[0]
    assert inner.calls[0]["alpha"] == 1.4


def _fake_beam_tune(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[dict[str, Any]]]:
    import nmt.tune

    ran: list[str] = []
    kws: list[dict[str, Any]] = []

    def fake(model_dir: Path, out_path: Path, **kw: Any) -> dict[str, Any]:
        ran.append(Path(out_path).stem)
        kws.append(kw)
        t = _tuning(42.0, **kw["extra"])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(t), encoding="utf-8")
        return t

    monkeypatch.setattr(nmt.tune, "run_tune", fake)
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: _Stub())
    return ran, kws


def test_stage1_tune_uses_the_staged_grid_and_records_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    ran, kws = _fake_beam_tune(monkeypatch)
    tdir = tmp_path / "tuning"
    assert fa.tune_candidate("main+A__beam", models, tdir) is True
    assert kws[0]["alphas"] == fa.FINAL_ALL_ALPHAS and kws[0]["beams"] == (4, 5)
    assert kws[0]["extra"]["members"] == ["main", "A"]
    assert set(kws[0]["extra"]["member_sha256"]) == {"main", "A"}
    assert fa.tune_candidate("main+A__beam", models, tdir) is False  # valid file: skipped


def test_stage1_tuning_on_the_pre_staging_grid_is_redone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    ran, _ = _fake_beam_tune(monkeypatch)
    tdir = tmp_path / "tuning"
    tdir.mkdir()
    old = _tuning(
        41.0,
        alpha_beam={"grid": _grid() + [{"alpha": 1.2, "beam": 1}]},
        candidate="main__beam",
        member_sha256=_hashes(models, ("main",)),
    )
    (tdir / "main__beam.json").write_text(json.dumps(old), encoding="utf-8")
    assert fa.tune_candidate("main__beam", models, tdir) is True and ran == ["main__beam"]


def test_beam_tuning_made_by_other_weights_is_redone_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    ran, _ = _fake_beam_tune(monkeypatch)
    tdir = tmp_path / "tuning"
    assert fa.tune_candidate("A__beam", models, tdir) is True
    assert fa.tune_candidate("A__beam", models, tdir) is False
    (models["A"] / "model.safetensors").write_bytes(b"a retrained model")
    assert fa.tune_candidate("A__beam", models, tdir) is True  # stale: redone
    assert ran == ["A__beam", "A__beam"]
    t = json.loads((tdir / "A__beam.json").read_text("utf-8"))
    del t["member_sha256"]  # a file with no hashes cannot be shown current
    (tdir / "A__beam.json").write_text(json.dumps(t), encoding="utf-8")
    assert fa.tune_candidate("A__beam", models, tdir) is True


def test_ensemble_tuning_is_stale_when_any_member_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    ran, _ = _fake_beam_tune(monkeypatch)
    tdir = tmp_path / "tuning"
    fa.tune_candidate("main+B__beam", models, tdir)
    (models["B"] / "model.safetensors").write_bytes(b"other B")
    assert fa.tune_candidate("main+B__beam", models, tdir) is True and len(ran) == 2
    (models["A"] / "model.safetensors").write_bytes(b"other A")  # not a member: still current
    assert fa.tune_candidate("main+B__beam", models, tdir) is False


def test_pool_tune_runs_the_real_procedure_with_a_stub_translator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    stub = _Stub()
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: stub)
    tdir = tmp_path / "tuning"
    tdir.mkdir()
    with pytest.raises(ev.EvalStepError, match="needs a valid full rule-1 tuning"):
        fa.tune_candidate("A__mbr_beam8", models, tdir)
    base = _tuning(41.0, candidate="A__beam", member_sha256=_hashes(models, ("A",)))
    base["winner"]["alpha"] = 1.8
    (tdir / "A__beam.json").write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(ev, "_full_selection_sizes", lambda: (1940, 1000))
    assert fa.tune_candidate("A__mbr_beam8", models, tdir, batch_size=64) is True
    out = json.loads((tdir / "A__mbr_beam8.json").read_text("utf-8"))
    assert out["winner"]["alpha"] == 1.8 and out["winner"]["beam"] == 8
    assert out["winner"]["mbr"]["kind"] == "beam" and out["members"] == ["A"]
    assert out["alpha_beam"]["alpha"] == 1.8 and ev.tuning_is_full(out)
    assert set(out["segmentation"]["scores"]) == {
        "no_segmentation",
        "T=64",
        "T=128",
        "T=192",
        "T=256",
    }
    assert {c["alpha"] for c in stub.calls} == {1.8}
    assert fa.tune_candidate("A__mbr_beam8", models, tdir) is False


# --- decode --------------------------------------------------------------------------------------


def test_decode_winner_builds_the_winning_ensemble_pool_and_decodes_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []
    stub = _Stub()

    def fake_build(models: Any, members: Any, mbr: Any = None, device: Any = None) -> Any:
        built.append((list(members), mbr))
        return stub

    monkeypatch.setattr(fa, "build_translator", fake_build)
    d = tmp_path / "eval"
    d.mkdir()
    win = {
        "candidate": "main+B__mbr_eps0.02_n16",
        "alpha": 1.4,
        "beam": None,
        "segment_threshold": None,
        "members": ["main", "B"],
        "mbr": {"kind": "sample", "n": 16, "epsilon": 0.02, "seed": 1234},
    }
    (d / "selection.json").write_text(json.dumps({"winner": win}), encoding="utf-8")
    test_inputs = tmp_path / "t.jsonl"
    test_inputs.write_text(json.dumps({"id": "test_0", "source": "x"}), encoding="utf-8")
    summary = fa.decode_winner(
        d,
        MODELS,
        include_test=True,
        load_rows=lambda split: [{"id": f"{split}_0", "source": "s"}],
        test_inputs=test_inputs,
    )
    assert built == [(["main", "B"], win["mbr"])]
    assert summary["candidate"] == "main+B__mbr_eps0.02_n16" and summary["test_included"]
    assert (d / "test_predictions.json").is_file()
    assert (d / "predictions" / "seg_off" / "e3_predictions.json").is_file()


def test_decode_refuses_without_a_selection(tmp_path: Path) -> None:
    with pytest.raises(ev.EvalStepError, match="selection.json"):
        fa.decode_winner(tmp_path, MODELS)


# --- report: bootstrap + latency -----------------------------------------------------------------


def _selection_for_report(winner_is_production: bool = False) -> dict[str, Any]:
    def cfg(name: str, **kw: Any) -> dict[str, Any]:
        c = fa.candidate_by_name(name)
        return {
            "candidate": name,
            "members": list(c.members),
            "mbr": c.mbr,
            "alpha": 1.6,
            "beam": 8 if c.mbr else 5,
            "segment_threshold": kw.get("t"),
            "objective": 50.0,
        }

    win = cfg("main__beam" if winner_is_production else "main+A__mbr_beam8", t=192)
    return {
        "winner": win,
        "runner_up": cfg("A__beam"),
        "production": cfg("main__beam", t=None),
    }


def test_report_bootstrap_and_latency_with_stub_translators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nmt.evaluate import load_split

    sel = _selection_for_report()
    d = tmp_path / "eval"
    d.mkdir()
    (d / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    tags: list[str] = []

    def fake_build(models: Any, members: Any, mbr: Any = None, device: Any = None) -> Any:
        tags.append("+".join(members) + ("/mbr" if mbr else ""))
        return _Stub(tag=" " + "".join(members))

    monkeypatch.setattr(fa, "build_translator", fake_build)
    rep = fa.run_report(d, MODELS, batch_size=64, n_bootstrap=20)
    assert tags == ["main+A/mbr", "A", "main"]  # winner, runner-up, production: one build each
    assert rep["winner"] == "main+A__mbr_beam8" and rep["production"] == "main__beam"
    assert rep["winner_is_production"] is False
    b = rep["bootstrap"]
    for key, other in (("winner_vs_runner_up", "A__beam"), ("winner_vs_production", "main__beam")):
        assert b[key]["a"] == "main+A__mbr_beam8" and b[key]["b"] == other
        lo, hi = b[key]["ci95"]
        assert lo <= hi and 0.0 <= b[key]["p_value"] <= 1.0
        assert b[key]["n_resamples"] == 20 and b[key]["seed"] == 1234
    lat = rep["latency"]["winner"]
    assert (
        lat["n_sentences"] == 200 and lat["settings"]["includes_pool_generation_and_chrf_utility"]
    )
    assert (
        lat["settings"]["members"] == ["main", "A"] and lat["settings"]["segment_threshold"] == 192
    )
    assert lat["settings"]["batch_size"] == 64 and "gpu" in lat and lat["sentences_per_second"] > 0
    assert (
        rep["latency"]["production"]["settings"]["includes_pool_generation_and_chrf_utility"]
        is False
    )
    # E1 is decoded without segmentation, E2 at the config's T (the objective's definition)
    e1_ids = [r["id"] for r in load_split("e1")[0]]
    assert json.loads(
        (d / "report/predictions/main+A__mbr_beam8/e1_predictions.json").read_text("utf-8")
    ).keys() == set(e1_ids)
    assert (d / "report.json").is_file() and fa.report_is_valid(
        d / "report.json", d / "selection.json"
    )
    # resumable: a second run builds nothing and decodes nothing
    tags.clear()
    assert fa.run_report(d, MODELS, n_bootstrap=20) == rep and tags == []


def test_report_when_the_winner_is_the_production_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sel = _selection_for_report(winner_is_production=True)
    d = tmp_path / "eval"
    d.mkdir()
    (d / "selection.json").write_text(json.dumps(sel), encoding="utf-8")
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: _Stub())
    rep = fa.run_report(d, MODELS, n_bootstrap=10)
    assert rep["winner_is_production"] is True
    assert "note" in rep["bootstrap"]["winner_vs_production"]
    assert rep["bootstrap"]["winner_vs_runner_up"]["b"] == "A__beam"


def test_report_refuses_without_a_staged_selection(tmp_path: Path) -> None:
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ev.EvalStepError, match="staged selection"):
        fa.run_report(tmp_path, MODELS)


def test_latency_benchmark_fields_and_pool_flag() -> None:
    stub = _Stub()
    out = fa.latency_benchmark(
        stub,
        ["a b c", "d e"] * 10,
        alpha=1.4,
        beam=None,
        segment_threshold=128,
        mbr={"kind": "sample", "n": 8},
        members=["A", "B"],
        batch_size=16,
        warmup=2,
    )
    assert out["n_sentences"] == 20 and out["output_tokens"] == 50  # 2 tokens per "x y" etc.
    assert out["settings"]["includes_pool_generation_and_chrf_utility"] is True
    assert out["settings"]["decode_precision"].startswith("fp32")
    assert len(stub.calls) == 2 and stub.calls[0]["n"] == 2 and stub.calls[1]["n"] == 20
    assert stub.calls[1]["segment_threshold"] == 128 and stub.calls[1]["batch_size"] == 16


# --- upload / manifest / hf-verify ---------------------------------------------------------------


def _eval_dir(
    tmp_path: Path, members: tuple[str, ...] = ("main", "A")
) -> tuple[Path, dict[str, Path]]:
    d = tmp_path / "eval"
    d.mkdir()
    models = _fake_dirs(tmp_path)
    (d / "tuning").mkdir()
    stage2 = [f"{'+'.join(members)}__{p}" for p in fa.POOL_LABELS]
    for c in [*S1, *stage2]:
        (d / "tuning" / f"{c}.json").write_text("{}", encoding="utf-8")
    for rel in (
        "bench.json",
        "decode_summary.json",
        "run_meta.json",
        "test_predictions.json",
        "stage1.json",
        "report.json",
    ):
        (d / rel).write_text("{}", encoding="utf-8")
    (d / "validation.json").write_text(
        json.dumps({"valid": True, "n_ids": 330, "empty_strings": 0}), encoding="utf-8"
    )
    (d / "selection.json").write_text(
        json.dumps(
            {
                "candidate_order": ["x"] * 10,  # the 10 final candidates; NOT the manifest order
                "stage2_candidates": stage2,
                "winner": {"candidate": stage2[0], "members": list(members)},
            }
        ),
        encoding="utf-8",
    )
    for v in ev.VARIANTS:
        for s in ev.DECODE_SPLITS:
            p = d / "predictions" / v / f"{s}_predictions.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")
    return d, models


def test_upload_final_all_lists_stage_files_and_the_winner_members(tmp_path: Path) -> None:
    d, models = _eval_dir(tmp_path)
    api = FakeApi()
    rec = ev.upload_run(api, REPO, "final_all", d, models)
    assert rec["revision"] == REVISION and api.titles == [ev.commit_message("final_all")]
    manifest = json.loads(api.remote["runs/final_all/manifest.json"])
    assert manifest["winner_members"] == ["main", "A"]
    assert manifest["candidate_order"] == list(S1)  # the pre-registered stage-1 order
    assert len(manifest["stage2_candidates"]) == 4
    files = set(manifest["files"])
    assert {"stage1.json", "report.json", "models/main/model.safetensors"} <= files
    assert "tuning/main+A__mbr_beam8.json" in files and "tuning/main+A+B__beam.json" in files
    assert not any(f.startswith("models/B/") or f.startswith("model/") for f in files)
    assert ev.verify_run_on_hf(api, REPO, "final_all")["complete"] is True
    again = FakeApi()
    again.remote, again.titles = dict(api.remote), list(api.titles)
    ev.upload_run(again, REPO, "final_all", d, models)
    assert again.commits == []


def test_upload_final_all_needs_model_dirs_for_every_winner_member(tmp_path: Path) -> None:
    d, models = _eval_dir(tmp_path, ("main", "A", "B"))
    with pytest.raises(ev.EvalStepError, match="model dir for every winner member"):
        ev.upload_run(FakeApi(), REPO, "final_all", d, None)
    with pytest.raises(ev.EvalStepError, match="model dir for every winner member"):
        ev.upload_run(FakeApi(), REPO, "final_all", d, {"main": models["main"]})


@pytest.mark.parametrize(
    "damage",
    ["no_members", "no_stage2", "missing_stage2_file", "missing_member_file", "validation"],
)
def test_verify_final_all_not_complete_on_damage(tmp_path: Path, damage: str) -> None:
    d, models = _eval_dir(tmp_path)
    api = FakeApi()
    ev.upload_run(api, REPO, "final_all", d, models)
    key = "runs/final_all/manifest.json"
    manifest = json.loads(api.remote[key])
    if damage == "no_members":
        del manifest["winner_members"]
    elif damage == "no_stage2":
        del manifest["stage2_candidates"]
    elif damage == "missing_stage2_file":
        del manifest["files"]["tuning/main+A__mbr_beam8.json"]
    elif damage == "missing_member_file":
        del manifest["files"]["models/A/model.safetensors"]
    else:
        bad = json.dumps({"valid": True, "n_ids": 329}).encode()
        api.remote["runs/final_all/validation.json"] = bad
        manifest["files"]["validation.json"] = {
            "sha256": hashlib.sha256(bad).hexdigest(),
            "bytes": len(bad),
        }
    api.remote[key] = json.dumps(manifest).encode()
    assert ev.verify_run_on_hf(api, REPO, "final_all")["complete"] is False


def test_eval_l4_cli_accepts_final_all_and_a_custom_grid() -> None:
    p = ev._parser()
    assert p.parse_args(["hf-verify", "--repo", "o/r", "--run", "final_all"]).run == "final_all"
    cmd = ["tune", "--model", "m", "--out", "o", "--alphas", "1.2", "2.0", "--beams", "5"]
    t = p.parse_args(cmd)
    assert t.alphas == [1.2, 2.0] and t.beams == [5]
    d = p.parse_args(["tune", "--model", "m", "--out", "o"])
    assert d.alphas is None and d.beams is None  # default: nmt.tune's unchanged grid


def test_run_tune_step_passes_the_grid_only_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nmt.tune

    seen: list[dict[str, Any]] = []

    def fake(model_dir: Path, out_path: Path, **kw: Any) -> dict[str, Any]:
        seen.append(kw)
        t = _tuning(1.0)
        Path(out_path).write_text(json.dumps(t), encoding="utf-8")
        return t

    monkeypatch.setattr(nmt.tune, "run_tune", fake)
    ev.run_tune_step(tmp_path / "m", tmp_path / "a.json")
    ev.run_tune_step(tmp_path / "m", tmp_path / "b.json", alphas=fa.FINAL_ALL_ALPHAS)
    assert seen[0] == {"batch_size": ev.EVAL_BATCH_SIZE}
    assert seen[1]["alphas"] == fa.FINAL_ALL_ALPHAS and "beams" not in seen[1]


# --- estimate (staged, measured L4 rates) --------------------------------------------------------


def _workload() -> dict[str, Any]:
    return json.loads((REPO_ROOT / "colab" / "eval_workload.json").read_text("utf-8"))


def test_estimate_stage1_arithmetic_uses_the_measured_l4_rates() -> None:
    assert fa.MEASURED_L4_RATES == {"greedy": 2305.93, "beam": 1198.86}  # bench.json, HF c3d85982
    wl = _workload()
    est = fa.estimate_final_all(wl)
    e12 = wl["splits"]["e1"]["out_tokens"] + wl["splits"]["e2"]["out_tokens"]
    e2 = wl["splits"]["e2"]["out_tokens"]
    per_model = 10 * e12 + 5 * e2  # 5 alphas x 2 beams on E1+E2, T step = 5 E2 decodes
    assert est["stage1"]["per_model_tokens"] == per_model
    assert est["stage1"]["sum_members"] == 12  # 1+1+1+2+2+2+3 members over the 7 sets
    assert est["stage1"]["seconds"] == pytest.approx(12 * per_model / 1198.86)
    # a faster beam rate halves stage 1; rates are inputs, not baked in
    fast = fa.estimate_final_all(wl, {"greedy": 5000.0, "beam": 2397.72}, "test")
    assert fast["stage1"]["seconds"] == pytest.approx(est["stage1"]["seconds"] / 2)


def test_estimate_stage2_scenarios_and_comparison_with_the_exhaustive_figure() -> None:
    wl = _workload()
    est = fa.estimate_final_all(wl)
    cheap, dear = est["stage2"]["cheapest"], est["stage2"]["dearest"]
    assert cheap["sum_members"] == 2 and dear["sum_members"] == 5
    assert dear["gpu_seconds"] == pytest.approx(cheap["gpu_seconds"] * 5 / 2)
    e12 = wl["splits"]["e1"]["out_tokens"] + wl["splits"]["e2"]["out_tokens"]
    pool_tokens = e12 + 5 * wl["splits"]["e2"]["out_tokens"]
    per_model = sum(pool_tokens * n / 5 / 1198.86 for n in (8, 16, 8, 16))  # N/5 x beam-5 time
    assert cheap["gpu_seconds"] == pytest.approx(2 * per_model)
    sc = est["scenarios"]
    assert sc["cheapest"]["seconds"] < sc["dearest"]["seconds"]
    assert sc["dearest"]["hours"] < fa.PRE_STAGED_ESTIMATE_HOURS == 20.6
    for s in sc.values():
        assert s["seconds"] == pytest.approx(sum(s["parts_seconds"].values()))
        assert s["cu"] == pytest.approx(s["hours"] * 1.54)
    lines = fa.format_final_estimate(est)
    text = "\n".join(lines)
    assert lines[0].startswith("ESTIMATE") and "not a measurement" in lines[0]
    assert "beam 4 ASSUMED equal" in text and "20.6 h" in text and "stage 2 (cheapest" in text


# --- notebook support: exports in the plan, plan --json, summary -------------------------------


def test_plan_with_checkpoints_adds_the_three_exports_right_after_hf_check() -> None:
    ckpts = fa.final_all_checkpoint_paths(Path("/d/runs"), "")
    plan = fa.plan_final_all(
        eval_root=Path("/d/eval/final_all"),
        hf_repo="o/r",
        models={n: Path("/d/eval/final_all/models") / n for n in fa.MODEL_NAMES},
        python="py",
        ckpt_files=ckpts,
        repo_dir=Path("/repo"),
    )
    names = [n for n, _ in plan]
    assert names[:6] == ["hf-verify", "hf-check", "export:main", "export:A", "export:B", "bench"]
    assert len(names) == 34
    exports = dict(plan)
    for name, (run, step, cfg) in fa.FINAL_ALL_CHECKPOINTS.items():
        argv = exports[f"export:{name}"]
        assert argv[:4] == ["py", "-m", "nmt.eval_l4", "candidates"]
        assert argv[argv.index("--candidate") + 1] == f"{name}={step}"
        assert Path(argv[argv.index("--config") + 1]) == Path("/repo/configs") / f"{cfg}.yaml"
        assert Path(argv[argv.index("--ckpt-dir") + 1]) == Path("/d/runs") / run / "ckpt"
        assert Path(argv[argv.index("--out-dir") + 1]) == Path("/d/eval/final_all/models")
        assert ev._parser().parse_args(argv[3:]).cmd == "candidates"
    # without checkpoints: the 24 steps up to the upload + the 7 COMET steps
    assert len(_plan()) == 31


def test_describe_plan_marks_only_the_stage2_steps() -> None:
    lines = fa.describe_plan(_plan())
    marked = [ln for ln in lines if fa.STAGE2_NOTE in ln]
    assert len(marked) == 8 and all("[tune-stage2:" in ln for ln in marked)
    assert fa.STAGE2_NOTE == "depends on stage1-select top-2"


def test_cli_plan_json_and_summary_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = [f"{n}={tmp_path / n / 'ckpt' / 'x.pt'}" for n in fa.MODEL_NAMES]
    args = ["plan", "--json", "--eval-root", str(tmp_path / "e"), "--repo", "o/r"]
    for c in ckpt:
        args += ["--ckpt-file", c]
    assert fa.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [n for n, _ in plan][:3] == ["hf-verify", "hf-check", "export:main"]
    assert any(a == f"A={tmp_path / 'e' / 'models' / 'A'}" for n, argv in plan for a in argv)
    # summary of an empty dir: every missing piece says NOT DONE, and nothing crashes
    assert fa.main(["summary", "--eval-root", str(tmp_path / "none"), "--hf-repo", "o/r"]) == 0
    out = capsys.readouterr().out
    for needle in ("bench: NOT DONE", "stage 1: NOT DONE", "selection: NOT DONE"):
        assert needle in out
    assert "report (bootstrap + latency): NOT DONE" in out and "HF upload to o/r: NOT DONE" in out


def test_checkpoint_paths_are_the_three_final_files() -> None:
    paths = fa.final_all_checkpoint_paths(Path("/runs"), "notebook_")
    assert {n: (p.parent.parent.name, p.name) for n, p in paths.items()} == {
        "main": ("notebook_main", "step_00024645.pt"),
        "A": ("notebook_ext_branch_a_l4", "step_00037500.pt"),
        "B": ("notebook_ext_branch_b_l4", "step_00050000.pt"),
    }
