from __future__ import annotations

# nmt/final_all.py: the exhaustive PREREG rule-5 candidate set, its plan / selection / decode /
# tune steps (stub translators, no model, no GPU), the final_all upload + hf-verify manifest
# extension in nmt/eval_l4.py (fake HfApi), and the cost estimate. No real decoding is performed.
import json
import types
from pathlib import Path
from typing import Any

import pytest

import nmt.eval_l4 as ev
import nmt.final_all as fa
from tests.test_eval_l4 import REPO, REVISION, FakeApi

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZES = (1940, 1000)


@pytest.fixture(autouse=True)
def _canonical_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "_full_selection_sizes", lambda: SIZES)


# --- the candidate set ---------------------------------------------------------------------------


def test_candidate_set_is_exactly_rule_5_in_the_documented_order() -> None:
    cands = fa.final_all_candidates()
    assert len(cands) == 35 == len(set(fa.FINAL_ALL_CANDIDATES))
    assert [c.members for c in cands[::5]] == [
        ("main",),
        ("A",),
        ("B",),
        ("main", "A"),
        ("main", "B"),
        ("A", "B"),
        ("main", "A", "B"),
    ]
    assert [c.name for c in cands[:5]] == [
        "main__beam",
        "main__mbr_beam8",
        "main__mbr_beam16",
        "main__mbr_eps0.02_n8",
        "main__mbr_eps0.02_n16",
    ]
    assert cands[-1].name == "main+A+B__mbr_eps0.02_n16"
    assert [c.pool for c in cands[:5]] == [
        None,
        ("beam", 8),
        ("beam", 16),
        ("sample", 8),
        ("sample", 16),
    ]
    assert cands[1].mbr == {"kind": "beam", "n": 8, "epsilon": 0.02, "seed": 1234}
    assert cands[1].beam_name == "main__beam" and cands[-1].beam_name == "main+A+B__beam"


def test_constants_match_rules_1_and_2_and_mbr_defaults() -> None:
    from nmt import mbr

    assert fa.FINAL_ALL_ALPHAS == (1.2, 1.4, 1.6, 1.8, 2.0) and fa.FINAL_ALL_BEAMS == (1, 4, 5)
    assert fa.SAMPLING_EPSILON == mbr.DEFAULT_EPSILON and fa.SAMPLING_SEED == 1234
    assert fa.SAMPLING_SEED == mbr.DEFAULT_SAMPLING_SEED
    assert [mbr.MBRConfig(k, n).label for k, n in fa.POOL_SPECS] == [
        fa.pool_label(k, n) for k, n in fa.POOL_SPECS
    ]


def test_existing_runs_and_default_grid_are_unchanged() -> None:
    from nmt.tune import DEFAULT_ALPHAS, DEFAULT_BEAMS

    assert ev.RUNS == ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
    assert ev.ALL_RUNS[-1] == "final_all" and ev.ALL_RUNS[:-1] == ev.RUNS
    assert DEFAULT_ALPHAS == (0.6, 0.8, 1.0, 1.2) and DEFAULT_BEAMS == (1, 4, 5)
    assert ev.expected_candidates("final_all") == fa.FINAL_ALL_CANDIDATES
    assert ev.expected_candidates("main") == ev.MAIN_CANDIDATES
    assert "model/model.safetensors" not in ev.required_rels("final_all", ())
    assert "validation.json" in ev.required_rels("final_all", ())
    assert ev.final_all_model_rels(["A", "B"])[0] == "models/A/model.safetensors"


def test_grid_matches_only_the_rule_1_grid() -> None:
    grid = [{"alpha": a, "beam": b} for a in fa.FINAL_ALL_ALPHAS for b in fa.FINAL_ALL_BEAMS]
    assert fa.grid_matches({"alpha_beam": {"grid": grid}})
    old = [{"alpha": a, "beam": b} for a in (0.6, 0.8, 1.0, 1.2) for b in (1, 4, 5)]
    assert not fa.grid_matches({"alpha_beam": {"grid": old}})
    assert not fa.grid_matches({}) and not fa.grid_matches(None)


def test_parse_model_args_refuses_bad_input() -> None:
    assert fa.parse_model_args(["main=/a", "A=/b"]) == {"main": Path("/a"), "A": Path("/b")}
    for bad in (["main"], ["C=/x"], ["main=/a", "main=/b"], ["=x"]):
        with pytest.raises(ev.EvalStepError):
            fa.parse_model_args(bad)


# --- plan ----------------------------------------------------------------------------------------

MODELS = {"main": Path("/m/main"), "A": Path("/m/A"), "B": Path("/m/B")}


def _plan() -> list[tuple[str, list[str]]]:
    return fa.plan_final_all(
        eval_root=Path("/eval/final_all"), hf_repo="o/r", models=MODELS, python="py"
    )


def test_plan_order_gates_before_gpu_work_and_private_upload_last() -> None:
    names = [n for n, _ in _plan()]
    assert names[:3] == ["hf-verify", "hf-check", "bench"]
    assert names[3:38] == [f"tune:{c}" for c in fa.FINAL_ALL_CANDIDATES]
    assert names[38:] == ["select", "decode", "validate-test", "upload"]
    # an MBR candidate's tune comes after its model-set's beam tune (it reads its alpha)
    for c in fa.final_all_candidates():
        assert names.index(f"tune:{c.beam_name}") <= names.index(f"tune:{c.name}")


def test_plan_argvs_are_real_subcommands_with_the_right_run() -> None:
    plan = dict(_plan())
    assert plan["hf-verify"][-2:] == ["--run", "final_all"]
    assert plan["upload"][3] == "upload" and "--model" in plan["upload"]
    assert plan["decode"][-1] == "--test"
    # every argv parses with the CLI it names
    for name, argv in plan.items():
        parser = fa._parser() if argv[2] == "nmt.final_all" else ev._parser()
        args = parser.parse_args(argv[3:])
        assert args.cmd == argv[3], name
    tune = fa._parser().parse_args(plan["tune:A+B__mbr_beam16"][3:])
    assert tune.name == "A+B__mbr_beam16" and len(tune.model) == 3


def test_plan_module_imports_only_the_stdlib_at_top_level() -> None:
    src = (REPO_ROOT / "nmt" / "final_all.py").read_text(encoding="utf-8")
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("torch" in ln or "nmt.mbr" in ln or "nmt.ensemble" in ln for ln in top)


# --- tuning --------------------------------------------------------------------------------------


def _tuning(objective: float, **over: Any) -> dict[str, Any]:
    t: dict[str, Any] = {
        "limit_e1": None,
        "limit_e2": None,
        "n_e1": SIZES[0],
        "n_e2": SIZES[1],
        "alpha_beam": {
            "grid": [{"alpha": a, "beam": b} for a in fa.FINAL_ALL_ALPHAS for b in (1, 4, 5)]
        },
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


def _write_all_tunings(d: Path, objectives: dict[str, float] | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for c in fa.final_all_candidates():
        t = _tuning(
            (objectives or {}).get(c.name, 40.0),
            candidate=c.name,
            member_sha256={m: f"h_{m}" for m in c.members},
        )
        if c.pool is not None:
            t["alpha_beam"] = {"note": "fixed", "alpha": 1.6}
            t["winner"] = {**t["winner"], "mbr": c.mbr}
        (d / f"{c.name}.json").write_text(json.dumps(t), encoding="utf-8")


def test_select_picks_the_best_and_records_everything(tmp_path: Path) -> None:
    _write_all_tunings(tmp_path / "t", {"A+B__mbr_eps0.02_n8": 50.0})
    out = tmp_path / "selection.json"
    sel = fa.run_final_select(tmp_path / "t", out)
    on_disk = json.loads(out.read_text("utf-8"))
    assert on_disk == sel
    assert sel["candidate_order"] == list(fa.FINAL_ALL_CANDIDATES) and len(sel["candidates"]) == 35
    win = sel["winner"]
    assert win["candidate"] == "A+B__mbr_eps0.02_n8" and win["members"] == ["A", "B"]
    assert win["mbr"] == {"kind": "sample", "n": 8, "epsilon": 0.02, "seed": 1234}
    assert sel["tie_rule"] == fa.FINAL_ALL_TIE_RULE and "earliest" in sel["tie_rule"]
    assert sel["candidates"]["main+A__beam"]["members"] == ["main", "A"]
    assert fa.final_selection_is_valid(out, tmp_path / "t")


def test_select_ties_go_to_the_earlier_listed_candidate(tmp_path: Path) -> None:
    _write_all_tunings(tmp_path / "t", {"B__beam": 45.0, "main+A__beam": 45.0 + 1e-12})
    sel = fa.run_final_select(tmp_path / "t", tmp_path / "s.json")
    assert sel["winner"]["candidate"] == "B__beam"
    assert sel["tied_candidates"] == ["B__beam", "main+A__beam"]


@pytest.mark.parametrize("damage", ["missing", "smoke", "old_grid", "wrong_candidate"])
def test_select_refuses_unfit_tunings(tmp_path: Path, damage: str) -> None:
    d = tmp_path / "t"
    _write_all_tunings(d)
    victim = d / "A__beam.json"
    t = json.loads(victim.read_text("utf-8"))
    if damage == "missing":
        victim.unlink()
    elif damage == "smoke":
        t["limit_e1"] = 10
    elif damage == "old_grid":
        t["alpha_beam"]["grid"] = [{"alpha": 0.6, "beam": 5}]
    else:
        v2 = d / "A__mbr_beam8.json"
        t2 = json.loads(v2.read_text("utf-8"))
        t2["candidate"] = "B__mbr_beam8"
        v2.write_text(json.dumps(t2), encoding="utf-8")
    if damage != "missing":
        victim.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(ev.EvalStepError, match="select"):
        fa.run_final_select(d, tmp_path / "s.json")
    assert not (tmp_path / "s.json").exists()


def test_selection_is_stale_when_a_tuning_file_is_newer(tmp_path: Path) -> None:
    import os

    _write_all_tunings(tmp_path / "t")
    out = tmp_path / "s.json"
    fa.run_final_select(tmp_path / "t", out)
    newer = tmp_path / "t" / "B__beam.json"
    os.utime(newer, (out.stat().st_mtime + 10, out.stat().st_mtime + 10))
    assert not fa.final_selection_is_valid(out, tmp_path / "t")


class _Stub:
    """Translator stand-in: upper-cases sources, records its calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stats = types.SimpleNamespace(n_beam=0, n_greedy_fallback=0, n_copy_fallback=0)

    def translate(self, texts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append({"n": len(texts), **kwargs})
        self.stats.n_beam += len(texts)
        return [t.upper() or "x" for t in texts]


def test_pool_translator_forwards_mbr_and_drops_beam() -> None:
    inner = _Stub()
    pt = fa.PoolTranslator(inner, "CFG")
    out = pt.translate(["a b"], batch_size=4, beam=7, alpha=1.4, segment_threshold=None)
    assert out == ["A B"] and pt.stats is inner.stats
    assert inner.calls[0]["mbr"] == "CFG" and "beam" not in inner.calls[0]
    assert inner.calls[0]["alpha"] == 1.4


def _fake_dirs(tmp_path: Path) -> dict[str, Path]:
    models = {}
    for m in fa.MODEL_NAMES:
        d = tmp_path / "models" / m
        d.mkdir(parents=True)
        for f in ("model.safetensors", "config.json", "spm.model"):
            (d / f).write_bytes(f"{m}-{f}".encode())
        models[m] = d
    return models


def test_tune_beam_candidate_uses_the_rule_1_grid_and_records_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nmt.tune

    models, seen = _fake_dirs(tmp_path), {}
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: _Stub())

    def fake_run_tune(model_dir: Path, out_path: Path, **kw: Any) -> dict[str, Any]:
        seen.update(kw, model_dir=model_dir)
        t = _tuning(41.0, **kw["extra"])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(t), encoding="utf-8")
        return t

    monkeypatch.setattr(nmt.tune, "run_tune", fake_run_tune)
    tdir = tmp_path / "tuning"
    assert fa.tune_candidate("main+A__beam", models, tdir) is True
    assert seen["alphas"] == fa.FINAL_ALL_ALPHAS and seen["beams"] == fa.FINAL_ALL_BEAMS
    assert seen["extra"]["members"] == ["main", "A"] and set(seen["extra"]["member_sha256"]) == {
        "main",
        "A",
    }
    assert fa.tune_candidate("main+A__beam", models, tdir) is False  # valid file: skipped


def test_tune_refuses_a_tuning_made_on_the_old_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nmt.tune

    models = _fake_dirs(tmp_path)
    tdir = tmp_path / "tuning"
    tdir.mkdir()
    old = _tuning(41.0, alpha_beam={"grid": [{"alpha": 0.6, "beam": 5}]}, candidate="main__beam")
    (tdir / "main__beam.json").write_text(json.dumps(old), encoding="utf-8")
    ran: list[bool] = []

    def fake_run_tune(model_dir: Path, out_path: Path, **kw: Any) -> dict[str, Any]:
        ran.append(True)
        t = _tuning(42.0)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(t), encoding="utf-8")
        return t

    monkeypatch.setattr(nmt.tune, "run_tune", fake_run_tune)
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: _Stub())
    assert fa.tune_candidate("main__beam", models, tdir) is True and ran == [True]


def test_tune_pool_candidate_needs_its_beam_tuning_and_fixes_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    tdir = tmp_path / "tuning"
    stub = _Stub()
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: stub)
    with pytest.raises(ev.EvalStepError, match="needs a valid full rule-1 tuning"):
        fa.tune_candidate("A__mbr_beam8", models, tdir)
    tdir.mkdir()
    base = _tuning(41.0, candidate="A__beam", member_sha256=fa._member_sha256(models, ["A"]))
    base["winner"]["alpha"] = 1.8
    (tdir / "A__beam.json").write_text(json.dumps(base), encoding="utf-8")
    # the real data sets and scorer, a stub translator, no model
    monkeypatch.setattr(ev, "_full_selection_sizes", lambda: (1940, 1000))
    assert fa.tune_candidate("A__mbr_beam8", models, tdir, batch_size=64) is True
    out = json.loads((tdir / "A__mbr_beam8.json").read_text("utf-8"))
    assert out["winner"]["alpha"] == 1.8 and out["winner"]["beam"] == 8
    assert out["winner"]["mbr"]["kind"] == "beam" and out["members"] == ["A"]
    assert out["n_e1"] == 1940 and out["n_e2"] == 1000 and ev.tuning_is_full(out)
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


# --- upload / manifest / hf-verify ---------------------------------------------------------------


def _eval_dir(
    tmp_path: Path, members: tuple[str, ...] = ("main", "A")
) -> tuple[Path, dict[str, Path]]:
    d = tmp_path / "eval"
    d.mkdir()
    models = _fake_dirs(tmp_path)
    (d / "tuning").mkdir()
    for c in fa.FINAL_ALL_CANDIDATES:
        (d / "tuning" / f"{c}.json").write_text("{}", encoding="utf-8")
    for rel in ("bench.json", "decode_summary.json", "run_meta.json", "test_predictions.json"):
        (d / rel).write_text("{}", encoding="utf-8")
    (d / "validation.json").write_text(
        json.dumps({"valid": True, "n_ids": 330, "empty_strings": 0}), encoding="utf-8"
    )
    winner = fa.candidate_name(members, None)
    (d / "selection.json").write_text(
        json.dumps(
            {
                "candidate_order": list(fa.FINAL_ALL_CANDIDATES),
                "winner": {"candidate": winner, "members": list(members)},
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


def test_upload_final_all_lists_the_winner_members_and_verifies(tmp_path: Path) -> None:
    d, models = _eval_dir(tmp_path)
    api = FakeApi()
    rec = ev.upload_run(api, REPO, "final_all", d, models)
    assert rec["revision"] == REVISION and api.titles == [ev.commit_message("final_all")]
    manifest = json.loads(api.remote["runs/final_all/manifest.json"])
    assert manifest["winner_members"] == ["main", "A"]
    assert manifest["candidate_order"] == list(fa.FINAL_ALL_CANDIDATES)
    files = set(manifest["files"])
    assert "models/main/model.safetensors" in files and "models/A/spm.model" in files
    assert not any(f.startswith("models/B/") or f.startswith("model/") for f in files)
    assert ev.verify_run_on_hf(api, REPO, "final_all")["complete"] is True
    # a second upload is skipped on HF evidence
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
    assert not FakeApi().commits


@pytest.mark.parametrize("damage", ["no_members", "missing_member_file", "bad_validation"])
def test_verify_final_all_not_complete_on_damage(tmp_path: Path, damage: str) -> None:
    d, models = _eval_dir(tmp_path)
    api = FakeApi()
    ev.upload_run(api, REPO, "final_all", d, models)
    key = "runs/final_all/manifest.json"
    manifest = json.loads(api.remote[key])
    if damage == "no_members":
        del manifest["winner_members"]
        api.remote[key] = json.dumps(manifest).encode()
    elif damage == "missing_member_file":
        del manifest["files"]["models/A/model.safetensors"]
        api.remote[key] = json.dumps(manifest).encode()
    else:
        bad = json.dumps({"valid": True, "n_ids": 329}).encode()
        api.remote["runs/final_all/validation.json"] = bad
        import hashlib

        manifest["files"]["validation.json"] = {
            "sha256": hashlib.sha256(bad).hexdigest(),
            "bytes": len(bad),
        }
        api.remote[key] = json.dumps(manifest).encode()
    assert ev.verify_run_on_hf(api, REPO, "final_all")["complete"] is False


def test_eval_l4_cli_accepts_final_all_and_a_custom_grid() -> None:
    p = ev._parser()
    assert p.parse_args(["hf-verify", "--repo", "o/r", "--run", "final_all"]).run == "final_all"
    t = p.parse_args(
        ["tune", "--model", "m", "--out", "o", "--alphas", "1.2", "2.0", "--beams", "5"]
    )
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


# --- estimate ------------------------------------------------------------------------------------


def test_estimate_scales_with_members_and_is_labelled_an_estimate() -> None:
    workload = json.loads((REPO_ROOT / "colab" / "eval_workload.json").read_text("utf-8"))
    est = fa.estimate_final_all(workload)
    per = est["per_model_set_seconds"]
    assert per["main+A+B"]["beam"] == pytest.approx(3 * per["main"]["beam"])
    assert per["main+A"]["mbr_beam16"] == pytest.approx(2 * per["main"]["mbr_beam16"])
    assert per["main"]["mbr_beam16"] == pytest.approx(2 * per["main"]["mbr_beam8"])
    assert est["seconds"] == pytest.approx(sum(est["parts_seconds"].values()))
    assert est["n_candidates"] == 35
    fast = fa.estimate_final_all(workload, {"greedy": 5000.0, "beam": 2000.0}, "test")
    assert fast["parts_seconds"]["tuning_gpu"] == pytest.approx(
        est["parts_seconds"]["tuning_gpu"] / 2
    )
    lines = fa.format_final_estimate(est)
    assert lines[0].startswith("ESTIMATE (ASSUMED rates") and "not a measurement" in lines[0]


# --- stale tunings on resume ---------------------------------------------------------------------


def _fake_beam_tune(monkeypatch: pytest.MonkeyPatch, models: dict[str, Path]) -> list[str]:
    """Patch nmt.tune.run_tune to record calls and write a fresh, correct beam tuning."""
    import nmt.tune

    ran: list[str] = []

    def fake(model_dir: Path, out_path: Path, **kw: Any) -> dict[str, Any]:
        ran.append(Path(out_path).stem)
        t = _tuning(42.0, **kw["extra"])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(t), encoding="utf-8")
        return t

    monkeypatch.setattr(nmt.tune, "run_tune", fake)
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: _Stub())
    return ran


def test_beam_tuning_made_by_other_weights_is_redone_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    ran = _fake_beam_tune(monkeypatch, models)
    tdir = tmp_path / "tuning"
    assert fa.tune_candidate("A__beam", models, tdir) is True
    assert fa.tune_candidate("A__beam", models, tdir) is False  # same weights: skipped
    (models["A"] / "model.safetensors").write_bytes(b"a retrained model")
    assert fa.tune_candidate("A__beam", models, tdir) is True  # stale: redone
    assert ran == ["A__beam", "A__beam"]
    # a tuning file that records no hashes at all cannot be shown current
    t = json.loads((tdir / "A__beam.json").read_text("utf-8"))
    del t["member_sha256"]
    (tdir / "A__beam.json").write_text(json.dumps(t), encoding="utf-8")
    assert fa.tune_candidate("A__beam", models, tdir) is True


def test_ensemble_tuning_is_stale_when_any_member_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    ran = _fake_beam_tune(monkeypatch, models)
    tdir = tmp_path / "tuning"
    fa.tune_candidate("main+B__beam", models, tdir)
    (models["B"] / "model.safetensors").write_bytes(b"other B")
    assert fa.tune_candidate("main+B__beam", models, tdir) is True and len(ran) == 2
    (models["A"] / "model.safetensors").write_bytes(b"other A")  # not a member: still current
    assert fa.tune_candidate("main+B__beam", models, tdir) is False


def test_pool_tuning_is_stale_when_the_beam_winner_alpha_or_weights_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _fake_dirs(tmp_path)
    stub = _Stub()
    monkeypatch.setattr(fa, "build_translator", lambda *a, **k: stub)
    monkeypatch.setattr(ev, "_full_selection_sizes", lambda: (1940, 1000))
    tdir = tmp_path / "tuning"
    tdir.mkdir()
    hashes = fa._member_sha256(models, ["A"])

    def write_base(alpha: float, sha: dict[str, str]) -> None:
        base = _tuning(41.0, candidate="A__beam", member_sha256=sha)
        base["winner"]["alpha"] = alpha
        (tdir / "A__beam.json").write_text(json.dumps(base), encoding="utf-8")

    write_base(1.6, hashes)
    assert fa.tune_candidate("A__mbr_beam8", models, tdir, batch_size=64) is True
    assert fa.tune_candidate("A__mbr_beam8", models, tdir, batch_size=64) is False
    write_base(1.2, hashes)  # the beam winner's alpha moved: the pool file used 1.6
    assert fa.tune_candidate("A__mbr_beam8", models, tdir, batch_size=64) is True
    out = json.loads((tdir / "A__mbr_beam8.json").read_text("utf-8"))
    assert out["alpha_beam"]["alpha"] == 1.2 and out["member_sha256"] == hashes
    write_base(1.2, {"A": "stale"})  # the beam file was made by other weights
    with pytest.raises(ev.EvalStepError, match="current model weights"):
        fa.tune_candidate("A__mbr_beam8", models, tdir, batch_size=64)


def test_select_refuses_a_mix_of_stale_and_fresh_tunings(tmp_path: Path) -> None:
    d = tmp_path / "t"
    _write_all_tunings(d)
    f = d / "main+A__beam.json"
    t = json.loads(f.read_text("utf-8"))
    t["member_sha256"]["A"] = "h_other"  # A's weights differ from what A__beam recorded
    f.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(ev.EvalStepError, match="different weights"):
        fa.run_final_select(d, tmp_path / "s.json")
    _write_all_tunings(d)
    g = d / "B__mbr_beam16.json"
    t = json.loads(g.read_text("utf-8"))
    t["alpha_beam"]["alpha"] = 1.2  # base winner alpha is 1.6
    g.write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(ev.EvalStepError, match="different alpha"):
        fa.run_final_select(d, tmp_path / "s.json")
    assert not (tmp_path / "s.json").exists()
