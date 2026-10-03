from __future__ import annotations

# The stage-1 INTERIM result of the final_all session (nmt/stage1_interim.py): the stage-1 winner's
# test predictions are decoded, validated and uploaded privately BEFORE stage 2. Offline: a stub
# translator, a fake HfApi, the real data/test inputs and sample submission (330 ids).
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import nmt.eval_l4 as ev
import nmt.final_all as fa
import nmt.stage1_interim as si
from tests.test_eval_l4 import READ, REPO, REVISION, FakeApi, _DecodeStub
from tests.test_final_all import MODELS, _eval_dir

CFG = {"alpha": 1.4, "beam": 5, "segment_threshold": 96}


def stage1_dict(top: str = "A__beam", cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cands = {
        n: {"members": list(fa.candidate_by_name(n).members), "config": dict(cfg or CFG)}
        for n in fa.STAGE1_CANDIDATES
    }
    cands["main__beam"]["config"] = {"alpha": 1.2, "beam": 4, "segment_threshold": None}
    return {"top2": [top, "main__beam"], "candidates": cands}


@pytest.fixture
def stage1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """stage1.json on disk; _require_stage1 (which re-derives it from 7 tuning files) is stubbed."""
    data = stage1_dict()
    ev._write_json(tmp_path / "stage1.json", data)
    monkeypatch.setattr(fa, "_require_stage1", lambda *_a, **_k: data)
    return data


def decode(tmp_path: Path, stub: _DecodeStub, **kw: Any) -> dict[str, Any]:
    built: list[Any] = []

    def build(models: Any, members: Any, device: Any = None) -> _DecodeStub:
        built.append(list(members))
        return stub

    out = si.decode_stage1_test(
        tmp_path / "eval", MODELS, tmp_path / "stage1.json", tmp_path / "tuning", build=build, **kw
    )
    stub.built = built  # type: ignore[attr-defined]
    return out


def make_valid(tmp_path: Path) -> Path:
    ev.validate_test_predictions(
        si.interim_dir(tmp_path / "eval") / "test_predictions.json",
        si.interim_dir(tmp_path / "eval") / "validation.json",
    )
    return si.interim_dir(tmp_path / "eval")


# --- decode ---------------------------------------------------------------------------------------


def test_decodes_only_the_test_set_with_the_stage1_winner_at_its_stage1_config(
    tmp_path: Path, stage1: dict[str, Any]
) -> None:
    stub = _DecodeStub()
    meta = decode(tmp_path, stub)
    assert stub.built == [["A"]]  # rank 1 model set, plain (no MBR)
    assert [c[1:] for c in stub.calls] == [(5, 1.4, 96)]  # its stage-1 beam, alpha, T
    assert stub.calls[0][0] == ev.EXPECTED_TEST_IDS  # only the 330 test sentences
    out = si.interim_dir(tmp_path / "eval") / "test_predictions.json"
    assert len(json.loads(out.read_text("utf-8"))) == 330
    assert meta["candidate"] == "A__beam" and meta["done"] is True
    assert not (tmp_path / "eval" / "predictions").exists()  # no other split was decoded
    assert stub.calls[0][0] == 330


def test_an_ensemble_winner_is_decoded_with_all_its_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = stage1_dict("main+A+B__beam")
    ev._write_json(tmp_path / "stage1.json", data)
    monkeypatch.setattr(fa, "_require_stage1", lambda *_a, **_k: data)
    stub = _DecodeStub()
    decode(tmp_path, stub)
    assert stub.built == [["main", "A", "B"]]


def test_decode_is_resumable_and_never_reuses_another_winner_or_config(
    tmp_path: Path, stage1: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    decode(tmp_path, _DecodeStub())
    again = _DecodeStub()
    decode(tmp_path, again)
    assert again.calls == [] and "SKIP" in capsys.readouterr().out
    # a different stage-1 config (e.g. stage 1 was re-run) discards the old predictions
    other = stage1_dict(cfg={"alpha": 1.8, "beam": 4, "segment_threshold": None})
    monkeypatch.setattr(fa, "_require_stage1", lambda *_a, **_k: other)
    redo = _DecodeStub()
    decode(tmp_path, redo)
    assert [c[1:] for c in redo.calls] == [(4, 1.8, None)]
    assert "discarding" in capsys.readouterr().out
    # a damaged prediction file is redone
    out = si.interim_dir(tmp_path / "eval") / "test_predictions.json"
    out.write_text("{}", encoding="utf-8")
    fix = _DecodeStub()
    decode(tmp_path, fix)
    assert len(fix.calls) == 1


def test_decode_refuses_a_stale_stage1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stale(*_a: Any, **_k: Any) -> Any:
        raise ev.EvalStepError("stage1.json is missing or stale")

    monkeypatch.setattr(fa, "_require_stage1", stale)
    with pytest.raises(ev.EvalStepError, match="stale"):
        decode(tmp_path, _DecodeStub())


# --- validation -----------------------------------------------------------------------------------


def test_validation_accepts_exactly_the_330_ids_and_fails_loudly_otherwise(
    tmp_path: Path, stage1: dict[str, Any]
) -> None:
    decode(tmp_path, _DecodeStub())
    d = make_valid(tmp_path)
    assert json.loads((d / "validation.json").read_text("utf-8"))["valid"] is True
    pred = d / "test_predictions.json"
    data = json.loads(pred.read_text("utf-8"))
    first = next(iter(data))
    data[first] = "   "  # one empty string
    pred.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ev.EvalStepError, match="INVALID"):
        make_valid(tmp_path)
    assert json.loads((d / "validation.json").read_text("utf-8"))["valid"] is False


def test_the_plan_step_runs_the_existing_validate_test_on_the_interim_file() -> None:
    plan = dict(fa.plan_final_all(eval_root=Path("/e"), hf_repo="o/r", models=MODELS, python="py"))
    argv = plan["validate-stage1-test"]
    assert argv[:4] == ["py", "-m", "nmt.eval_l4", "validate-test"]
    assert Path(argv[argv.index("--pred") + 1]) == Path("/e/stage1_interim/test_predictions.json")
    assert ev._parser().parse_args(argv[3:]).cmd == "validate-test"
    assert fa._parser().parse_args(plan["decode-stage1-test"][3:]).cmd == "decode-stage1-test"
    assert fa._parser().parse_args(plan["upload-stage1"][3:]).cmd == "upload-stage1"


# --- upload ---------------------------------------------------------------------------------------


def ready(tmp_path: Path, stage1: dict[str, Any]) -> tuple[Path, FakeApi]:
    decode(tmp_path, _DecodeStub())
    make_valid(tmp_path)
    return tmp_path / "eval", FakeApi()


def up(api: FakeApi, tmp_path: Path) -> dict[str, Any]:
    return si.upload_stage1(api, REPO, tmp_path / "eval", tmp_path / "stage1.json", tmp_path / "t")


def test_upload_is_a_private_commit_under_runs_final_all_stage1_with_a_manifest(
    tmp_path: Path, stage1: dict[str, Any]
) -> None:
    _root, api = ready(tmp_path, stage1)
    rec = up(api, tmp_path)
    assert api.titles == [si.COMMIT_MESSAGE] and rec["revision"] == REVISION
    assert set(api.commits[0]) == {
        "runs/final_all_stage1/test_predictions.json",
        "runs/final_all_stage1/validation.json",
        "runs/final_all_stage1/stage1.json",
        "runs/final_all_stage1/manifest.json",
    }
    manifest = json.loads(api.remote["runs/final_all_stage1/manifest.json"])
    assert manifest["winner"] == "A__beam" and manifest["winner_members"] == ["A"]
    assert manifest["winner_config"] == CFG
    for rel, entry in manifest["files"].items():
        data = api.remote[f"runs/final_all_stage1/{rel}"]
        assert entry == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    saved = json.loads((tmp_path / "eval" / si.UPLOAD_RECORD).read_text("utf-8"))
    assert saved["private"] is True and saved["revision"] == REVISION
    assert api.created and all(c["private"] is True for c in api.created)  # created private


def test_upload_prints_the_revision_and_is_skipped_on_hf_evidence(
    tmp_path: Path, stage1: dict[str, Any], capsys: Any
) -> None:
    _root, api = ready(tmp_path, stage1)
    up(api, tmp_path)
    assert f"HF_STAGE1_REVISION={REVISION}" in capsys.readouterr().out
    up(api, tmp_path)
    assert len(api.commits) == 1 and "SKIP" in capsys.readouterr().out
    # a local-only record never skips: a fresh repo gets the commit
    fresh = FakeApi()
    up(fresh, tmp_path)
    assert len(fresh.commits) == 1


def test_privacy_is_checked_before_and_after_and_a_public_repo_is_refused(
    tmp_path: Path, stage1: dict[str, Any]
) -> None:
    _root, api = ready(tmp_path, stage1)
    api.private = False
    with pytest.raises(ev.HFNotPrivateError):
        up(api, tmp_path)
    assert api.commits == []  # refused BEFORE any commit
    flip = FakeApi(flip_after_commit=True)
    with pytest.raises(ev.HFNotPrivateError, match="after the interim upload"):
        up(flip, tmp_path)


def test_a_read_only_token_and_a_denied_commit_are_write_token_errors(
    tmp_path: Path, stage1: dict[str, Any]
) -> None:
    _root, api = ready(tmp_path, stage1)
    api.who = READ
    with pytest.raises(ev.HFWriteTokenError):
        up(api, tmp_path)
    assert api.commits == []


def test_upload_refuses_unvalidated_stale_or_foreign_predictions(
    tmp_path: Path, stage1: dict[str, Any]
) -> None:
    root, api = ready(tmp_path, stage1)
    d = si.interim_dir(root)
    pred = d / "test_predictions.json"
    data = json.loads(pred.read_text("utf-8"))
    data[next(iter(data))] = "changed after validation"
    pred.write_text(json.dumps(data), encoding="utf-8")  # validation.json no longer matches
    with pytest.raises(ev.EvalStepError, match="does not validate"):
        up(api, tmp_path)
    (d / "validation.json").unlink()
    with pytest.raises(ev.EvalStepError, match="does not validate"):
        up(api, tmp_path)
    ev._write_json(d / "decode_meta.json", {"candidate": "B__beam", "config": CFG})
    with pytest.raises(ev.EvalStepError, match="not the stage-1 winner"):
        up(api, tmp_path)
    assert api.commits == []


def test_the_main_final_all_upload_is_unaffected_by_the_interim_files(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    d, models = _eval_dir(tmp_path / "x")
    api = FakeApi()
    ev.upload_run(api, REPO, "final_all", d, models)
    api.remote["runs/final_all_stage1/manifest.json"] = b"{}"
    api.remote["runs/final_all_stage1/test_predictions.json"] = b"{}"
    assert ev.verify_run_on_hf(api, REPO, "final_all")["complete"] is True


# --- estimate, summary, notebook text -------------------------------------------------------------


def test_estimate_includes_the_small_interim_decode_and_upload() -> None:
    workload = json.loads((ev.REPO_ROOT / "colab" / "eval_workload.json").read_text("utf-8"))
    est = fa.estimate_final_all(workload)
    for label, sc in est["scenarios"].items():
        parts = sc["parts_seconds"]
        members = 3 if label == "dearest" else 1
        want = members * workload["splits"]["test"]["out_tokens"] / est["rates"]["beam"]
        assert parts["stage1_interim_decode"] == pytest.approx(want)
        assert parts["stage1_interim_upload"] == 60.0
        assert parts["stage1_interim_decode"] < 120  # 330 sentences: seconds, not hours
        assert sc["seconds"] == pytest.approx(sum(parts.values()))


def test_summary_lines_show_the_interim_revision_or_not_done(
    tmp_path: Path, stage1: dict[str, Any]
) -> None:
    pre = {"preflight_gpu": "L4", "preflight_precision": "bf16"}
    assert "stage-1 interim result: NOT DONE" in fa.format_final_all_summary(
        tmp_path / "none", "sha", pre, "o/r"
    )
    root, api = ready(tmp_path, stage1)
    up(api, tmp_path)
    text = "\n".join(fa.format_final_all_summary(root, "sha", pre, "o/r"))
    assert "stage-1 interim winner: A__beam" in text and "validation: OK" in text
    assert f"HF_STAGE1_REVISION={REVISION}" in text and "runs/final_all_stage1/" in text


def test_dry_run_plan_marks_the_interim_steps_as_a_safety_net_before_stage_2() -> None:
    lines = fa.describe_plan(
        fa.plan_final_all(eval_root=Path("/e"), hf_repo="o/r", models=MODELS, python="py")
    )
    marked = [ln for ln in lines if fa.INTERIM_NOTE in ln]
    assert [ln.split("]")[0].strip(" [") for ln in marked] == list(fa.INTERIM_STEPS)
    first_stage2 = next(i for i, ln in enumerate(lines) if "[tune-stage2:" in ln)
    assert all(lines.index(ln) < first_stage2 for ln in marked)
