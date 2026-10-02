from __future__ import annotations

# Tests for nmt/eval_l4.py, the Colab-side steps of CONFIG = "eval_l4": candidate export, bench,
# selection (winner + tie rule), decoding both variants with skip-if-valid resume, test-prediction
# validation, and the private-HF upload guards (read-only token, public repo, privacy flipping
# after upload) against a FAKE HfApi. No network, no GPU, no real decoding: translators are stubs
# and the real eval data under data/ supplies the ids.
import hashlib
import json
import re
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest
import torch

import nmt.eval_l4 as ev
from nmt.model.transformer import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "test" / "sample_submission.json"
SIZES = (1940, 1000)  # |E1|, |E2|
REVISION = "a" * 40


# --- candidates -----------------------------------------------------------------------------------


def test_parse_candidate_specs_keeps_order_and_rejects_bad_input() -> None:
    specs = ev.parse_candidate_specs(["final=24645", "avg=1,2,3"])
    assert list(specs) == ["final", "avg"] and specs["avg"] == [1, 2, 3]
    for bad in ("final", "=1", "final=", "x=1,2;x=2"):
        with pytest.raises((ev.EvalStepError, ValueError)):
            ev.parse_candidate_specs([bad])
    with pytest.raises(ev.EvalStepError, match="duplicate"):
        ev.parse_candidate_specs(["a=1", "a=2"])


def test_missing_checkpoints_are_all_named_and_nothing_is_substituted(tmp_path: Path) -> None:
    (tmp_path / "step_00000002.pt").write_bytes(b"x")
    specs = {"c1": [1, 2], "c2": [2, 3]}
    with pytest.raises(ev.EvalStepError) as err:
        ev.check_candidate_files(tmp_path, specs)
    msg = str(err.value)
    assert "step_00000001.pt" in msg and "step_00000003.pt" in msg and "step_00000002.pt" not in msg
    (tmp_path / "step_00000001.pt").write_bytes(b"x")
    (tmp_path / "step_00000003.pt").write_bytes(b"x")
    ev.check_candidate_files(tmp_path, specs)


def _tiny_config_yaml(tmp_path: Path) -> Path:
    # smoke.yaml's model section is the tiny one (2.7M params); reuse it verbatim.
    path = tmp_path / "tiny.yaml"
    path.write_bytes((REPO_ROOT / "configs" / "smoke.yaml").read_bytes())
    return path


def _write_ckpt(ckpt_dir: Path, step: int, seed: int) -> dict[str, torch.Tensor]:
    from nmt.train import _build_model_config, load_config

    torch.manual_seed(seed)
    cfg = _build_model_config(load_config(REPO_ROOT / "configs" / "smoke.yaml"))
    state = Transformer(cfg).state_dict()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": state}, ckpt_dir / f"step_{step:08d}.pt")
    return state


def test_build_candidate_averages_exactly_the_listed_steps_and_resumes(tmp_path: Path) -> None:
    ckpts = tmp_path / "ckpt"
    states = {s: _write_ckpt(ckpts, s, seed=s) for s in (10, 20, 30)}
    cfg = _tiny_config_yaml(tmp_path)
    out = tmp_path / "cands" / "avg"
    assert ev.build_candidate("avg", [10, 20], ckpts, cfg, out) is True
    assert ev.candidate_is_valid(out, [10, 20]) and not ev.candidate_is_valid(out, [10, 30])
    meta = json.loads((out / "candidate_meta.json").read_text(encoding="utf-8"))
    assert meta["ckpt_files"] == ["step_00000010.pt", "step_00000020.pt"]

    from safetensors.torch import load_file

    saved = load_file(out / "model.safetensors")
    key = next(k for k in states[10] if states[10][k].is_floating_point())
    expected = (states[10][key].float() + states[20][key].float()) / 2
    got = saved[f"transformer.{key}"]  # NMTModel wraps the Transformer under `transformer`
    assert torch.allclose(got, expected, atol=1e-6)
    # a valid export is skipped (no checkpoint is even opened the second time)
    (ckpts / "step_00000010.pt").unlink()
    assert ev.build_candidate("avg", [10, 20], ckpts, cfg, out) is False


def test_build_candidate_fails_on_a_missing_file_without_writing_an_export(
    tmp_path: Path,
) -> None:
    _write_ckpt(tmp_path / "ckpt", 10, seed=1)
    with pytest.raises(ev.EvalStepError, match="step_00000020.pt"):
        ev.build_candidate(
            "avg", [10, 20], tmp_path / "ckpt", _tiny_config_yaml(tmp_path), tmp_path / "out"
        )
    assert not (tmp_path / "out" / "candidate_meta.json").exists()


# --- bench ----------------------------------------------------------------------------------------


class _StubSp:
    def encode(self, text: str, out_type: Any = int) -> list[int]:
        return list(range(len(text.split())))


class _StubTranslator:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.sp = _StubSp()
        self.calls: list[tuple[int, int | None]] = []

    def translate(
        self,
        texts: list[str],
        batch_size: int = 32,
        beam: int = 5,
        alpha: float = 0.6,
        segment_threshold: int | None = None,
    ) -> list[str]:
        self.calls.append((beam, segment_threshold))
        return [f"{t} out" for t in texts]


def test_bench_times_greedy_and_beam5_without_segmentation_after_a_warmup() -> None:
    stub = _StubTranslator()
    texts = ["un deux trois", "quatre cinq"]
    result = ev.bench_translator(stub, texts, batch_size=2, warmup=1)
    assert stub.calls == [(1, None), (1, None), (5, None)]  # warm-up, greedy, beam 5
    assert result["n_sentences"] == 2 and result["source_tokens"] == 5
    assert set(result["modes"]) == {"greedy", "beam5"}
    assert result["modes"]["beam5"]["alpha"] == 0.6 and result["modes"]["greedy"]["alpha"] is None
    assert result["modes"]["greedy"]["output_tokens"] == 7  # 4 + 3 words after " out"
    assert result["modes"]["greedy"]["output_tokens_per_second"] > 0


def test_bench_works_with_the_real_translator_attributes(tmp_path: Path) -> None:
    from nmt.hub import export_checkpoint
    from nmt.train import _build_model_config, load_config
    from nmt.translate import Translator

    ckpts = tmp_path / "ckpt"
    _write_ckpt(ckpts, 1, seed=3)
    cfg = _build_model_config(load_config(REPO_ROOT / "configs" / "smoke.yaml"))
    export_checkpoint(
        [ckpts / "step_00000001.pt"], tmp_path / "m", cfg, REPO_ROOT / "tokenizer" / "spm.model"
    )
    translator = Translator.from_pretrained(str(tmp_path / "m"), device="cpu")
    result = ev.bench_translator(translator, ["Bonjour le monde."], batch_size=1, warmup=1)
    assert result["modes"]["beam5"]["wall_seconds"] > 0
    assert result["source_tokens"] > 0


def test_bench_validity_requires_200_sentences_and_both_rates(tmp_path: Path) -> None:
    path = tmp_path / "bench.json"
    assert not ev.bench_is_valid(path)
    good = {
        "n_sentences": 200,
        "modes": {m: {"output_tokens_per_second": 10.0} for m in ("greedy", "beam5")},
    }
    path.write_text(json.dumps(good), encoding="utf-8")
    assert ev.bench_is_valid(path)
    good["n_sentences"] = 8
    path.write_text(json.dumps(good), encoding="utf-8")
    assert not ev.bench_is_valid(path)
    path.write_text("{broken", encoding="utf-8")
    assert not ev.bench_is_valid(path)


# --- tuning validity + selection ------------------------------------------------------------------


def _tuning(objective: float, threshold: int | None = None, **over: Any) -> dict[str, Any]:
    key = "no_segmentation" if threshold is None else f"T={threshold}"
    t: dict[str, Any] = {
        "limit_e1": None,
        "limit_e2": None,
        "n_e1": SIZES[0],
        "n_e2": SIZES[1],
        "model_sha256": "s" * 8,
        "winner": {"alpha": 0.8, "beam": 5, "segment_threshold": threshold},
        "segmentation": {
            "best": key,
            "scores": {
                key: {
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


@pytest.fixture(autouse=True)
def _canonical_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "_full_selection_sizes", lambda: SIZES)


def test_the_canonical_sizes_used_by_the_tests_are_the_real_ones() -> None:
    from nmt.selection import load_selection_set

    assert (len(load_selection_set("e1")), len(load_selection_set("e2"))) == SIZES


def test_tuning_is_full_refuses_limited_or_incomplete_runs() -> None:
    assert ev.tuning_is_full(_tuning(30.0), SIZES)
    assert not ev.tuning_is_full(_tuning(30.0, limit_e1=100), SIZES)
    assert not ev.tuning_is_full(_tuning(30.0, n_e2=200), SIZES)
    assert not ev.tuning_is_full(_tuning(30.0, winner={"alpha": 1}), SIZES)
    assert not ev.tuning_is_full(None, SIZES)


def test_candidate_objective_reads_a_real_nmt_tune_report() -> None:
    # reports/smoke/selection_grid.json is a real `nmt.tune.run_tune` output (limit 200 smoke run):
    # the schema this module consumes (segmentation.best / .scores, winner) is nmt.tune's own.
    real = json.loads((REPO_ROOT / "reports" / "smoke" / "selection_grid.json").read_text("utf-8"))
    got = ev.candidate_objective(real)
    seg = real["segmentation"]
    assert got["objective"] == seg["scores"][seg["best"]]["objective"]
    assert got["config"] == real["winner"] and got["n_e1"] == real["n_e1"]
    assert not ev.tuning_is_full(real, SIZES)  # a limit-200 smoke tuning can never be selected


def test_select_winner_picks_the_highest_objective_and_reports_every_candidate() -> None:
    out = ev.select_winner(
        {"final": _tuning(30.0), "avg_last5": _tuning(31.5, 128), "avg_decay": _tuning(31.0)}
    )
    assert out["winner"]["candidate"] == "avg_last5"
    assert out["winner"]["segment_threshold"] == 128 and out["winner"]["objective"] == 31.5
    assert set(out["candidates"]) == {"final", "avg_last5", "avg_decay"}
    assert out["candidates"]["final"]["objective"] == 30.0 and out["tie_rule"]
    assert out["selection_sets"].startswith("E1 + E2 only")


def test_select_winner_ties_go_to_the_earliest_candidate() -> None:
    out = ev.select_winner(
        {"final": _tuning(30.0), "avg_last5": _tuning(30.0), "avg_decay": _tuning(29.0)}
    )
    assert out["winner"]["candidate"] == "final"
    assert out["tied_candidates"] == ["final", "avg_last5"]
    out = ev.select_winner({"final": _tuning(29.0), "avg_last5": _tuning(30.0 - 1e-12)})
    assert out["winner"]["candidate"] == "avg_last5"  # within 1e-9 of the best but not tied w/ it


def test_select_winner_refuses_a_smoke_tuning_and_an_empty_set() -> None:
    with pytest.raises(ev.EvalStepError, match="not full"):
        ev.select_winner({"final": _tuning(30.0, limit_e2=5)})
    with pytest.raises(ev.EvalStepError, match="no candidates"):
        ev.select_winner({})


def test_run_select_reads_files_writes_selection_and_skip_checks_candidate_order(
    tmp_path: Path,
) -> None:
    for name, obj in (("final", 30.0), ("avg_last5", 32.0)):
        (tmp_path / f"{name}.json").write_text(json.dumps(_tuning(obj)), encoding="utf-8")
    out = tmp_path / "selection.json"
    ev.run_select(tmp_path, ["final", "avg_last5"], out)
    assert ev.selection_is_valid(out, ["final", "avg_last5"])
    assert not ev.selection_is_valid(out, ["final"])
    with pytest.raises(ev.EvalStepError, match="missing"):
        ev.run_select(tmp_path, ["final", "nope"], tmp_path / "x.json")


def test_tune_step_is_skipped_when_a_valid_full_tuning_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "tuning" / "final.json"
    out.parent.mkdir()
    out.write_text(json.dumps(_tuning(30.0)), encoding="utf-8")
    import nmt.tune

    def boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a valid tuning must not be redone")

    monkeypatch.setattr(nmt.tune, "run_tune", boom)
    assert ev.run_tune_step(tmp_path / "model", out) is False

    def fake(model_dir: Path, out_path: Path, batch_size: int) -> dict[str, Any]:
        t = _tuning(31.0)
        out_path.write_text(json.dumps(t), encoding="utf-8")
        return t

    monkeypatch.setattr(nmt.tune, "run_tune", fake)
    out.write_text(json.dumps(_tuning(30.0, limit_e1=10)), encoding="utf-8")  # a smoke leftover
    assert ev.run_tune_step(tmp_path / "model", out) is True


# --- decode ---------------------------------------------------------------------------------------


class _DecodeStub:
    """Translator stand-in: upper-cases sources, records (n, beam, alpha, threshold) per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float, int | None]] = []
        self.stats = types.SimpleNamespace(n_beam=0, n_greedy_fallback=0, n_copy_fallback=0)

    def translate(
        self,
        texts: list[str],
        batch_size: int,
        beam: int,
        alpha: float,
        segment_threshold: int | None,
    ) -> list[str]:
        self.calls.append((len(texts), beam, alpha, segment_threshold))
        self.stats.n_beam += len(texts)
        return [t.upper() or "x" for t in texts]


def _rows(split: str) -> list[dict[str, Any]]:
    return [{"id": f"{split}_{i}", "source": f"source {split} {i}"} for i in range(3)]


def _selection(candidate: str = "final", threshold: int | None = None) -> dict[str, Any]:
    return {
        "winner": {"candidate": candidate, "alpha": 0.8, "beam": 4, "segment_threshold": threshold}
    }


def _decode(
    tmp_path: Path, stub: _DecodeStub, selection: dict[str, Any], test: bool = False
) -> dict[str, Any]:
    test_inputs = tmp_path / "test_inputs.jsonl"
    test_inputs.write_text(
        "\n".join(json.dumps({"id": f"test_{i}", "source": f"t {i}"}) for i in range(2)),
        encoding="utf-8",
    )
    return ev.run_decode(
        stub,  # type: ignore[arg-type]
        selection,
        tmp_path / "eval",
        include_test=test,
        batch_size=7,
        load_rows=_rows,
        test_inputs=test_inputs,
    )


def test_decode_off_threshold_decodes_once_per_split_and_copies_the_tuned_variant(
    tmp_path: Path,
) -> None:
    stub = _DecodeStub()
    summary = _decode(tmp_path, stub, _selection(threshold=None))
    assert len(stub.calls) == len(ev.DECODE_SPLITS)  # one decode per split, not two
    assert all(c[1:] == (4, 0.8, None) for c in stub.calls)
    assert summary["variants_identical"] is True and summary["tuned_threshold_is_off"] is True
    for split in ev.DECODE_SPLITS:
        off = tmp_path / "eval" / "predictions" / "seg_off" / f"{split}_predictions.json"
        tuned = tmp_path / "eval" / "predictions" / "seg_tuned" / f"{split}_predictions.json"
        assert off.read_bytes() == tuned.read_bytes()
    assert not (tmp_path / "eval" / "test_predictions.json").exists()


def test_decode_with_a_threshold_runs_both_variants_and_the_test_set_at_the_tuned_t(
    tmp_path: Path,
) -> None:
    stub = _DecodeStub()
    summary = _decode(tmp_path, stub, _selection(threshold=128), test=True)
    thresholds = [c[3] for c in stub.calls]
    assert thresholds.count(None) == len(ev.DECODE_SPLITS)
    assert thresholds.count(128) == len(ev.DECODE_SPLITS) + 1  # + the test set
    assert stub.calls[-1][3] == 128 and stub.calls[-1][0] == 2  # test: full tuned config incl. T
    assert summary["variants_identical"] is False and summary["test_included"] is True
    test_pred = json.loads((tmp_path / "eval" / "test_predictions.json").read_text("utf-8"))
    assert test_pred == {"test_0": "T 0", "test_1": "T 1"}


def test_decode_rerun_skips_valid_files_and_redoes_only_a_missing_one(tmp_path: Path) -> None:
    sel = _selection(threshold=128)
    _decode(tmp_path, _DecodeStub(), sel)
    again = _DecodeStub()
    _decode(tmp_path, again, sel)
    assert again.calls == []  # everything validated -> nothing decoded
    victim = tmp_path / "eval" / "predictions" / "seg_tuned" / "e2_predictions.json"
    victim.write_text("{}", encoding="utf-8")  # truncated by a disconnect
    third = _DecodeStub()
    _decode(tmp_path, third, sel)
    assert [c[3] for c in third.calls] == [128] and len(third.calls) == 1


def test_decode_discards_predictions_of_a_different_candidate_or_config(tmp_path: Path) -> None:
    _decode(tmp_path, _DecodeStub(), _selection("final", 128))
    other = _DecodeStub()
    _decode(tmp_path, other, _selection("avg_last5", 128))
    assert len(other.calls) == 2 * len(ev.DECODE_SPLITS)  # nothing reused
    summary = json.loads((tmp_path / "eval" / "decode_summary.json").read_text("utf-8"))
    assert summary["candidate"] == "avg_last5"


def test_prediction_file_valid_cases(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    ids = ["a", "b"]
    assert not ev.prediction_file_valid(path, ids)
    for content, ok in (
        ({"a": "x", "b": "y"}, True),
        ({"a": "x"}, False),
        ({"a": "x", "b": "y", "c": "z"}, False),
        ({"a": "x", "b": "  "}, False),
        ({"a": "x", "b": 3}, False),
        (["a", "b"], False),
    ):
        path.write_text(json.dumps(content), encoding="utf-8")
        assert ev.prediction_file_valid(path, ids) is ok, content


# --- validate-test --------------------------------------------------------------------------------


def _sample_ids() -> list[str]:
    return list(json.loads(SAMPLE.read_text(encoding="utf-8")))


def test_validate_test_accepts_exactly_the_330_ids(tmp_path: Path) -> None:
    ids = _sample_ids()
    assert len(ids) == 330
    pred = tmp_path / "test_predictions.json"
    pred.write_text(json.dumps({i: "ok" for i in ids}), encoding="utf-8")
    rec = ev.validate_test_predictions(pred, tmp_path / "validation.json")
    assert rec["valid"] is True and rec["n_ids"] == 330 and rec["empty_strings"] == 0
    saved = json.loads((tmp_path / "validation.json").read_text("utf-8"))
    assert saved["pred_sha256"] and saved["sample_sha256"]


@pytest.mark.parametrize(
    "problem", ["missing", "extra", "empty", "blank", "non_string", "bad_utf8"]
)
def test_validate_test_fails_loudly_and_records_why(tmp_path: Path, problem: str) -> None:
    ids = _sample_ids()
    data: dict[str, Any] = {i: "ok" for i in ids}
    pred = tmp_path / "test_predictions.json"
    if problem == "missing":
        del data[ids[0]]
    elif problem == "extra":
        data["test_99999"] = "x"
    elif problem == "empty":
        data[ids[0]] = ""
    elif problem == "blank":
        data[ids[0]] = "   "
    elif problem == "non_string":
        data[ids[0]] = None
    if problem == "bad_utf8":
        pred.write_bytes(b'{"test_00000": "\xff\xfe"}')
    else:
        pred.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ev.EvalStepError, match="INVALID"):
        ev.validate_test_predictions(pred, tmp_path / "validation.json")
    saved = json.loads((tmp_path / "validation.json").read_text("utf-8"))
    assert saved["valid"] is False and saved["error"]


def test_test_inputs_ids_equal_the_sample_submission_ids() -> None:
    rows = (REPO_ROOT / "data" / "test" / "inputs.jsonl").read_text("utf-8").splitlines()
    assert {json.loads(r)["id"] for r in rows if r} == set(_sample_ids())


# --- HF guards (fake HfApi) -----------------------------------------------------------------------


class _HttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.response = types.SimpleNamespace(status_code=status)


WRITE = {"auth": {"accessToken": {"role": "write"}}, "name": "gg"}
READ = {"auth": {"accessToken": {"role": "read"}}, "name": "gg"}
REPO = "owner/eval"


def _fine(perms: list[str], name: str = REPO) -> dict[str, Any]:
    scoped = [{"entity": {"type": "model", "name": name}, "permissions": perms}]
    return {"auth": {"accessToken": {"role": "fineGrained", "fineGrained": {"scoped": scoped}}}}


class FakeApi:
    """HfApi stand-in recording every mutating call. `private` is what repo_info reports;
    `flip_after_commit` makes the repo look public once a commit has happened."""

    def __init__(
        self,
        who: dict[str, Any] | None = None,
        private: bool | None = True,
        exists: bool = True,
        flip_after_commit: bool = False,
        commit_error: Exception | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self.who = who or WRITE
        self.private = private
        self.exists = exists
        self.flip = flip_after_commit
        self.commit_error = commit_error
        self.create_error = create_error
        self.created: list[dict[str, Any]] = []
        self.commits: list[list[str]] = []
        self.titles: list[str] = []
        self.remote: dict[str, bytes] = {}  # path in repo -> content at the (single) head
        self.list_error: Exception | None = None
        self._cache = tempfile.TemporaryDirectory()  # stands in for the HF download cache

    def whoami(self) -> dict[str, Any]:
        return self.who

    def create_repo(self, **kwargs: Any) -> None:
        if self.create_error:
            raise self.create_error
        self.created.append(kwargs)
        if not self.exists:
            self.exists, self.private = True, kwargs["private"]

    def repo_info(self, **_kwargs: Any) -> Any:
        if not self.exists:
            raise _HttpError(404)
        private = False if (self.flip and self.commits) else self.private
        return types.SimpleNamespace(private=private, sha=REVISION)

    def create_commit(self, **kwargs: Any) -> Any:
        if self.commit_error:
            raise self.commit_error
        self.commits.append([op.path_in_repo for op in kwargs["operations"]])
        self.titles.append(kwargs["commit_message"])
        for op in kwargs["operations"]:
            self.remote[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        return types.SimpleNamespace(oid=REVISION)

    def list_repo_tree(self, **kwargs: Any) -> Any:
        if self.list_error:
            raise self.list_error
        prefix = kwargs["path_in_repo"] + "/"
        found = {p: b for p, b in self.remote.items() if p.startswith(prefix)}
        if not found:
            raise _HttpError(404)
        for path, data in found.items():
            lfs = None
            if path.endswith(".safetensors"):  # the hub reports a sha256 only for LFS files
                lfs = types.SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
            yield types.SimpleNamespace(path=path, size=len(data), lfs=lfs)

    def list_repo_commits(self, **_kwargs: Any) -> list[Any]:
        return [types.SimpleNamespace(title=t, commit_id=REVISION) for t in self.titles]

    def hf_hub_download(self, **kwargs: Any) -> str:
        path = Path(self._cache.name) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.remote[kwargs["filename"]])
        return str(path)


@pytest.mark.parametrize(
    ("who", "refused"),
    [
        (WRITE, False),
        ({"auth": {"accessToken": {"role": "admin"}}}, False),
        (READ, True),
        (_fine(["repo.write"]), False),
        (_fine(["repo.content.read"]), True),
        (_fine(["repo.write"], name="owner/other"), True),
        ({"auth": {}}, False),  # unrecognised shape: the upload's own 401/403 is the backstop
    ],
)
def test_write_scope_problem(who: dict[str, Any], refused: bool) -> None:
    assert (ev.write_scope_problem(who, REPO) is not None) is refused


def test_read_only_token_fails_with_the_instruction_for_gg() -> None:
    with pytest.raises(ev.HFWriteTokenError) as err:
        ev.check_write_token(FakeApi(who=READ), REPO)
    msg = str(err.value)
    assert "fine-grained" in msg and REPO in msg and "HF_TOKEN_WRITE" in msg and "GG" in msg
    with pytest.raises(ev.HFWriteTokenError):
        ev.hf_check(FakeApi(who=_fine(["repo.content.read"])), REPO)


def test_hf_check_passes_for_a_missing_repo_and_a_private_one_but_not_a_public_one() -> None:
    assert ev.hf_check(FakeApi(exists=False), REPO) == {
        "account": "gg",
        "repo": REPO,
        "exists": False,
    }
    assert ev.hf_check(FakeApi(), REPO)["exists"] is True
    with pytest.raises(ev.HFNotPrivateError):
        ev.hf_check(FakeApi(private=False), REPO)
    with pytest.raises(ev.EvalStepError, match="<owner>/<name>"):
        ev.hf_check(FakeApi(), "not-a-repo-id")


def _eval_dir(tmp_path: Path, run: str = "main", winner: str = "avg_decay") -> Path:
    d = tmp_path / "eval"
    (d / "candidates" / winner).mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "spm.model"):
        (d / "candidates" / winner / name).write_bytes(b"w")
    cands = ["final", "avg_last5", "avg_decay"] if run == "main" else ["final"]
    for c in cands:
        (d / "tuning").mkdir(exist_ok=True)
        (d / "tuning" / f"{c}.json").write_text("{}", encoding="utf-8")
    for rel in ("bench.json", "decode_summary.json", "run_meta.json"):
        (d / rel).write_text("{}", encoding="utf-8")
    (d / "selection.json").write_text(
        json.dumps({"candidate_order": cands, "winner": {"candidate": winner}}), encoding="utf-8"
    )
    for v in ev.VARIANTS:
        for s in ev.DECODE_SPLITS:
            p = d / "predictions" / v / f"{s}_predictions.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")
    if run == "main":
        (d / "test_predictions.json").write_text("{}", encoding="utf-8")
        (d / "validation.json").write_text(
            json.dumps({"valid": True, "n_ids": 330, "empty_strings": 0}), encoding="utf-8"
        )
    return d


def test_collect_upload_files_lists_the_prefix_and_only_the_selected_model(tmp_path: Path) -> None:
    files = dict((r, p) for p, r in ev.collect_upload_files(_eval_dir(tmp_path), "main"))
    assert "runs/main/test_predictions.json" in files and "runs/main/validation.json" in files
    assert "runs/main/model/model.safetensors" in files and "runs/main/selection.json" in files
    assert "runs/main/predictions/seg_tuned/e3_predictions.json" in files
    assert "runs/main/tuning/avg_last5.json" in files
    assert all(r.startswith("runs/main/") for r in files)
    abl = dict(
        (r, p)
        for p, r in ev.collect_upload_files(
            _eval_dir(tmp_path / "a", "s1_sin_l4", "final"), "s1_sin_l4"
        )
    )
    assert "runs/s1_sin_l4/test_predictions.json" not in abl


def test_collect_upload_files_names_every_missing_file(tmp_path: Path) -> None:
    d = _eval_dir(tmp_path)
    (d / "bench.json").unlink()
    (d / "test_predictions.json").unlink()
    with pytest.raises(ev.EvalStepError) as err:
        ev.collect_upload_files(d, "main")
    assert "bench.json" in str(err.value) and "test_predictions.json" in str(err.value)


def test_upload_creates_a_private_repo_commits_once_and_records_the_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d, api = _eval_dir(tmp_path), FakeApi(exists=False)
    rec = ev.upload_run(api, REPO, "main", d)
    assert api.created == [
        {"repo_id": REPO, "repo_type": "model", "private": True, "exist_ok": True}
    ]
    assert len(api.commits) == 1 and "runs/main/model/model.safetensors" in api.commits[0]
    assert rec["revision"] == REVISION and rec["private"] is True
    assert f"HF_EVAL_REVISION={REVISION}" in capsys.readouterr().out
    saved = json.loads((d / "hf_upload.json").read_text("utf-8"))
    assert saved["revision"] == REVISION and saved["repo"] == REPO


def test_upload_refuses_a_public_repo_before_uploading_anything(tmp_path: Path) -> None:
    api = FakeApi(private=False)
    with pytest.raises(ev.HFNotPrivateError, match="NOT private"):
        ev.upload_run(api, REPO, "main", _eval_dir(tmp_path))
    assert api.commits == []
    assert not (tmp_path / "eval" / "hf_upload.json").exists()


def test_upload_refuses_when_privacy_cannot_be_read_back(tmp_path: Path) -> None:
    api = FakeApi(private=None)  # a missing flag is a refusal, not a pass
    with pytest.raises(ev.HFNotPrivateError):
        ev.upload_run(api, REPO, "main", _eval_dir(tmp_path))
    assert api.commits == []


def test_upload_verifies_privacy_again_after_the_commit(tmp_path: Path) -> None:
    api = FakeApi(flip_after_commit=True)
    with pytest.raises(ev.HFNotPrivateError, match="after upload"):
        ev.upload_run(api, REPO, "main", _eval_dir(tmp_path))
    assert len(api.commits) == 1
    assert not (tmp_path / "eval" / "hf_upload.json").exists()  # never recorded as a success


def test_upload_refuses_a_read_only_token_before_touching_the_repo(tmp_path: Path) -> None:
    api = FakeApi(who=READ)
    with pytest.raises(ev.HFWriteTokenError):
        ev.upload_run(api, REPO, "main", _eval_dir(tmp_path))
    assert api.created == [] and api.commits == []


def test_upload_maps_a_403_on_commit_to_the_write_token_message(tmp_path: Path) -> None:
    api = FakeApi(commit_error=_HttpError(403))
    with pytest.raises(ev.HFWriteTokenError, match="fine-grained"):
        ev.upload_run(api, REPO, "main", _eval_dir(tmp_path))
    other = FakeApi(commit_error=_HttpError(500))
    with pytest.raises(_HttpError):
        ev.upload_run(other, REPO, "main", _eval_dir(tmp_path / "b"))


def test_upload_tolerates_a_create_403_for_an_existing_private_repo(tmp_path: Path) -> None:
    api = FakeApi(create_error=_HttpError(403))
    assert ev.upload_run(api, REPO, "main", _eval_dir(tmp_path))["revision"] == REVISION
    missing = FakeApi(create_error=_HttpError(403), exists=False)
    with pytest.raises(ev.HFNotPrivateError):  # cannot create and cannot read it back: refuse
        ev.upload_run(missing, REPO, "main", _eval_dir(tmp_path / "b"))


def _uploaded(tmp_path: Path, run: str = "main", winner: str = "avg_decay") -> tuple[Path, FakeApi]:
    """An eval dir whose run was uploaded once into a fresh FakeApi (the remote store holds it)."""
    d = _eval_dir(tmp_path, run, winner)
    api = FakeApi()
    ev.upload_run(api, REPO, run, d)
    return d, api


def test_upload_is_skipped_on_rerun_only_on_hf_evidence_and_still_checks_privacy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d, api = _uploaded(tmp_path)
    again = FakeApi()
    again.remote, again.titles = dict(api.remote), list(api.titles)  # the repo as HF holds it
    rec = ev.upload_run(again, REPO, "main", d)
    assert again.commits == [] and "SKIP (verified complete on HF" in capsys.readouterr().out
    assert rec["revision"] == REVISION and rec["verified_existing"] is True
    with pytest.raises(ev.HFNotPrivateError):
        ev.upload_run(FakeApi(private=False), REPO, "main", d)


def test_a_local_hf_upload_json_alone_never_skips_the_upload(tmp_path: Path) -> None:
    d, _ = _uploaded(tmp_path)  # hf_upload.json now exists on "Drive"
    fresh_repo = FakeApi()  # ... but this HF repo holds nothing for the run
    ev.upload_run(fresh_repo, REPO, "main", d)
    assert len(fresh_repo.commits) == 1


def test_upload_commits_the_manifest_with_every_sha_and_records_its_hash(tmp_path: Path) -> None:
    d, api = _uploaded(tmp_path)
    assert api.titles == [ev.commit_message("main")]
    manifest = json.loads(api.remote["runs/main/manifest.json"])
    assert manifest["run"] == "main" and manifest["candidate_order"] == list(ev.MAIN_CANDIDATES)
    assert set(manifest["files"]) == set(ev.required_rels("main", ev.MAIN_CANDIDATES))
    weights = manifest["files"]["model/model.safetensors"]
    assert weights == {"sha256": hashlib.sha256(b"w").hexdigest(), "bytes": 1}
    saved = json.loads((d / "hf_upload.json").read_text("utf-8"))
    assert (
        saved["manifest_sha256"]
        == hashlib.sha256(api.remote["runs/main/manifest.json"]).hexdigest()
    )


def test_verify_complete_run_and_ablation(tmp_path: Path) -> None:
    _, api = _uploaded(tmp_path)
    state = ev.verify_run_on_hf(api, REPO, "main")
    assert state == {
        "complete": True,
        "reason": "manifest and sha256 verified",
        "revision": REVISION,
    }
    _, abl = _uploaded(tmp_path / "abl", "s1_sin_l4", "final")
    assert ev.verify_run_on_hf(abl, REPO, "s1_sin_l4")["complete"] is True


@pytest.mark.parametrize(
    "damage",
    [
        "no_manifest",
        "tampered_small",
        "tampered_lfs",
        "missing_file",
        "no_commit",
        "bad_validation",
    ],
)
def test_verify_says_not_complete_for_every_kind_of_damage(tmp_path: Path, damage: str) -> None:
    _, api = _uploaded(tmp_path)
    if damage == "no_manifest":
        del api.remote["runs/main/manifest.json"]
    elif damage == "tampered_small":
        data = api.remote["runs/main/selection.json"]
        api.remote["runs/main/selection.json"] = data[:-1] + b"X"  # same size, other bytes
    elif damage == "tampered_lfs":
        api.remote["runs/main/model/model.safetensors"] = b"x"  # lfs sha256 is computed from it
    elif damage == "missing_file":
        del api.remote["runs/main/predictions/seg_tuned/e3_predictions.json"]
    elif damage == "no_commit":
        api.titles = ["some other commit"]
    elif damage == "bad_validation":
        manifest = json.loads(api.remote["runs/main/manifest.json"])
        bad = json.dumps({"valid": True, "n_ids": 329, "empty_strings": 0}).encode()
        api.remote["runs/main/validation.json"] = bad
        manifest["files"]["validation.json"] = {
            "sha256": hashlib.sha256(bad).hexdigest(),
            "bytes": len(bad),
        }
        api.remote["runs/main/manifest.json"] = json.dumps(manifest).encode()
    state = ev.verify_run_on_hf(api, REPO, "main")
    assert state["complete"] is False and state["revision"] is None and state["reason"]


def test_verify_a_missing_repo_or_an_unuploaded_run_is_not_complete() -> None:
    assert ev.verify_run_on_hf(FakeApi(exists=False), REPO, "main")["complete"] is False
    assert ev.verify_run_on_hf(FakeApi(), REPO, "s2_rope_l4")["complete"] is False


def test_verify_fails_closed_when_the_evidence_cannot_be_read(tmp_path: Path) -> None:
    _, api = _uploaded(tmp_path)
    api.list_error = _HttpError(500)
    with pytest.raises(ev.EvalStepError, match="cannot list"):
        ev.verify_run_on_hf(api, REPO, "main")
    with pytest.raises(ev.HFNotPrivateError):
        ev.verify_run_on_hf(FakeApi(private=None), REPO, "main")
    with pytest.raises(ev.EvalStepError, match="cannot read"):
        broken = FakeApi()
        broken.repo_info = lambda **_k: (_ for _ in ()).throw(_HttpError(500))  # type: ignore[method-assign]
        ev.verify_run_on_hf(broken, REPO, "main")


def test_hf_verify_cli_prints_the_marker_line_the_notebook_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, api = _uploaded(tmp_path)
    monkeypatch.setattr(ev, "_hf_api", lambda: api)
    assert ev.main(["hf-verify", "--repo", REPO, "--run", "main"]) == 0
    out = capsys.readouterr().out
    assert "HF_RUN_COMPLETE=true" in out and f"HF_EVAL_REVISION={REVISION}" in out
    assert ev.main(["hf-verify", "--repo", REPO, "--run", "s3_rope_concat_l4"]) == 0
    assert "HF_RUN_COMPLETE=false" in capsys.readouterr().out


def test_missing_hf_token_message_names_the_colab_secret_and_never_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ev.EvalStepError) as err:
        ev._hf_api()
    assert "HF_TOKEN_WRITE" in str(err.value) and "HF_TOKEN" in str(err.value)


def test_upload_rejects_an_unknown_run_and_a_bad_repo_id(tmp_path: Path) -> None:
    with pytest.raises(ev.EvalStepError, match="RUN"):
        ev.upload_run(FakeApi(), REPO, "nope", _eval_dir(tmp_path / "a"))
    with pytest.raises(ev.EvalStepError, match="<owner>/<name>"):
        ev.upload_run(FakeApi(), "bad", "main", _eval_dir(tmp_path / "b"))


# --- workload + CLI -------------------------------------------------------------------------------


def test_committed_workload_matches_a_fresh_measurement() -> None:
    committed = json.loads((REPO_ROOT / "colab" / "eval_workload.json").read_text("utf-8"))
    assert committed == ev.measure_workload()


def test_cli_exit_codes_and_validate_test_roundtrip(tmp_path: Path) -> None:
    pred = tmp_path / "p.json"
    pred.write_text("{}", encoding="utf-8")
    assert ev.main(["validate-test", "--pred", str(pred), "--out", str(tmp_path / "v.json")]) == 1
    pred.write_text(json.dumps({i: "x" for i in _sample_ids()}), encoding="utf-8")
    assert ev.main(["validate-test", "--pred", str(pred), "--out", str(tmp_path / "v.json")]) == 0
    assert (
        ev.main(
            [
                "candidates",
                "--ckpt-dir",
                str(tmp_path),
                "--config",
                "c.yaml",
                "--out-dir",
                str(tmp_path / "o"),
                "--candidate",
                "final=5",
            ]
        )
        == 1
    )  # missing file


def test_constants_match_the_preregistration() -> None:
    assert ev.RUNS == ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4")
    assert ev.DECODE_SPLITS == ("dev", "e1", "e2", "e2synth", "e3")
    assert ev.BENCH_N_SENTENCES == 200 and ev.BENCH_BEAM == 5 and ev.BENCH_ALPHA == 0.6


def test_module_scores_nothing_itself_and_imports_only_the_set_loader_from_selection() -> None:
    # E3, dev and E2-synth are DECODED here (reporting) but never scored: selection reads the
    # E1+E2-only objectives nmt.tune already computed through nmt.selection.
    source = (REPO_ROOT / "nmt" / "eval_l4.py").read_text(encoding="utf-8")
    assert "score_slice" not in source
    assert re.findall(r"from nmt\.selection import (.+)", source) == ["load_selection_set"]


def test_model_config_for_every_run_exists() -> None:
    for run in ev.RUNS:
        assert (REPO_ROOT / "configs" / f"{run}.yaml").is_file()


# --- order of work and the selection surface ----------------------------------------------------


def test_selection_can_only_ever_see_e1_and_e2_and_decoding_needs_a_finished_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nmt import selection

    assert set(selection._ALLOWED_SETS) == {"e1", "e2"}  # dev, E3, E2-synth, test: not loadable
    tune_src = (REPO_ROOT / "nmt" / "tune.py").read_text(encoding="utf-8")
    assert "load_split" not in tune_src and '"e3"' not in tune_src and '"dev"' not in tune_src
    # `decode` (the only step that touches dev/E3/test) refuses to start before selection.json
    assert ev.main(["decode", "--eval-dir", str(tmp_path)]) == 1
    assert "selection.json missing or unreadable" in capsys.readouterr().err
    assert not (tmp_path / "predictions").exists()
