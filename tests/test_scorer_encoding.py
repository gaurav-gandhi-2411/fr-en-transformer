from __future__ import annotations

# Local scoring runs the vendored official/score.py through nmt.evaluate.run_official_scorer, which
# forces UTF-8 mode. The scorer opens its inputs with the platform default encoding, so without
# PYTHONUTF8=1 a UTF-8 file with literal non-ASCII text is read as cp1252 mojibake on Windows
# (incident: a reference-identical dev prediction file scored BLEU 97.5). These tests are
# OS-agnostic: they pass on Linux (UTF-8 default) and on Windows (where they are the real guard).
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nmt import evaluate

# Accented French, curly quotes, an em dash, a non-Latin-1 character and an emoji: every one of
# these is mangled when decoded as cp1252 (bytes 0x81/0x8D/0x8F/0x90/0x9D are even undefined).
REFERENCES = {
    "t1": "Le garçon a mangé « une crème brûlée » à l’hôtel — très tard.",
    "t2": "L’œuvre d’art coûte 5 € ; ça vaut Zoë.",
    "t3": "Naïve café: déjà vu, Ångström, 日本語 🙂 and ŝ.",
    "t4": "Plain ASCII sentence number four.",
}


def _write_gold_and_pred(tmp_path: Path) -> tuple[Path, Path]:
    gold = tmp_path / "labels.jsonl"
    rows = [{"id": i, "reference": r, "slice": "seen"} for i, r in REFERENCES.items()]
    gold.write_bytes(
        ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    )
    pred = tmp_path / "pred.json"
    pred.write_bytes(json.dumps(REFERENCES, ensure_ascii=False).encode("utf-8"))  # literal UTF-8
    return gold, pred


def test_reference_identical_utf8_predictions_score_bleu_and_chrf_100(tmp_path: Path) -> None:
    gold, pred = _write_gold_and_pred(tmp_path)
    assert any(ord(ch) > 127 for ch in pred.read_text(encoding="utf-8"))  # non-ASCII on disk
    report = evaluate.run_official_scorer_cli(gold, pred, tmp_path / "out.json")
    assert report["all"]["bleu"] == pytest.approx(100.0)
    assert report["all"]["chrf"] == pytest.approx(100.0)
    stdout = evaluate.run_official_scorer(gold, pred).stdout
    assert "OVERALL" in stdout


def test_wrapper_sets_utf8_mode_even_when_the_parent_env_lacks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("PYTHONUTF8", "PYTHONIOENCODING"):
        monkeypatch.delenv(name, raising=False)
    env = evaluate.official_scorer_env()
    assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
    assert evaluate.official_scorer_env({"PYTHONUTF8": "0", "KEEP": "x"}) == {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "KEEP": "x",
    }  # a parent that disables UTF-8 mode is overridden, other variables pass through


def test_the_subprocess_really_receives_utf8_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(cmd=cmd, env=kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    evaluate.run_official_scorer(tmp_path / "g", tmp_path / "p", tmp_path / "o")
    assert seen["env"]["PYTHONUTF8"] == "1" and seen["env"]["PYTHONIOENCODING"] == "utf-8"
    assert seen["cmd"][1] == str(evaluate.OFFICIAL_SCORE_PY)  # the vendored file, as shipped
    assert seen["cmd"][-2:] == ["--out", str(tmp_path / "o")]


def test_no_code_outside_the_wrapper_launches_the_official_scorer() -> None:
    root = Path(__file__).resolve().parents[1]
    allowed = {root / "nmt" / "evaluate.py"}
    offenders = []
    for folder in ("nmt", "scripts", "colab"):
        for path in (root / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            # launching a script needs an interpreter: sys.executable next to the scorer path
            if path not in allowed and "sys.executable" in text and "OFFICIAL_SCORE_PY" in text:
                offenders.append(path.name)
    assert offenders == []
