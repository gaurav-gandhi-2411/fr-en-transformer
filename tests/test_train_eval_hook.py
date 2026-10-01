from __future__ import annotations

# Tests for nmt.evaluate.build_train_eval_fn: the nmt.train periodic eval hook (spec §11) --
# greedy BLEU/chrF on fixed subsets of E1/E2/E3/dev, E3 keys tagged reporting-only, a W&B Table
# of sample translations logged to the run (not the returned dict), and full wiring through a
# real (tiny, synthetic-data) nmt.train.train() call.
from pathlib import Path

import pytest
import torch

from nmt.evaluate import TrainEvalConfig, build_train_eval_fn
from nmt.model.transformer import ModelConfig, Transformer
from nmt.train import (
    BatchSection,
    CkptSection,
    EvalSection,
    LoggingSection,
    ModelSection,
    OptimSection,
    TrainConfig,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TOKENIZER = REPO_ROOT / "tokenizer" / "spm.model"

pytestmark = pytest.mark.skipif(
    not REAL_TOKENIZER.is_file(), reason="tokenizer/spm.model not built yet (P2)"
)

_TINY_ROWS = {
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
        [
            {"id": "dev_0", "source": "Merci.", "slice": "seen", "length": 6},
            {"id": "dev_1", "source": "Je ne sais pas.", "slice": "unseen_domain", "length": 15},
        ],
        [
            {"id": "dev_0", "reference": "Thanks.", "slice": "seen"},
            {"id": "dev_1", "reference": "I don't know.", "slice": "unseen_domain"},
        ],
    ),
}


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int]] = []

    def log(self, row: dict, step: int) -> None:
        self.logged.append((row, step))


def _tiny_model() -> Transformer:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16000, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32)
    return Transformer(cfg)


def test_build_train_eval_fn_returns_expected_metric_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmt.evaluate.load_split", lambda name: _TINY_ROWS[name])
    cfg = TrainEvalConfig(tokenizer_path=REAL_TOKENIZER, e1_n=1, e2_n=1, e3_n=1, device="cpu")
    eval_fn = build_train_eval_fn(cfg)
    model = _tiny_model()

    metrics = eval_fn(model, step=10, wandb_run=None)

    for key in ("e1_bleu", "e1_chrf", "e1_n", "e2_bleu", "e2_chrf", "e2_n"):
        assert key in metrics
    for key in (
        "dev_seen_bleu",
        "dev_seen_chrf",
        "dev_unseen_domain_bleu",
        "dev_unseen_domain_chrf",
    ):
        assert key in metrics
    # every E3 key is prefixed "e3_reporting_only_" so it can never be mistaken for a selection
    # signal downstream (spec §11: "reporting only; tagged so in W&B keys").
    e3_keys = [k for k in metrics if k.startswith("e3")]
    assert e3_keys and all(k.startswith("e3_reporting_only_") for k in e3_keys)
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_build_train_eval_fn_logs_a_wandb_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmt.evaluate.load_split", lambda name: _TINY_ROWS[name])
    cfg = TrainEvalConfig(
        tokenizer_path=REAL_TOKENIZER, e1_n=1, e2_n=1, e3_n=1, n_samples_table=2, device="cpu"
    )
    eval_fn = build_train_eval_fn(cfg)
    model = _tiny_model()
    fake_run = _FakeWandbRun()

    metrics = eval_fn(model, step=5, wandb_run=fake_run)

    assert "sample_table_error" not in metrics  # Table logging must not have failed
    assert len(fake_run.logged) == 1
    logged_row, logged_step = fake_run.logged[0]
    assert logged_step == 5
    assert "eval/sample_translations" in logged_row


def test_eval_fn_restores_model_training_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nmt.evaluate.load_split", lambda name: _TINY_ROWS[name])
    cfg = TrainEvalConfig(tokenizer_path=REAL_TOKENIZER, e1_n=1, e2_n=1, e3_n=1, device="cpu")
    eval_fn = build_train_eval_fn(cfg)
    model = _tiny_model()
    model.train()
    eval_fn(model, step=1, wandb_run=None)
    assert model.training is True


def test_train_loop_wires_eval_fn_and_writes_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full nmt.train.train() call with a real build_train_eval_fn hook: eval-interval rows in
    metrics.jsonl carry the hook's keys."""
    monkeypatch.setattr("nmt.evaluate.load_split", lambda name: _TINY_ROWS[name])
    eval_cfg = TrainEvalConfig(tokenizer_path=REAL_TOKENIZER, e1_n=1, e2_n=1, e3_n=1, device="cpu")
    eval_fn = build_train_eval_fn(eval_cfg)

    run_dir, ckpt_dir = tmp_path / "run", tmp_path / "ckpt"
    cfg = TrainConfig(
        name="hook-test",
        group="smoke",
        seed=3,
        device="cpu",
        model=ModelSection(
            vocab_size=16000, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32
        ),
        batch=BatchSection(max_tokens=256, tokens_per_step=256, chunk_size=16),
        optim=OptimSection(lr=1e-3, warmup_steps=1, planned_steps=2, cooldown_frac=0.2),
        ckpt=CkptSection(dir=str(ckpt_dir), ckpt_minutes=999, ckpt_steps=None),
        logging=LoggingSection(run_dir=str(run_dir)),
        eval=EvalSection(eval_every=2),
    )
    train(cfg, wandb_mode="disabled", synthetic=True, max_steps=2, eval_fn=eval_fn)

    import json

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    eval_rows = [r["eval"] for r in rows if "eval" in r]
    assert len(eval_rows) == 1
    assert eval_rows[0]["step"] == 2
    assert "e1_bleu" in eval_rows[0]
    assert "e3_reporting_only_bleu" in eval_rows[0]
