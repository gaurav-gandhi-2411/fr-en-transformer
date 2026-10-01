from __future__ import annotations

# Tests for the precision / kernel / provenance layer of nmt/train.py: precision resolution,
# autocast + GradScaler wiring, the checkpoint precision guard, the SDPA kernel report, RNG
# fingerprints, W&B entity + provenance plumbing, and the no-torch.compile guarantee. CUDA-only
# checks skip on the CPU/CI env and run under `uv run --project envs/cuda`.
import ast
import json
import os
import socket
import sys
import types
import uuid
import warnings
from pathlib import Path

import pytest
import torch
import yaml

import nmt.train as train_module
from nmt.model.transformer import ModelConfig
from nmt.train import (
    BatchSection,
    CkptSection,
    LoggingSection,
    ModelSection,
    OptimSection,
    TrainConfig,
    _flash_reason,
    autocast_context,
    collect_run_provenance,
    load_config,
    probe_sdpa_kernel,
    resolve_precision,
    rng_fingerprint,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
CUDA = torch.device("cuda")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cfg(
    run_dir: Path, ckpt_dir: Path, precision: str = "fp32", device: str = "cpu"
) -> TrainConfig:
    return TrainConfig(
        name="prec",
        group="smoke",
        seed=7,
        device=device,
        precision=precision,
        model=ModelSection(
            vocab_size=32, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32, dropout=0.1
        ),
        batch=BatchSection(max_tokens=128, tokens_per_step=128, chunk_size=32),
        optim=OptimSection(lr=1e-3, warmup_steps=2, planned_steps=20, cooldown_frac=0.2),
        ckpt=CkptSection(dir=str(ckpt_dir), ckpt_minutes=999, ckpt_steps=3),
        logging=LoggingSection(run_dir=str(run_dir)),
    )


# ---- resolution ---------------------------------------------------------------------------


def test_auto_resolves_per_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_precision("auto", CPU) == "fp32"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert resolve_precision("auto", CUDA) == "bf16"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    assert resolve_precision("auto", CUDA) == "fp16"


def test_explicit_precision_rules() -> None:
    assert resolve_precision("fp32", CPU) == "fp32"
    assert resolve_precision("bf16", CUDA) == "bf16"
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_precision("bf16", CPU)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_precision("fp16", CPU)
    with pytest.raises(ValueError, match="precision must be one of"):
        resolve_precision("fp8", CPU)


def test_load_config_rejects_unknown_precision(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"name": "x", "precision": "half"}), encoding="utf-8")
    with pytest.raises(ValueError, match="precision must be one of"):
        load_config(path)


def test_autocast_context_fp32_is_a_noop_on_cpu() -> None:
    with autocast_context(CPU, "fp32"):
        assert not torch.is_autocast_enabled("cpu")


# ---- checkpoint precision guard -----------------------------------------------------------


def test_checkpoint_records_precision_and_resume_with_other_precision_fails(
    tmp_path: Path,
) -> None:
    run, ckpt = tmp_path / "run", tmp_path / "ckpt"
    train(_cfg(run, ckpt), wandb_mode="disabled", synthetic=True, max_steps=3)
    path = sorted(ckpt.glob("step_*.pt"))[-1]
    payload = torch.load(path, weights_only=False)
    assert payload["precision"] == "fp32"

    payload["precision"] = "bf16"  # pretend it was written by a bf16 run
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="cannot resume.*precision='bf16'"):
        train(_cfg(run, ckpt), wandb_mode="disabled", synthetic=True, resume=True, max_steps=6)


# ---- RNG fingerprint ----------------------------------------------------------------------


def test_rng_fingerprint_is_stable_and_state_sensitive() -> None:
    sampler = {"epoch": 0, "batch_cursor": 3, "aug_rng_state": {"state": {"state": 1, "inc": 2}}}
    torch.manual_seed(1)
    a = rng_fingerprint(train_module._rng_state(), sampler)
    assert len(a) == 12 and int(a, 16) >= 0
    torch.manual_seed(1)
    assert rng_fingerprint(train_module._rng_state(), sampler) == a
    torch.rand(1)
    assert rng_fingerprint(train_module._rng_state(), sampler) != a
    torch.manual_seed(1)
    assert rng_fingerprint(train_module._rng_state(), {**sampler, "batch_cursor": 4}) != a


# ---- SDPA report --------------------------------------------------------------------------


def test_sdpa_probe_is_a_noop_off_cuda() -> None:
    report = probe_sdpa_kernel(ModelConfig(vocab_size=32), CPU, "fp32")
    assert report["kernel"] == "n/a"
    assert report["summary"].startswith("SDPA kernel:")


def test_flash_reason_keeps_only_the_flash_section() -> None:
    msgs = [
        "Memory efficient kernel not used because: (Triggered internally at x.cpp:1.)",
        "Memory Efficient attention has been runtime disabled. (Triggered internally at y.h:2.)",
        "Flash attention kernel not used because: (Triggered internally at x.cpp:3.)",
        "Torch was not compiled with flash attention. (Triggered internally at z.cpp:4.)",
        "cuDNN attention kernel not used because: (Triggered internally at x.cpp:5.)",
        "cuDNN attention has been runtime disabled. (Triggered internally at w.cpp:6.)",
    ]
    assert _flash_reason(msgs) == "Torch was not compiled with flash attention."
    assert _flash_reason([]) == "unsupported"


@needs_cuda
def test_sdpa_probe_on_cuda_names_a_backend_for_every_attention_site() -> None:
    report = probe_sdpa_kernel(ModelConfig(vocab_size=32), CUDA, "bf16")
    assert set(report["per_site"]) == {"encoder_self", "decoder_self", "cross"}
    assert report["kernel"] in {"efficient", "flash", "cudnn", "math", "mixed"}
    assert report["summary"].startswith("SDPA kernel: ")


# ---- CUDA: bf16 wiring --------------------------------------------------------------------


@needs_cuda
def test_bf16_train_step_on_cuda_has_no_grad_scaling_and_finite_loss(tmp_path: Path) -> None:
    run, ckpt = tmp_path / "run", tmp_path / "ckpt"
    train(
        _cfg(run, ckpt, precision="bf16", device="cuda"),
        wandb_mode="disabled",
        synthetic=True,
        max_steps=4,
    )
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in rows if "loss" in r]
    assert len(steps) == 4
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in steps)
    assert all(r["loss_scale"] == 1.0 for r in steps)  # GradScaler disabled under bf16
    info = json.loads((run / "run_info.json").read_text())
    assert info["precision"] == "bf16" and info["tf32"] is True
    assert info["sdpa_kernel"] in {"efficient", "flash", "cudnn", "math", "mixed"}
    payload = torch.load(sorted(ckpt.glob("step_*.pt"))[-1], weights_only=False)
    assert payload["precision"] == "bf16"
    with pytest.raises(RuntimeError, match="cannot resume"):
        train(
            _cfg(run, ckpt, precision="fp16", device="cuda"),
            wandb_mode="disabled",
            synthetic=True,
            resume=True,
            max_steps=6,
        )


# ---- provenance and W&B -------------------------------------------------------------------


def test_run_provenance_fields_and_no_hostname(tmp_path: Path) -> None:
    from nmt.model.transformer import Transformer

    model = Transformer(
        ModelConfig(vocab_size=32, d_model=16, n_heads=2, enc_layers=1, dec_layers=1)
    )
    sdpa = probe_sdpa_kernel(model.cfg, CPU, "fp32")
    prov = collect_run_provenance(
        _cfg(tmp_path, tmp_path), CPU, "fp32", False, sdpa, model, tmp_path
    )
    expected = {
        "git_sha",
        "git_dirty",
        "precision",
        "tf32",
        "sdpa_kernel",
        "torch_version",
        "torch_cuda_version",
        "gpu_name",
        "gpu_total_memory_mb",
        "driver_version",
        "platform",
        "param_count",
        "data_manifest_sha256",
        "shard_manifest_sha256",
    }
    assert expected <= set(prov)
    assert prov["param_count"] == model.param_count()
    assert prov["gpu_name"] is None and prov["shard_manifest_sha256"] is None  # CPU, no manifest
    assert prov["git_dirty"] in (True, False, None)
    assert socket.gethostname().lower() not in str(prov["platform"]).lower()
    # the repo's committed data manifest is hashed (64 hex chars) when present
    if (REPO_ROOT / "data" / "data_manifest.json").is_file():
        assert len(prov["data_manifest_sha256"]) == 64


def test_wandb_init_passes_entity_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}
    fake = types.SimpleNamespace(init=lambda **kw: captured.update(kw) or "RUN")
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = _cfg(tmp_path, tmp_path)
    cfg.logging.wandb_entity = "some-entity"
    from nmt.model.transformer import Transformer

    model = Transformer(
        ModelConfig(vocab_size=32, d_model=16, n_heads=2, enc_layers=1, dec_layers=1)
    )
    run = train_module.init_wandb(
        cfg, "offline", 7, model, CPU, {"precision": "bf16", "git_dirty": False}
    )
    assert run == "RUN"
    assert captured["entity"] == "some-entity"
    assert captured["config"]["precision"] == "bf16"  # resolved value wins over raw "auto"/"fp32"
    assert captured["config"]["git_dirty"] is False


def test_every_committed_config_sets_wandb_entity_explicitly() -> None:
    paths = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert paths
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["logging"].get("wandb_entity") == "gauravgandhi429-gaurav-gandhi", path.name
        assert load_config(path).logging.wandb_entity == "gauravgandhi429-gaurav-gandhi"


# ---- torch.compile stays off --------------------------------------------------------------


def test_no_torch_compile_anywhere_in_the_package() -> None:
    """AST scan (comments and docstrings don't count): no `torch.compile` attribute use and no
    `from torch import compile` anywhere under nmt/."""
    offenders = []
    for path in (REPO_ROOT / "nmt").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_attr = (
                isinstance(node, ast.Attribute)
                and node.attr == "compile"
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            )
            is_import = (
                isinstance(node, ast.ImportFrom)
                and node.module == "torch"
                and any(a.name == "compile" for a in node.names)
            )
            if is_attr or is_import:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, offenders
    assert train_module.TORCH_COMPILE is False


# ---- determinism setup --------------------------------------------------------------------


def test_seed_everything_sets_cublas_workspace_on_cuda_without_overriding_user_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    train_module.seed_everything(1)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    train_module.seed_everything(1)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_nondeterminism_warnings_are_deduplicated_across_call_sites() -> None:
    train_module.seed_everything(1)
    text = f"op_{uuid.uuid4().hex} does not have a deterministic implementation"
    with warnings.catch_warnings(record=True) as caught:
        for _ in range(3):
            warnings.warn(text, UserWarning, stacklevel=1)
        warnings.warn(text, UserWarning, stacklevel=1)  # a different call site, same text
    assert len(caught) == 1
