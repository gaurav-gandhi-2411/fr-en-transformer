from __future__ import annotations

# Launcher for `python -m nmt.train` that merges a preflight YAML into the W&B run config
# WITHOUT touching nmt/. Why a launcher and not the env var: with wandb 0.30 the documented
# `WANDB_CONFIG_PATHS` env var is NOT split into a list, pydantic rejects the bare string
# ("'str' instances are not allowed as a Sequence value") and nmt.train then silently continues
# WITHOUT W&B logging -- so the notebook must never set it. `wandb.setup(settings=Settings(
# config_paths=[...]))` is the typed route; the later `wandb.init` inside nmt.train inherits it.
#
# The notebook kernel stays stdlib-only after the install cell, so it hands the preflight values
# over as JSON; this fresh interpreter converts them to the YAML wandb wants.
#
# Usage: python colab/run_train.py <preflight.json> [nmt.train args ...]   (cwd = repo root)
import json
import os
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def write_wandb_config_yaml(path: Path, values: Mapping[str, Any]) -> None:
    """Write `values` in the format wandb's config-file loader expects: `key: {value: x}`
    (a bare `key: x` raises KeyError('value') inside wandb). Keys are used verbatim.
    """
    # str() first: torch.__version__ is a str *subclass* (TorchVersion) that yaml.safe_dump rejects.
    payload = {str(k): {"value": str(v) if isinstance(v, str) else v} for k, v in values.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def prepare_preflight_yaml(json_path: Path) -> Path:
    """Convert the notebook's preflight JSON to `<same name>.yaml` next to it; return that path."""
    values = json.loads(json_path.read_text(encoding="utf-8"))
    yaml_path = json_path.with_suffix(".yaml")
    write_wandb_config_yaml(yaml_path, values)
    return yaml_path


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit("usage: run_train.py <preflight.json> [nmt.train args ...]")
    json_path, train_args = Path(args[0]), args[1:]
    sys.path.insert(0, os.getcwd())  # `nmt` importable even if the editable install is missing
    try:
        import wandb

        yaml_path = prepare_preflight_yaml(json_path)
        wandb.setup(settings=wandb.Settings(config_paths=[str(yaml_path)]))
    except Exception as exc:  # noqa: BLE001 - preflight metadata must never block training
        print(f"WARNING: could not register preflight W&B config ({exc!r}); continuing.")
    sys.argv = ["nmt.train", *train_args]
    runpy.run_module("nmt.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
