# envs/cuda

Separate `uv` project for running training on the local NVIDIA GPU (RTX 3070 Laptop, driver
581.42 / CUDA 13.0). The root `pyproject.toml` / `uv.lock` (CPU torch, used by CI) are untouched.

- Same pinned versions as the root `uv.lock` for every package (direct deps pinned in
  `pyproject.toml`; every transitive package pinned via `constraint-dependencies`), except
  `torch==2.14.0`, which comes from the `https://download.pytorch.org/whl/cu130` index
  (`2.14.0+cu130`, win_amd64 cp312 wheel verified present; cu128/cu129 have no 2.14.0 wheel,
  cu126 does but cu130 matches the driver's CUDA 13.0).
  `tests/test_cuda_env_pins.py` fails if `envs/cuda/uv.lock` drifts from the root `uv.lock`.
- `package = false`: the repo's `nmt` package is not installed. Run from the REPO ROOT:
  `python -m <module>` puts the current directory on `sys.path`, so `nmt`, `scripts` and `tests`
  import straight from the working tree (always the live code, no stale editable install).

```
uv run --project envs/cuda python -m nmt.train --config configs/pilot_3070.yaml --wandb online
uv run --project envs/cuda python -m scripts.probe_microbatch --out runs/pilot_3070/probe.json
uv run --project envs/cuda python -m pytest -q tests/test_train*.py tests/test_model.py
```

`uv run --project envs/cuda` leaves the cwd alone (it only selects the project), and the venv is
created at `envs/cuda/.venv` (gitignored). Never use `python script.py` form for repo modules:
that puts the script's directory, not the repo root, on `sys.path`.
