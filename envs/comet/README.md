# envs/comet

Isolated `uv` project for COMET-22 (`Unbabel/wmt22-comet-da`) scoring only (spec §8). Kept
separate from the main repo's `pyproject.toml` because `unbabel-comet` pins `numpy<2` and an old
`torchmetrics`/`pytorch-lightning` stack that does not co-resolve with the main repo's pinned
`torch==2.14.0` / `numpy==2.5.3`.

Invoked by `nmt.evaluate.run_comet` via:

```
uv run --project envs/comet python envs/comet/score_comet.py --in <triples.json> --out <out.json>
```

`--in` is a JSON list of `{"src": ..., "mt": ..., "ref": ...}` objects. Eval-only: this script's
output is never consumed by `nmt/selection.py` or training. The model checkpoint is ~2.3 GB and
is downloaded once to the local Hugging Face cache on first use.
