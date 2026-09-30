from __future__ import annotations

"""CLI entry point wiring the stages together: prepare|tokenize|train|evaluate|
predict|analyze|export|all. Reproduce command:
`python -m nmt.pipeline --config configs/main.yaml --stage all --seed 1234`. Spec §2.
"""
