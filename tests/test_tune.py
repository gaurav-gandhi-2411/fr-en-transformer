from __future__ import annotations

# Tests for nmt/tune.py's segmentation grid: the default T grid and the always-present "off"
# (`no_segmentation`) candidate. Uses a stub translator, so no model or decoding is needed.
from nmt.selection import limited_selection_set, load_selection_set
from nmt.tune import (
    DEFAULT_SEG_THRESHOLDS,
    NO_SEGMENTATION_KEY,
    _parse_args,
    _segmentation_tune,
)


class _EchoTranslator:
    """Stub with `Translator.translate`'s signature; echoes sources and records thresholds."""

    def __init__(self) -> None:
        self.thresholds: list[int | None] = []

    def translate(
        self,
        sources: list[str],
        batch_size: int,
        beam: int,
        alpha: float,
        segment_threshold: int | None,
    ) -> list[str]:
        self.thresholds.append(segment_threshold)
        return list(sources)


def _run(thresholds: tuple[int, ...]) -> tuple[dict, _EchoTranslator]:
    e1 = limited_selection_set(load_selection_set("e1"), 3)
    e2 = limited_selection_set(load_selection_set("e2"), 3)
    stub = _EchoTranslator()
    fixed_e1 = dict(zip(e1.ids, e1.sources, strict=True))
    out = _segmentation_tune(stub, e1, e2, fixed_e1, 0.6, 5, thresholds, 4)  # type: ignore[arg-type]
    return out, stub


def test_default_grid_is_64_128_192_256_and_cli_default_matches() -> None:
    assert DEFAULT_SEG_THRESHOLDS == (64, 128, 192, 256)
    args = _parse_args(["--model", "m", "--out", "o"])
    assert args.seg_thresholds == [64, 128, 192, 256]


def test_off_candidate_is_always_scored_alongside_every_threshold() -> None:
    out, stub = _run(DEFAULT_SEG_THRESHOLDS)
    assert list(out["scores"]) == [NO_SEGMENTATION_KEY, "T=64", "T=128", "T=192", "T=256"]
    assert stub.thresholds == [None, 64, 128, 192, 256]


def test_off_candidate_present_even_with_an_empty_threshold_grid() -> None:
    out, stub = _run(())
    assert list(out["scores"]) == [NO_SEGMENTATION_KEY]
    assert out["best"] == NO_SEGMENTATION_KEY and out["best_segment_threshold"] is None
    assert stub.thresholds == [None]
