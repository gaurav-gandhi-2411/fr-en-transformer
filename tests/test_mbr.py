from __future__ import annotations

# nmt/mbr.py: the chrF utility equals official/score.py's chrf_sentence, the MBR rule
# (self-excluded mean utility, first-in-pool tie-break, empty handling), deterministic pools.
import random
import time

import pytest
import torch

from nmt.evaluate import load_official_module
from nmt.hub import NMTModel
from nmt.mbr import (
    MBRConfig,
    beam_pool,
    chrf_matrix,
    mbr_select,
    mbr_utilities,
    sample_pool,
)

# Mixed ASCII / accented / CJK / uppercase-special letters / tabs, so lowercasing and the
# whitespace strip are exercised on non-ASCII text.
_ALPHABET = list("abcdeé fgh ij,. àç漢字ñ ") + ["ß", "İ", "Σ", "\t"]


def _random_text(rng: random.Random) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 40)))


def test_utility_equals_official_chrf_on_random_pairs() -> None:
    official = load_official_module()
    rng = random.Random(42)
    n_pairs, worst = 0, 0.0
    saw_non_ascii = False
    for _ in range(40):
        texts = [_random_text(rng) for _ in range(rng.randint(2, 8))]
        texts.append(texts[0])  # a duplicate string
        saw_non_ascii = saw_non_ascii or any(not t.isascii() for t in texts)
        m = chrf_matrix(texts)
        for i, h in enumerate(texts):
            for j, r in enumerate(texts):
                want = official.chrf_sentence(h, r)
                assert m[i][j] == want, (h, r, m[i][j], want)  # exact, not approximate
                worst = max(worst, abs(m[i][j] - want))
                n_pairs += 1
    assert n_pairs >= 200 and worst == 0.0 and saw_non_ascii


def test_utility_edge_cases_match_official() -> None:
    official = load_official_module()
    texts = ["", "   ", "a", "ab", "abcdefg", "ABCDEFG", "x y z", "é", "漢字漢字漢字"]
    m = chrf_matrix(texts)
    for i, h in enumerate(texts):
        for j, r in enumerate(texts):
            assert m[i][j] == official.chrf_sentence(h, r)


def test_identical_candidates_return_first() -> None:
    assert mbr_select(["the cat sat"] * 5) == 0


def test_pool_of_one_returns_it() -> None:
    assert mbr_select(["only"]) == 0
    assert mbr_utilities(["only"]) == [0.0]


def test_empty_pool_raises() -> None:
    with pytest.raises(ValueError):
        mbr_select([])


def test_consensus_candidate_wins() -> None:
    pool = [
        "a completely different thing",
        "the cat sat on the mat",
        "the cat sat on a mat",
        "the cat sat on the mat.",
    ]
    assert mbr_select(pool) in (1, 3)  # a near-identical majority member, never the outlier


def test_self_exclusion_and_first_tie_break() -> None:
    pool = ["abc def", "abc def"]
    assert mbr_utilities(pool) == [100.0, 100.0]
    assert mbr_select(pool) == 0  # tie -> first in pool order
    # a candidate does not vote for itself: with one outlier and two equal copies, the copies
    # support each other (utility 100 + x) and beat the outlier
    pool = ["zzzz qqqq", "the quick fox", "the quick fox"]
    assert mbr_select(pool) == 1


def test_empty_candidates_dropped_unless_all_empty() -> None:
    assert mbr_select(["", "hello world", "  ", "hello world"]) == 1
    assert mbr_select(["", "hello"]) == 1
    assert mbr_select(["", "  "]) == 0


def test_deterministic() -> None:
    rng = random.Random(1)
    pool = [_random_text(rng) + "x" for _ in range(16)]
    assert len({mbr_select(pool) for _ in range(3)}) == 1
    assert mbr_utilities(pool) == mbr_utilities(list(pool))


def test_n16_pool_speed_smoke() -> None:
    rng = random.Random(3)
    base = "".join(rng.choice("abcdefgh ") for _ in range(120))
    pools = [[base[: 100 + i] + str(k) for i in range(16)] for k in range(50)]
    t0 = time.perf_counter()
    for p in pools:
        mbr_select(p)
    assert time.perf_counter() - t0 < 30.0  # generous CI bound for 50 pools of N=16


def _tiny(seed: int) -> NMTModel:
    torch.manual_seed(seed)
    return NMTModel(
        vocab_size=48, d_model=16, n_heads=2, enc_layers=1, dec_layers=1, d_ff=32
    ).eval()


def _src() -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.tensor([[5, 6, 7, 3], [8, 9, 3, 0], [4, 3, 0, 0]])
    return src, src != 0


def _toks(pools: list) -> list[list[list[int]]]:
    return [[h.tokens for h in p] for p in pools]


def test_sampling_pool_is_deterministic_given_seed_and_batch_independent() -> None:
    model = _tiny(0)
    src, mask = _src()
    keys = ["a", "b", "c"]
    a = sample_pool(model, src, mask, 2, 3, keys, n=4, epsilon=0.02, seed=1234)
    b = sample_pool(model, src, mask, 2, 3, keys, n=4, epsilon=0.02, seed=1234)
    c = sample_pool(model, src, mask, 2, 3, keys, n=4, epsilon=0.02, seed=99)
    assert _toks(a) == _toks(b)
    assert _toks(a) != _toks(c)
    # source 1 alone (same key) draws the same samples as inside the batch of 3
    solo = sample_pool(model, src[1:2], mask[1:2], 2, 3, ["b"], n=4, epsilon=0.02, seed=1234)
    assert _toks(solo)[0] == _toks(a)[1]
    assert all(len(p) == 4 for p in a)


def test_epsilon_near_one_keeps_only_the_argmax_token() -> None:
    model = _tiny(1)
    src, mask = _src()
    pool = sample_pool(model, src, mask, 2, 3, ["a", "b", "c"], n=8, epsilon=0.999, seed=1)
    other = sample_pool(model, src, mask, 2, 3, ["a", "b", "c"], n=1, epsilon=0.999, seed=2)
    for p, o in zip(pool, other, strict=True):
        assert len({tuple(h.tokens) for h in p}) == 1  # every sample is the greedy path
        assert p[0].tokens == o[0].tokens


def test_beam_pool_best_equals_beam_search_and_is_sorted() -> None:
    from nmt.decode import DecodeConfig, beam_search_decode

    model = _tiny(2)
    src, mask = _src()
    pools = beam_pool(model, src, mask, 2, 3, 0, n=4, alpha=1.2)
    best = beam_search_decode(model, src, mask, 2, 3, 0, DecodeConfig(beam_size=4, alpha=1.2))
    for pool, ref in zip(pools, best, strict=True):
        assert pool[0].tokens == ref.tokens and pool[0].score == ref.score
        assert [h.score for h in pool] == sorted((h.score for h in pool), reverse=True)


def test_mbr_config_validation() -> None:
    with pytest.raises(ValueError):
        MBRConfig(kind="nope")
    with pytest.raises(ValueError):
        MBRConfig(n=0)
    assert MBRConfig("sample", 16).label == "mbr_eps0.02_n16"
    assert MBRConfig("beam", 8).label == "mbr_beam8"
