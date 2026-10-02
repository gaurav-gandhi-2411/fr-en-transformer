from __future__ import annotations

# Minimum-Bayes-risk (MBR) decoding (PREREG post-selection amendment, rule 2).
#
# Utility: sentence-level chrF EXACTLY as `official/score.py::chrf_sentence` computes it (lowercase,
# all whitespace removed, character n-grams n=1..6, beta=2, F averaged over the orders for which
# both sides have n-grams, empty-vs-empty = 100, one empty = 0). `official/score.py` is vendored and
# byte-pinned, so this module re-implements the same arithmetic (same expressions, same operation
# order) with the n-gram statistics of every candidate computed ONCE per sentence, and
# `tests/test_mbr.py` asserts equality with the official function on 600+ random pairs.
#
# Rule: pseudo-references = the candidate pool. For each candidate i the expected utility is the
# mean of chrF(hyp=candidate i, ref=candidate j) over every OTHER pool member j != i (self-
# exclusion: a candidate never votes for itself; a duplicate of it does, so repeated samples act as
# frequency weights, the standard MBR estimate). The output is the argmax; ties (within TIE_EPS)
# go to the FIRST such member in pool order, so the result is deterministic. A whitespace-only
# (empty) candidate is dropped from the pool before scoring when any non-empty candidate exists
# (an empty hypothesis is never a valid translation and would only dilute the pseudo-references);
# an all-empty pool returns index 0 and the caller's fallback chain handles it.
#
# Cost: chrF overlap is symmetric, so N=16 needs 120 unordered overlaps (x 6 orders), both
# directions of the F-score come from the same overlap, and identical strings are scored once.
import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from nmt.decode import (
    DecodeConfig,
    Hypothesis,
    _repeat_memory,
    _step_log_probs,
    beam_search_nbest,
    decode_max_len,
    gnmt_length_penalty,
)

CHRF_MAX_N = 6  # official/score.py chrf_sentence default
CHRF_BETA = 2.0  # official/score.py chrf_sentence default
TIE_EPS = 1e-12  # utilities within this of the best are tied (float noise, not a tolerance on BLEU)
POOL_KINDS = ("beam", "sample")
DEFAULT_SAMPLING_SEED = 1234  # PREREG rule 2
DEFAULT_EPSILON = 0.02  # PREREG rule 2


@dataclass(frozen=True)
class MBRConfig:
    """One MBR pool: `kind="beam"` = beam n-best with beam size n (the n best finished
    hypotheses; the alpha is the decoding alpha of the caller), `kind="sample"` = n independent
    epsilon samples."""

    kind: str = "beam"
    n: int = 8
    epsilon: float = DEFAULT_EPSILON
    seed: int = DEFAULT_SAMPLING_SEED

    def __post_init__(self) -> None:
        if self.kind not in POOL_KINDS:
            raise ValueError(f"MBRConfig.kind must be one of {POOL_KINDS}, got {self.kind!r}")
        if self.n < 1:
            raise ValueError("MBRConfig.n must be >= 1")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("MBRConfig.epsilon must be in [0, 1)")

    @property
    def label(self) -> str:
        return f"mbr_beam{self.n}" if self.kind == "beam" else f"mbr_eps{self.epsilon:g}_n{self.n}"


# ---------------------------------------------------------------------------------------------
# chrF utility
# ---------------------------------------------------------------------------------------------


class _Stats:
    """A candidate's chrF statistics: the normalized character string and its n-gram counters."""

    __slots__ = ("grams", "text", "totals")

    def __init__(self, s: str) -> None:
        self.text = re.sub(r"\s+", "", (s or "").lower())
        self.grams: list[Counter[str]] = []
        self.totals: list[int] = []
        for n in range(1, CHRF_MAX_N + 1):
            c = Counter(self.text[i : i + n] for i in range(len(self.text) - n + 1))
            self.grams.append(c)
            self.totals.append(sum(c.values()))


def _overlap(a: Counter[str], b: Counter[str]) -> int:
    if len(a) > len(b):
        a, b = b, a
    return sum(min(c, b[g]) for g, c in a.items())


def _f_score(ov: int, hyp_total: int, ref_total: int) -> float:
    p = ov / max(hyp_total, 1)
    rec = ov / max(ref_total, 1)
    if p + rec == 0:
        return 0.0
    b2 = CHRF_BETA * CHRF_BETA
    return (1 + b2) * p * rec / (b2 * p + rec)


def _pair_chrf(h: _Stats, r: _Stats, ovs: list[int | None]) -> float:
    """chrF(hyp=h, ref=r) from the shared per-order overlaps (None = an order skipped because a
    side has no n-gram of that order), mirroring official chrf_sentence's control flow."""
    if not h.text and not r.text:
        return 100.0
    if not h.text or not r.text:
        return 0.0
    f_scores = [
        _f_score(ov, h.totals[n], r.totals[n]) for n, ov in enumerate(ovs) if ov is not None
    ]
    return 100.0 * (sum(f_scores) / len(f_scores)) if f_scores else 0.0


def chrf_matrix(texts: Sequence[str]) -> list[list[float]]:
    """M[i][j] = official chrF(hyp=texts[i], ref=texts[j]) for every ordered pair (diagonal
    included). N-gram statistics are computed once per distinct string, and each unordered pair's
    overlaps once."""
    uniq: dict[str, int] = {}
    owner = [uniq.setdefault(t, len(uniq)) for t in texts]
    stats = [_Stats(t) for t in uniq]
    u = len(stats)
    um = [[0.0] * u for _ in range(u)]
    for i in range(u):
        for j in range(i, u):
            a, b = stats[i], stats[j]
            ovs: list[int | None] = [
                None if (not a.grams[n] or not b.grams[n]) else _overlap(a.grams[n], b.grams[n])
                for n in range(CHRF_MAX_N)
            ]
            um[i][j] = _pair_chrf(a, b, ovs)
            if j != i:
                um[j][i] = _pair_chrf(b, a, ovs)
    return [[um[owner[i]][owner[j]] for j in range(len(texts))] for i in range(len(texts))]


def mbr_utilities(texts: Sequence[str]) -> list[float]:
    """Mean chrF of each candidate against every OTHER member (self-excluded); a pool of one has
    utility 0.0 (no pseudo-reference)."""
    n = len(texts)
    if n <= 1:
        return [0.0] * n
    m = chrf_matrix(texts)
    return [sum(m[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]


def mbr_select(texts: Sequence[str]) -> int:
    """Index (into `texts`) of the MBR output: highest mean utility, first in pool order on ties.
    Whitespace-only candidates are skipped when a non-empty one exists (module docstring)."""
    if not texts:
        raise ValueError("mbr_select: empty pool")
    keep = [i for i, t in enumerate(texts) if (t or "").strip()]
    if not keep:
        return 0
    if len(keep) == 1:
        return keep[0]
    utils = mbr_utilities([texts[i] for i in keep])
    best = max(utils)
    return next(i for i, u in zip(keep, utils, strict=True) if best - u <= TIE_EPS)


# ---------------------------------------------------------------------------------------------
# pool builders
# ---------------------------------------------------------------------------------------------


def beam_pool(
    model: object,
    src: Tensor,
    src_mask: Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    n: int,
    alpha: float,
    no_repeat_ngram_size: int = 3,
) -> list[list[Hypothesis]]:
    """The n best finished hypotheses per source (beam size n, GNMT `alpha`), best first. Fewer
    than n only when the search hit max_len first."""
    cfg = DecodeConfig(beam_size=n, alpha=alpha, no_repeat_ngram_size=no_repeat_ngram_size)
    return beam_search_nbest(model, src, src_mask, bos_id, eos_id, pad_id, cfg)  # type: ignore[arg-type]


def _row_seed(seed: int, key: str, j: int) -> int:
    digest = hashlib.sha256(f"{seed}|{j}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


@torch.no_grad()
def sample_pool(
    model: object,
    src: Tensor,
    src_mask: Tensor,
    bos_id: int,
    eos_id: int,
    keys: Sequence[str],
    n: int,
    epsilon: float = DEFAULT_EPSILON,
    seed: int = DEFAULT_SAMPLING_SEED,
    max_len_a: float = 1.5,
    max_len_b: int = 10,
) -> list[list[Hypothesis]]:
    """n independent epsilon samples per source (Hewitt et al. 2022): at every step tokens with
    probability < epsilon are removed (the arg-max token is always kept), the rest renormalized
    and sampled. Ancestral over the model's own distribution (an ensemble's averaged log-probs are
    renormalized by log_softmax first); no repetition block; max_len as in beam search.

    Deterministic and independent of batch composition: the uniform draws for sample j of source b
    come from a CPU generator seeded by sha256(seed | j | keys[b]) (`keys[b]` = a stable string
    for the source, e.g. its text), one uniform per step used by inverse-CDF sampling.
    """
    device = src.device
    b_size = src.size(0)
    if len(keys) != b_size:
        raise ValueError("sample_pool: one key per source row is required")
    was_training = model.training  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
    rows = b_size * n
    src_lens = src_mask.sum(dim=1).tolist()
    max_lens = [decode_max_len(int(length), max_len_a, max_len_b) for length in src_lens]
    row_max = torch.tensor([m for m in max_lens for _ in range(n)], device=device)
    global_max = max(max_lens) if max_lens else 0

    uniforms = torch.empty(rows, max(global_max, 1))
    for b in range(b_size):
        for j in range(n):
            g = torch.Generator(device="cpu")
            g.manual_seed(_row_seed(seed, keys[b], j))
            uniforms[b * n + j] = torch.rand(uniforms.size(1), generator=g)
    uniforms = uniforms.to(device)

    memory = model.encode(src, src_mask)  # type: ignore[attr-defined]
    cache = model.init_decode_cache(  # type: ignore[attr-defined]
        _repeat_memory(memory, n), src_mask.repeat_interleave(n, dim=0)
    )
    cur = torch.full((rows, 1), bos_id, dtype=torch.long, device=device)
    out_tokens: list[list[int]] = [[] for _ in range(rows)]
    logp_sum = torch.zeros(rows, device=device)
    done = torch.zeros(rows, dtype=torch.bool, device=device)
    for step in range(global_max):
        if bool(done.all()):
            break
        lp = torch.log_softmax(_step_log_probs(model, cur, cache), dim=-1)  # type: ignore[arg-type]
        probs = lp.exp()
        keep = probs >= epsilon
        keep.scatter_(1, probs.argmax(dim=1, keepdim=True), True)
        filtered = torch.where(keep, probs, torch.zeros_like(probs))
        cdf = filtered.cumsum(dim=1)
        target = uniforms[:, step : step + 1] * cdf[:, -1:]
        tok = torch.searchsorted(cdf, target, right=True).clamp_(max=probs.size(1) - 1).squeeze(1)
        # right=True: the first index whose cdf is strictly above the target, so a filtered
        # (zero-mass) token can never be drawn, even for u == 0. Force EOS at the length cap.
        tok = torch.where(step + 1 >= row_max, torch.full_like(tok, eos_id), tok)
        chosen_lp = lp.gather(1, tok[:, None]).squeeze(1)
        logp_sum = logp_sum + torch.where(done, torch.zeros_like(chosen_lp), chosen_lp)
        tok = torch.where(done, torch.full_like(tok, eos_id), tok)
        for r, (t, was_done) in enumerate(zip(tok.tolist(), done.tolist(), strict=True)):
            if not was_done:
                out_tokens[r].append(t)
        done = done | (tok == eos_id)
        cur = tok[:, None]
    if was_training:
        model.train()  # type: ignore[attr-defined]
    return [
        [
            Hypothesis(
                tokens=out_tokens[b * n + j],
                score=float(logp_sum[b * n + j])
                / gnmt_length_penalty(len(out_tokens[b * n + j]), 1.0),
            )
            for j in range(n)
        ]
        for b in range(b_size)
    ]
