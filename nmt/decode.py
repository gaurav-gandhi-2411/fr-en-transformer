from __future__ import annotations

# Batched decoding: beam search (GNMT length penalty, 3-gram repetition block), greedy decode,
# and the long-input sentence-segmentation fallback. Spec §7.
#
# Beam search reduces exactly to greedy at beam_size=1 by construction (once the single active
# slot for an example finds EOS, that example's target active-slot count drops to zero and its
# group is frozen for the rest of the run -- no "keep searching for a second candidate" step, so
# there is nothing for a token-level equivalence test to disagree on). `greedy_decode` below is a
# separately-written, simpler/faster loop (no top-2K bookkeeping, no beam-index cache reorder);
# `tests/test_decode.py::test_beam_size_1_equals_greedy` checks the two independent
# implementations actually agree.
import re
from dataclasses import dataclass

import torch
from torch import Tensor

from nmt.model.transformer import Transformer

# Sentence-ending punctuation for the segmentation fallback (spec §7): ". ! ? ; : and …
# followed by space". The lookbehind keeps the punctuation attached to the preceding segment
# (delimiters are never dropped) and only the whitespace itself is consumed by the split.
_SENT_END_CHARS = ".!?;:…"
_SENT_SPLIT_RE = re.compile(rf"(?<=[{re.escape(_SENT_END_CHARS)}])\s+")


def split_sentences(text: str) -> list[str]:
    """Split `text` on sentence-ending punctuation followed by whitespace, keeping the
    delimiter attached to the segment before it. Never returns an empty segment (segments are
    filtered on `.strip() != ""`); falls back to `[text]` whole when there is no valid split
    point (0 or 1 non-empty parts found).
    """
    parts = [p for p in _SENT_SPLIT_RE.split(text) if p.strip() != ""]
    return parts if len(parts) > 1 else [text]


def gnmt_length_penalty(length: int, alpha: float) -> float:
    """GNMT length penalty lp(Y) = ((5 + |Y|) / 6) ** alpha (spec §7). `length` is the number of
    generated tokens (excluding BOS; conventionally including EOS, matching how `length` is
    computed at every call site below).
    """
    return ((5.0 + length) / 6.0) ** alpha


def decode_max_len(src_len: int, max_len_a: float = 1.5, max_len_b: int = 10) -> int:
    """max_len = floor(1.5 * src_len) + 10 (spec §7). `src_len` excludes padding, includes the
    appended source EOS (matches `src_mask.sum(dim=1)`, the loader's padding-mask convention).
    """
    return int(max_len_a * src_len) + max_len_b


def _blocked_tokens(history: list[int], n: int) -> set[int]:
    """Token ids that would complete an n-gram already present in `history` if appended next.

    `history` excludes BOS. With fewer than `n - 1` prior tokens no n-gram can yet be completed.
    Scans every existing n-gram in `history` for one sharing the trailing `n - 1` tokens as its
    prefix (an O(len(history)) scan per call -- fine at eval-set/dev decode lengths).
    """
    if n <= 0 or len(history) < n - 1:
        return set()
    prefix = tuple(history[-(n - 1) :]) if n > 1 else ()
    banned: set[int] = set()
    for i in range(len(history) - n + 1):
        if tuple(history[i : i + n - 1]) == prefix:
            banned.add(history[i + n - 1])
    return banned


@dataclass
class DecodeConfig:
    """Beam search hyperparameters (defaults from spec §7; its tuning grid overrides
    alpha/beam_size)."""

    beam_size: int = 5
    alpha: float = 0.6
    no_repeat_ngram_size: int = 3
    max_len_a: float = 1.5
    max_len_b: int = 10


@dataclass
class Hypothesis:
    """One decoded hypothesis. `tokens` excludes BOS; includes the trailing EOS only if the
    hypothesis actually terminated by emitting one (a hypothesis that hit `max_len` without ever
    emitting EOS has no trailing EOS token). `score` is the GNMT length-penalized log-probability.
    """

    tokens: list[int]
    score: float


def _step_log_probs(model: Transformer, cur_tok: Tensor, cache: dict) -> Tensor:
    """One decode step's (B, V) log-probabilities. A model that sets `step_returns_log_probs`
    (nmt.ensemble.Ensemble) already returns log-probs from `decode_step`, used as is (no second
    log_softmax: it would renormalize the averaged log-probs and break exact single-model parity);
    every other model returns logits and gets the log_softmax applied here, as before."""
    out = model.decode_step(cur_tok, cache)[:, 0, :]
    if getattr(model, "step_returns_log_probs", False):
        return out
    return torch.log_softmax(out, dim=-1)


def _repeat_memory(memory: Tensor | tuple[Tensor, ...], k: int) -> Tensor | tuple[Tensor, ...]:
    """`repeat_interleave(k, dim=0)` of an encoder memory; an ensemble's memory is a tuple with
    one tensor per member."""
    if isinstance(memory, tuple):
        return tuple(m.repeat_interleave(k, dim=0) for m in memory)
    return memory.repeat_interleave(k, dim=0)


def _reorder_cache(cache: dict, index: Tensor) -> None:
    """Reorder every per-beam tensor in a `Transformer.init_decode_cache` cache along dim 0 by
    `index` (new slot j pulls from old slot `index[j]`) -- self-attention K/V (the only state that
    actually differs across beams within one source example, since cross-attention K/V and the
    cross mask are identical across the K beams of a group and reordering them is a no-op).
    An ensemble cache (`{"members": [cache, ...]}`) reorders each member's cache.
    """
    if "members" in cache:
        for member_cache in cache["members"]:
            _reorder_cache(member_cache, index)
        return
    cache["memory_kv"] = [
        (k.index_select(0, index), v.index_select(0, index)) for k, v in cache["memory_kv"]
    ]
    for self_cache in cache["self_caches"]:
        if "k" in self_cache:
            self_cache["k"] = self_cache["k"].index_select(0, index)
            self_cache["v"] = self_cache["v"].index_select(0, index)
    cache["cross_mask"] = cache["cross_mask"].index_select(0, index)


def beam_search_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    cfg: DecodeConfig | None = None,
) -> list[Hypothesis]:
    """Batched beam search: one model.encode call plus one model.decode_step call per step across
    the *whole* (B * beam_size) beam array, with per-example GNMT length penalty, 3-gram
    repetition block and a per-example `max_len` derived from that example's own source length.
    Returns exactly one (the best) `Hypothesis` per source example, in input order.
    """
    nbest = beam_search_nbest(model, src, src_mask, bos_id, eos_id, pad_id, cfg)
    return [hyps[0] for hyps in nbest]


@torch.no_grad()
def beam_search_nbest(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    cfg: DecodeConfig | None = None,
) -> list[list[Hypothesis]]:
    """`beam_search_decode`'s search, returning every finished hypothesis per example (up to
    `beam_size`), best first (ties keep finish order, so element 0 is exactly the hypothesis
    `beam_search_decode` returns). An example that never finished returns its single best live beam.
    """
    cfg = cfg or DecodeConfig()
    device = src.device
    b_size, _ = src.shape
    k = cfg.beam_size
    was_training = model.training
    model.eval()

    memory = model.encode(src, src_mask)
    src_lens = src_mask.sum(dim=1).tolist()
    max_lens = [decode_max_len(int(n), cfg.max_len_a, cfg.max_len_b) for n in src_lens]

    memory_bk = _repeat_memory(memory, k)
    mask_bk = src_mask.repeat_interleave(k, dim=0)
    cache = model.init_decode_cache(memory_bk, mask_bk)

    tokens: list[list[int]] = [[bos_id] for _ in range(b_size * k)]
    scores = torch.full((b_size * k,), float("-inf"), device=device)
    for b in range(b_size):
        scores[b * k] = 0.0  # only beam 0 of each group starts finite: all K copies of BOS are
        # identical, so without this every group's first-step top-k would just pick the same
        # token K times over instead of the true top-K distinct continuations.

    finished: list[list[Hypothesis]] = [[] for _ in range(b_size)]
    active = [True] * b_size
    cur_tok = torch.full((b_size * k, 1), bos_id, dtype=torch.long, device=device)
    step = 0
    global_max_len = max(max_lens) if max_lens else 0

    while step < global_max_len and any(active):
        log_probs = _step_log_probs(model, cur_tok, cache)  # (B*K, V)
        vocab_size = log_probs.size(-1)

        for i in range(b_size * k):
            b = i // k
            if not active[b]:
                continue
            if cfg.no_repeat_ngram_size > 0:
                banned = _blocked_tokens(tokens[i][1:], cfg.no_repeat_ngram_size)
                if banned:
                    log_probs[i, list(banned)] = float("-inf")
            cur_len = len(tokens[i]) - 1  # generated so far, excluding BOS
            if cur_len + 1 >= max_lens[b]:
                # this beam must terminate on its next token: max_len is a hard cap (spec §7).
                forced = torch.full((vocab_size,), float("-inf"), device=device)
                forced[eos_id] = 0.0
                log_probs[i] = forced

        candidate_scores = (scores.unsqueeze(1) + log_probs).view(b_size, k * vocab_size)
        k_req = min(2 * k, k * vocab_size)
        top_scores, top_idx = candidate_scores.topk(k_req, dim=1)

        new_tokens: list[list[int]] = list(tokens)  # placeholder; every slot overwritten below
        new_scores = scores.clone()
        new_cache_index = torch.arange(b_size * k, device=device)

        for b in range(b_size):
            if not active[b]:
                continue
            target = k - len(finished[b])  # new active slots this group still needs
            filled = 0
            for c in range(k_req):
                if filled >= target:
                    break
                idx = int(top_idx[b, c].item())
                beam_local, token_id = idx // vocab_size, idx % vocab_size
                old_slot = b * k + beam_local
                seq = tokens[old_slot] + [token_id]
                sc = float(top_scores[b, c].item())
                if token_id == eos_id:
                    length = len(seq) - 1
                    finished[b].append(
                        Hypothesis(
                            tokens=seq[1:], score=sc / gnmt_length_penalty(length, cfg.alpha)
                        )
                    )
                    target -= 1
                    continue
                j = b * k + filled
                new_tokens[j] = seq
                new_scores[j] = sc
                new_cache_index[j] = old_slot
                filled += 1
            for slot in range(filled, k):
                j = b * k + slot
                new_tokens[j] = tokens[j]
                new_scores[j] = float("-inf")
                new_cache_index[j] = j
            if len(finished[b]) >= k:
                active[b] = False

        _reorder_cache(cache, new_cache_index)
        tokens = new_tokens
        scores = new_scores
        cur_tok = torch.tensor([[t[-1]] for t in tokens], dtype=torch.long, device=device)
        step += 1

    results: list[list[Hypothesis]] = []
    for b in range(b_size):
        if finished[b]:
            results.append(sorted(finished[b], key=lambda h: h.score, reverse=True))
        else:
            best_local = max(range(k), key=lambda loc: scores[b * k + loc].item())
            seq = tokens[b * k + best_local][1:]
            sc = float(scores[b * k + best_local].item())
            results.append(
                [Hypothesis(tokens=seq, score=sc / gnmt_length_penalty(len(seq), cfg.alpha))]
            )

    if was_training:
        model.train()
    return results


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len_a: float = 1.5,
    max_len_b: int = 10,
    no_repeat_ngram_size: int = 3,
) -> list[Hypothesis]:
    """Greedy (argmax) decode, batched across the whole source batch in one KV-cache pass per
    step -- an independent, simpler implementation from `beam_search_decode`; see module docstring
    for why beam_size=1 must (and does, per test) produce token-identical output to this.
    """
    del pad_id  # accepted for API symmetry with beam_search_decode; unused (no padding emitted)
    device = src.device
    b_size, _ = src.shape
    was_training = model.training
    model.eval()

    memory = model.encode(src, src_mask)
    src_lens = src_mask.sum(dim=1).tolist()
    max_lens = [decode_max_len(int(n), max_len_a, max_len_b) for n in src_lens]
    cache = model.init_decode_cache(memory, src_mask)

    tokens: list[list[int]] = [[bos_id] for _ in range(b_size)]
    finished = [False] * b_size
    scores = [0.0] * b_size
    cur_tok = torch.full((b_size, 1), bos_id, dtype=torch.long, device=device)

    step = 0
    global_max_len = max(max_lens) if max_lens else 0
    while step < global_max_len and not all(finished):
        log_probs = _step_log_probs(model, cur_tok, cache)
        next_tokens: list[int] = []
        for b in range(b_size):
            if finished[b]:
                next_tokens.append(eos_id)  # frozen; decode_step output discarded for this row
                continue
            row = log_probs[b].clone()
            if no_repeat_ngram_size > 0:
                banned = _blocked_tokens(tokens[b][1:], no_repeat_ngram_size)
                if banned:
                    row[list(banned)] = float("-inf")
            cur_len = len(tokens[b]) - 1
            if cur_len + 1 >= max_lens[b]:
                tok, lp = eos_id, 0.0
            else:
                tok = int(row.argmax().item())
                lp = float(row[tok].item())
            tokens[b].append(tok)
            scores[b] += lp
            if tok == eos_id:
                finished[b] = True
            next_tokens.append(tok)
        cur_tok = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
        step += 1

    if was_training:
        model.train()
    results = []
    for b in range(b_size):
        seq = tokens[b][1:]
        results.append(Hypothesis(tokens=seq, score=scores[b] / gnmt_length_penalty(len(seq), 1.0)))
    return results
