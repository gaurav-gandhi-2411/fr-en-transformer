from __future__ import annotations

# =============================================================================================
# EXPLORATORY / POST-HOC (not pre-registered): gap analysis v2 for main's final checkpoint.
#
# Why: the pre-registered OLS decomposition of the E1->E3 chrF gap (reports/final/*/analysis.json,
# `gap_decomposition`; nmt/analysis.py) left almost all of the gap in the residual domain term
# (main, seg_tuned: gap 12.67 chrF, residual 12.88). That result is reported UNCHANGED and is not
# recomputed or replaced here. v2 adds richer, post-hoc feature groups to see whether any of the
# residual can be attributed. Everything below was fixed BEFORE any v2 result was looked at.
#
# FIXED BEFORE LOOKING AT RESULTS
#   Training-target frequency bands. Count = occurrences in the English (target) side of the
#   training shards (HF dataset revision pinned in scripts/gap_v2_pull.py).
#     unseen   : count == 0
#     rare     : 1 <= count <= 9                       (RARE_MAX_COUNT)
#     top1k    : count >= 10 and frequency rank < 1000       (TOP_K)
#     mid_1k_10k: count >= 10 and 1000 <= rank < 10000       (MID_K)
#     tail_ge10: count >= 10 and rank >= 10000
#   rank = position in the descending-count order, ties broken by id (subwords) / by the word
#   string (words). The same thresholds are used for subword tokens (a) and words (b); (b)'s unit
#   is the lower-cased word core (see `word_core`) built from SentencePiece word-start markers.
#   Function vs content words: sklearn 1.7.2 ENGLISH_STOP_WORDS (318 words, BSD-3-Clause), vendored
#   at data/lexicons/english_stop_words_sklearn_1.7.2.txt. A word is "function" iff its lower-cased
#   alphanumeric core, split at apostrophes, has every piece in the list or in CLITIC_PIECES.
#   A subword token is "punct" iff its piece has no word character; otherwise it takes the
#   category of its word (function / content).
#   Alignment-noise thresholds on r = source words / reference words (whitespace words of the
#   normalised text): moderate = r < 0.5 or r > 2.0; severe = r < 0.33 or r > 3.0.
#   Register heuristics (word lists below): dialogue-start flag, quote density, pronoun rate,
#   first-person rate (English reference and French source).
#   Feature groups (<= 7): rarity (a,b), function_content (a), alignment (c), register (d),
#   fertility (e). Primary Shapley uses exactly these 5; a SENSITIVITY run adds a 6th control
#   group `length` (log source words), which is not part of (a)-(e).
#   Outcome: sentence chrF (official/score.py chrf_sentence) of main's seg_tuned predictions,
#   second outcome: per-sentence mean teacher-forced NLL (fp32, eval mode, no label smoothing,
#   EOS excluded). Bootstrap: 1000 stratified sentence resamples, seed 1234, refit each time.
# =============================================================================================
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm

from nmt.data.normalize import normalize_text
from nmt.evaluate import load_official_module, run_official_scorer
from scripts import gap_v2_stats as st

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL = st.LABEL
OUT_DIR = REPO_ROOT / "reports" / "final" / "gap_v2"
WORD_LIST_PATH = REPO_ROOT / "data" / "lexicons" / "english_stop_words_sklearn_1.7.2.txt"

TOP_K = 1000
MID_K = 10000
RARE_MAX_COUNT = 9
BANDS = ("unseen", "rare", "top1k", "mid_1k_10k", "tail_ge10")
CATEGORIES = ("function", "content", "punct")
RATIO_MODERATE = (0.5, 2.0)
RATIO_SEVERE = (0.33, 3.0)
SEED = 1234
N_BOOT = 1000

# n't stems and contraction suffixes: apostrophe-split pieces that count as function material
# (the sklearn list has none of them: "don't" -> "don", "t").
_CLITIC_TEXT = (
    "s t d ll ve m re don doesn didn isn aren wasn weren haven hasn hadn won wouldn couldn "
    "shouldn mustn needn ain"
)
CLITIC_PIECES = frozenset(_CLITIC_TEXT.split())
_PRONOUN_TEXT = (
    "i me my mine myself you your yours yourself he him his himself she her hers herself it its "
    "itself we us our ours ourselves they them their theirs themselves thou thee thy thine"
)
EN_PRONOUNS = frozenset(_PRONOUN_TEXT.split())
EN_FIRST_PERSON = frozenset({"i", "my", "me", "we"})  # "I'm" tokenises to i + m
FR_FIRST_PERSON = frozenset({"je", "j", "mon", "ma", "mes", "nous"})  # "j'ai" -> j + ai
DIALOGUE_START_RE = re.compile(r"^\s*(?:--|[-–—―]|\")")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_EDGE_RE = re.compile(r"^\W+|\W+$", re.UNICODE)

FEATURE_NAMES: dict[str, list[str]] = {
    "rarity": [
        "tok_share_top1k",
        "tok_share_rare",
        "tok_share_unseen",
        "word_share_unseen",
        "word_share_rare",
    ],
    "function_content": ["func_word_share", "punct_token_share"],
    "alignment": ["abs_log_word_ratio", "ratio_moderate_flag", "ratio_severe_flag"],
    "register": [
        "src_dialogue_start",
        "src_quote_density",
        "ref_quote_density",
        "ref_pronoun_rate",
        "ref_first_person_rate",
        "src_first_person_rate",
    ],
    "fertility": ["ref_fertility", "src_fertility"],
}
CONTROL_GROUP = {"length": ["log_src_words"]}


# ---------------------------------------------------------------------------------------------
# small pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------------------------


def load_word_list(path: Path = WORD_LIST_PATH) -> frozenset[str]:
    """Read the vendored one-word-per-line list; '#' lines are comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(ln.strip() for ln in lines if ln.strip() and not ln.startswith("#"))


def assign_bands(counts: Sequence[int], tiebreak: Sequence[Any]) -> np.ndarray:
    """Band index (into BANDS) per item from its training count. Items are ranked by descending
    count, ties by `tiebreak` ascending; rank only matters for items with count >= 10."""
    n = len(counts)
    order = sorted(range(n), key=lambda i: (-counts[i], tiebreak[i]))
    out = np.empty(n, dtype=np.int64)
    for rank, i in enumerate(order):
        c = counts[i]
        if c == 0:
            out[i] = 0
        elif c <= RARE_MAX_COUNT:
            out[i] = 1
        elif rank < TOP_K:
            out[i] = 2
        elif rank < MID_K:
            out[i] = 3
        else:
            out[i] = 4
    return out


def word_core(surface: str) -> str:
    """Lower-cased word with leading/trailing non-word characters stripped ('' if none left)."""
    return _EDGE_RE.sub("", surface.replace("▁", "")).lower()


def is_function_word(core: str, word_set: frozenset[str]) -> bool:
    """Function word iff every apostrophe-separated piece of `core` is in `word_set` or is a
    clitic piece. Empty core is not a word."""
    pieces = [p for p in core.split("'") if p]
    if not pieces:
        return False
    return all(p in word_set or p in CLITIC_PIECES for p in pieces)


def word_spans(start_flags: Sequence[bool]) -> list[tuple[int, int]]:
    """[start, end) spans of words given per-token word-start flags; token 0 always starts one."""
    starts = [i for i, f in enumerate(start_flags) if f or i == 0]
    ends = starts[1:] + [len(start_flags)]
    return list(zip(starts, ends, strict=True))


def ratio_flags(src_words: int, ref_words: int) -> dict[str, float]:
    """src/ref whitespace-word ratio and the moderate / severe extreme flags (declared
    thresholds). An empty reference is flagged severe (ratio set to +inf-like 99)."""
    r = src_words / ref_words if ref_words else 99.0
    mod = r < RATIO_MODERATE[0] or r > RATIO_MODERATE[1]
    sev = r < RATIO_SEVERE[0] or r > RATIO_SEVERE[1]
    return {"ratio": r, "moderate": float(mod), "severe": float(sev)}


def register_features(src: str, ref: str) -> dict[str, float]:
    """Heuristic register markers on NORMALISED text (normalize_text maps guillemets and curly
    quotes to '"' and apostrophes to "'")."""
    src_w = _WORD_RE.findall(src.lower())
    ref_w = _WORD_RE.findall(ref.lower())
    return {
        "src_dialogue_start": float(bool(DIALOGUE_START_RE.match(src))),
        "src_quote_density": src.count('"') / max(len(src), 1),
        "ref_quote_density": ref.count('"') / max(len(ref), 1),
        "ref_pronoun_rate": sum(w in EN_PRONOUNS for w in ref_w) / max(len(ref_w), 1),
        "ref_first_person_rate": sum(w in EN_FIRST_PERSON for w in ref_w) / max(len(ref_w), 1),
        "src_first_person_rate": sum(w in FR_FIRST_PERSON for w in src_w) / max(len(src_w), 1),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """UTF-8, LF line endings on every OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# tokenizer context and training-side counts
# ---------------------------------------------------------------------------------------------


class Tok:
    """SentencePiece + per-id tables (piece strings, word-start flags, punct flags)."""

    def __init__(self, sp: spm.SentencePieceProcessor) -> None:
        self.sp = sp
        n = sp.get_piece_size()
        self.pieces = [sp.id_to_piece(i) for i in range(n)]
        self.is_start = np.array([p.startswith("▁") for p in self.pieces], dtype=bool)
        byte_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
        self.is_punct = np.array(
            [
                (not byte_re.match(p)) and not _WORD_RE.search(p.replace("▁", ""))
                for p in self.pieces
            ],
            dtype=bool,
        )

    def surface(self, ids: Sequence[int]) -> str:
        return "".join(self.pieces[i] for i in ids)


def count_train_side(
    shard_files: Sequence[Path], key: str, tok: Tok, want_words: bool
) -> dict[str, Any]:
    """Token counts per id, token / word totals (word = word-start piece, i.e. whitespace word)
    and, if `want_words`, a Counter of lower-cased word cores, over the `key` side of the shards."""
    vocab = len(tok.pieces)
    tok_counts = np.zeros(vocab, dtype=np.int64)
    n_tokens = n_words = n_sent = 0
    word_counts: Counter[tuple[int, ...]] = Counter()
    for f in shard_files:
        with np.load(f) as z:
            ids = z[key].astype(np.int64)
            off = z[f"{key}_off"]
        tok_counts += np.bincount(ids, minlength=vocab)[:vocab]
        mask = tok.is_start[ids]
        mask[off[:-1][off[:-1] < len(ids)]] = True  # a sentence always opens a word
        n_tokens += len(ids)
        n_sent += len(off) - 1
        n_words += int(mask.sum())
        if want_words:
            pos = np.flatnonzero(mask).tolist()
            lst = ids.tolist()
            ends = pos[1:] + [len(lst)]
            for a, b in zip(pos, ends, strict=True):
                word_counts[tuple(lst[a:b])] += 1
    out: dict[str, Any] = {
        "tok_counts": tok_counts,
        "n_tokens": n_tokens,
        "n_words": n_words,
        "n_sent": n_sent,
    }
    if want_words:
        cores: Counter[str] = Counter()
        for ids_t, c in word_counts.items():
            core = word_core(tok.surface(ids_t))
            if core:
                cores[core] += c
        out["word_core_counts"] = cores
    return out


# ---------------------------------------------------------------------------------------------
# per-sentence features
# ---------------------------------------------------------------------------------------------


class FeatureContext:
    """Everything the per-sentence extractor needs (tokenizer, bands, word table, word list)."""

    def __init__(
        self,
        tok: Tok,
        tgt_tok_counts: np.ndarray,
        word_counts: Mapping[str, int],
        word_set: frozenset[str],
    ) -> None:
        self.tok, self.word_set = tok, word_set
        self.tok_band = assign_bands(list(tgt_tok_counts), list(range(len(tgt_tok_counts))))
        words = sorted(word_counts)
        wb = assign_bands([word_counts[w] for w in words], words)
        self.word_band = dict(zip(words, wb.tolist(), strict=True))

    def band_of_word(self, core: str) -> int:
        return self.word_band.get(core, 0)  # not in the training table at all = unseen


def sentence_features(src_raw: str, ref_raw: str, ctx: FeatureContext) -> dict[str, Any]:
    """Scalar features of one (source, reference) pair plus the reference's per-token band and
    category codes (aligned with the encoded reference, EOS excluded)."""
    src, ref = normalize_text(src_raw), normalize_text(ref_raw)
    tok = ctx.tok
    ref_ids = tok.sp.encode(ref, out_type=int)
    src_ids = tok.sp.encode(src, out_type=int)
    n_tok = max(len(ref_ids), 1)
    spans = word_spans([bool(tok.is_start[i]) for i in ref_ids]) if ref_ids else []
    tok_band = [int(ctx.tok_band[i]) for i in ref_ids]
    tok_cat = [2] * len(ref_ids)  # default punct
    word_bands: list[int] = []
    n_alpha_words = n_func_words = 0
    for a, b in spans:
        core = word_core(tok.surface(ref_ids[a:b]))
        if not core:
            continue  # pure-punctuation word: tokens stay 'punct'
        is_func = is_function_word(core, ctx.word_set)
        n_alpha_words += 1
        n_func_words += is_func
        word_bands.append(ctx.band_of_word(core))
        for j in range(a, b):
            if not tok.is_punct[ref_ids[j]]:
                tok_cat[j] = 0 if is_func else 1
    n_words_alpha = max(n_alpha_words, 1)
    src_ws, ref_ws = len(src.split()), len(ref.split())
    rf = ratio_flags(src_ws, ref_ws)
    src_starts = int(tok.is_start[src_ids].sum()) if src_ids else 0
    ref_starts = int(tok.is_start[ref_ids].sum()) if ref_ids else 0
    feats: dict[str, float] = {
        "tok_share_top1k": tok_band.count(2) / n_tok,
        "tok_share_rare": tok_band.count(1) / n_tok,
        "tok_share_unseen": tok_band.count(0) / n_tok,
        "word_share_unseen": word_bands.count(0) / n_words_alpha,
        "word_share_rare": word_bands.count(1) / n_words_alpha,
        "func_word_share": n_func_words / n_words_alpha,
        "punct_token_share": tok_cat.count(2) / n_tok,
        "abs_log_word_ratio": abs(math.log(rf["ratio"])) if rf["ratio"] > 0 else 0.0,
        "ratio_moderate_flag": rf["moderate"],
        "ratio_severe_flag": rf["severe"],
        "ref_fertility": len(ref_ids) / max(ref_starts, 1),
        "src_fertility": len(src_ids) / max(src_starts, 1),
        "log_src_words": math.log(max(src_ws, 1)),
        **register_features(src, ref),
    }
    return {
        "feats": feats,
        "ref_ids": ref_ids,
        "src_ids": src_ids,
        "tok_band": tok_band,
        "tok_cat": tok_cat,
        "word_bands": word_bands,
        "n_words_ws_src": src_ws,
        "n_words_ws_ref": ref_ws,
        "ratio": rf["ratio"],
    }


# ---------------------------------------------------------------------------------------------
# teacher-forced NLL
# ---------------------------------------------------------------------------------------------


def teacher_forced_nll(
    model: Any,
    src_ids: Sequence[Sequence[int]],
    ref_ids: Sequence[Sequence[int]],
    batch_size: int = 32,
) -> list[tuple[np.ndarray, float]]:
    """Per reference token NLL (natural log, fp32, eval mode, no label smoothing, no dropout) of
    every sentence, plus the EOS NLL. Returns [(nll over content tokens, eos_nll), ...] in input
    order. Source gets EOS appended, decoder input is BOS + reference (nmt.data.loader contract)."""
    import torch
    import torch.nn.functional as fn

    cfg = model.config
    model.eval()
    order = sorted(range(len(src_ids)), key=lambda i: len(src_ids[i]) + len(ref_ids[i]))
    out: list[tuple[np.ndarray, float] | None] = [None] * len(src_ids)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            srcs = [list(src_ids[i]) + [cfg.eos_id] for i in idx]
            tin = [[cfg.bos_id] + list(ref_ids[i]) for i in idx]
            tout = [list(ref_ids[i]) + [cfg.eos_id] for i in idx]

            def pad(seqs: list[list[int]]) -> Any:
                m = max(len(q) for q in seqs)
                t = torch.full((len(seqs), m), cfg.pad_id, dtype=torch.long)
                for k, q in enumerate(seqs):
                    t[k, : len(q)] = torch.tensor(q, dtype=torch.long)
                return t

            logits = model(pad(srcs), pad(tin)).float()
            logp = fn.log_softmax(logits, dim=-1)
            tgt = pad(tout)
            nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for k, i in enumerate(idx):
                n = len(tout[k])
                row = nll[k, :n].numpy().astype(np.float64)
                out[i] = (row[:-1], float(row[-1]))
    return [o for o in out if o is not None]


# ---------------------------------------------------------------------------------------------
# NLL band / category breakdown (descriptive, post-hoc)
# ---------------------------------------------------------------------------------------------


def mix_rate_breakdown(
    sums: np.ndarray, counts: np.ndarray, domain: np.ndarray, labels: Sequence[str]
) -> dict[str, Any]:
    """Pooled-token mean NLL by class. sums/counts: (sentences x classes) NLL sums and token
    counts. gap = mean(E3) - mean(E1) (positive = E3 harder) splits exactly as
        sum_b (s3_b - s1_b) * n1_b   [composition / mix: E3 has a different class mix]
      + sum_b  s3_b * (n3_b - n1_b)  [rate: higher NLL inside the class]
    with s = token share, n = class mean NLL (a class empty in E1 uses n1 := n3)."""
    e1, e3 = domain == 0, domain == 1
    c1, c3 = counts[e1].sum(axis=0), counts[e3].sum(axis=0)
    t1, t3 = sums[e1].sum(axis=0), sums[e3].sum(axis=0)
    s1, s3 = c1 / c1.sum(), c3 / c3.sum()
    n3 = np.divide(t3, c3, out=np.zeros_like(t3), where=c3 > 0)
    n1 = np.divide(t1, c1, out=n3.copy(), where=c1 > 0)
    mix, rate = (s3 - s1) * n1, s3 * (n3 - n1)
    return {
        "labels": list(labels),
        "token_share_e1": s1.tolist(),
        "token_share_e3": s3.tolist(),
        "mean_nll_e1": n1.tolist(),
        "mean_nll_e3": n3.tolist(),
        "mix_term": mix.tolist(),
        "rate_term": rate.tolist(),
        "gap_e3_minus_e1": float(t3.sum() / c3.sum() - t1.sum() / c1.sum()),
        "check_sum_terms": float(mix.sum() + rate.sum()),
        "n_tokens_e1": c1.tolist(),
        "n_tokens_e3": c3.tolist(),
    }


def bootstrap_breakdown(
    sums: np.ndarray,
    counts: np.ndarray,
    domain: np.ndarray,
    labels: Sequence[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile 95% CIs of the mix / rate terms and the pooled gap (stratified sentence
    resampling, as in the decomposition bootstrap)."""
    rng = np.random.default_rng(seed)
    mix, rate, gap = [], [], []
    for _ in range(n_boot):
        idx = st.stratified_resample(domain, rng)
        r = mix_rate_breakdown(sums[idx], counts[idx], domain[idx], labels)
        mix.append(r["mix_term"])
        rate.append(r["rate_term"])
        gap.append(r["gap_e3_minus_e1"])
    mix_a, rate_a, gap_a = np.array(mix), np.array(rate), np.array(gap)

    def ci(a: np.ndarray) -> list[list[float]] | list[float]:
        lo, hi = np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)
        return [lo.tolist(), hi.tolist()] if a.ndim > 1 else [float(lo), float(hi)]

    return {"mix_ci": ci(mix_a), "rate_ci": ci(rate_a), "gap_ci": ci(gap_a)}


# ---------------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _summ(a: np.ndarray) -> dict[str, float]:
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "median": float(np.median(a))}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--pulls", type=Path, required=True, help="dest of scripts.gap_v2_pull")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    import torch

    t_start = time.perf_counter()
    timing: dict[str, float] = {}

    def lap(name: str, t0: float) -> None:
        timing[name] = round(time.perf_counter() - t0, 2)

    # ---- provenance / verification ---------------------------------------------------------
    t0 = time.perf_counter()
    pulls = json.loads((args.pulls / "pull_records.json").read_text(encoding="utf-8"))
    ev, data_rec = pulls["eval"], pulls["data"]
    assert ev["verified"] is True and ev["private"] is True and ev["model_files_checked"] is True
    assert re.fullmatch(r"[0-9a-f]{40}", ev["hf_revision"])
    assert data_rec["private"] is True and re.fullmatch(r"[0-9a-f]{40}", data_rec["hf_revision"])
    committed_pull = json.loads(
        (REPO_ROOT / "reports/final/main/pull_record.json").read_text(encoding="utf-8")
    )
    assert committed_pull["hf_revision"] == ev["hf_revision"]
    data_dir = args.pulls / "data_repo"
    model_dir = args.pulls / "eval_main/source/runs/main/model"
    manifest = json.loads((REPO_ROOT / "data/data_manifest.json").read_text(encoding="utf-8"))
    sha_inputs: dict[str, str] = {}
    for rel, want in manifest["output_files_sha256"].items():
        if rel.startswith("data/eval/e1/") or rel.startswith("data/eval/e3/"):
            got_repo = sha256_file(REPO_ROOT / rel)
            got_hf = sha256_file(data_dir / rel)
            assert got_repo == want == got_hf, f"{rel}: repo/manifest/HF sha mismatch"
            sha_inputs[rel] = got_repo
    spm_shas = {
        "tokenizer/spm.model (repo)": sha256_file(REPO_ROOT / "tokenizer/spm.model"),
        "tokenizer/spm.model (HF data repo)": sha256_file(data_dir / "tokenizer/spm.model"),
        "spm.model (HF eval repo, model/)": sha256_file(model_dir / "spm.model"),
    }
    assert len(set(spm_shas.values())) == 1, spm_shas
    lap("verify", t0)

    # ---- data, predictions, chrF -----------------------------------------------------------
    t0 = time.perf_counter()
    rows: dict[str, list[dict[str, str]]] = {}
    pred_shas: dict[str, str] = {}
    module = load_official_module()
    seg = REPO_ROOT / "reports/final/main/seg_tuned"
    official_check: dict[str, Any] = {}
    for sname in ("e1", "e3"):
        inputs = {r["id"]: r for r in _read_jsonl(REPO_ROOT / f"data/eval/{sname}/inputs.jsonl")}
        labels = _read_jsonl(REPO_ROOT / f"data/eval/{sname}/labels.jsonl")
        pred_path = seg / f"{sname}_predictions.json"
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        sha = sha256_file(pred_path)
        want = committed_pull["files"][f"predictions/seg_tuned/{sname}_predictions.json"]["sha256"]
        got_fresh = ev["files"][f"predictions/seg_tuned/{sname}_predictions.json"]["sha256"]
        assert sha == want == got_fresh, f"{sname} predictions sha mismatch"
        pred_shas[f"reports/final/main/seg_tuned/{sname}_predictions.json"] = sha
        rows[sname] = [
            {
                "id": lab["id"],
                "source": inputs[lab["id"]]["source"],
                "reference": lab["reference"],
                "hyp": pred[lab["id"]],
            }
            for lab in labels
        ]
        # the official scorer, as shipped (PYTHONUTF8=1 via run_official_scorer)
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "o.json"
            run_official_scorer(REPO_ROOT / f"data/eval/{sname}/labels.jsonl", pred_path, out_json)
            off = json.loads(out_json.read_text(encoding="utf-8"))
        prev = json.loads((seg / f"official_{sname}.json").read_text(encoding="utf-8"))
        assert off["all"] == prev["all"], f"{sname}: official scorer differs from committed report"
        official_check[sname] = {"official_chrf": off["all"]["chrf"], "n": off["all"]["n"]}
    all_rows = rows["e1"] + rows["e3"]
    domain = np.array([0.0] * len(rows["e1"]) + [1.0] * len(rows["e3"]))
    chrf = np.array([float(module.chrf_sentence(r["hyp"], r["reference"])) for r in all_rows])
    prev_an = json.loads((seg / "analysis.json").read_text(encoding="utf-8"))
    prereg_gap = prev_an["gap_decomposition"]["total_gap_chrf_e1_minus_e3"]
    my_gap = float(chrf[domain == 0].mean() - chrf[domain == 1].mean())
    assert abs(my_gap - prereg_gap) < 1e-9, (my_gap, prereg_gap)
    lap("data_chrf", t0)

    # ---- training-side counts --------------------------------------------------------------
    t0 = time.perf_counter()
    sp = spm.SentencePieceProcessor()
    sp.load(str(REPO_ROOT / "tokenizer/spm.model"))
    tok = Tok(sp)
    shard_files = sorted((data_dir / "data/shards/train").glob("shard_*.npz"))
    tgt_side = count_train_side(shard_files, "tgt", tok, want_words=True)
    src_side = count_train_side(shard_files, "src", tok, want_words=False)
    freq_committed = np.load(REPO_ROOT / "tokenizer/freq_tgt.npy")
    freq_hf = np.load(data_dir / "tokenizer/freq_tgt.npy")
    assert np.array_equal(tgt_side["tok_counts"], freq_committed)
    assert np.array_equal(freq_committed, freq_hf)
    word_set = load_word_list()
    ctx = FeatureContext(tok, tgt_side["tok_counts"], tgt_side["word_core_counts"], word_set)
    train_info = {
        "n_train_pairs": tgt_side["n_sent"],
        "tgt_tokens": tgt_side["n_tokens"],
        "tgt_words_ws": tgt_side["n_words"],
        "src_tokens": src_side["n_tokens"],
        "src_words_ws": src_side["n_words"],
        "tgt_word_types_core": len(tgt_side["word_core_counts"]),
        "tgt_tok_counts_equal_committed_freq_tgt_npy": True,
        "band_sizes_subword_types": {
            b: int((ctx.tok_band == i).sum()) for i, b in enumerate(BANDS)
        },
        "band_sizes_word_types": {
            b: sum(1 for v in ctx.word_band.values() if v == i) for i, b in enumerate(BANDS)
        },
    }
    lap("train_counts", t0)

    # ---- per-sentence features -------------------------------------------------------------
    t0 = time.perf_counter()
    per = [sentence_features(r["source"], r["reference"], ctx) for r in all_rows]
    names_all = [n for g in FEATURE_NAMES.values() for n in g] + CONTROL_GROUP["length"]
    fmat = {n: np.array([p["feats"][n] for p in per]) for n in names_all}
    lap("features", t0)

    # ---- NLL -------------------------------------------------------------------------------
    t0 = time.perf_counter()
    from nmt.hub import load_pretrained

    model, _ = load_pretrained(str(model_dir))
    model = model.float().eval()
    max_len = model.config.max_len
    assert all(len(p["ref_ids"]) + 1 <= max_len and len(p["src_ids"]) + 1 <= max_len for p in per)
    nll = teacher_forced_nll(model, [p["src_ids"] for p in per], [p["ref_ids"] for p in per])
    # padding-invariance spot check: 8 sentences scored alone vs inside batches
    solo = [
        teacher_forced_nll(model, [per[i]["src_ids"]], [per[i]["ref_ids"]], batch_size=1)[0][0]
        for i in range(0, len(per), len(per) // 8)
    ]
    pad_diff = max(
        float(np.abs(a - nll[i][0]).max())
        for a, i in zip(solo, range(0, len(per), len(per) // 8), strict=True)
    )
    n_params = int(sum(p.numel() for p in model.parameters()))
    lap("nll", t0)
    sent_nll = np.array([x[0].mean() if len(x[0]) else 0.0 for x in nll])
    eos_nll = np.array([x[1] for x in nll])
    n_tok = np.array([len(x[0]) for x in nll])
    for p, (row, _) in zip(per, nll, strict=True):
        assert len(row) == len(p["ref_ids"])

    def class_arrays(code_key: str, n_cls: int) -> tuple[np.ndarray, np.ndarray]:
        sums = np.zeros((len(per), n_cls))
        cnts = np.zeros((len(per), n_cls))
        for i, (p, (row, _)) in enumerate(zip(per, nll, strict=True)):
            for c, v in zip(p[code_key], row, strict=True):
                sums[i, c] += v
                cnts[i, c] += 1
        return sums, cnts

    t0 = time.perf_counter()
    nll_break: dict[str, Any] = {"label": LABEL}
    for key, labels, code in (
        ("by_frequency_band", BANDS, "tok_band"),
        ("by_word_category", CATEGORIES, "tok_cat"),
    ):
        sums, cnts = class_arrays(code, len(labels))
        res = mix_rate_breakdown(sums, cnts, domain, labels)
        res["bootstrap"] = bootstrap_breakdown(sums, cnts, domain, labels, args.n_boot, args.seed)
        nll_break[key] = res
    e1m, e3m = domain == 0, domain == 1
    tok_sum = np.array([x[0].sum() for x in nll])
    nll_break["overall"] = {
        "pooled_token_mean_nll_e1": float(tok_sum[e1m].sum() / n_tok[e1m].sum()),
        "pooled_token_mean_nll_e3": float(tok_sum[e3m].sum() / n_tok[e3m].sum()),
        "per_sentence_mean_nll_e1": _summ(sent_nll[e1m]),
        "per_sentence_mean_nll_e3": _summ(sent_nll[e3m]),
        "eos_nll_mean_e1": float(eos_nll[e1m].mean()),
        "eos_nll_mean_e3": float(eos_nll[e3m].mean()),
        "n_sentences": [int(e1m.sum()), int(e3m.sum())],
        "n_content_tokens": [int(n_tok[e1m].sum()), int(n_tok[e3m].sum())],
        "definition": "natural-log NLL of reference tokens, teacher forced; EOS excluded",
    }
    lap("nll_breakdown", t0)

    # ---- decomposition ---------------------------------------------------------------------
    groups = {g: np.column_stack([fmat[n] for n in ns]) for g, ns in FEATURE_NAMES.items()}
    groups_ctrl = {**groups, "length": fmat["log_src_words"].reshape(-1, 1)}
    fnames = {**FEATURE_NAMES, **CONTROL_GROUP}
    shares: dict[str, Any] = {
        "label": LABEL,
        "definition": "scripts/gap_v2_stats.py header: Shapley over the mean difference",
        "n": {"e1": int(e1m.sum()), "e3": int(e3m.sum())},
        "preregistered_reference": {
            "source": "reports/final/main/seg_tuned/analysis.json",
            "key": "gap_decomposition",
            "total_gap_chrf_e1_minus_e3": prereg_gap,
            "residual_domain_chrf": prev_an["gap_decomposition"]["contributions_chrf"][
                "residual_domain"
            ],
            "r_squared": prev_an["gap_decomposition"]["r_squared"],
            "note": "cited unchanged, not recomputed or replaced; my_gap reproduces it",
            "v2_reproduced_gap": my_gap,
        },
    }
    boot_rt: dict[str, float] = {}
    for variant, gdict, ycol_name, y in (
        ("primary_5groups", groups, "chrf", chrf),
        ("sensitivity_6groups_plus_length_control", groups_ctrl, "chrf", chrf),
        ("second_outcome_nll_5groups", groups, "sentence_mean_nll", sent_nll),
    ):
        t1 = time.perf_counter()
        dec = st.decompose(y, gdict, domain)
        boot = st.bootstrap_decompose(y, gdict, domain, args.n_boot, args.seed)
        boot_rt[variant] = round(time.perf_counter() - t1, 1)
        shares[variant] = {
            "outcome": ycol_name,
            "gap_e1_minus_e3": dec["gap"],
            "groups": {
                g: {
                    "contribution": dec["contribution"][g],
                    "contribution_ci95": boot["contribution_ci"][g],
                    "share_of_gap": dec["contribution"][g] / dec["gap"],
                    "share_ci95": boot["share_ci"][g],
                    "r2_shapley": dec["r2_shapley"][g],
                    "r2_shapley_ci95": boot["r2_shapley_ci"][g],
                }
                for g in gdict
            },
            "explained_total": dec["explained_total"],
            "explained_share": dec["explained_total"] / dec["gap"],
            "residual": dec["residual"],
            "residual_ci95": boot["residual_ci"],
            "residual_share": dec["residual"] / dec["gap"],
            "residual_share_ci95": boot["residual_share_ci"],
            "gap_ci95": boot["gap_ci"],
            "r2_full": dec["r2_full"],
            "r2_domain_only": dec["r2_domain_only"],
            "bootstrap": {"n_boot": boot["n_boot"], "seed": boot["seed"], "strata": "domain"},
        }
        shares[variant]["sum_check"] = {
            "sum_group_contributions_plus_residual_minus_gap": sum(dec["contribution"].values())
            + dec["residual"]
            - dec["gap"]
        }
    shares["bootstrap_runtime_seconds"] = boot_rt
    ols = {
        "label": LABEL,
        "chrf_full_model_5groups": st.hc3_table(chrf, groups, fnames, domain),
        "nll_full_model_5groups": st.hc3_table(sent_nll, groups, fnames, domain),
        "note": "standardised features (coef per 1 SD), HC3 robust SEs (statsmodels)",
    }
    coll = {"label": LABEL, **st.collinearity_report(groups, fnames, domain)}
    coll["group_ctrl_condition"] = st.collinearity_report(groups_ctrl, fnames, domain)[
        "condition_number_standardised_design"
    ]
    lap("decomposition", t0)

    # ---- feature summary (descriptive) -----------------------------------------------------
    ncm = {"e1": e1m, "e3": e3m}
    feat_summary: dict[str, Any] = {"label": LABEL, "features": {}}
    for n in names_all:
        feat_summary["features"][n] = {d: _summ(fmat[n][m]) for d, m in ncm.items()} | {
            "mean_diff_e1_minus_e3": float(fmat[n][e1m].mean() - fmat[n][e3m].mean())
        }
    ratio = np.array([p["ratio"] for p in per])
    ref_tok = np.array([len(p["ref_ids"]) for p in per])
    src_tok = np.array([len(p["src_ids"]) for p in per])
    ref_w = np.array([p["n_words_ws_ref"] for p in per])
    src_w = np.array([p["n_words_ws_src"] for p in per])
    ext = {}
    for d, m in ncm.items():
        mod = ratio[m]
        ext[d] = {
            "n": int(m.sum()),
            "moderate_rate": float(((mod < RATIO_MODERATE[0]) | (mod > RATIO_MODERATE[1])).mean()),
            "severe_rate": float(((mod < RATIO_SEVERE[0]) | (mod > RATIO_SEVERE[1])).mean()),
            "ratio_median": float(np.median(mod)),
            "mean_chrf_moderate": float(
                chrf[m][(mod < RATIO_MODERATE[0]) | (mod > RATIO_MODERATE[1])].mean()
            )
            if ((mod < RATIO_MODERATE[0]) | (mod > RATIO_MODERATE[1])).any()
            else None,
            "mean_chrf_non_extreme": float(
                chrf[m][~((mod < RATIO_MODERATE[0]) | (mod > RATIO_MODERATE[1]))].mean()
            ),
        }
    artifact = {}
    for run in ("main", "s1_sin_l4", "s2_rope_l4", "s3_rope_concat_l4"):
        for var in ("seg_off", "seg_tuned"):
            a = json.loads(
                (REPO_ROOT / f"reports/final/{run}/{var}/analysis.json").read_text(encoding="utf-8")
            )["metric_artifact_share"]
            artifact[f"{run}/{var}"] = {
                "e3_delta_chrf": a["e3"]["delta_chrf"],
                "dev_unseen_delta_chrf": a["dev_unseen_domain"]["delta_chrf"],
            }
    feat_summary["alignment_noise"] = {
        "thresholds": {"moderate": RATIO_MODERATE, "severe": RATIO_SEVERE, "ratio": "src/ref ws"},
        "by_domain": ext,
        "reference_normalisation_artifact": {
            "source": "reports/final/<run>/<variant>/analysis.json",
            "key": "metric_artifact_share.{e3,dev_unseen_domain}.delta_chrf",
            "values": artifact,
        },
    }
    nov: dict[str, Any] = {}
    for d, m in ncm.items():
        tb = np.array([c for p, mm in zip(per, m, strict=True) if mm for c in p["tok_band"]])
        wb = [c for p, mm in zip(per, m, strict=True) if mm for c in p["word_bands"]]
        types: dict[str, int] = {}
        for p, mm in zip(per, m, strict=True):
            if not mm:
                continue
            for a_, b_ in word_spans([bool(tok.is_start[i]) for i in p["ref_ids"]]):
                core = word_core(tok.surface(p["ref_ids"][a_:b_]))
                if core:
                    types[core] = ctx.band_of_word(core)
        tv = np.array(list(types.values()))
        nov[d] = {
            "subword_token_share_by_band": {
                b: float((tb == i).mean()) for i, b in enumerate(BANDS)
            },
            "word_token_share_by_band": {
                b: float((np.array(wb) == i).mean()) for i, b in enumerate(BANDS)
            },
            "word_type_share_by_band": {b: float((tv == i).mean()) for i, b in enumerate(BANDS)},
            "n_word_tokens": len(wb),
            "n_word_types": int(len(tv)),
        }
    feat_summary["target_novelty_and_bands"] = {
        "bands": BANDS,
        "thresholds": {"top_k": TOP_K, "mid_k": MID_K, "rare_max_count": RARE_MAX_COUNT},
        "by_domain": nov,
        "train": train_info,
    }
    feat_summary["fertility_subword_tokens_per_whitespace_word"] = {
        "e1_ref_pooled": float(ref_tok[e1m].sum() / ref_w[e1m].sum()),
        "e3_ref_pooled": float(ref_tok[e3m].sum() / ref_w[e3m].sum()),
        "train_tgt_pooled": tgt_side["n_tokens"] / tgt_side["n_words"],
        "e1_src_pooled": float(src_tok[e1m].sum() / src_w[e1m].sum()),
        "e3_src_pooled": float(src_tok[e3m].sum() / src_w[e3m].sum()),
        "train_src_pooled": src_side["n_tokens"] / src_side["n_words"],
        "per_sentence_mean_ref": {
            "e1": float(fmat["ref_fertility"][e1m].mean()),
            "e3": float(fmat["ref_fertility"][e3m].mean()),
        },
        "prior_claim_in_task_brief": "E3 1.71 vs train 1.54 (source file not found in repo)",
    }
    feat_summary["word_lists"] = {
        "function_words": f"{WORD_LIST_PATH.relative_to(REPO_ROOT).as_posix()} "
        f"(sha256 {sha256_file(WORD_LIST_PATH)})",
        "clitic_pieces": sorted(CLITIC_PIECES),
        "en_pronouns": sorted(EN_PRONOUNS),
        "en_first_person": sorted(EN_FIRST_PERSON),
        "fr_first_person": sorted(FR_FIRST_PERSON),
        "dialogue_start_regex": DIALOGUE_START_RE.pattern,
    }
    feat_summary["official_scorer_check"] = official_check
    feat_summary["nll_model_checks"] = {
        "n_params": n_params,
        "max_abs_diff_padded_vs_single_nll": pad_diff,
        "torch": torch.__version__,
    }

    # ---- write ------------------------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", "scripts", "nmt", "data", "tokenizer"))
    timing["total"] = round(time.perf_counter() - t_start, 2)
    provenance = {
        "label": LABEL,
        "code_commit_sha": head,
        "code_tree_dirty": dirty,
        "command": "uv run python -m scripts.gap_v2 --pulls <dest of scripts.gap_v2_pull>",
        "eval_repo": {k: ev[k] for k in ("hf_repo", "hf_revision", "private", "manifest_sha256")},
        "eval_model_files_verified": ev["model_files_checked"],
        "eval_model_sha256": {
            k: v["sha256"] for k, v in ev["files"].items() if k.startswith("model/")
        },
        "data_repo": {k: data_rec[k] for k in ("hf_repo", "hf_revision", "private")},
        "data_repo_files_sha256": data_rec["files_sha256"],
        "input_files_sha256": {
            **sha_inputs,
            **pred_shas,
            "spm.model": next(iter(spm_shas.values())),
        },
        "word_list_sha256": sha256_file(WORD_LIST_PATH),
        "runtime_seconds": timing,
        "python": sys.version.split()[0],
        "seed": args.seed,
        "n_boot": args.n_boot,
    }
    for name, obj in (
        ("features_summary.json", feat_summary),
        ("nll_breakdown.json", nll_break),
        ("gap_shares.json", shares),
        ("ols_hc3.json", ols),
        ("collinearity.json", coll),
        ("provenance.json", provenance),
    ):
        write_json(args.out / name, obj)
    make_figure(shares, nll_break, args.out)
    write_readme(args.out, shares, nll_break, feat_summary, coll, provenance)
    print(json.dumps(timing))
    return 0


# ---------------------------------------------------------------------------------------------
# figure + README
# ---------------------------------------------------------------------------------------------


def make_figure(shares: dict[str, Any], nll_break: dict[str, Any], out: Path) -> None:
    """Gap components with bootstrap 95% CIs: chrF gap (left) and, descriptively, the pooled
    token-NLL gap by frequency band split into mix and rate terms (right)."""
    prim = shares["primary_5groups"]
    names = list(prim["groups"]) + ["unexplained residual"]
    vals = [prim["groups"][g]["contribution"] for g in prim["groups"]] + [prim["residual"]]
    cis = [prim["groups"][g]["contribution_ci95"] for g in prim["groups"]] + [prim["residual_ci95"]]
    err = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(vals, cis, strict=True)]).T
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2))
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, xerr=err, color=["#4C78A8"] * (len(names) - 1) + ["#999999"], capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"chrF points of the E1-E3 gap ({prim['gap_e1_minus_e3']:.2f}); 95% bootstrap CI")
    ax.set_title("Shapley share of the chrF gap by feature group")
    band = nll_break["by_frequency_band"]
    b = band["bootstrap"]
    x = np.arange(len(band["labels"]))
    mix, rate = np.array(band["mix_term"]), np.array(band["rate_term"])
    for off, arr, ci, lab, col in (
        (-0.2, mix, b["mix_ci"], "composition (mix) term", "#F58518"),
        (0.2, rate, b["rate_ci"], "within-band (rate) term", "#54A24B"),
    ):
        e = np.array([arr - np.array(ci[0]), np.array(ci[1]) - arr])
        bx.bar(x + off, arr, width=0.4, yerr=e, capsize=3, label=lab, color=col)
    bx.set_xticks(x)
    bx.set_xticklabels(band["labels"], rotation=20)
    bx.axhline(0, color="k", lw=0.8)
    bx.set_ylabel(f"nats per token (pooled gap E3-E1 = {band['gap_e3_minus_e1']:.2f})")
    bx.set_title("Token-NLL gap by training-frequency band")
    bx.legend()
    fig.suptitle(f"{LABEL}: E1-E3 gap analysis v2 (main, final checkpoint)", fontsize=12)
    fig.text(
        0.5,
        0.005,
        f"{LABEL}. Left: chrF gap split by Shapley over 5 feature groups (heuristic features); "
        "the grey bar is the unexplained residual. Right: descriptive NLL mix/rate split. "
        "Bars show 95% stratified sentence-bootstrap CIs (1000 resamples, seed 1234).",
        ha="center",
        fontsize=7.5,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "gap_v2_components.png", dpi=150)
    fig.savefig(out / "gap_v2_components.svg", metadata={"Date": None})
    plt.close(fig)


def write_readme(
    out: Path,
    shares: dict[str, Any],
    nll_break: dict[str, Any],
    feat: dict[str, Any],
    coll: dict[str, Any],
    prov: dict[str, Any],
) -> None:
    """README.md: label, hand-written findings (scripts/gap_v2_findings.md), generated tables,
    provenance footer. LF line endings."""
    prim = shares["primary_5groups"]
    sens = shares["sensitivity_6groups_plus_length_control"]
    lines = [
        f"# Gap analysis v2 -- {LABEL}",
        "",
        f"> **{LABEL}.** Post-hoc, run after the pre-registered OLS decomposition left almost all "
        "of the E1->E3 gap unexplained. The pre-registered result is cited unchanged "
        f"(`reports/final/main/seg_tuned/analysis.json`, `gap_decomposition`: gap "
        f"{shares['preregistered_reference']['total_gap_chrf_e1_minus_e3']:.2f} chrF, residual "
        f"{shares['preregistered_reference']['residual_domain_chrf']:.2f}). All features below "
        "are heuristics defined in `scripts/gap_v2.py` before results were inspected.",
        "",
        "## Findings",
        "",
        (REPO_ROOT / "scripts" / "gap_v2_findings.md").read_text(encoding="utf-8").strip(),
        "",
        f"## Table 1 -- Shapley share of the E1-E3 chrF gap ({LABEL})",
        "",
        "Gap = mean sentence chrF(E1) - mean sentence chrF(E3) = "
        f"{prim['gap_e1_minus_e3']:.2f} (n = {shares['n']['e1']} + {shares['n']['e3']}). Share = "
        "Shapley value of the group's mean-difference contribution b_k * (xbar_E1 - xbar_E3) in "
        "OLS with a domain dummy (exact over all 2^5 subsets); the groups' contributions plus the "
        "residual sum to the gap exactly. 95% CIs: 1000 stratified sentence-bootstrap refits, "
        "seed 1234.",
        "",
        "| group | chrF points [95% CI] | share of gap [95% CI] | Shapley R^2 [95% CI] |",
        "|---|---|---|---|",
    ]
    for g, v in prim["groups"].items():
        lines.append(
            f"| {g} | {v['contribution']:+.2f} [{v['contribution_ci95'][0]:+.2f}, "
            f"{v['contribution_ci95'][1]:+.2f}] | {v['share_of_gap'] * 100:+.1f}% "
            f"[{v['share_ci95'][0] * 100:+.1f}, {v['share_ci95'][1] * 100:+.1f}] | "
            f"{v['r2_shapley']:.4f} [{v['r2_shapley_ci95'][0]:.4f}, "
            f"{v['r2_shapley_ci95'][1]:.4f}] |"
        )
    lines += [
        f"| **explained (all groups)** | {prim['explained_total']:+.2f} | "
        f"{prim['explained_share'] * 100:+.1f}% | R^2 {prim['r2_full']:.4f} "
        f"(domain-only {prim['r2_domain_only']:.4f}) |",
        f"| **unexplained residual (domain)** | {prim['residual']:+.2f} "
        f"[{prim['residual_ci95'][0]:+.2f}, {prim['residual_ci95'][1]:+.2f}] | "
        f"{prim['residual_share'] * 100:+.1f}% [{prim['residual_share_ci95'][0] * 100:+.1f}, "
        f"{prim['residual_share_ci95'][1] * 100:+.1f}] | - |",
        "",
        "Sensitivity (adds a `length` control group, log source words; not part of (a)-(e)): "
        f"explained {sens['explained_total']:+.2f} chrF "
        f"({sens['explained_share'] * 100:+.1f}%), residual {sens['residual']:+.2f} "
        f"[{sens['residual_ci95'][0]:+.2f}, {sens['residual_ci95'][1]:+.2f}]; length "
        f"{sens['groups']['length']['contribution']:+.2f} "
        f"[{sens['groups']['length']['contribution_ci95'][0]:+.2f}, "
        f"{sens['groups']['length']['contribution_ci95'][1]:+.2f}].",
        "",
        f"## Table 2 -- Token NLL by training-frequency band ({LABEL})",
        "",
        "Teacher-forced NLL of the reference (fp32, eval mode, no label smoothing, EOS excluded). "
        f"Pooled token gap E3-E1 = {nll_break['by_frequency_band']['gap_e3_minus_e1']:.3f} nats "
        "(mix + rate terms sum to it exactly).",
        "",
        "| band | token share E1 | token share E3 | mean NLL E1 | mean NLL E3 | mix term | "
        "rate term |",
        "|---|---|---|---|---|---|---|",
    ]
    bb = nll_break["by_frequency_band"]
    for i, lab in enumerate(bb["labels"]):
        lines.append(
            f"| {lab} | {bb['token_share_e1'][i] * 100:.1f}% | {bb['token_share_e3'][i] * 100:.1f}%"
            f" | {bb['mean_nll_e1'][i]:.2f} | {bb['mean_nll_e3'][i]:.2f} | "
            f"{bb['mix_term'][i]:+.3f} | {bb['rate_term'][i]:+.3f} |"
        )
    lines += [
        "",
        f"## Figure ({LABEL})",
        "",
        f"![{LABEL}: gap components](gap_v2_components.png)",
        "",
        f"*{LABEL}.* Left: Shapley contribution of each feature group to the chrF gap with 95% "
        "bootstrap CIs; the grey bar is what remains unexplained. Right: descriptive split of the "
        "pooled token-NLL gap by training-frequency band.",
        "",
        "## Files",
        "",
        "`features_summary.json` (feature means/sd per domain, novelty, fertility, alignment, word "
        "lists), `nll_breakdown.json`, `gap_shares.json`, `ols_hc3.json`, `collinearity.json`, "
        "`provenance.json`. All carry the key `label`.",
        "",
        "## Provenance",
        "",
        f"- code commit (the commit the numbers were generated from): `{prov['code_commit_sha']}`"
        f" (tree dirty: {prov['code_tree_dirty']})",
        f"- eval repo `{prov['eval_repo']['hf_repo']}` @ `{prov['eval_repo']['hf_revision']}` "
        f"(private={prov['eval_repo']['private']}, manifest sha256 "
        f"`{prov['eval_repo']['manifest_sha256']}`, model files verified: "
        f"{prov['eval_model_files_verified']})",
        f"- data repo `{prov['data_repo']['hf_repo']}` @ `{prov['data_repo']['hf_revision']}` "
        f"(private={prov['data_repo']['private']}; shard + spm sha256 verified against "
        "`data/shards/manifest.json`)",
        f"- word list sha256 `{prov['word_list_sha256']}`; runtime {prov['runtime_seconds']}",
        "- input file sha256 values: `provenance.json` (`input_files_sha256`, "
        "`eval_model_sha256`, `data_repo_files_sha256`)",
        "- line endings: LF (files are written as bytes)",
        "",
    ]
    (out / "README.md").write_bytes("\n".join(lines).encode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
