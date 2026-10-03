from __future__ import annotations

# The `final_all` run: STAGED final selection (PREREG 2026-10-03, which supersedes rule 5 and the
# beam grid of rule 1 of the 2026-10-02 post-selection amendment; rules 2-4 and 6 stand).
#
#   Stage 1: each of 7 model sets {main final, A, B, {main,A}, {main,B}, {A,B}, {main,A,B}} tuned
#            with BEAM only (alpha {1.2,1.4,1.6,1.8,2.0} x beam {4,5}, then the T step) on the full
#            E1 + E2 by the section 2 objective; the 2 best sets go on (ties: earlier model set).
#   Stage 2: the 4 MBR pools (beam n-best N=8/16, epsilon-0.02 sampling N=8/16, seed 1234) on those
#            2 sets only; alpha = that set's stage-1 winner alpha; T re-tuned.
#   Final:   2 stage-1 beam winners + 8 MBR configs = 10 candidates, same objective; the winner is
#            decoded on dev / E1 / E2 / E2-synth / E3 (+ the 330-id test set) and uploaded with a
#            manifest under the run name `final_all`. Ties: model-set order, beam before MBR.
#   Report:  paired bootstrap (nmt.compare, 1,000 resamples, seed 1234) winner vs runner-up and
#            winner vs the production config (best single model, beam only), and the latency of
#            the winner and of the production config. Reported, not gates.
#   COMET:   after the private upload, a NON-FATAL eval-only stage (nmt/comet_stage.py) scores the
#            final_all decodes, the report-step predictions, the 4 existing runs (pulled from the
#            private HF eval repo at pinned revisions) and the copy-the-source baseline with
#            COMET-22 on the session's GPU, then makes a second private upload commit. The plan
#            lists its steps (names `comet:*`) after `upload`; hf-verify of the main upload does
#            not depend on them.
#
# Interpretations of the rule text (also in the PR description, none changes the candidate set):
#   * An MBR candidate takes the GNMT alpha of the winning (alpha, beam) of the SAME model-set's
#     stage-1 beam candidate. The pool size N, not that beam width, sets the beam width of the
#     n-best pool (an N-best list needs beam >= N). Sampling pools do not use alpha. T is tuned for
#     the MBR candidate like the T step of section 1 (E1 undecoded-segmented, E2 per T).
#   * The candidate NAMES still span the full 35-name space (7 sets x {beam + 4 pools}), but only
#     the 7 beam names and the 8 MBR names of the top-2 sets are ever tuned.
#
# Step CLI (each step is idempotent: output written last, skipped when it validates; a tuning file
# is also checked against the CURRENT weights' sha256 and, for MBR, the current stage-1 alpha):
#   python -m nmt.final_all tune|stage1-select|tune-stage2|select|report|decode|plan|estimate ...
# The planning function `plan_final_all` returns the ordered (step name, argv) list; this module's
# top level imports only the stdlib (the Colab kernel rule): torch and the model code are imported
# inside the step functions, which run in their own subprocesses.
import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmt import eval_l4 as ev

FINAL_ALL_ALPHAS: tuple[float, ...] = (1.2, 1.4, 1.6, 1.8, 2.0)  # PREREG rule 1 (extended)
FINAL_ALL_BEAMS: tuple[int, ...] = (4, 5)  # PREREG 2026-10-03: beam 1 is dropped
STAGE1_TOP_N = 2  # model sets that go on to stage 2
MODEL_NAMES: tuple[str, ...] = ("main", "A", "B")
# PREREG rules 3 and 5: the single models, then exactly the four ensembles, in this order.
MODEL_SETS: tuple[tuple[str, ...], ...] = (
    ("main",),
    ("A",),
    ("B",),
    ("main", "A"),
    ("main", "B"),
    ("A", "B"),
    ("main", "A", "B"),
)
# (kind, N); kind "beam" = n-best from a beam of width N, "sample" = N epsilon samples.
POOL_SPECS: tuple[tuple[str, int], ...] = (("beam", 8), ("beam", 16), ("sample", 8), ("sample", 16))
SAMPLING_EPSILON = 0.02  # PREREG rule 2 (== nmt.mbr.DEFAULT_EPSILON, checked by a test)
SAMPLING_SEED = 1234  # PREREG rule 2 (== nmt.mbr.DEFAULT_SAMPLING_SEED)
BEAM_KIND = "beam"


def pool_label(kind: str, n: int) -> str:
    """The decode-kind part of a candidate name for the MBR pool (kind, n)."""
    return f"mbr_beam{n}" if kind == "beam" else f"mbr_eps{SAMPLING_EPSILON:g}_n{n}"


@dataclass(frozen=True)
class FinalAllCandidate:
    """One candidate of the 35-name space: which models, beam search (`pool is None`) or MBR."""

    name: str
    members: tuple[str, ...]
    pool: tuple[str, int] | None

    @property
    def beam_name(self) -> str:
        """The name of this model-set's plain-beam candidate (an MBR candidate's alpha source)."""
        return candidate_name(self.members, None)

    @property
    def mbr(self) -> dict[str, Any] | None:
        if self.pool is None:
            return None
        kind, n = self.pool
        return {"kind": kind, "n": n, "epsilon": SAMPLING_EPSILON, "seed": SAMPLING_SEED}


def candidate_name(members: Sequence[str], pool: tuple[str, int] | None) -> str:
    kind = BEAM_KIND if pool is None else pool_label(*pool)
    return "+".join(members) + "__" + kind


def candidate_space() -> list[FinalAllCandidate]:
    """The 35 NAMES of the candidate space (7 model sets x {beam, 4 MBR pools}) in tie-break order.
    Only 7 + 8 of them are ever tuned (stage 1 and stage 2)."""
    pools: list[tuple[str, int] | None] = [None, *POOL_SPECS]
    return [FinalAllCandidate(candidate_name(m, p), m, p) for m in MODEL_SETS for p in pools]


STAGE1_CANDIDATES: tuple[str, ...] = tuple(candidate_name(m, None) for m in MODEL_SETS)
POOL_LABELS = tuple(pool_label(k, n) for k, n in POOL_SPECS)


def candidate_by_name(name: str) -> FinalAllCandidate:
    for c in candidate_space():
        if c.name == name:
            return c
    raise ev.EvalStepError(f"unknown final_all candidate {name!r}")


# The three models of the run and where their FINAL checkpoints live on Drive:
# model name -> (run dir under runs/, checkpoint step, configs/<name>.yaml of that run).
# main: the main run's last checkpoint (the `final` candidate of its own eval); A and B: the ends of
# the PREREG rule-4 extension branches (A decays 30,000 -> 37,500; B 40,000 -> 50,000).
FINAL_ALL_CHECKPOINTS: dict[str, tuple[str, int, str]] = {
    "main": ("main", 24645, "main"),
    "A": ("ext_branch_a_l4", 37500, "ext_branch_a_l4"),
    "B": ("ext_branch_b_l4", 50000, "ext_branch_b_l4"),
}


def final_all_checkpoint_paths(runs_base: Path, run_prefix: str = "") -> dict[str, Path]:
    """The checkpoint FILE of each model: <runs_base>/<run_prefix><run>/ckpt/step_<step>.pt."""
    return {
        name: ev.checkpoint_path(Path(runs_base) / f"{run_prefix}{run}" / "ckpt", step)
        for name, (run, step, _cfg) in FINAL_ALL_CHECKPOINTS.items()
    }


def check_final_all_checkpoints(paths: Mapping[str, Path]) -> None:
    """Raise EvalStepError naming EVERY missing checkpoint file (before any GPU time is spent);
    no neighbouring step may stand in for a missing one."""
    missing = [str(p) for p in paths.values() if not Path(p).is_file()]
    if missing:
        raise ev.EvalStepError(
            f"final_all: {len(missing)} required checkpoint file(s) missing, refusing before any "
            "GPU time; the three models are fixed by PREREG:\n  " + "\n  ".join(missing)
        )


def parse_model_args(specs: Sequence[str]) -> dict[str, Path]:
    """`["main=DIR", "A=DIR", "B=DIR"]` -> {"main": Path, ...}; refuses unknown/duplicate names."""
    out: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ev.EvalStepError(f"bad --model {spec!r}; expected name=DIR")
        if name not in MODEL_NAMES:
            raise ev.EvalStepError(f"--model name {name!r} is not one of {MODEL_NAMES}")
        if name in out:
            raise ev.EvalStepError(f"duplicate --model {name!r}")
        out[name] = Path(path)
    return out


def check_model_dirs(models: Mapping[str, Path], members: Sequence[str]) -> None:
    """Every member must have a finished HF-style export (weights, config, tokenizer)."""
    for m in members:
        if m not in models:
            raise ev.EvalStepError(f"no --model {m}=DIR given")
        missing = [f for f in ev._MODEL_FILES if not (Path(models[m]) / f).is_file()]
        if missing:
            raise ev.EvalStepError(f"model {m} ({models[m]}) lacks {', '.join(missing)}")


# ---------------------------------------------------------------------------------------------
# translators
# ---------------------------------------------------------------------------------------------


class PoolTranslator:
    """A Translator whose `translate` always decodes with one MBR pool, so nmt.tune's helpers and
    `eval_l4.run_decode` (which call `translate(beam=, alpha=, segment_threshold=)`) can drive MBR
    unchanged. `beam` is ignored; everything else is delegated to the wrapped translator."""

    def __init__(self, translator: Any, mbr: Any) -> None:
        self._inner = translator
        self._mbr = mbr

    def translate(self, texts: list[str], **kwargs: Any) -> list[str]:
        kwargs.pop("beam", None)
        return self._inner.translate(texts, mbr=self._mbr, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_translator(
    models: Mapping[str, Path],
    members: Sequence[str],
    mbr: Mapping[str, Any] | None = None,
    device: str | None = None,
) -> Any:
    """Translator for one model (plain) or several (nmt.ensemble), wrapped in a PoolTranslator when
    `mbr` (a dict of MBRConfig fields) is given."""
    check_model_dirs(models, members)
    dirs = [models[m] for m in members]
    if len(dirs) == 1:
        from nmt.translate import Translator

        translator = Translator.from_pretrained(str(dirs[0]), device=device)
    else:
        from nmt.ensemble import load_ensemble_translator

        translator = load_ensemble_translator(dirs, device=device)
    if mbr is None:
        return translator
    from nmt.mbr import MBRConfig

    return PoolTranslator(translator, MBRConfig(**mbr))


# ---------------------------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------------------------


def grid_matches(tuning: Any) -> bool:
    """True iff a beam candidate's tuning file used exactly the staged grid (alpha x beam {4,5})."""
    if not isinstance(tuning, dict):
        return False
    grid = (tuning.get("alpha_beam") or {}).get("grid") or []
    want = {(a, b) for a in FINAL_ALL_ALPHAS for b in FINAL_ALL_BEAMS}
    return {(g.get("alpha"), g.get("beam")) for g in grid} == want


def _tuning_ok(
    tuning: Any,
    cand: FinalAllCandidate,
    member_sha256: Mapping[str, str] | None = None,
    base_alpha: float | None = None,
) -> bool:
    """A finished full tuning of `cand` on the right grid. When `member_sha256` (the CURRENT
    weights' hashes) is given, the file must have been made by exactly those weights, and when
    `base_alpha` (the CURRENT beam winner's alpha) is given, an MBR file must have used that alpha;
    otherwise it is stale (a model or an upstream tuning changed) and is not accepted. A file that
    records no hash / alpha cannot be shown current, so it fails too."""
    if not ev.tuning_is_full(tuning):
        return False
    if member_sha256 is not None and tuning.get("member_sha256") != dict(member_sha256):
        return False
    if cand.pool is not None:
        if base_alpha is not None and (tuning.get("alpha_beam") or {}).get("alpha") != base_alpha:
            return False
        return tuning.get("candidate") == cand.name
    return grid_matches(tuning)


def _member_sha256(models: Mapping[str, Path], members: Sequence[str]) -> dict[str, str]:
    return {m: ev._sha256_file(Path(models[m]) / "model.safetensors") for m in members}


def run_pool_tune(
    translator: Any,
    cand: FinalAllCandidate,
    base_tuning: dict[str, Any],
    out_path: Path,
    batch_size: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Tune one MBR candidate on full E1 + E2: alpha fixed to `base_tuning`'s rule-1 winner, E1
    decoded once without segmentation, then the T step (nmt.tune._segmentation_tune) on E2. Writes
    a tuning file with the same schema as nmt.tune.run_tune (so selection reads it unchanged)."""
    from nmt.selection import load_selection_set
    from nmt.tune import (
        DEFAULT_SEG_THRESHOLDS,
        _decode_ids,
        _git_sha,
        _segmentation_tune,
    )

    e1, e2 = load_selection_set("e1"), load_selection_set("e2")
    alpha = base_tuning["winner"]["alpha"]
    assert cand.pool is not None
    n = cand.pool[1]
    e1_preds, e1_seconds = _decode_ids(
        translator, e1.ids, e1.sources, n, alpha, batch_size, segment_threshold=None
    )
    seg = _segmentation_tune(
        translator, e1, e2, e1_preds, alpha, n, DEFAULT_SEG_THRESHOLDS, batch_size
    )
    result: dict[str, Any] = {
        "model_dir": None,
        "model_sha256": None,
        "git_sha": _git_sha(),
        "limit_e1": None,
        "limit_e2": None,
        "n_e1": len(e1),
        "n_e2": len(e2),
        "batch_size": batch_size,
        "alpha_beam": {
            "note": "MBR candidate: alpha is the rule-1 winner of the beam candidate "
            f"{cand.beam_name}; no alpha x beam grid is searched",
            "alpha_source": cand.beam_name,
            "alpha": alpha,
            "e1_decode_seconds": round(e1_seconds, 3),
        },
        "segmentation": seg,
        "winner": {
            "alpha": alpha,
            # beam: the n-best pool's width for beam pools; sampling has no beam (None)
            "beam": n if cand.pool[0] == "beam" else None,
            "segment_threshold": seg["best_segment_threshold"],
            "mbr": cand.mbr,
        },
        "fallback_counts": {
            "beam": translator.stats.n_beam,
            "greedy": translator.stats.n_greedy_fallback,
            "copy": translator.stats.n_copy_fallback,
        },
        **extra,
    }
    ev._write_json(out_path, result)
    return result


def tune_candidate(
    name: str,
    models: Mapping[str, Path],
    tuning_dir: Path,
    batch_size: int = ev.EVAL_BATCH_SIZE,
    device: str | None = None,
) -> bool:
    """Tune one candidate into `<tuning_dir>/<name>.json`; skipped when a valid full tuning (on the
    right grid / for this candidate) exists. Returns True when it ran. An MBR candidate needs its
    model-set's beam tuning first (its alpha)."""
    cand = candidate_by_name(name)
    out = Path(tuning_dir) / f"{name}.json"
    check_model_dirs(models, cand.members)
    hashes = _member_sha256(models, cand.members)
    base = None
    if cand.pool is not None:
        base = ev._read_json(Path(tuning_dir) / f"{cand.beam_name}.json")
        if not _tuning_ok(base, candidate_by_name(cand.beam_name), hashes):
            raise ev.EvalStepError(
                f"tune {name}: needs a valid full rule-1 tuning of {cand.beam_name} made by the "
                "current model weights first"
            )
    base_alpha = base["winner"]["alpha"] if base else None
    existing = ev._read_json(out)
    if _tuning_ok(existing, cand, hashes, base_alpha):
        print(f"tune {name}: SKIP (valid full tuning at {out})")
        return False
    if existing is not None:
        print(f"tune {name}: existing {out} is stale or invalid for the current models; redoing")
    extra = {
        "candidate": name,
        "members": list(cand.members),
        "member_sha256": hashes,
        "mbr": cand.mbr,
    }
    translator = build_translator(models, cand.members, mbr=cand.mbr, device=device)
    if cand.pool is None:
        from nmt.tune import run_tune

        result = run_tune(
            Path(models[cand.members[0]]),
            out,
            alphas=FINAL_ALL_ALPHAS,
            beams=FINAL_ALL_BEAMS,
            batch_size=batch_size,
            translator=translator,
            extra=extra,
        )
    else:
        assert base is not None
        result = run_pool_tune(translator, cand, base, out, batch_size, extra)
    w = result["winner"]
    print(
        f"tune {name}: winner alpha={w['alpha']} beam={w['beam']} "
        f"segment_threshold={w['segment_threshold']}"
    )
    return True


# ---------------------------------------------------------------------------------------------
# stage 1 (beam, 7 model sets) -> top 2
# ---------------------------------------------------------------------------------------------

STAGE1_TIE_RULE = (
    f"model sets within {ev.TIE_EPSILON:g} of an objective are tied; the tie goes to the "
    "earlier-listed model set (main, A, B, main+A, main+B, A+B, main+A+B)"
)
FINAL_TIE_RULE = (
    f"candidates within {ev.TIE_EPSILON:g} of the best objective are tied; the tie goes to the "
    "earlier-listed candidate: model-set order (main, A, B, main+A, main+B, A+B, main+A+B), "
    "beam before MBR (mbr_beam8, mbr_beam16, mbr_eps0.02_n8, mbr_eps0.02_n16)"
)


def stage1_candidates() -> list[FinalAllCandidate]:
    """The 7 stage-1 candidates (one beam candidate per model set), in model-set order."""
    return [candidate_by_name(n) for n in STAGE1_CANDIDATES]


def _rank(order: Sequence[str], objectives: Mapping[str, float]) -> list[str]:
    """`order` sorted by objective, best first; at every rank the earliest-listed candidate within
    TIE_EPSILON of the best remaining objective goes first (the tie rule, applied at each rank)."""
    remaining = list(order)
    out: list[str] = []
    while remaining:
        best = max(objectives[n] for n in remaining)
        pick = next(n for n in remaining if best - objectives[n] <= ev.TIE_EPSILON)
        out.append(pick)
        remaining.remove(pick)
    return out


def _check_consistent(tunings: Mapping[str, dict[str, Any]]) -> None:
    """Refuse a mix of stale and fresh tuning files (no model dirs are needed here): every file must
    record the same hash for the same member, and an MBR candidate's alpha must be its beam
    candidate's current winner alpha (the beam candidate must be among `tunings`)."""
    seen: dict[str, str] = {}
    for name, t in tunings.items():
        for member, sha in (t.get("member_sha256") or {}).items():
            if seen.setdefault(member, sha) != sha:
                raise ev.EvalStepError(
                    f"select: {name} was tuned with different weights for member {member!r} than "
                    "another candidate (a model changed between tuning steps); redo the tunes"
                )
    for name in tunings:
        cand = candidate_by_name(name)
        if cand.pool is None:
            continue
        base = tunings.get(cand.beam_name)
        if base is None:
            raise ev.EvalStepError(f"select: {name} needs the stage-1 tuning {cand.beam_name}")
        want = base["winner"]["alpha"]
        if (tunings[name].get("alpha_beam") or {}).get("alpha") != want:
            raise ev.EvalStepError(
                f"select: {name} used a different alpha than the current winner of "
                f"{cand.beam_name} ({want}); redo its tune"
            )


def _read_stage1_tunings(tuning_dir: Path) -> dict[str, dict[str, Any]]:
    tunings: dict[str, dict[str, Any]] = {}
    for cand in stage1_candidates():
        data = ev._read_json(Path(tuning_dir) / f"{cand.name}.json")
        if not _tuning_ok(data, cand):
            raise ev.EvalStepError(
                f"stage1: {tuning_dir}/{cand.name}.json is missing, not a full E1+E2 tuning, or "
                "not on the pre-registered grid"
            )
        tunings[cand.name] = data
    _check_consistent(tunings)
    return tunings


def _stage1_data(tuning_dir: Path) -> dict[str, Any]:
    """The stage-1 result computed from the 7 tuning files: objectives, ranking, top 2."""
    tunings = _read_stage1_tunings(tuning_dir)
    entries = {n: ev.candidate_objective(t) for n, t in tunings.items()}
    for n, e in entries.items():
        e["members"] = list(candidate_by_name(n).members)
    ranking = _rank(STAGE1_CANDIDATES, {n: e["objective"] for n, e in entries.items()})
    return {
        "stage": 1,
        "grid": {"alphas": list(FINAL_ALL_ALPHAS), "beams": list(FINAL_ALL_BEAMS)},
        "candidate_order": list(STAGE1_CANDIDATES),
        "candidates": entries,
        "ranking": ranking,
        "top2": ranking[:STAGE1_TOP_N],
        "tie_rule": STAGE1_TIE_RULE,
    }


def run_stage1_select(tuning_dir: Path, out_path: Path) -> dict[str, Any]:
    """stage1.json: the 7 beam candidates' objectives and the top 2 model sets (PREREG staged
    selection, stage 1). Refuses a missing / smoke / wrong-grid / mixed-weights tuning."""
    data = _stage1_data(tuning_dir)
    ev._write_json(out_path, data)
    print(
        "stage1-select: top 2 = "
        + ", ".join(f"{n} ({data['candidates'][n]['objective']:.4f})" for n in data["top2"])
    )
    return data


def stage1_is_valid(path: Path, tuning_dir: Path) -> bool:
    """stage1.json exists, equals what the 7 CURRENT tuning files give, and no tuning file is
    newer than it (so a re-tuned stage-1 candidate invalidates it, and with it stage 2)."""
    p = Path(path)
    data = ev._read_json(p)
    if not isinstance(data, dict):
        return False
    try:
        if data != _stage1_data(tuning_dir):
            return False
    except ev.EvalStepError:
        return False
    return all(
        (Path(tuning_dir) / f"{n}.json").stat().st_mtime <= p.stat().st_mtime
        for n in STAGE1_CANDIDATES
    )


def _require_stage1(stage1_path: Path, tuning_dir: Path) -> dict[str, Any]:
    if not stage1_is_valid(stage1_path, tuning_dir):
        raise ev.EvalStepError(
            f"{stage1_path} is missing or stale (a stage-1 tuning changed or is invalid); "
            "run stage1-select again before any stage-2 step"
        )
    return ev._read_json(stage1_path)  # type: ignore[return-value]


def stage2_candidate_name(stage1: Mapping[str, Any], rank: int, pool: str) -> str:
    """Name of the stage-2 candidate for the model set ranked `rank` (1 or 2) and pool label."""
    if rank not in range(1, STAGE1_TOP_N + 1):
        raise ev.EvalStepError(f"--rank must be 1..{STAGE1_TOP_N}, got {rank}")
    if pool not in POOL_LABELS:
        raise ev.EvalStepError(f"unknown pool {pool!r}; one of {POOL_LABELS}")
    members = candidate_by_name(stage1["top2"][rank - 1]).members
    return "+".join(members) + "__" + pool


def tune_stage2(
    pool: str,
    rank: int,
    models: Mapping[str, Path],
    tuning_dir: Path,
    stage1_path: Path,
    batch_size: int = ev.EVAL_BATCH_SIZE,
    device: str | None = None,
) -> tuple[str, bool]:
    """Tune the MBR candidate (`pool`) of the model set ranked `rank` in the CURRENT stage 1. The
    model set is read from stage1.json at run time (the plan is a static argv list); a missing or
    stale stage1.json refuses. Returns (candidate name, ran)."""
    stage1 = _require_stage1(stage1_path, tuning_dir)
    name = stage2_candidate_name(stage1, rank, pool)
    return name, tune_candidate(name, models, tuning_dir, batch_size, device)


# ---------------------------------------------------------------------------------------------
# final selection (10 candidates) + decode
# ---------------------------------------------------------------------------------------------


def final_candidates(stage1: Mapping[str, Any]) -> list[FinalAllCandidate]:
    """The 10 final candidates: for each of the top-2 model sets (in model-set order) its beam
    winner then its 4 MBR configs."""
    sets = sorted(
        (candidate_by_name(n).members for n in stage1["top2"]), key=lambda m: MODEL_SETS.index(m)
    )
    pools: list[tuple[str, int] | None] = [None, *POOL_SPECS]
    return [candidate_by_name(candidate_name(m, p)) for m in sets for p in pools]


def production_config(stage1: Mapping[str, Any]) -> dict[str, Any]:
    """The production config: the best SINGLE model (main / A / B) by stage-1 objective (ties to
    the earlier-listed), with its stage-1 beam winner; no MBR, no ensemble."""
    singles = [n for n in STAGE1_CANDIDATES if len(candidate_by_name(n).members) == 1]
    best = _rank(singles, {n: stage1["candidates"][n]["objective"] for n in singles})[0]
    entry = stage1["candidates"][best]
    return {
        "candidate": best,
        "objective": entry["objective"],
        "members": entry["members"],
        "mbr": None,
        **entry["config"],
    }


def _stage2_tunings(tuning_dir: Path, cands: Sequence[FinalAllCandidate]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in cands:
        data = ev._read_json(Path(tuning_dir) / f"{c.name}.json")
        base = ev._read_json(Path(tuning_dir) / f"{c.beam_name}.json")
        alpha = base["winner"]["alpha"] if isinstance(base, dict) and base.get("winner") else None
        if not _tuning_ok(data, c, None, alpha):
            raise ev.EvalStepError(
                f"select: {tuning_dir}/{c.name}.json is missing, not a full E1+E2 tuning, not on "
                "the pre-registered grid, or used another alpha than its stage-1 winner"
            )
        out[c.name] = data
    return out


def run_final_select(tuning_dir: Path, stage1_path: Path, out_path: Path) -> dict[str, Any]:
    """selection.json: stage-1 objectives and top 2, stage-2 objectives, the 10-candidate ranking
    (objective on E1 + E2), winner, runner-up, production config, tie rules. Refuses a stale
    stage 1 and any missing / stale stage-2 tuning of the top-2 sets."""
    stage1 = _require_stage1(stage1_path, tuning_dir)
    cands = final_candidates(stage1)
    tunings = _stage2_tunings(tuning_dir, cands)
    stage1_tunings = _read_stage1_tunings(tuning_dir)
    tunings = {c.name: tunings[c.name] for c in cands}
    for c in cands:  # the beam candidates ARE the stage-1 tunings (same file)
        if c.pool is None:
            tunings[c.name] = stage1_tunings[c.name]
    _check_consistent({**stage1_tunings, **tunings})
    result = ev.select_winner(tunings)
    for name, entry in result["candidates"].items():
        entry["members"] = list(candidate_by_name(name).members)
    ranking = _rank(list(tunings), {n: e["objective"] for n, e in result["candidates"].items()})
    winner = result["winner"]["candidate"]
    assert ranking[0] == winner  # select_winner and _rank apply the same tie rule
    runner = ranking[1]
    result["winner"]["members"] = list(candidate_by_name(winner).members)
    result["runner_up"] = {
        "candidate": runner,
        "objective": result["candidates"][runner]["objective"],
        "members": list(candidate_by_name(runner).members),
        **result["candidates"][runner]["config"],
    }
    result.update(
        {
            "run": "final_all staged selection (PREREG 2026-10-03)",
            "stage1": stage1,
            "top2": list(stage1["top2"]),
            "stage2_candidates": [c.name for c in cands if c.pool is not None],
            "ranking": ranking,
            "production": production_config(stage1),
            "tie_rule": FINAL_TIE_RULE,
        }
    )
    ev._write_json(out_path, result)
    w, r = result["winner"], result["runner_up"]
    print(
        f"select: winner {w['candidate']} objective={w['objective']:.4f} alpha={w['alpha']} "
        f"beam={w['beam']} segment_threshold={w['segment_threshold']} mbr={w.get('mbr')}; "
        f"runner-up {r['candidate']} objective={r['objective']:.4f}"
    )
    return result


def final_selection_is_valid(path: Path, tuning_dir: Path, stage1_path: Path) -> bool:
    """selection.json is for the CURRENT stage 1's 10 candidates and no input is newer than it."""
    p = Path(path)
    data = ev._read_json(p)
    if not stage1_is_valid(stage1_path, tuning_dir):
        return False
    stage1 = ev._read_json(stage1_path)
    cands = final_candidates(stage1)  # type: ignore[arg-type]
    if not (
        isinstance(data, dict)
        and data.get("candidate_order") == [c.name for c in cands]
        and data.get("stage1") == stage1
        and (data.get("winner") or {}).get("members")
        and (data.get("runner_up") or {}).get("candidate")
    ):
        return False
    inputs = [Path(stage1_path)] + [Path(tuning_dir) / f"{c.name}.json" for c in cands]
    return all(i.stat().st_mtime <= p.stat().st_mtime for i in inputs)


def decode_winner(
    eval_dir: Path,
    models: Mapping[str, Path],
    batch_size: int = ev.EVAL_BATCH_SIZE,
    include_test: bool = True,
    device: str | None = None,
    **run_decode_kwargs: Any,
) -> dict[str, Any]:
    """Decode dev / E1 / E2 / E2-synth / E3 (+ test) for the selected candidate at its tuned
    config, through eval_l4.run_decode (per-file resumable)."""
    selection = ev._read_json(Path(eval_dir) / "selection.json")
    if not isinstance(selection, dict) or not (selection.get("winner") or {}).get("members"):
        raise ev.EvalStepError(f"decode: {eval_dir}/selection.json missing or has no winner")
    win = selection["winner"]
    translator = build_translator(models, win["members"], mbr=win.get("mbr"), device=device)
    return ev.run_decode(
        translator,
        selection,
        Path(eval_dir),
        include_test=include_test,
        batch_size=batch_size,
        **run_decode_kwargs,
    )


# ---------------------------------------------------------------------------------------------
# report: paired bootstrap (winner vs runner-up, vs production) + latency
# ---------------------------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 1000  # PREREG staged selection: nmt/compare.py --objective
BOOTSTRAP_SEED = 1234
LATENCY_N_SENTENCES = ev.BENCH_N_SENTENCES  # the first 200 E2 sentences, like bench.json
REPORT_NAME = "report.json"


def latency_benchmark(
    translator: Any,
    texts: Sequence[str],
    *,
    alpha: float,
    beam: int | None,
    segment_threshold: int | None,
    mbr: Mapping[str, Any] | None,
    members: Sequence[str],
    batch_size: int = ev.EVAL_BATCH_SIZE,
    warmup: int = 8,
) -> dict[str, Any]:
    """Wall-clock decode of `texts` with the config exactly as deployed: one `translate()` call
    (length-sorted batches of `batch_size`) after one untimed warm-up call. For an MBR winner the
    time includes pool generation AND the chrF utility; for an ensemble every member runs every
    step. Output tokens are SentencePiece tokens of the detokenized outputs (as bench.json)."""
    texts = list(texts)
    kw: dict[str, Any] = {
        "batch_size": batch_size,
        "beam": beam if beam is not None else 1,
        "alpha": alpha,
        "segment_threshold": segment_threshold,
    }
    translator.translate(texts[:warmup], **kw)
    ev._sync(translator.device)
    t0 = time.perf_counter()
    outputs = translator.translate(texts, **kw)
    ev._sync(translator.device)
    wall = time.perf_counter() - t0
    out_tokens = sum(len(translator.sp.encode(h, out_type=int)) for h in outputs)
    return {
        "n_sentences": len(texts),
        "wall_seconds": round(wall, 4),
        "sentences_per_second": round(len(texts) / wall, 3) if wall > 0 else None,
        "output_tokens": out_tokens,
        "output_tokens_per_second": round(out_tokens / wall, 2) if wall > 0 else None,
        "settings": {
            "split": "e2 (first 200 sentences in file order)",
            "batch_size": batch_size,
            "alpha": alpha,
            "beam": beam,
            "mbr": dict(mbr) if mbr else None,
            "segment_threshold": segment_threshold,
            "members": list(members),
            "includes_pool_generation_and_chrf_utility": mbr is not None,
            "decode_precision": "fp32 (no autocast in nmt.translate)",
            "warmup": f"one untimed translate of the first {warmup} sentences",
        },
    }


def _hardware() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
    }


def report_is_valid(path: Path, selection_path: Path) -> bool:
    data = ev._read_json(path)
    sel = ev._read_json(selection_path)
    if not (isinstance(data, dict) and isinstance(sel, dict)):
        return False
    return bool(
        data.get("winner") == sel["winner"]["candidate"]
        and data.get("runner_up") == sel["runner_up"]["candidate"]
        and data.get("production") == sel["production"]["candidate"]
        and Path(path).stat().st_mtime >= Path(selection_path).stat().st_mtime
    )


def run_report(
    eval_dir: Path,
    models: Mapping[str, Path],
    batch_size: int = ev.EVAL_BATCH_SIZE,
    device: str | None = None,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    load_e12: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Reported alongside the selection (not gates): paired bootstrap of the selection objective
    (nmt.compare.paired_bootstrap_objective, 1,000 resamples, seed 1234) of winner vs runner-up and
    winner vs the production config, and the latency of the winner and the production config.

    Each distinct config is decoded on E1 (no segmentation) and E2 (its tuned T) -- exactly how the
    selection objective was defined -- into <eval_dir>/report/predictions/<candidate>/ (per-file
    resumable), and its latency is measured once. report.json is written last."""
    from nmt import compare
    from nmt.evaluate import load_split

    eval_dir = Path(eval_dir)
    sel_path = eval_dir / "selection.json"
    sel = ev._read_json(sel_path)
    if not isinstance(sel, dict) or "runner_up" not in sel:
        raise ev.EvalStepError(f"report: {sel_path} missing or not a staged selection")
    if report_is_valid(eval_dir / REPORT_NAME, sel_path):
        print(f"report: SKIP (valid {eval_dir / REPORT_NAME})")
        return ev._read_json(eval_dir / REPORT_NAME)  # type: ignore[return-value]
    roles = {
        "winner": sel["winner"],
        "runner_up": sel["runner_up"],
        "production": sel["production"],
    }
    load = load_e12 or (lambda split: load_split(split)[0])
    e1_rows, e2_rows = load("e1"), load("e2")
    configs: dict[str, dict[str, Any]] = {}
    for cfg in roles.values():
        configs.setdefault(cfg["candidate"], cfg)
    latency: dict[str, Any] = {}
    for name, cfg in configs.items():
        translator = build_translator(models, cfg["members"], mbr=cfg.get("mbr"), device=device)
        pdir = eval_dir / "report" / "predictions" / name
        for split, rows, seg in (("e1", e1_rows, None), ("e2", e2_rows, cfg["segment_threshold"])):
            path = pdir / f"{split}_predictions.json"
            if ev.prediction_file_valid(path, [r["id"] for r in rows]):
                print(f"report {name}/{split}: SKIP (valid)")
                continue
            ev._decode_to_file(
                translator,
                rows,
                path,
                beam=cfg["beam"] if cfg["beam"] is not None else 1,
                alpha=cfg["alpha"],
                segment_threshold=seg,
                batch_size=batch_size,
            )
        lat_path = eval_dir / "report" / "latency" / f"{name}.json"
        lat = ev._read_json(lat_path)
        if not (isinstance(lat, dict) and lat.get("config") == _config_key(cfg)):
            lat = latency_benchmark(
                translator,
                [r["source"] for r in e2_rows[:LATENCY_N_SENTENCES]],
                alpha=cfg["alpha"],
                beam=cfg["beam"],
                segment_threshold=cfg["segment_threshold"],
                mbr=cfg.get("mbr"),
                members=cfg["members"],
                batch_size=batch_size,
            )
            lat.update({"candidate": name, "config": _config_key(cfg), **_hardware()})
            ev._write_json(lat_path, lat)
        latency[name] = lat

    def inputs(name: str) -> tuple[list[str], list[str], list[str], list[str]]:
        d = eval_dir / "report" / "predictions" / name
        h1, r1 = compare._objective_inputs(d, "e1")
        h2, r2 = compare._objective_inputs(d, "e2")
        return h1, h2, r1, r2

    def boot(a: str, b: str) -> dict[str, Any]:
        if a == b:
            return {"note": "same candidate: no comparison", "delta": 0.0}
        h1a, h2a, r1, r2 = inputs(a)
        h1b, h2b, _, _ = inputs(b)
        res = compare.paired_bootstrap_objective(
            (h1a, h2a), (h1b, h2b), (r1, r2), n_bootstrap, seed
        )
        return {
            "a": a,
            "b": b,
            "delta": res["delta"],
            "ci95": [res["ci_low"], res["ci_high"]],
            "p_value": res["p_value"],
            "objective_a": res["a"]["objective"],
            "objective_b": res["b"]["objective"],
            "n_resamples": res["n_resamples"],
            "seed": seed,
            "resampling": res["resampling"],
        }

    w, r, p = (roles[k]["candidate"] for k in ("winner", "runner_up", "production"))
    report = {
        "winner": w,
        "runner_up": r,
        "production": p,
        "winner_is_production": w == p,
        "bootstrap": {"winner_vs_runner_up": boot(w, r), "winner_vs_production": boot(w, p)},
        "latency": {"winner": latency[w], "production": latency[p]},
        "note": "E1 + E2 objectives are optimistic for the winner (it was chosen on them); "
        "dev and E3 are the unbiased view. E1 is decoded without segmentation and E2 at each "
        "config's tuned T, as in the selection objective.",
    }
    ev._write_json(eval_dir / REPORT_NAME, report)
    return report


def _config_key(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: cfg.get(k) for k in ("candidate", "alpha", "beam", "segment_threshold", "mbr", "members")
    }


# ---------------------------------------------------------------------------------------------
# plan (what the notebook's eval_plan does for the other runs)
# ---------------------------------------------------------------------------------------------


def _model_args(models: Mapping[str, Path]) -> list[str]:
    out: list[str] = []
    for name in MODEL_NAMES:
        out += ["--model", f"{name}={models[name]}"]
    return out


def final_all_export_steps(
    ckpt_files: Mapping[str, Path], models_root: Path, repo_dir: Path, python: str | None = None
) -> list[tuple[str, list[str]]]:
    """One `nmt.eval_l4 candidates` export per model (the existing export machinery: a single
    checkpoint is its own weights) into <models_root>/<name>/, from its own run's config."""
    py = python or sys.executable
    steps = []
    for name, (_run, step, cfg) in FINAL_ALL_CHECKPOINTS.items():
        steps.append(
            (
                f"export:{name}",
                [
                    py,
                    "-m",
                    "nmt.eval_l4",
                    "candidates",
                    "--ckpt-dir",
                    str(Path(ckpt_files[name]).parent),
                    "--config",
                    str(Path(repo_dir) / "configs" / f"{cfg}.yaml"),
                    "--out-dir",
                    str(models_root),
                    "--candidate",
                    f"{name}={step}",
                ],
            )
        )
    return steps


def plan_final_all(
    *,
    eval_root: Path,
    hf_repo: str,
    models: Mapping[str, Path],
    batch_size: int = ev.EVAL_BATCH_SIZE,
    python: str | None = None,
    ckpt_files: Mapping[str, Path] | None = None,
    repo_dir: Path | None = None,
    comet_batch_size: int = 64,
    comet_precision: str = "fp32",
) -> list[tuple[str, list[str]]]:
    """The ordered (step name, argv) list of the staged `final_all` session, each argv a fresh
    interpreter: hf-verify (the notebook's skip check: it parses HF_RUN_COMPLETE and skips the rest
    when true), hf-check (token + private repo + write probe, BEFORE any GPU work), bench, the 7
    stage-1 tunes, stage1-select (writes stage1.json: the top 2 model sets), the 8 stage-2 tunes
    (`tune-stage2 --pool P --rank 1|2`, which read the top 2 from stage1.json at run time), select
    (10 candidates), report (bootstrap + latency), decode (+test) for the winner, validate-test,
    and the PRIVATE upload, then the COMET stage (`comet:install`, `comet:pull:<run>` x 4,
    `comet:score`, `comet:upload`; non-fatal for everything before it, see nmt/comet_stage.py).
    The plan is static: only the model sets of the stage-2 steps
    depend on stage 1, and they are resolved when those steps run. With `ckpt_files` (and
    `repo_dir`) the three `export:<name>` steps go right after hf-check (so the export, which
    reads Drive, comes after the token check but before bench); `models` must then point at
    <eval_root>/models/<name>."""
    py = python or sys.executable
    root = Path(eval_root)
    base = [py, "-m", "nmt.eval_l4"]
    fa = [py, "-m", "nmt.final_all"]
    mdl = _model_args(models)
    batch = ["--batch-size", str(batch_size)]
    tdir = ["--tuning-dir", str(root / "tuning")]
    s1 = ["--stage1", str(root / "stage1.json")]
    plan: list[tuple[str, list[str]]] = [
        ("hf-verify", [*base, "hf-verify", "--repo", hf_repo, "--run", ev.FINAL_ALL_RUN]),
        ("hf-check", [*base, "hf-check", "--repo", hf_repo]),
        *(
            final_all_export_steps(ckpt_files, root / "models", repo_dir or ev.REPO_ROOT, py)
            if ckpt_files is not None
            else []
        ),
        (
            "bench",  # the main model, beam 5: the measured rates feed estimate_final_all
            [
                *base,
                "bench",
                "--model",
                str(models["main"]),
                "--out",
                str(root / "bench.json"),
                "--run",
                ev.FINAL_ALL_RUN,
                *batch,
            ],
        ),
    ]
    for name in STAGE1_CANDIDATES:
        plan.append((f"tune:{name}", [*fa, "tune", "--name", name, *mdl, *tdir, *batch]))
    plan.append(
        ("stage1-select", [*fa, "stage1-select", *tdir, "--out", str(root / "stage1.json")])
    )
    for rank in range(1, STAGE1_TOP_N + 1):
        for pool in POOL_LABELS:
            plan.append(
                (
                    f"tune-stage2:rank{rank}:{pool}",
                    [*fa, "tune-stage2", "--pool", pool, "--rank", str(rank), *mdl, *tdir, *s1]
                    + batch,
                )
            )
    plan += [
        ("select", [*fa, "select", *tdir, *s1, "--out", str(root / "selection.json")]),
        ("report", [*fa, "report", "--eval-dir", str(root), *mdl, *batch]),
        ("decode", [*fa, "decode", "--eval-dir", str(root), *mdl, *batch, "--test"]),
        (
            "validate-test",
            [
                *base,
                "validate-test",
                "--pred",
                str(root / "test_predictions.json"),
                "--out",
                str(root / "validation.json"),
            ],
        ),
        (
            "upload",
            [
                *base,
                "upload",
                "--repo",
                hf_repo,
                "--run",
                ev.FINAL_ALL_RUN,
                "--eval-dir",
                str(root),
                *mdl,
            ],
        ),
    ]
    from nmt import comet_stage  # lazy: the module top level stays stdlib + eval_l4

    plan += comet_stage.plan_steps(
        eval_root=root,
        hf_repo=hf_repo,
        python=py,
        batch_size=comet_batch_size,
        precision=comet_precision,
    )
    return plan


STAGE2_NOTE = "depends on stage1-select top-2"


def describe_plan(plan: Sequence[tuple[str, Sequence[str]]]) -> list[str]:
    """Printable lines of a plan; the stage-2 steps are marked as depending on the stage-1 top 2
    (their model set is read from stage1.json when they run)."""
    lines = []
    for name, argv in plan:
        mark = f"   <- {STAGE2_NOTE}" if name.startswith("tune-stage2") else ""
        mark = mark or (
            "   <- COMET stage: non-fatal, after the upload" if name == "comet:install" else ""
        )
        lines.append(f"  [{name}] {' '.join(argv)}{mark}")
    return lines


def _read(path: Path) -> dict[str, Any] | None:
    data = ev._read_json(path)
    return data if isinstance(data, dict) else None


def format_final_all_summary(
    eval_root: Path, git_sha: str, preflight: Mapping[str, Any], hf_repo: str
) -> list[str]:
    """Summary-cell lines of a final_all session: SHA, GPU, bench, stage-1 objectives and top 2,
    stage-2 objectives, winner / runner-up / production, the bootstrap deltas, latency, the test
    validation, the HF repo + revision + private flag. Missing pieces say NOT DONE."""
    root = Path(eval_root)
    lines = [
        "eval run: final_all (staged selection, PREREG 2026-10-03)",
        f"git commit SHA: {git_sha}",
        f"GPU: {preflight.get('preflight_gpu')} (runtime-resolved precision "
        f"{preflight.get('preflight_precision')}; decoding itself runs in fp32)",
    ]
    bench = _read(root / "bench.json")
    if bench:
        for label, mode in bench.get("modes", {}).items():
            lines.append(
                f"bench {label} (main final): {mode.get('sentences_per_second')} sent/s, "
                f"{mode.get('output_tokens_per_second')} out-tok/s ({bench.get('gpu')})"
            )
    else:
        lines.append("bench: NOT DONE")
    stage1 = _read(root / "stage1.json")
    if stage1:
        for name in stage1["candidate_order"]:
            c = stage1["candidates"][name]
            t = c["config"]["segment_threshold"]
            lines.append(
                f"stage 1 {name}: objective {c['objective']:.4f} alpha={c['config']['alpha']} "
                f"beam={c['config']['beam']} T={'off' if t is None else t}"
            )
        lines.append("stage 1 TOP 2: " + ", ".join(stage1["top2"]))
    else:
        lines.append("stage 1: NOT DONE")
    sel = _read(root / "selection.json")
    if sel and "runner_up" in sel:
        for name in sel["stage2_candidates"]:
            lines.append(f"stage 2 {name}: objective {sel['candidates'][name]['objective']:.4f}")
        lines.append(f"final ranking ({len(sel['ranking'])}): " + " > ".join(sel["ranking"]))
        w, r = sel["winner"], sel["runner_up"]
        wt = w["segment_threshold"]
        lines.append(
            f"WINNER: {w['candidate']} objective {w['objective']:.4f}; alpha={w['alpha']} "
            f"beam={w['beam']} T={'off' if wt is None else wt} mbr={w.get('mbr')}"
        )
        lines.append(f"runner-up: {r['candidate']} objective {r['objective']:.4f}")
        lines.append(f"production config: {sel['production']['candidate']}")
        lines.append(f"tie rule: {sel['tie_rule']}")
    else:
        lines.append("selection: NOT DONE")
    report = _read(root / REPORT_NAME)
    if report:
        for key, b in report["bootstrap"].items():
            if "ci95" in b:
                lines.append(
                    f"bootstrap {key}: delta {b['delta']:+.3f} [{b['ci95'][0]:+.3f}, "
                    f"{b['ci95'][1]:+.3f}] p={b['p_value']} "
                    f"(n={b['n_resamples']}, seed {b['seed']})"
                )
            else:
                lines.append(f"bootstrap {key}: {b.get('note')}")
        for role, lat in report["latency"].items():
            lines.append(
                f"latency {role}: {lat['sentences_per_second']} sent/s, "
                f"{lat['output_tokens_per_second']} out-tok/s ({lat.get('gpu')}, "
                f"batch {lat['settings']['batch_size']}, 200 E2 sentences)"
            )
    else:
        lines.append("report (bootstrap + latency): NOT DONE")
    val = _read(root / "validation.json")
    if val and val.get("valid"):
        lines.append(f"test validation: OK ({val['n_ids']} ids, {val['empty_strings']} empty)")
    else:
        lines.append(
            f"test validation: {'FAILED: ' + str(val.get('error')) if val else 'NOT DONE'}"
        )
    up = _read(root / "hf_upload.json")
    if up:
        lines.append(
            f"HF repo: {up['repo']} revision: {up['revision']} private: {up['private']} "
            "(prefix runs/final_all/)"
        )
        lines.append(f"HF_EVAL_REVISION={up['revision']}")
    else:
        lines.append(f"HF upload to {hf_repo}: NOT DONE")
    lines += _comet_summary_lines(root)
    return lines


def _comet_summary_lines(root: Path) -> list[str]:
    """COMET-22 lines of the summary: counts, model, device, then one line per system / variant
    with the per-split means [95% bootstrap CI]; the second upload's revision. NOT DONE when the
    stage has not produced them (it is non-fatal, so this is a normal state after a failure)."""
    summary = _read(root / "comet" / "comet_summary.json")
    if not summary:
        return ["COMET-22: NOT DONE (see the COMET FAILED line of the run cell, if any)"]
    model = summary.get("model") or {}
    runtime = summary.get("runtime") or {}
    lines = [
        f"COMET-22: {summary['n_sets']} sets, {summary['n_segments']} segments, "
        f"{summary['n_distinct_triples_all_sets']} distinct triples; {model.get('name')}@"
        f"{str(model.get('revision'))[:8]}; device {summary.get('device')}; precision "
        f"{summary.get('precision')}; batch {summary.get('batch_size')}; scoring "
        f"{runtime.get('scoring_wall_seconds')} s"
    ]
    groups: dict[tuple[str, str], list[str]] = {}
    for row in summary["sets"]:
        lo, hi = row["ci95"]
        groups.setdefault((row["system"], row["variant"]), []).append(
            f"{row['split']} {row['mean']:.4f} [{lo:.4f}, {hi:.4f}]"
        )
    for (system, variant), cells in groups.items():
        lines.append(f"COMET {system}/{variant}: " + "; ".join(cells))
    up = _read(root / "comet_hf_upload.json")
    if up:
        lines.append(
            f"HF COMET upload: {up['repo']} revision: {up['revision']} private: {up['private']} "
            "(prefix runs/final_all/comet/)"
        )
        lines.append(f"HF_COMET_REVISION={up['revision']}")
    else:
        lines.append("HF COMET upload: NOT DONE")
    return lines


# ---------------------------------------------------------------------------------------------
# cost ESTIMATE
# ---------------------------------------------------------------------------------------------

# MEASURED on the L4 for the main final model (bench.json, 200 first E2 sentences, batch 32, fp32,
# alpha 0.6): runs/main/bench.json at HF eval revision c3d8598252853fcd7df1ef4a00e8b0382b8f4351
# (pulled read-only; its numbers equal the ones in the PREREG 2026-10-03 amendment).
MEASURED_L4_RATES = {"greedy": 2305.93, "beam": 1198.86}  # output tokens / s, one model
MEASURED_L4_SENTENCES_PER_SECOND = {"greedy": 41.097, "beam5": 21.345}
# MEASURED on the dev laptop CPU (not Colab, which may be slower): mean seconds of one mbr_select
# over a pool of N distinct strings of ~160 characters (the mean E1/E2 reference length), 300 pools
# each; the command and output are in the PR description.
MEASURED_MBR_CPU_SECONDS_PER_POOL = {8: 0.0120, 16: 0.0354}
ASSUMED_FIXED_SECONDS = 900.0  # ASSUMED, same as the notebook (clone, install, loads, upload)
CU_PER_HOUR = 1.54  # reported by GG for the L4 pilot
PRE_STAGED_ESTIMATE_HOURS = 20.6  # the exhaustive 35-candidate set under ASSUMED rates 2500/1000


def estimate_final_all(
    workload: Mapping[str, Any],
    rates: Mapping[str, float] | None = None,
    basis: str = "MEASURED L4 beam-5 rate (main final, bench.json at HF revision c3d85982), "
    "everything else ASSUMED as listed",
    mbr_cpu_seconds: Mapping[int, float] | None = None,
    comet_triples: int | None = None,
) -> dict[str, Any]:
    """ESTIMATE (not a measurement) of the wall time of the STAGED session (PREREG 2026-10-03).

    Token counts: colab/eval_workload.json (reference SentencePiece tokens as the proxy of output
    length). `rates`: single-model output tokens/s (default: the MEASURED L4 numbers). Assumptions:
      * beam 4 costs the same per token as beam 5 (ASSUMED);
      * an M-member ensemble costs M x a single model per token (ASSUMED; encoder cost ignored);
      * an MBR pool of N costs like a beam of width N: N/5 x the beam-5 time per token (ASSUMED,
        linear; sampling pools are costed the same);
      * the chrF utility time per pool is MEASURED once on a laptop CPU;
      * COMET stage (GG 2026-10-03: on the L4 after the upload): `comet_triples` distinct triples
        (default: the UNDEDUPED segment count of every set from the workload, a conservative upper
        bound) at an ASSUMED central 100 triples/s (range 50-150, NO L4 COMET rate has been
        measured; see `comet.rows`) + ASSUMED setup (install, model download + load, pulls) and
        second-upload times.
    Stage 1 is exact (all 7 model sets are known). Stage 2 depends on WHICH 2 sets win stage 1, so
    it is given for the cheapest (two single models) and the dearest (the 3- and 2-member
    ensembles) outcome, with the report/final decode costed on the dearest pool (N=16, the
    stage-2 set with most members)."""
    rates = dict(rates or MEASURED_L4_RATES)
    cpu = dict(mbr_cpu_seconds or MEASURED_MBR_CPU_SECONDS_PER_POOL)
    sp = workload["splits"]
    e12 = sp["e1"]["out_tokens"] + sp["e2"]["out_tokens"]
    e2 = sp["e2"]["out_tokens"]
    beam_rate = rates["beam"]
    sizes = [len(m) for m in MODEL_SETS]
    n_grid = len(FINAL_ALL_ALPHAS) * len(FINAL_ALL_BEAMS)
    stage1_tokens_per_model = n_grid * e12 + 5 * e2  # grid on E1+E2, T step = 5 E2 decodes
    stage1 = {
        "per_model_tokens": stage1_tokens_per_model,
        "sum_members": sum(sizes),
        "seconds": sum(sizes) * stage1_tokens_per_model / beam_rate,
    }
    n_sent = sp["e1"]["n"] + sp["e2"]["n"] + 5 * sp["e2"]["n"]
    pool_ns = [n for _, n in POOL_SPECS]
    pool_tokens = e12 + 5 * e2  # E1+E2 once + 5 E2 decodes (T step), per pool and model
    # time of one model over the 4 pools: sum_N tokens * (N / 5) / beam_rate
    per_model_pools = sum(pool_tokens * n / 5.0 / beam_rate for n in pool_ns)
    cpu_per_set = sum(n_sent * cpu[n] for n in pool_ns)

    def stage2(members: Sequence[int]) -> dict[str, float]:
        gpu = sum(members) * per_model_pools
        return {
            "sum_members": float(sum(members)),
            "gpu_seconds": gpu,
            "chrf_cpu_seconds": len(members) * cpu_per_set,
        }

    scenarios = {"cheapest": stage2([1, 1]), "dearest": stage2([3, 2])}
    split_tokens = sum(sp[s]["out_tokens"] for s in ev.DECODE_SPLITS)
    final_tokens = 2 * split_tokens + sp["test"]["out_tokens"]  # two variants per split, test once
    final_sentences = 2 * sum(sp[s]["n"] for s in ev.DECODE_SPLITS) + sp["test"]["n"]
    from nmt import comet_stage  # lazy: the module top level stays stdlib + eval_l4

    seg_all = sum(sp[s]["n"] for s in ev.DECODE_SPLITS)
    if comet_triples is None:  # 4 runs x 2 variants + baseline + winner x 2 variants + 3 report
        comet_triples = (
            len(comet_stage.PINNED_RUNS) * len(ev.VARIANTS) + 1 + len(ev.VARIANTS)
        ) * seg_all + comet_stage.FINAL_ALL_REPORT_CONFIGS * (sp["e1"]["n"] + sp["e2"]["n"])
    comet = comet_stage.estimate_comet(comet_triples)
    # report: E1+E2 for the winner, runner-up and production (3 configs, costed at the dearest
    # pool) + 2 latency runs of the first 200 E2 sentences
    report_tokens = 3 * e12 + 2 * workload["bench_e2_first200_out_tokens"]
    out: dict[str, Any] = {
        "basis": basis,
        "rates": rates,
        "stage1": stage1,
        "stage2": scenarios,
        "comet": comet,
        "scenarios": {},
    }
    for label, s2 in scenarios.items():
        m_max = 3 if label == "dearest" else 1
        final_gpu = m_max * final_tokens * 16 / 5.0 / beam_rate  # worst pool: N=16 x m_max members
        report_gpu = m_max * report_tokens * 16 / 5.0 / beam_rate
        final_cpu = final_sentences * cpu[16]
        parts = {
            "stage1_gpu": stage1["seconds"],
            "stage2_gpu": s2["gpu_seconds"],
            "stage2_chrf_cpu": s2["chrf_cpu_seconds"],
            "report_upper_bound": report_gpu,
            "final_decode_upper_bound": final_gpu,
            "final_decode_chrf_cpu_upper_bound": final_cpu,
            "bench": workload["bench_e2_first200_out_tokens"]
            * (1 / rates["greedy"] + 1 / beam_rate),
            "fixed_overhead": ASSUMED_FIXED_SECONDS,
            "comet_gpu": comet_triples / comet_stage.ASSUMED_COMET_CENTRAL,
            "comet_setup": comet_stage.ASSUMED_COMET_SETUP_SECONDS,
            "comet_second_upload": comet_stage.ASSUMED_COMET_UPLOAD_SECONDS,
        }
        seconds = sum(parts.values())
        out["scenarios"][label] = {
            "parts_seconds": parts,
            "seconds": seconds,
            "hours": seconds / 3600,
            "cu": seconds / 3600 * CU_PER_HOUR,
        }
    out["pre_staged_hours"] = PRE_STAGED_ESTIMATE_HOURS
    return out


def format_final_estimate(est: Mapping[str, Any]) -> list[str]:
    """Printable ESTIMATE lines: the arithmetic per stage, then hours / CU per scenario."""
    s1 = est["stage1"]
    r = est["rates"]
    lines = [
        f"ESTIMATE ({est['basis']}) -- not a measurement of this run:",
        f"  rates (one model, out-tok/s): beam5 {r['beam']} (beam 4 ASSUMED equal), "
        f"greedy {r['greedy']}",
        f"  stage 1: sum over the 7 sets of members x {s1['per_model_tokens']:,} tokens "
        f"(10 grid decodes of E1+E2 + 5 E2 for T) / {r['beam']} tok/s = "
        f"{s1['sum_members']} x {s1['per_model_tokens'] / r['beam']:.0f} s = "
        f"{s1['seconds'] / 60:.1f} min",
    ]
    for label, s2 in est["stage2"].items():
        lines.append(
            f"  stage 2 ({label}: member counts summing to {s2['sum_members']:.0f}): "
            f"GPU {s2['gpu_seconds'] / 60:.1f} min + chrF CPU {s2['chrf_cpu_seconds'] / 60:.1f} min"
        )
    for label, sc in est["scenarios"].items():
        lines.append(
            f"  total ({label}): "
            + ", ".join(f"{k} {v / 60:.1f}" for k, v in sc["parts_seconds"].items())
            + f" (min) = {sc['seconds'] / 60:.0f} min = {sc['hours']:.1f} h ~ {sc['cu']:.1f} CU "
            f"at {CU_PER_HOUR} CU/h"
        )
    c = est["comet"]
    lines.append(
        f"  COMET stage (after the upload; {c['n_triples']:,} distinct triples at most; NO L4 "
        f"COMET rate measured): setup + second upload ASSUMED {c['fixed_seconds'] / 60:.0f} min; "
        "scoring ASSUMED "
        + ", ".join(
            f"{r['triples_per_second']:.0f}/s -> {r['scoring_seconds'] / 60:.1f} min "
            f"({r['total_hours']:.2f} h with setup)"
            for r in c["rows"]
        )
        + "; the totals above use the central rate"
    )
    lines.append(
        f"  before staging: {est['pre_staged_hours']} h (exhaustive 35 candidates, ASSUMED rates "
        "2500/1000)"
    )
    return lines


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "final_all steps")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tune", help="stage 1: tune one model set with beam (full E1+E2)")
    s.add_argument("--name", required=True, choices=STAGE1_CANDIDATES)
    s.add_argument("--model", action="append", required=True, help="main|A|B=DIR (export dir)")
    s.add_argument("--tuning-dir", required=True, type=Path)
    s.add_argument("--batch-size", type=int, default=ev.EVAL_BATCH_SIZE)

    s = sub.add_parser("stage1-select", help="write stage1.json: the top 2 model sets")
    s.add_argument("--tuning-dir", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)

    s = sub.add_parser("tune-stage2", help="stage 2: one MBR pool on the model set at --rank")
    s.add_argument("--pool", required=True, choices=POOL_LABELS)
    s.add_argument("--rank", required=True, type=int, choices=range(1, STAGE1_TOP_N + 1))
    s.add_argument("--model", action="append", required=True)
    s.add_argument("--tuning-dir", required=True, type=Path)
    s.add_argument("--stage1", required=True, type=Path)
    s.add_argument("--batch-size", type=int, default=ev.EVAL_BATCH_SIZE)

    s = sub.add_parser("select", help="selection.json over the 10 final candidates")
    s.add_argument("--tuning-dir", required=True, type=Path)
    s.add_argument("--stage1", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)

    s = sub.add_parser("report", help="paired bootstrap + latency of winner / runner-up / prod")
    s.add_argument("--eval-dir", required=True, type=Path)
    s.add_argument("--model", action="append", required=True)
    s.add_argument("--batch-size", type=int, default=ev.EVAL_BATCH_SIZE)

    s = sub.add_parser("decode", help="decode every split (+test) for the winner")
    s.add_argument("--eval-dir", required=True, type=Path)
    s.add_argument("--model", action="append", required=True)
    s.add_argument("--batch-size", type=int, default=ev.EVAL_BATCH_SIZE)
    s.add_argument("--test", action="store_true")

    s = sub.add_parser("plan", help="print the ordered step list (the notebook uses --json)")
    s.add_argument("--eval-root", required=True, type=Path)
    s.add_argument("--repo", required=True)
    s.add_argument(
        "--model",
        action="append",
        default=[],
        help="main|A|B=DIR; default <eval-root>/models/<name> (where the export steps write)",
    )
    s.add_argument(
        "--ckpt-file",
        action="append",
        default=[],
        help="main|A|B=checkpoint file: adds the three export steps after hf-check",
    )
    s.add_argument("--repo-dir", type=Path, default=ev.REPO_ROOT, help="for configs/<name>.yaml")
    s.add_argument("--comet-batch-size", type=int, default=64)
    s.add_argument("--comet-precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    s.add_argument("--json", action="store_true", help="print [[name, argv], ...] as JSON")

    s = sub.add_parser("summary", help="print the Summary-cell lines of a finished session")
    s.add_argument("--eval-root", required=True, type=Path)
    s.add_argument("--hf-repo", required=True)
    s.add_argument("--git-sha", default="?")
    s.add_argument("--gpu", default=None)
    s.add_argument("--precision", default=None)

    s = sub.add_parser("estimate", help="print the cost ESTIMATE of the staged flow")
    s.add_argument("--workload", type=Path, default=ev.REPO_ROOT / "colab" / "eval_workload.json")
    s.add_argument("--bench", type=Path, default=None, help="bench.json: use its measured rates")
    s.add_argument(
        "--runs-root",
        type=Path,
        default=ev.REPO_ROOT / "reports" / "final",
        help="the 4 runs' predictions (checkout): exact COMET distinct-triple count if present",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except ev.EvalStepError as exc:
        print(f"final_all {args.cmd}: FAILED: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "tune":
        tune_candidate(args.name, parse_model_args(args.model), args.tuning_dir, args.batch_size)
    elif args.cmd == "stage1-select":
        if stage1_is_valid(args.out, args.tuning_dir):
            print(f"stage1-select: SKIP (valid {args.out})")
        else:
            run_stage1_select(args.tuning_dir, args.out)
    elif args.cmd == "tune-stage2":
        name, _ = tune_stage2(
            args.pool,
            args.rank,
            parse_model_args(args.model),
            args.tuning_dir,
            args.stage1,
            args.batch_size,
        )
        print(f"tune-stage2: rank {args.rank} pool {args.pool} -> {name}")
    elif args.cmd == "select":
        if final_selection_is_valid(args.out, args.tuning_dir, args.stage1):
            print(f"select: SKIP (valid {args.out})")
        else:
            run_final_select(args.tuning_dir, args.stage1, args.out)
    elif args.cmd == "report":
        run_report(args.eval_dir, parse_model_args(args.model), args.batch_size)
    elif args.cmd == "decode":
        decode_winner(
            args.eval_dir, parse_model_args(args.model), args.batch_size, include_test=args.test
        )
    elif args.cmd == "plan":
        models = (
            parse_model_args(args.model)
            if args.model
            else {n: args.eval_root / "models" / n for n in MODEL_NAMES}
        )
        plan = plan_final_all(
            eval_root=args.eval_root,
            hf_repo=args.repo,
            models=models,
            ckpt_files=parse_model_args(args.ckpt_file) if args.ckpt_file else None,
            repo_dir=args.repo_dir,
            comet_batch_size=args.comet_batch_size,
            comet_precision=args.comet_precision,
        )
        if args.json:
            print(json.dumps([[name, argv] for name, argv in plan]))
        else:
            print("\n".join(describe_plan(plan)))
    elif args.cmd == "summary":
        pre = {"preflight_gpu": args.gpu, "preflight_precision": args.precision}
        print("\n".join(format_final_all_summary(args.eval_root, args.git_sha, pre, args.hf_repo)))
    elif args.cmd == "estimate":
        workload = json.loads(args.workload.read_text(encoding="utf-8"))
        rates, basis = None, None
        bench = ev._read_json(args.bench) if args.bench else None
        if bench:
            m = bench["modes"]
            rates = {
                "greedy": float(m["greedy"]["output_tokens_per_second"]),
                "beam": float(m["beam5"]["output_tokens_per_second"]),
            }
            basis = f"MEASURED rates from {args.bench}, everything else ASSUMED as listed"
        from nmt import comet_stage

        triples = comet_stage.planned_counts(Path("."), args.runs_root)["distinct_upper_bound"]
        est = (
            estimate_final_all(workload, rates, basis, comet_triples=triples)
            if basis
            else estimate_final_all(workload, comet_triples=triples)
        )
        print("\n".join(format_final_estimate(est)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
