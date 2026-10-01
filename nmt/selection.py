from __future__ import annotations

# Checkpoint/decoding-config selection, restricted by construction to two hard-coded eval sets
# (spec §8, §15: "Selection is restricted ... by code path"). This module must never import or
# reference any other eval set's paths or loaders -- enforced both by review (no such reference
# exists below) and by tests/test_selection.py's source scan. Every string literal that names the
# two allowed sets appears only as "e1"/"e2"; nothing else is ever named here.
import functools
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hard-coded mapping: the ONLY two names `load_selection_set` will ever accept. Not derived from
# config, a CLI arg, or any caller-supplied path -- a SelectionSet literally cannot be built for
# anything else through this function.
_ALLOWED_SETS: dict[str, Path] = {
    "e1": REPO_ROOT / "data" / "eval" / "e1",
    "e2": REPO_ROOT / "data" / "eval" / "e2",
}


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class SelectionSet:
    """An immutable, provenance-checked view of one of the two allowed eval sets, and nothing
    else. `load_selection_set` is the only way to construct one -- there is no public constructor
    path here that accepts an arbitrary directory or name.
    """

    name: Literal["e1", "e2"]
    sources: tuple[str, ...]
    references: tuple[str, ...]
    ids: tuple[str, ...]
    inputs_sha256: str
    labels_sha256: str

    def __len__(self) -> int:
        return len(self.ids)


def load_selection_set(name: Literal["e1", "e2"]) -> SelectionSet:
    """The only constructor path for a `SelectionSet`. `name` must be exactly one of the two
    hard-coded keys in `_ALLOWED_SETS` -- anything else (including an arbitrary path) raises
    `ValueError`. Records both source files' sha256, and refuses content whose ids are not ALL
    prefixed `<name>_` (a tampered/substituted file is caught here, not silently used).
    """
    if name not in _ALLOWED_SETS:
        raise ValueError(
            f"load_selection_set only accepts one of {sorted(_ALLOWED_SETS)!r}, got {name!r}"
        )
    base = _ALLOWED_SETS[name]
    inputs_path, labels_path = base / "inputs.jsonl", base / "labels.jsonl"
    inputs_rows = _read_jsonl(inputs_path)
    labels_by_id = {r["id"]: r for r in _read_jsonl(labels_path)}

    prefix = f"{name}_"
    bad_ids = [r["id"] for r in inputs_rows if not r["id"].startswith(prefix)]
    if bad_ids:
        raise ValueError(
            f"load_selection_set({name!r}): {len(bad_ids)} id(s) not prefixed {prefix!r}, "
            f"e.g. {bad_ids[:5]} -- refusing possibly-tampered/substituted content"
        )

    sources, references, ids = [], [], []
    for row in inputs_rows:
        label_row = labels_by_id.get(row["id"])
        if label_row is None:
            raise ValueError(f"load_selection_set({name!r}): id {row['id']!r} has no label row")
        sources.append(row["source"])
        references.append(label_row["reference"])
        ids.append(row["id"])

    return SelectionSet(
        name=name,
        sources=tuple(sources),
        references=tuple(references),
        ids=tuple(ids),
        inputs_sha256=_sha256_of(inputs_path),
        labels_sha256=_sha256_of(labels_path),
    )


@functools.lru_cache(maxsize=2)
def _canonical(name: str) -> SelectionSet:
    return load_selection_set(name)  # type: ignore[arg-type]


def _require_selection_set(obj: object, expected_name: str) -> None:
    # SelectionSet is a plain frozen dataclass, so a caller could build one by hand with
    # name="e1" but arbitrary content. Checking the declared name alone would let that through;
    # instead the object must be content-identical to what `load_selection_set` reads from the
    # hard-coded files right now -- OR a content-consistent PREFIX of it (same sha256es, same
    # ids/sources/references in the same order, just fewer of them): `limited_selection_set`
    # below is the only place that ever builds one of those, for CPU-bound smoke/CI runs.
    if not isinstance(obj, SelectionSet):
        raise TypeError(f"expected a SelectionSet, got {type(obj).__name__}")
    if obj.name != expected_name:
        raise TypeError(f"expected a SelectionSet(name={expected_name!r}), got name={obj.name!r}")
    canonical = _canonical(expected_name)
    n = len(obj)
    is_consistent_prefix = (
        obj.inputs_sha256 == canonical.inputs_sha256
        and obj.labels_sha256 == canonical.labels_sha256
        and n <= len(canonical)
        and obj.ids == canonical.ids[:n]
        and obj.sources == canonical.sources[:n]
        and obj.references == canonical.references[:n]
    )
    if not is_consistent_prefix:
        raise TypeError(
            f"SelectionSet(name={expected_name!r}) does not match the canonical "
            f"{expected_name} files -- refusing hand-built or modified content"
        )


def limited_selection_set(sel: SelectionSet, limit: int | None) -> SelectionSet:
    """A verified PREFIX subset of an already-loaded canonical `sel` (its first `limit` ids,
    same order) -- not a new loader path and not arbitrary caller content: `sel` must already be
    a genuine `load_selection_set("e1"|"e2")` result. Returns `sel` unchanged whenever `limit` is
    `None` or `>= len(sel)`. Exists so CPU-bound runs (`nmt/tune.py`'s `--limit-e1`/`--limit-e2`)
    can decode/score a small, fast subset while still passing `select`/`selection_objective`'s
    canonical-content guard above, which accepts exactly this kind of consistent prefix.
    """
    if limit is None or limit >= len(sel):
        return sel
    return replace(
        sel,
        sources=sel.sources[:limit],
        references=sel.references[:limit],
        ids=sel.ids[:limit],
    )


def selection_objective(
    hyps_by_id: dict[str, str], e1: SelectionSet, e2: SelectionSet
) -> dict[str, float]:
    """0.4*BLEU(union) + 0.4*chrF(union) + 0.2*chrF(e1), mirroring the official OVERALL formula
    with `e1` standing in for the slice this module is never allowed to see. Uses
    `official/score.py`'s own BLEU/chrF implementations (via `nmt.evaluate`'s importlib loader)
    for consistency with the primary metric.
    """
    _require_selection_set(e1, "e1")
    _require_selection_set(e2, "e2")
    from nmt.evaluate import load_official_module

    module = load_official_module()
    hyps_e1 = [hyps_by_id[i] for i in e1.ids]
    hyps_e2 = [hyps_by_id[i] for i in e2.ids]
    combined_hyps = hyps_e1 + hyps_e2
    combined_refs = list(e1.references) + list(e2.references)
    combined = module.score_slice(combined_hyps, combined_refs)
    e1_only = module.score_slice(hyps_e1, list(e1.references))
    objective = 0.40 * combined["bleu"] + 0.40 * combined["chrf"] + 0.20 * e1_only["chrf"]
    return {
        "objective": objective,
        "bleu_union": combined["bleu"],
        "chrf_union": combined["chrf"],
        "chrf_e1": e1_only["chrf"],
    }


def select(
    candidates: dict[object, dict[str, str]], e1: SelectionSet, e2: SelectionSet
) -> dict[str, object]:
    """`candidates`: mapping of an opaque candidate key (e.g. a `(checkpoint, alpha, beam)`
    tuple) to that candidate's predictions (`{id: hypothesis}`, covering at least every id in
    `e1`/`e2`). Returns `{"best": <winning key>, "scores": {<key>: selection_objective(...)}}`.

    Raises `TypeError` if `e1`/`e2` are not `SelectionSet`s of the matching name -- the only two
    objects this function (or anything it calls) is ever allowed to score against.
    """
    _require_selection_set(e1, "e1")
    _require_selection_set(e2, "e2")
    if not candidates:
        raise ValueError("select: candidates must be non-empty")
    scored = {key: selection_objective(hyps, e1, e2) for key, hyps in candidates.items()}
    best_key = max(scored, key=lambda k: scored[k]["objective"])
    return {"best": best_key, "scores": scored}
