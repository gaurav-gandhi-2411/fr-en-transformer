from __future__ import annotations

# Mirrors the numeric scalar history of the private W&B runs into a NEW, PRIVATE project so a
# sanitised copy exists for a later publication decision. Nothing here makes anything public and
# nothing touches the source project (read-only API calls only on it).
#
# What is copied: scalar numeric history keys (same `_step` values) and a WHITELISTED config.
# What is not: tables, media, artifacts, code, output.log, requirements, machine metadata,
# sample translations (they carry dev sentences), paths, host names, emails.
#
# Order of safety: (1) create the destination project PRIVATE and prove it via GraphQL, abort if
# not provable; (2) log one run per source run (deterministic run id, so a re-run cannot create a
# duplicate); (3) scan every new run's files, config, summary and history for forbidden strings.
#
# CLI: `python scripts/mirror_wandb.py [--dry-run] [--log reports/final/wandb_mirror.json]`
import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

ENTITY = "gauravgandhi429-gaurav-gandhi"
SOURCE_PROJECT = "fr-en-transformer"
DEST_PROJECT = "fr-en-transformer-public"

# Source run id -> expected name (checked against the API; a mismatch aborts the run).
SOURCE_RUNS: dict[str, str] = {
    "5d9fc85d": "main",
    "a150f75a": "s1_sin_l4",
    "ba29d416": "s2_rope_l4",
    "f8200e87": "s3_rope_concat_l4",
    "86e3ddfa": "ext_stable_l4",
    "ba8c6bf4": "ext_branch_a_l4",
    "1f7d5761": "ext_branch_b_l4",
}

# Dotted paths into the source config. A path that names a dict keeps the whole sub-dict, so keep
# these specific: anything not listed (paths, platform, git, entity, preflight, ...) is dropped.
CONFIG_WHITELIST: tuple[str, ...] = (
    "name",
    "model",
    "optim.lr",
    "optim.eps",
    "optim.betas",
    "optim.grad_clip",
    "optim.warmup_steps",
    "optim.weight_decay",
    "optim.cooldown_frac",
    "optim.planned_steps",
    "batch.max_tokens",
    "batch.tokens_per_step",
    "batch.concat_prob",
    "batch.concat_max_len",
    "label_smoothing",
    "seed",
    "precision",
    "tf32",
    "param_count",
    "shard_manifest_sha256",
    "torch_version",
    "gpu_name",
)

# Bookkeeping keys W&B adds itself; `_step` is passed to `log(step=)` instead of as a metric.
_DROP_HISTORY_KEYS = frozenset({"_runtime", "_timestamp", "_wandb", "_step"})

# Scan targets: strings that must never appear in a mirrored run (the entity name inside URLs is
# deliberately not matched).
FORBIDDEN_PATTERNS: dict[str, str] = {
    "sponsor_name": r"(?i)four[\s_-]*kites",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}",
    "windows_user_path": r"(?i)C:[\\/]+Users",
    "host_name": r"(?i)legion",
    "account_email_prefix": r"gauravgandhi429@",
    "drive_path": r"/content/drive|MyDrive",
    "dev_test_id": r"\b(?:dev|test)_\d{5}\b",
}

_PRIVATE_QUERY = """
query ProjectAccess($name: String!, $entity: String!) {
  project(name: $name, entityName: $entity) { id name entityName access createdAt }
}
"""
_CREATE_MUTATION = """
mutation CreateProject($input: UpsertModelInput!) {
  result: upsertModel(input: $input) {
    project { id name entityName access }
    inserted
  }
}
"""


# --------------------------------------------------------------------------- pure functions
def mirror_tag(run_id: str) -> str:
    """Tag that marks a run as the mirror of source run `run_id` (used for idempotency)."""
    return f"mirrored-from-private-run-{run_id}"


def mirror_notes(run_id: str) -> str:
    """Run notes: states plainly where the run came from."""
    return f"mirrored from private run {run_id}"


def mirror_run_id(run_id: str) -> str:
    """Deterministic destination run id, so re-running can never create a second copy."""
    return f"m-{run_id}"


def _lookup(config: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def filter_config(
    config: Mapping[str, Any], whitelist: Iterable[str] = CONFIG_WHITELIST
) -> dict[str, Any]:
    """Return a nested dict holding only the whitelisted dotted paths present in `config`."""
    out: dict[str, Any] = {}
    for dotted in whitelist:
        found, value = _lookup(config, dotted)
        if not found:
            continue
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _is_numeric_scalar(value: Any) -> bool:
    # bool is an int subclass but is a flag, not a curve; NaN/inf cannot be plotted or compared.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def scalar_row(row: Mapping[str, Any]) -> dict[str, float | int]:
    """Keep only finite numeric scalar entries of a history row, dropping W&B bookkeeping keys."""
    return {k: v for k, v in row.items() if k not in _DROP_HISTORY_KEYS and _is_numeric_scalar(v)}


def iter_mirror_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield `(step, scalars)` for rows that have a `_step` and at least one scalar metric.
    Steps must be non-decreasing (W&B rejects going backwards); out-of-order rows are skipped.
    """
    last = -1
    for row in rows:
        step = row.get("_step")
        if not isinstance(step, int) or isinstance(step, bool) or step < last:
            continue
        scalars = scalar_row(row)
        if not scalars:
            continue
        last = step
        yield step, scalars


def scan_text(text: str, literals: Iterable[str] = ()) -> dict[str, int]:
    """Count forbidden-pattern matches and verbatim literal hits in `text`.

    `literals` are dev/test sentences or ids; each is checked as a plain substring.
    """
    counts = {name: len(re.findall(pat, text)) for name, pat in FORBIDDEN_PATTERNS.items()}
    counts["dev_test_literal"] = sum(1 for lit in literals if lit and lit in text)
    return counts


def merge_counts(*parts: Mapping[str, int]) -> dict[str, int]:
    """Sum several count dicts key-wise."""
    out: dict[str, int] = {}
    for part in parts:
        for key, value in part.items():
            out[key] = out.get(key, 0) + value
    return out


# --------------------------------------------------------------------------- repo data
def load_literals(root: Path) -> list[str]:
    """Dev/test sources, dev references and ids that must not leak. Short strings are skipped
    for sentences (a 3-word reply would false-positive on any text); ids are always kept.
    """
    literals: list[str] = []
    for rel in ("data/dev/inputs.jsonl", "data/dev/labels.jsonl", "data/test/inputs.jsonl"):
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"scan input missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for key, value in rec.items():
                if not isinstance(value, str):
                    continue
                if key == "id" or len(value) >= 25:
                    literals.append(value)
    return sorted(set(literals))


# --------------------------------------------------------------------------- W&B access
def graphql(api: Any, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
    """Run one GraphQL call through the Api's service client."""
    return api._service_api.execute_graphql(query, dict(variables))  # noqa: SLF001


def project_access(api: Any, entity: str, project: str) -> dict[str, Any] | None:
    """Return `{id, name, entityName, access, createdAt}` or None if the project is missing."""
    data = graphql(api, _PRIVATE_QUERY, {"name": project, "entity": entity})
    return (data or {}).get("project")


def ensure_private_project(api: Any, entity: str, project: str) -> dict[str, Any]:
    """Create the destination project as PRIVATE if absent, then PROVE it is PRIVATE.

    Fails closed: any lookup that is not exactly access == PRIVATE raises, and the caller must
    not log a single run after that.
    """
    info = project_access(api, entity, project)
    created = False
    if info is None:
        graphql(
            api,
            _CREATE_MUTATION,
            {"input": {"name": project, "entityName": entity, "access": "PRIVATE"}},
        )
        created = True
        info = project_access(api, entity, project)
    if info is None:
        raise RuntimeError(f"project {entity}/{project} not found after creation attempt")
    if str(info.get("access")).upper() != "PRIVATE":
        raise RuntimeError(f"project {entity}/{project} access={info.get('access')!r}, not PRIVATE")
    return {**info, "created_this_run": created}


# --------------------------------------------------------------------------- mirroring
def build_plan(api: Any, entity: str, source_project: str, run_id: str) -> dict[str, Any]:
    """Read one source run (read-only) and compute what would be mirrored."""
    run = api.run(f"{entity}/{source_project}/{run_id}")
    if run.name != SOURCE_RUNS[run_id]:
        raise RuntimeError(f"run {run_id} is named {run.name!r}, expected {SOURCE_RUNS[run_id]!r}")
    rows = list(iter_mirror_rows(run.scan_history()))
    keys = sorted({k for _, scalars in rows for k in scalars})
    return {
        "source_id": run_id,
        "name": run.name,
        "config": filter_config(dict(run.config)),
        "rows": rows,
        "keys": keys,
        "n_steps": len(rows),
        "first_step": rows[0][0] if rows else None,
        "last_step": rows[-1][0] if rows else None,
    }


def existing_mirror(api: Any, entity: str, project: str, source_id: str) -> Any | None:
    """Return the destination run already mirroring `source_id` (by tag), else None."""
    for run in api.runs(f"{entity}/{project}"):
        if mirror_tag(source_id) in (run.tags or []):
            return run
    return None


def upload_run(entity: str, project: str, plan: Mapping[str, Any]) -> str:
    """Log one planned run to the destination project; returns the new run's URL."""
    import wandb

    settings = wandb.Settings(
        disable_code=True,
        disable_git=True,
        save_code=False,
        x_disable_meta=True,
        x_save_requirements=False,
        x_disable_stats=True,
        console="off",
        silent=True,
    )
    run = wandb.init(
        entity=entity,
        project=project,
        id=mirror_run_id(plan["source_id"]),
        resume="never",
        name=plan["name"],
        notes=mirror_notes(plan["source_id"]),
        tags=[mirror_tag(plan["source_id"])],
        config=plan["config"],
        settings=settings,
        reinit=True,
    )
    try:
        for step, scalars in plan["rows"]:
            run.log(scalars, step=step)
    finally:
        url = run.url
        run.finish()
    return str(url)


KEEP_FILES = frozenset({"config.yaml", "wandb-summary.json"})


def scan_run(
    api: Any, path: str, literals: list[str], delete_extra_files: bool = True
) -> dict[str, Any]:
    """Scan a mirrored run's file names, file contents, config, summary and history."""
    run = api.run(path)
    counts: dict[str, int] = {}
    files_seen: list[str] = []
    deleted: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in list(run.files()):
            files_seen.append(f.name)
            if f.name not in KEEP_FILES and delete_extra_files:
                f.delete()
                deleted.append(f.name)
                continue
            counts = merge_counts(counts, scan_text(f.name, literals))
            local = Path(f.download(root=tmp, replace=True).name)
            text = local.read_text(encoding="utf-8", errors="replace")
            counts = merge_counts(counts, scan_text(text, literals))
    blobs = [
        json.dumps(dict(run.config), default=str),
        json.dumps(dict(run.summary), default=str),
        json.dumps(run.notes or ""),
        json.dumps(list(run.tags or [])),
        run.name or "",
    ]
    for blob in blobs:
        counts = merge_counts(counts, scan_text(blob, literals))
    n_hist = 0
    for row in run.scan_history():
        n_hist += 1
        counts = merge_counts(counts, scan_text(json.dumps(row, default=str), literals))
    return {
        "files": files_seen,
        "deleted_files": deleted,
        "history_rows_scanned": n_hist,
        "counts": counts,
        "total": sum(counts.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mirror W&B scalar history to a PRIVATE project.")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--source-project", default=SOURCE_PROJECT)
    parser.add_argument("--dest-project", default=DEST_PROJECT)
    parser.add_argument("--dry-run", action="store_true", help="read only; write nothing")
    parser.add_argument("--log", type=Path, default=Path("reports/final/wandb_mirror.json"))
    parser.add_argument("--only", nargs="*", default=None, help="source run ids to process")
    args = parser.parse_args(argv)

    import wandb

    root = Path(__file__).resolve().parent.parent
    api = wandb.Api(timeout=120)
    ids = args.only or list(SOURCE_RUNS)
    unknown = [i for i in ids if i not in SOURCE_RUNS]
    if unknown:
        print(f"unknown run ids: {unknown}", file=sys.stderr)
        return 2

    print("config whitelist:", json.dumps(CONFIG_WHITELIST))
    plans = [build_plan(api, args.entity, args.source_project, i) for i in ids]
    for p in plans:
        print(
            f"{p['source_id']} {p['name']}: {p['n_steps']} steps "
            f"({p['first_step']}..{p['last_step']}), {len(p['keys'])} keys"
        )
        print("  keys:", ", ".join(p["keys"]))
        print("  config:", json.dumps(p["config"], sort_keys=True))
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    visibility = ensure_private_project(api, args.entity, args.dest_project)
    print("destination project PRIVATE proven:", json.dumps(visibility))

    literals = load_literals(root)
    runs_log: list[dict[str, Any]] = []
    for plan in plans:
        sid = plan["source_id"]
        prior = existing_mirror(api, args.entity, args.dest_project, sid)
        if prior is not None:
            if prior.state != "finished":
                raise RuntimeError(f"existing mirror of {sid} is {prior.state!r}; resolve by hand")
            print(f"{sid}: already mirrored as {prior.id}, skipping upload")
            url = prior.url
        else:
            url = upload_run(args.entity, args.dest_project, plan)
            print(f"{sid}: uploaded -> {url}")
        dest_path = f"{args.entity}/{args.dest_project}/{mirror_run_id(sid)}"
        scan = scan_run(api, dest_path, literals)
        print(f"{sid}: scan total={scan['total']} counts={scan['counts']}")
        runs_log.append(
            {
                "source_id": sid,
                "name": plan["name"],
                "mirror_run_id": mirror_run_id(sid),
                "url": url,
                "steps_logged": plan["n_steps"],
                "first_step": plan["first_step"],
                "last_step": plan["last_step"],
                "keys_logged": plan["keys"],
                "config_logged": plan["config"],
                "scan": scan,
            }
        )

    final_vis = project_access(api, args.entity, args.dest_project)
    record = {
        "entity": args.entity,
        "project": args.dest_project,
        "project_url": f"https://wandb.ai/{args.entity}/{args.dest_project}",
        "project_visibility_proof": {
            "at_creation": visibility,
            "after_upload": final_vis,
        },
        "config_whitelist": list(CONFIG_WHITELIST),
        "scan_patterns": FORBIDDEN_PATTERNS,
        "scan_literals_count": len(literals),
        "runs": runs_log,
        "scan_total_all_runs": sum(r["scan"]["total"] for r in runs_log),
    }
    log_path = args.log if args.log.is_absolute() else root / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"wrote {log_path}")
    if str((final_vis or {}).get("access")).upper() != "PRIVATE":
        print("FAIL: project not PRIVATE after upload", file=sys.stderr)
        return 1
    return 0 if record["scan_total_all_runs"] == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
