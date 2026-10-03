from __future__ import annotations

# The STAGE-1 INTERIM RESULT of the final_all session (GG requirement): right after `stage1-select`
# and BEFORE any stage-2 step, the session saves, validates and uploads the test predictions of the
# stage-1 WINNER (rank 1 model set at its stage-1 tuned beam config), so a usable submission exists
# if the session is interrupted later. Steps (names in the plan):
#   decode-stage1-test    decode ONLY the 330 test sentences -> <eval>/stage1_interim/
#                         test_predictions.json (+ decode_meta.json naming candidate and config)
#   validate-stage1-test  nmt.eval_l4 validate-test on that file -> stage1_interim/validation.json
#   upload-stage1         a PRIVATE commit under runs/final_all_stage1/ (test_predictions.json,
#                         validation.json, stage1.json, manifest.json with sha256 + size)
# Each step is idempotent (skips when its output validates; the upload skips on HF evidence). A
# failure of any of them stops the session before stage 2 (the notebook raises). The final
# `upload` of runs/final_all/ is unchanged and still carries everything.
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nmt import eval_l4 as ev
from nmt import final_all as fa

STAGE1_RUN = "final_all_stage1"
INTERIM_DIR = "stage1_interim"
STEP_DECODE = "decode-stage1-test"
STEP_VALIDATE = "validate-stage1-test"
STEP_UPLOAD = "upload-stage1"
INTERIM_STEPS = (STEP_DECODE, STEP_VALIDATE, STEP_UPLOAD)
COMMIT_MESSAGE = "eval_l4: final_all_stage1 interim (stage-1 winner test predictions)"
MANIFEST_NAME = "manifest.json"
UPLOAD_RECORD = "stage1_hf_upload.json"  # in <eval>/, next to hf_upload.json
FILES = ("test_predictions.json", "validation.json", "stage1.json")


def interim_dir(eval_dir: Path) -> Path:
    return Path(eval_dir) / INTERIM_DIR


def winner_of(stage1: Mapping[str, Any]) -> dict[str, Any]:
    """The stage-1 winner: rank 1 of stage1.json with its members and tuned beam config."""
    name = stage1["top2"][0]
    entry = stage1["candidates"][name]
    cfg = entry["config"]
    return {
        "candidate": name,
        "members": list(entry["members"]),
        "config": {
            "alpha": cfg["alpha"],
            "beam": cfg["beam"],
            "segment_threshold": cfg["segment_threshold"],
        },
    }


def _test_rows(test_inputs: Path | None) -> list[dict[str, Any]]:
    path = test_inputs or ev.REPO_ROOT / "data" / "test" / "inputs.jsonl"
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln]


def decode_stage1_test(
    eval_dir: Path,
    models: Mapping[str, Path],
    stage1_path: Path,
    tuning_dir: Path,
    batch_size: int = ev.EVAL_BATCH_SIZE,
    *,
    device: str | None = None,
    test_inputs: Path | None = None,
    build: Any = None,
) -> dict[str, Any]:
    """Decode ONLY the test set with the stage-1 winner at its stage-1 beam config. Refuses a
    missing/stale stage1.json. Skipped when the file validates AND decode_meta.json names this
    candidate and config; anything else (another winner/config) is discarded, never reused.
    `build` replaces fa.build_translator (tests)."""
    stage1 = fa._require_stage1(stage1_path, tuning_dir)
    win = winner_of(stage1)
    d = interim_dir(eval_dir)
    out, meta_path = d / "test_predictions.json", d / "decode_meta.json"
    rows = _test_rows(test_inputs)
    ids = [r["id"] for r in rows]
    want = {"candidate": win["candidate"], "config": win["config"]}
    prior = ev._read_json(meta_path)
    if (
        isinstance(prior, dict)
        and {k: prior.get(k) for k in want} == want
        and ev.prediction_file_valid(out, ids)
    ):
        print(f"{STEP_DECODE}: SKIP (valid {out})")
        return prior
    if out.exists():
        print(f"{STEP_DECODE}: predictions of another candidate/config found; discarding them")
        out.unlink()
    ev._write_json(meta_path, {**want, "members": win["members"], "done": False})  # config first
    translator = (build or fa.build_translator)(models, win["members"], device=device)
    cfg = win["config"]
    stats = ev._decode_to_file(
        translator,
        rows,
        out,
        beam=cfg["beam"],
        alpha=cfg["alpha"],
        segment_threshold=cfg["segment_threshold"],
        batch_size=batch_size,
    )
    meta = {**want, "members": win["members"], "done": True, "batch_size": batch_size, **stats}
    ev._write_json(meta_path, meta)
    print(f"{STEP_DECODE}: {win['candidate']} {len(rows)} sentences in {stats['seconds']}s")
    return meta


def collect_files(eval_dir: Path, stage1_path: Path) -> list[tuple[Path, str]]:
    """(local path, repo path) of the upload. Refuses unless validation.json says valid for THIS
    test_predictions.json (sha256) and every file exists."""
    d = interim_dir(eval_dir)
    pred, val = d / "test_predictions.json", d / "validation.json"
    record = ev._read_json(val)
    if not (
        isinstance(record, dict)
        and record.get("valid") is True
        and record.get("n_ids") == ev.EXPECTED_TEST_IDS
        and pred.is_file()
        and record.get("pred_sha256") == ev._sha256_file(pred)
    ):
        raise ev.EvalStepError(
            f"{STEP_UPLOAD}: {val} does not validate the current {pred} (330 ids, 0 empty); "
            "run validate-stage1-test"
        )
    files = [
        (pred, "test_predictions.json"),
        (val, "validation.json"),
        (Path(stage1_path), "stage1.json"),
    ]
    missing = [str(p) for p, _ in files if not p.is_file()]
    if missing:
        raise ev.EvalStepError(f"{STEP_UPLOAD}: missing file(s): {', '.join(missing)}")
    return [(p, f"runs/{STAGE1_RUN}/{rel}") for p, rel in files]


def build_manifest(files: list[tuple[Path, str]], winner: Mapping[str, Any]) -> dict[str, Any]:
    prefix = f"runs/{STAGE1_RUN}/"
    return {
        "schema": 1,
        "run": STAGE1_RUN,
        "kind": "stage1_interim",
        "winner": winner["candidate"],
        "winner_members": list(winner["members"]),
        "winner_config": dict(winner["config"]),
        "files": {
            rel.removeprefix(prefix): {"sha256": ev._sha256_file(p), "bytes": p.stat().st_size}
            for p, rel in files
        },
    }


def verify_on_hf(api: Any, repo_id: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Is exactly this interim upload on the PRIVATE repo? (privacy asserted first.) True only when
    a commit titled COMMIT_MESSAGE exists, the remote manifest equals `manifest` and every listed
    file has the manifest's size and sha256. Unreadable evidence raises (fail closed)."""
    ev.assert_repo_private(api, repo_id, "interim upload check")
    prefix = f"runs/{STAGE1_RUN}"
    try:
        head = getattr(api.repo_info(repo_id=repo_id, repo_type="model"), "sha", None)
        entries = list(
            api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=prefix,
                recursive=True,
                repo_type="model",
                revision=head,
            )
        )
        commits = list(api.list_repo_commits(repo_id=repo_id, repo_type="model", revision=head))
    except Exception as exc:
        if ev._status(exc) == 404:
            return {"complete": False, "reason": f"nothing under {prefix}/", "revision": None}
        raise ev.EvalStepError(f"interim verify: cannot read {repo_id} ({exc})") from exc
    upload = next((c for c in commits if getattr(c, "title", None) == COMMIT_MESSAGE), None)
    if upload is None:
        return {"complete": False, "reason": "no interim commit", "revision": None}
    remote = {e.path: e for e in entries if getattr(e, "size", None) is not None}

    def fetch(rel: str) -> Path:
        return Path(
            api.hf_hub_download(
                repo_id=repo_id, filename=f"{prefix}/{rel}", repo_type="model", revision=head
            )
        )

    if f"{prefix}/{MANIFEST_NAME}" not in remote:
        return {"complete": False, "reason": "manifest missing", "revision": None}
    if ev._read_json(fetch(MANIFEST_NAME)) != dict(manifest):
        return {"complete": False, "reason": "remote manifest differs", "revision": None}
    for rel, want in manifest["files"].items():
        entry = remote.get(f"{prefix}/{rel}")
        if entry is None or entry.size != want["bytes"]:
            return {"complete": False, "reason": f"{rel}: absent or wrong size", "revision": None}
        if (ev._lfs_sha256(entry) or ev._sha256_file(fetch(rel))) != want["sha256"]:
            return {"complete": False, "reason": f"{rel}: sha256 differs", "revision": None}
    return {
        "complete": True,
        "reason": "manifest and sha256 verified",
        "revision": upload.commit_id,
    }


def upload_stage1(
    api: Any, repo_id: str, eval_dir: Path, stage1_path: Path, tuning_dir: Path
) -> dict[str, Any]:
    """The interim PRIVATE upload commit under runs/final_all_stage1/. Order: refuse a stale
    stage1.json / unvalidated predictions, token scope, create_repo(private) + read-back (refuse
    unless private), skip on HF evidence, commit (files + manifest.json), privacy read-back again.
    Prints HF_STAGE1_REVISION=<sha> and records <eval>/stage1_hf_upload.json."""
    from huggingface_hub import CommitOperationAdd

    root = Path(eval_dir)
    if not ev._REPO_ID_RE.match(repo_id):
        raise ev.EvalStepError(f"HF_EVAL_REPO={repo_id!r} is not '<owner>/<name>'")
    stage1 = fa._require_stage1(stage1_path, tuning_dir)
    win = winner_of(stage1)
    meta = ev._read_json(interim_dir(root) / "decode_meta.json") or {}
    if (meta.get("candidate"), meta.get("config")) != (win["candidate"], win["config"]):
        raise ev.EvalStepError(
            f"{STEP_UPLOAD}: the interim predictions are not the stage-1 winner's"
        )
    files = collect_files(root, stage1_path)
    manifest = build_manifest(files, win)
    ev.check_write_token(api, repo_id)
    ev.ensure_private_repo(api, repo_id)  # create_repo(private=True, exist_ok) + read-back
    state = verify_on_hf(api, repo_id, manifest)
    if state["complete"]:
        ev.assert_repo_private(api, repo_id, "interim upload already done")
        print(f"{STEP_UPLOAD}: SKIP (verified on HF at revision {state['revision']})")
        revision, existing = state["revision"], True
    else:
        manifest_path = interim_dir(root) / MANIFEST_NAME
        ev._write_json(manifest_path, manifest)
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in files]
        ops.append(
            CommitOperationAdd(
                path_in_repo=f"runs/{STAGE1_RUN}/{MANIFEST_NAME}",
                path_or_fileobj=str(manifest_path),
            )
        )
        try:
            commit = api.create_commit(
                repo_id=repo_id, repo_type="model", operations=ops, commit_message=COMMIT_MESSAGE
            )
        except Exception as exc:
            if ev._is_auth_error(exc):
                raise ev.HFWriteTokenError(
                    ev.write_token_message(repo_id, f"interim upload refused: {exc}")
                ) from exc
            raise
        revision = getattr(commit, "oid", None) or getattr(commit, "commit_oid", None)
        ev.assert_repo_private(api, repo_id, "after the interim upload")
        if not revision or not ev._REVISION_RE.match(str(revision)):
            raise ev.EvalStepError(f"{STEP_UPLOAD}: no usable revision sha ({revision!r})")
        existing = False
    record = {
        "repo": repo_id,
        "run": STAGE1_RUN,
        "revision": revision,
        "private": True,
        "winner": win["candidate"],
        "files": [r for _, r in files] + [f"runs/{STAGE1_RUN}/{MANIFEST_NAME}"],
        "verified_existing": existing,
    }
    ev._write_json(root / UPLOAD_RECORD, record)
    print(f"HF_STAGE1_REVISION={revision}")
    return record
