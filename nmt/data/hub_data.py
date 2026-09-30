from __future__ import annotations

# Push/pull of the P2 tokenizer + shard artifacts to/from the private Hugging Face dataset repo
# (spec §1: "CPU work runs locally ... artifacts are pushed to a private HF dataset repo, and
# Colab only pulls shards and trains"). `push_to_hub` is used once locally after `tokenize.py`
# finishes; `pull_from_hub` is the function the Colab notebook (P5) calls, and it re-verifies
# every shard's sha256 against the downloaded `manifest.json` before returning, so a corrupted or
# truncated download fails loudly instead of silently training on bad data.
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def push_to_hub(
    repo_id: str,
    shards_dir: Path,
    tokenizer_dir: Path,
    eval_dir: Path,
    data_manifest_path: Path,
    commit_message: str = "Push P2 tokenizer + shards",
) -> dict[str, Any]:
    """Create (or reuse) a **private** HF dataset repo `repo_id` and upload `shards_dir` (as
    `data/shards/`), `tokenizer_dir` (as `tokenizer/`), `eval_dir` (as `data/eval/`) and
    `data_manifest_path` (as `data/data_manifest.json`). Verifies the repo is private
    immediately after creation, before any upload — raises if it is not, per the standing
    "never make anything public" instruction. Returns the final HF commit sha and the
    verified-private flag.
    """
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if not info.private:
        raise RuntimeError(
            f"refusing to upload: HF dataset repo {repo_id!r} is not private "
            f"(private={info.private})"
        )

    last_commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(shards_dir),
        path_in_repo="data/shards",
        commit_message=f"{commit_message}: data/shards",
    )
    last_commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(tokenizer_dir),
        path_in_repo="tokenizer",
        commit_message=f"{commit_message}: tokenizer",
    )
    last_commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(eval_dir),
        path_in_repo="data/eval",
        commit_message=f"{commit_message}: data/eval",
    )
    last_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(data_manifest_path),
        path_in_repo="data/data_manifest.json",
        commit_message=f"{commit_message}: data_manifest.json",
    )
    return {
        "repo_id": repo_id,
        "private": True,
        "commit_sha": getattr(last_commit, "oid", None),
    }


def verify_shards_against_manifest(local_dir: Path) -> dict[str, Any]:
    """Re-verify every shard file's sha256 (and `tokenizer/spm.model`'s, if the manifest
    records one) under `local_dir` against `local_dir/data/shards/manifest.json` — raising
    `RuntimeError` naming every mismatch/missing file, never silently passing on a partial or
    corrupted download. Pure/local/network-free (unlike `pull_from_hub`, which calls this after
    `snapshot_download`), so it's directly unit-testable. Returns the parsed manifest on success.
    """
    local_dir = Path(local_dir)
    manifest_path = local_dir / "data" / "shards" / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"verify_shards_against_manifest: manifest.json not found at {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mismatches: list[str] = []
    for rel_path, expected_sha in manifest["shard_files_sha256"].items():
        f = local_dir / "data" / "shards" / rel_path
        if not f.is_file():
            mismatches.append(f"{rel_path}: MISSING")
            continue
        actual_sha = _sha256_of(f)
        if actual_sha != expected_sha:
            mismatches.append(f"{rel_path}: expected {expected_sha}, got {actual_sha}")

    spm_path = local_dir / "tokenizer" / "spm.model"
    expected_spm_sha = manifest.get("spm_model_sha256")
    if expected_spm_sha is not None:
        if not spm_path.is_file():
            mismatches.append("tokenizer/spm.model: MISSING")
        elif _sha256_of(spm_path) != expected_spm_sha:
            mismatches.append(
                f"tokenizer/spm.model: expected {expected_spm_sha}, got {_sha256_of(spm_path)}"
            )

    if mismatches:
        raise RuntimeError(
            "verify_shards_against_manifest: sha256 verification failed for "
            f"{len(mismatches)} file(s):\n" + "\n".join(mismatches)
        )
    logger.info(
        "verify_shards_against_manifest: verified %d shard files + spm.model against manifest.json",
        len(manifest["shard_files_sha256"]),
    )
    return manifest


def pull_from_hub(repo_id: str, local_dir: Path, revision: str | None = None) -> Path:
    """Download `repo_id` (an HF dataset repo) into `local_dir`, then re-verify every shard
    file's sha256 against the downloaded `data/shards/manifest.json` (`verify_shards_against_
    manifest`) — failing loudly on any mismatch or missing file. This is the function the Colab
    notebook (P5) calls before training ever starts, so a truncated/corrupted download is caught
    before it can silently produce a bad run.
    """
    local_dir = Path(local_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        revision=revision,
    )
    verify_shards_against_manifest(local_dir)
    return local_dir
