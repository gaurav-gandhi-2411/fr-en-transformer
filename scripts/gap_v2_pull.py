from __future__ import annotations

# Read-only pulls for the EXPLORATORY / POST-HOC gap analysis v2 (scripts/gap_v2.py):
#   1. main's run from the private eval repo at the pinned revision WITH model weights, verified
#      against the uploaded manifest by scripts.eval_local.pull_run (private=True is checked there);
#   2. the private training-data dataset repo at the pinned revision (shards + tokenizer +
#      data_manifest.json), verified against shards/manifest.json (nmt.data.hub_data).
# Nothing is ever uploaded. The ambient HUGGINGFACEHUB_API_TOKEN / HF_TOKEN are removed from the
# process env first (the former is invalid); the cached `hf` login is used instead.
import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from nmt.data.hub_data import verify_shards_against_manifest
from scripts.eval_local import HubClient, _sha256, pull_run

EVAL_REPO = "OWNER/fr-en-transformer-eval"
EVAL_REVISION = "c3d8598252853fcd7df1ef4a00e8b0382b8f4351"
DATA_REPO = "OWNER/fr-en-transformer-data"
DATA_REVISION = "c40e393740f41dd3aac3e615952ac7b86d4e58e0"


def pull_data(dest: Path) -> dict[str, Any]:
    """Pull the pinned dataset revision into `dest`, verify shards + spm.model sha256 against
    data/shards/manifest.json, and return a record (repo, revision, private flag, file hashes)."""
    info = HfApi().repo_info(DATA_REPO, repo_type="dataset", revision=DATA_REVISION)
    if info.private is not True or info.sha != DATA_REVISION:
        raise RuntimeError(f"{DATA_REPO}: private={info.private} sha={info.sha}")
    snapshot_download(
        repo_id=DATA_REPO, repo_type="dataset", local_dir=str(dest), revision=DATA_REVISION
    )
    verify_shards_against_manifest(dest)
    files = {
        p.relative_to(dest).as_posix(): _sha256(p)
        for p in sorted(dest.rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }
    return {
        "hf_repo": DATA_REPO,
        "hf_revision": DATA_REVISION,
        "private": True,
        "verified_against": "data/shards/manifest.json (shard + spm.model sha256)",
        "files_sha256": files,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    for var in ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
        os.environ.pop(var, None)
    eval_dir = args.dest / "eval_main"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pull_run(HubClient(), EVAL_REPO, "main", EVAL_REVISION, eval_dir, with_model=True)
    record = {"eval": json.loads((eval_dir / "pull_record.json").read_text(encoding="utf-8"))}
    record["data"] = pull_data(args.dest / "data_repo")
    (args.dest / "pull_records.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("eval verified:", record["eval"]["verified"], "private:", record["eval"]["private"])
    print("model_files_checked:", record["eval"]["model_files_checked"])


if __name__ == "__main__":
    main()
