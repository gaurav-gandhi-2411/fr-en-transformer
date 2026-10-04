from __future__ import annotations

# Verifies the W&B project is PRIVATE before any online run (nothing is public until approved,
# Reads `access` straight from the W&B GraphQL API; exits 1 if the project is missing
# or not PRIVATE. `--out` records the answer for provenance.
#
# CLI: `python scripts/check_wandb_project.py [--entity E] [--project P] [--out FILE]`
import argparse
import json
from pathlib import Path

ENTITY = "gauravgandhi429-gaurav-gandhi"
PROJECT = "fr-en-transformer"
_QUERY = (
    "query P($e:String!,$p:String!){ project(name:$p, entityName:$e)"
    "{ name entityName access createdAt } }"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that the W&B project is PRIVATE.")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import wandb  # lazy: only needed when the check actually runs

    api = wandb.Api()
    result = api._service_api.execute_graphql(_QUERY, {"e": args.entity, "p": args.project})
    project = result.get("project")
    record = {"viewer": api.viewer.username, "project": project}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))
    if project is None or project.get("access") != "PRIVATE":
        print("FAIL: W&B project missing or not PRIVATE")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
