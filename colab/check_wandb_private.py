from __future__ import annotations

# Fails-closed check that a W&B project exists and is PRIVATE. Runs as a SUBPROCESS of the
# notebook (through run_step) so `import wandb` happens in a fresh interpreter that sees the
# packages pip just installed/upgraded on disk, never in the Colab kernel that imported the
# older preinstalled copies at startup. Prints `W&B project <entity>/<project>: access=<...>`.
#
# Usage: python colab/check_wandb_private.py <entity> <project>   (env WANDB_API_KEY required)
# Exit 0 only when the project is verified PRIVATE; any other outcome (missing, public, API
# error, unexpected exception) exits 1 with the reason on stderr.
import os
import sys
import time
from typing import Any

PROJECT_ACCESS_QUERY = """
query ProjectAccess($name: String!, $entity: String!) {
  project(name: $name, entityName: $entity) { id name entityName access }
}
"""


def verify_project_is_private(
    entity: str, project: str, api: Any = None, sleep: Any = time.sleep
) -> str:
    """Return the project's access level; raise unless it exists and is PRIVATE (fail closed:
    an unverifiable privacy state is treated as a failure, never as a pass).
    """
    if api is None:
        import wandb

        api = wandb.Api()
    info = None
    for _attempt in range(3):
        try:
            # `access` is not exposed on wandb.Project's public attributes, hence the direct query.
            data = api._service_api.execute_graphql(
                PROJECT_ACCESS_QUERY, variables={"name": project, "entity": entity}
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not verify W&B project {entity}/{project} visibility "
                f"({type(exc).__name__}: {exc}). Check WANDB_API_KEY belongs to entity "
                f"{entity!r}, and that the project exists (create it as Private in the W&B UI)."
            ) from exc
        info = (data or {}).get("project")
        if info:
            break
        # Observed once: the very first lookup returned null for an existing project and the
        # next identical call succeeded, so retry before declaring the project missing.
        sleep(2.0)
    if not info:
        raise RuntimeError(
            f"W&B project {entity}/{project} does not exist or is not visible to this API key "
            "(3 lookups). Create it as PRIVATE at https://wandb.ai/ (or fix WANDB_API_KEY / "
            "WANDB_ENTITY)."
        )
    access = str(info.get("access"))
    print(f"W&B project {entity}/{project}: access={access}")
    if access.upper() != "PRIVATE":
        raise RuntimeError(
            f"W&B project {entity}/{project} has access={access!r}, not PRIVATE. Everything stays "
            "private until GG approves (spec 13): set Project -> Settings -> Visibility to "
            "Private, then re-run."
        )
    return access


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: check_wandb_private.py <entity> <project>", file=sys.stderr)
        return 2
    entity, project = args
    try:
        import wandb

        wandb.login(key=os.environ["WANDB_API_KEY"])
        verify_project_is_private(entity, project)
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY error, never pass silently
        print(f"W&B privacy check FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
