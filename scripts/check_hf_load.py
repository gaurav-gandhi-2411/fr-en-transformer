from __future__ import annotations

# Smoke test for a staged or published model: load it exactly as the model card documents, with
# NO credentials, and translate three invented sentences. Meant to run inside a FRESH venv that
# has only this repository's package installed, so it proves a stranger could do the same.
#
# It creates nothing global: HF_HOME points at a temp dir that is deleted on exit, and every
# Hugging Face token variable is removed before `huggingface_hub` is imported. If a token is
# still resolvable after that isolation, it refuses to run (a test that silently used your login
# would pass for the wrong reason, and could not be trusted for a public repo).
#
# CLI: `python scripts/check_hf_load.py <repo_id_or_local_dir>`; exit 0 only on PASS.
import os
import shutil
import sys
import tempfile
from pathlib import Path

TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN_PATH",
)

# Invented for this check; none comes from the provided dev, test or evaluation data.
SENTENCES = (
    "Mon voisin répare son vélo dans la cour.",
    "Nous avons réservé une table pour huit heures ce soir.",
    "La neige a retardé le train de vingt minutes.",
)


def isolate_environment(tmp_home: Path) -> list[str]:
    """Drop token variables and point HF_HOME at `tmp_home`. Returns the names that were set.

    Must run before `huggingface_hub` is imported: it reads these variables at import time.
    """
    removed = [name for name in TOKEN_ENV_VARS if os.environ.get(name)]
    for name in TOKEN_ENV_VARS:
        os.environ.pop(name, None)
    os.environ["HF_HOME"] = str(tmp_home)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return removed


def run(target: str) -> int:
    """Load `target` unauthenticated, translate the sentences, print a PASS/FAIL line."""
    tmp_home = Path(tempfile.mkdtemp(prefix="hf_check_home_"))
    try:
        removed = isolate_environment(tmp_home)
        if removed:
            print(f"note: ignored token variables set in this shell: {', '.join(removed)}")

        from huggingface_hub import get_token

        if get_token() is not None:
            print("FAIL: a Hugging Face token is still visible after isolation; refusing to run")
            return 1

        # The documented snippet, verbatim apart from the model id.
        from nmt.translate import Translator

        tr = Translator.from_pretrained(target, device="cpu")
        outputs = tr.translate(list(SENTENCES), beam=5, alpha=1.2, segment_threshold=192)

        for src, out in zip(SENTENCES, outputs, strict=True):
            print(f"  {src}\n    -> {out}")
        if len(outputs) != len(SENTENCES) or any(not o.strip() for o in outputs):
            print("FAIL: empty or missing translation")
            return 1
        print(f"PASS: loaded {target!r} unauthenticated and translated {len(outputs)} sentences")
        return 0
    except Exception as exc:  # noqa: BLE001 - any failure is a FAIL line, never a traceback pass
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: check_hf_load.py <repo_id_or_local_dir>", file=sys.stderr)
        return 2
    return run(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
