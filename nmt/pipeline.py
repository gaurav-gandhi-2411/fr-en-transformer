from __future__ import annotations

# CLI entry point wiring every phase together: prepare|tokenize|train|evaluate|predict|analyze|
# export|all. Reproduce command:
# `python -m nmt.pipeline --config configs/main.yaml --stage all --seed 1234`.
#
# Each stage delegates to the already-tested module it wires up (nmt.data.prepare, nmt.data.
# tokenize, nmt.train, nmt.hub, nmt.evaluate, nmt.translate, nmt.analysis, nmt.submission) rather
# than re-implementing anything here -- this module's only job is argument/path plumbing.
import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nmt import evaluate as evaluate_mod
from nmt import submission as submission_mod
from nmt import tune as tune_mod
from nmt.analysis import run_analysis
from nmt.hub import export_checkpoint
from nmt.train import _build_model_config, build_eval_fn, default_eval_fn, load_config, train
from nmt.translate import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM,
    DEFAULT_SEGMENT_THRESHOLD,
    Translator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "prepare",
    "tokenize",
    "train",
    "evaluate",
    "predict",
    "analyze",
    "export",
    "tune",
    "all",
)

DEFAULT_TOKENIZER_PATH = REPO_ROOT / "tokenizer" / "spm.model"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_path(cfg_value: str, base: Path = REPO_ROOT) -> Path:
    p = Path(cfg_value)
    return p if p.is_absolute() else base / p


# ---------------------------------------------------------------------------------------------
# prepare / tokenize: thin delegation to each module's own CLI main()
# ---------------------------------------------------------------------------------------------


def stage_prepare(seed: int) -> int:
    from nmt.data.prepare import main as prepare_main

    return prepare_main(["--seed", str(seed)])


def stage_tokenize(seed: int) -> int:
    from nmt.data.tokenize import main as tokenize_main

    return tokenize_main(["--seed", str(seed)])


# ---------------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------------


def stage_train(
    config_path: Path,
    seed: int,
    resume: bool = False,
    wandb_mode: str = "offline",
    max_steps: int | None = None,
    synthetic: bool = False,
    cooldown_now: bool = False,
    run_dir: Path | None = None,
) -> None:
    cfg = load_config(config_path)
    if run_dir is not None:
        # CLI --run-dir override (never edits the committed config file itself): redirects both
        # the run/metrics dir and the checkpoint dir under one caller-chosen root, e.g. for a
        # fresh gate-run directory distinct from the committed configs/smoke.yaml's runs/smoke/.
        cfg.logging.run_dir = str(run_dir)
        cfg.ckpt.dir = str(Path(run_dir) / "ckpt")
    eval_fn = default_eval_fn if synthetic else build_eval_fn(cfg, seed)
    train(
        cfg,
        resume=resume,
        max_steps=max_steps,
        wandb_mode=wandb_mode,
        cooldown_now=cooldown_now,
        synthetic=synthetic,
        seed=seed,
        eval_fn=eval_fn,
    )


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------


def _latest_checkpoints(ckpt_dir: Path, n: int) -> list[Path]:
    files = sorted(Path(ckpt_dir).glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return files[-n:] if n > 0 else files


def stage_export(
    config_path: Path,
    out_dir: Path | None = None,
    average: bool = True,
    n_checkpoints: int | None = None,
    run_dir: Path | None = None,
) -> Path:
    cfg = load_config(config_path)
    ckpt_dir = (
        _resolve_path(str(Path(run_dir) / "ckpt"))
        if run_dir is not None
        else _resolve_path(cfg.ckpt.dir)
    )
    n = n_checkpoints if n_checkpoints is not None else cfg.ckpt.keep_last
    ckpts = _latest_checkpoints(ckpt_dir, n)
    if not ckpts:
        raise FileNotFoundError(f"stage_export: no checkpoints found under {ckpt_dir}")
    model_cfg = _build_model_config(cfg)
    out_dir = out_dir or (REPO_ROOT / "reports" / cfg.name / "export")
    return export_checkpoint(ckpts, out_dir, model_cfg, DEFAULT_TOKENIZER_PATH, average=average)


# ---------------------------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------------------------


def stage_tune(
    model_dir: Path,
    out_path: Path,
    alphas: tuple[float, ...] = tune_mod.DEFAULT_ALPHAS,
    beams: tuple[int, ...] = tune_mod.DEFAULT_BEAMS,
    seg_thresholds: tuple[int, ...] = tune_mod.DEFAULT_SEG_THRESHOLDS,
    limit_e1: int | None = None,
    limit_e2: int | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Decoding-tuning grid (alpha x beam, then segmentation threshold) on E1/E2 only, via
    `nmt.tune.run_tune` -- writes the full score table + winner to `out_path`."""
    return tune_mod.run_tune(
        model_dir,
        out_path,
        alphas=alphas,
        beams=beams,
        seg_thresholds=seg_thresholds,
        limit_e1=limit_e1,
        limit_e2=limit_e2,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------------------------


def stage_evaluate(
    model_dir: Path,
    run_name: str,
    ckpt_name: str,
    seed: int,
    beam: int = DEFAULT_BEAM,
    alpha: float = DEFAULT_ALPHA,
    segment_threshold: int | None = DEFAULT_SEGMENT_THRESHOLD,
    batch_size: int = 16,
    n_bootstrap: int = 1000,
    comet: bool = False,
    splits: tuple[str, ...] = ("dev", "e1", "e2", "e3", "e2synth"),
    out_dir: Path | None = None,
) -> dict[str, Any]:
    translator = Translator.from_pretrained(str(model_dir))
    decode_cfg = evaluate_mod.EvalRunConfig(
        beam_size=beam,
        alpha=alpha,
        segment_threshold=segment_threshold,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=seed,
        comet=comet,
    )
    return evaluate_mod.run_evaluation(
        translator,
        run_name,
        ckpt_name,
        decode_cfg,
        out_dir=out_dir,
        splits=splits,
        provenance={"model_dir": str(model_dir), "seed": seed},
    )


# ---------------------------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------------------------


def stage_predict(
    model_dir: Path,
    input_path: Path,
    output_path: Path,
    beam: int = DEFAULT_BEAM,
    alpha: float = DEFAULT_ALPHA,
    batch_size: int = 32,
    segment_threshold: int | None = DEFAULT_SEGMENT_THRESHOLD,
    validate: bool = True,
) -> dict[str, Any]:
    translator = Translator.from_pretrained(str(model_dir))
    rows = _read_jsonl(input_path)
    ids = [r["id"] for r in rows]
    sources = [r["source"] for r in rows]
    translations = translator.translate(
        sources,
        batch_size=batch_size,
        beam=beam,
        alpha=alpha,
        segment_threshold=segment_threshold,
    )
    result = dict(zip(ids, translations, strict=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if validate:
        return submission_mod.validate_submission(output_path)
    return {"n_ids": len(result)}


# ---------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------


def stage_analyze(
    eval_dir: Path,
    out_dir: Path | None = None,
    other_eval_json_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Reads `{e1,e3,dev}_predictions.json` written by `stage_evaluate`/`run_evaluation` under
    `eval_dir`, then runs the full analysis. `other_eval_json_paths` (optional, `{label:
    path}`) lets the length-bucket figure overlay multiple runs (e.g. S1 vs S2, once those
    exist)."""
    e1_pred = json.loads((eval_dir / "e1_predictions.json").read_text(encoding="utf-8"))
    e3_pred = json.loads((eval_dir / "e3_predictions.json").read_text(encoding="utf-8"))
    dev_pred = json.loads((eval_dir / "dev_predictions.json").read_text(encoding="utf-8"))
    other_eval_jsons = None
    if other_eval_json_paths:
        other_eval_jsons = {
            label: json.loads(path.read_text(encoding="utf-8"))
            for label, path in other_eval_json_paths.items()
        }
    elif (eval_dir / "eval.json").is_file():
        other_eval_jsons = {
            eval_dir.name: json.loads((eval_dir / "eval.json").read_text(encoding="utf-8"))
        }
    return run_analysis(e1_pred, e3_pred, dev_pred, out_dir or eval_dir, other_eval_jsons)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fr-en-transformer pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="Override run_dir (not the committed config)."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--wandb", choices=["offline", "online", "disabled"], default="offline")
    parser.add_argument("--model", type=Path, default=None, help="Export dir for evaluate/predict.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "test" / "inputs.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-threshold", type=int, default=DEFAULT_SEGMENT_THRESHOLD)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--eval-dir", type=Path, default=None, help="For --stage analyze.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg_for_names = load_config(args.config) if args.config.is_file() else None
    run_name = cfg_for_names.name if cfg_for_names else args.config.stem
    # --run-dir redirects every stage's output under one caller-chosen root instead of the
    # committed config's own runs/<name>/ + reports/<name>/ paths (never edits the config file).
    report_root = args.run_dir if args.run_dir is not None else (REPO_ROOT / "reports" / run_name)

    if args.stage in ("prepare", "all"):
        stage_prepare(args.seed)
    if args.stage in ("tokenize", "all"):
        stage_tokenize(args.seed)
    if args.stage in ("train", "all"):
        stage_train(
            args.config,
            args.seed,
            resume=args.resume,
            wandb_mode=args.wandb,
            max_steps=args.max_steps,
            synthetic=args.synthetic,
            run_dir=args.run_dir,
        )
    export_dir = args.model
    if args.stage in ("export", "all"):
        export_dir = stage_export(args.config, out_dir=report_root / "export", run_dir=args.run_dir)
        print(f"export: wrote {export_dir}")
    tune_result: dict[str, Any] | None = None
    if args.stage in ("tune", "all"):
        model_dir = export_dir or args.model
        if model_dir is None:
            raise ValueError("--stage tune requires --model (or run --stage export/all first)")
        tune_out = report_root / "selection_grid.json"
        tune_result = stage_tune(model_dir, tune_out)
        w = tune_result["winner"]
        print(
            f"tune: wrote {tune_out} (winner alpha={w['alpha']} beam={w['beam']} "
            f"segment_threshold={w['segment_threshold']})"
        )
    if args.stage in ("evaluate", "all"):
        model_dir = export_dir or args.model
        if model_dir is None:
            raise ValueError("--stage evaluate requires --model (or run --stage export/all first)")
        # `all` runs `tune` (the E1/E2-only search) right before this, so its winning
        # alpha/beam/segment_threshold replace the CLI defaults and the reported numbers are for
        # the tuned config. A standalone `--stage evaluate` run keeps using the CLI flags.
        beam, alpha, segment_threshold = args.beam, args.alpha, args.segment_threshold
        if tune_result is not None:
            w = tune_result["winner"]
            beam, alpha, segment_threshold = w["beam"], w["alpha"], w["segment_threshold"]
        eval_dir = report_root / "eval" / Path(model_dir).name
        stage_evaluate(
            model_dir,
            run_name,
            Path(model_dir).name,
            args.seed,
            beam=beam,
            alpha=alpha,
            segment_threshold=segment_threshold,
            batch_size=args.batch_size,
            n_bootstrap=args.n_bootstrap,
            comet=args.comet,
            out_dir=eval_dir,
        )
        print(f"evaluate: wrote {eval_dir / 'eval.json'}")
    if args.stage in ("predict", "all"):
        model_dir = export_dir or args.model
        if model_dir is None:
            raise ValueError("--stage predict requires --model (or run --stage export/all first)")
        out_path = args.output or (report_root / "test_predictions.json")
        beam, alpha, segment_threshold = args.beam, args.alpha, args.segment_threshold
        if tune_result is not None:  # as for `evaluate`: predict at the config tuning chose
            w = tune_result["winner"]
            beam, alpha, segment_threshold = w["beam"], w["alpha"], w["segment_threshold"]
        info = stage_predict(
            model_dir,
            args.input,
            out_path,
            beam=beam,
            alpha=alpha,
            batch_size=args.batch_size,
            segment_threshold=segment_threshold,
        )
        print(f"predict: wrote {out_path} ({info['n_ids']} ids), validated OK")
    if args.stage in ("analyze", "all"):
        model_dir = export_dir or args.model
        eval_dir = args.eval_dir or (
            report_root / "eval" / (Path(model_dir).name if model_dir else "export")
        )
        stage_analyze(eval_dir)
        print(f"analyze: wrote {eval_dir / 'analysis.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
