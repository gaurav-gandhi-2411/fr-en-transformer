from __future__ import annotations

# CPU production benchmark of the main run's `final` model: fp32 vs int8 dynamic quantization
# (PREREG staged-selection amendment, item "production config" reports latency).
#
# Everything is measured through the production API (`nmt.translate.Translator.translate`, i.e.
# normalization + SentencePiece tokenization + beam/greedy decode + detokenization), on CPU, in a
# FRESH subprocess per (phase, precision, thread-setting, repetition) so that peak RSS and warm-up
# state are never shared between configurations. Quality is scored only through
# `nmt.evaluate.run_official_scorer` (official/score.py as shipped, PYTHONUTF8=1) and
# `nmt.compare`'s paired bootstrap machinery.
#
# Subcommands (cwd must be the repo root; run with the repo venv):
#   run        launch every missing worker job (resumable; results land in --work-dir)
#   aggregate  score the quality predictions, build results.json / raw_latencies.json / README.md
#   worker     INTERNAL: one measurement in this process (spawned by `run`)
#
# int8 scheme: `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`:
# the weights of every nn.Linear (attention q/k/v/out projections, FFN fc1/fc2) are stored int8
# (per-tensor, symmetric), activations are quantized on the fly per call, biases stay fp32.
# NOT quantized: the (tied) token embedding nn.Embedding, all LayerNorms, and the tied OUTPUT
# PROJECTION, which the model computes as `F.linear(out, self.embed.weight)` (a functional call
# on the embedding weight, not an nn.Linear module, so module-based dynamic quantization cannot
# see it). Quantizing it would need a model code change, which is out of scope here.
import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "OWNER/fr-en-transformer-eval"  # private; read-only pull
HF_REVISION = "c3d8598252853fcd7df1ef4a00e8b0382b8f4351"
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "prod_pull" / "source" / "runs" / "main" / "model"
DEFAULT_PULL_RECORD = REPO_ROOT / "runs" / "prod_pull" / "pull_record.json"
DEFAULT_WORK_DIR = REPO_ROOT / "runs" / "prod_work"  # gitignored (runs/)
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "final" / "production"
SELECTION_JSON = REPO_ROOT / "reports" / "final" / "main" / "selection.json"
COLAB_E1_PREDICTIONS = (
    REPO_ROOT / "reports" / "final" / "main" / "seg_tuned" / "e1_predictions.json"
)

# Fixed BEFORE any measurement (the deterministic sample is a pure function of these constants).
SEED = 1234
N_LATENCY = 200  # timed single-sentence requests per mode
N_WARMUP = 10  # untimed single-sentence requests (disjoint from the timed sample) per mode
N_THROUGHPUT = 320  # sentences per throughput run (10 batches of 32)
BATCH_SIZE = 32
BEAM = 5
ALPHA = 1.2  # reports/final/main/selection.json winner (final)
SEG_THRESHOLD = 192  # same
N_BOOTSTRAP = 1000
BG_SAMPLE_SECONDS = 5.0
BG_FLAG_PCT = 30.0
SPREAD_FLAG_FRAC = 0.10  # (max - min) / median of per-run p50 above this is called "large"

PRECISIONS = ("fp32", "int8")
THREAD_SETTINGS = ("default", "1")
# mode -> (beam, segment_threshold)
MODES: dict[str, tuple[int, int | None]] = {
    "greedy": (1, SEG_THRESHOLD),
    "beam5": (BEAM, SEG_THRESHOLD),
    "beam5_noseg": (BEAM, None),
}
QUALITY_MODES = ("beam5", "greedy")


# --------------------------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """q-th percentile (0..100) by linear interpolation between order statistics (the same
    definition as numpy's default). With n=200 the p99 rests on the top two values."""
    if not values:
        raise ValueError("percentile of an empty list")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def summarize_latency(ms: list[float]) -> dict[str, float | int]:
    """n, mean, min, p50, p95, p99, max of per-request latencies in milliseconds."""
    return {
        "n": len(ms),
        "mean": float(statistics.fmean(ms)),
        "min": float(min(ms)),
        "p50": percentile(ms, 50),
        "p95": percentile(ms, 95),
        "p99": percentile(ms, 99),
        "max": float(max(ms)),
    }


def deterministic_split(
    n_rows: int,
    seed: int = SEED,
    n_latency: int = N_LATENCY,
    n_warmup: int = N_WARMUP,
    n_throughput: int = N_THROUGHPUT,
) -> dict[str, list[int]]:
    """Disjoint index sets from one seeded permutation of range(n_rows): timed latency sample,
    warm-up sample, throughput sample. Pure function of its arguments."""
    need = n_latency + n_warmup + n_throughput
    if need > n_rows:
        raise ValueError(f"need {need} rows but only {n_rows} available")
    perm = random.Random(seed).sample(range(n_rows), n_rows)
    return {
        "latency": perm[:n_latency],
        "warmup": perm[n_latency : n_latency + n_warmup],
        "throughput": perm[n_latency + n_warmup : need],
    }


def length_stats(values: list[int]) -> dict[str, float | int]:
    """n, mean, median, p95, min, max of integer lengths."""
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "median": percentile([float(v) for v in values], 50),
        "p95": percentile([float(v) for v in values], 95),
        "min": min(values),
        "max": max(values),
    }


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_int8_modules(model: nn.Module) -> dict[str, Any]:
    """Which modules `quantize_dynamic(model, {nn.Linear})` will quantize and which not, computed
    from the module tree BEFORE quantization. Parameter counts are exact."""
    quantized: list[str] = []
    linear_weight = 0
    linear_bias = 0
    skipped: list[dict[str, Any]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            quantized.append(name)
            linear_weight += mod.weight.numel()
            linear_bias += mod.bias.numel() if mod.bias is not None else 0
        elif list(mod.parameters(recurse=False)):
            skipped.append(
                {
                    "name": name,
                    "type": type(mod).__name__,
                    "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                }
            )
    total = sum(p.numel() for p in model.parameters())
    return {
        "quantized_module_type": "torch.nn.Linear",
        "n_quantized_modules": len(quantized),
        "quantized_modules": quantized,
        "quantized_weight_params": linear_weight,
        "linear_bias_params_kept_fp32": linear_bias,
        "total_params": total,
        "not_quantized_params": total - linear_weight,
        "not_quantized_modules_with_params": skipped,
    }


def quantize_int8(model: nn.Module) -> nn.Module:
    """In-place dynamic int8 quantization of every nn.Linear (see module header)."""
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8, inplace=True
    )


def median_min_max(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n_runs": len(values),
    }


def aggregate_runs(runs: list[dict[str, float]], keys: list[str]) -> dict[str, Any]:
    """Across-run summary: median/min/max of each per-run statistic in `keys`, plus the relative
    spread (max-min)/median of the first key and whether it exceeds SPREAD_FLAG_FRAC."""
    out: dict[str, Any] = {k: median_min_max([r[k] for r in runs]) for k in keys}
    first = out[keys[0]]
    spread = (first["max"] - first["min"]) / first["median"] if first["median"] else float("nan")
    out["relative_spread_of_" + keys[0]] = spread
    out["spread_large"] = bool(spread > SPREAD_FLAG_FRAC)
    return out


# --------------------------------------------------------------------------------------------
# system probes
# --------------------------------------------------------------------------------------------


class _PMC(ctypes.Structure):
    _fields_ = [  # noqa: RUF012 - ctypes layout
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _win_mem_counters() -> _PMC:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return pmc


def peak_rss_mb() -> tuple[float, str]:
    """(peak resident set of THIS process in MiB, which API produced it). Windows: psutil's
    `peak_wset` when psutil is importable, else ctypes GetProcessMemoryInfo.PeakWorkingSetSize
    (the same counter). Elsewhere: resource.ru_maxrss."""
    if sys.platform == "win32":
        try:
            import psutil

            return psutil.Process().memory_info().peak_wset / 1024**2, "psutil.peak_wset"
        except ImportError:
            return _win_mem_counters().PeakWorkingSetSize / 1024**2, "ctypes.PeakWorkingSetSize"
    import resource

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (kb / 1024.0 if sys.platform != "darwin" else kb / 1024.0**2), "resource.ru_maxrss"


def current_rss_mb() -> float:
    """Current working set of this process in MiB (psutil if importable, else ctypes on Windows)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        if sys.platform == "win32":
            return _win_mem_counters().WorkingSetSize / 1024**2
        return float("nan")


def cpu_utilisation_pct(seconds: float) -> tuple[float | None, str]:
    """Total (all logical CPUs) utilisation over `seconds`, from psutil, else Windows
    GetSystemTimes. Returns (percent or None, source)."""
    try:
        import psutil

        return float(psutil.cpu_percent(interval=seconds)), "psutil.cpu_percent"
    except ImportError:
        pass
    if sys.platform != "win32":
        return None, "unavailable"
    from ctypes import wintypes

    def times() -> tuple[int, int, int]:
        idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
            ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)
        )
        f = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime  # noqa: E731
        return f(idle), f(kern), f(user)

    i0, k0, u0 = times()
    time.sleep(seconds)
    i1, k1, u1 = times()
    total = (k1 - k0) + (u1 - u0)  # kernel time includes idle
    return (100.0 * (total - (i1 - i0)) / total if total else None), "ctypes.GetSystemTimes"


def _run_text(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


def cpu_model_name() -> str:
    """CPU model string read from the system (Windows registry; /proc/cpuinfo; platform)."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ac_power_status() -> str | None:
    """Windows GetSystemPowerStatus.ACLineStatus: 'AC', 'battery' or 'unknown'."""
    if sys.platform != "win32":
        return None

    class _SPS(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - ctypes layout
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    s = _SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):  # type: ignore[attr-defined]
        return None
    return {0: "battery", 1: "AC"}.get(s.ACLineStatus, "unknown")


def git_state() -> dict[str, Any]:
    sha = _run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    dirty = _run_text(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"]
    )
    return {"code_sha": sha, "tracked_files_dirty": bool(dirty)}


def default_thread_count() -> int:
    """torch's default intra-op thread count, read in a clean subprocess."""
    out = _run_text([sys.executable, "-c", "import torch;print(torch.get_num_threads())"])
    return int(out) if out else -1


def hardware_context() -> dict[str, Any]:
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        ram = psutil.virtual_memory().total / 1024**3
    except ImportError:
        physical, ram = None, None
    return {
        "cpu_model": cpu_model_name(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical,
        "ram_gib": ram,
        "torch_default_num_threads": default_thread_count(),
        "torch": torch.__version__,
        "torch_quantized_engine": torch.backends.quantized.engine,
        "torch_supported_quantized_engines": list(torch.backends.quantized.supported_engines),
        "python": platform.python_version(),
        "os": platform.platform(),
        "power_plan": _run_text(["powercfg", "/getactivescheme"])
        if sys.platform == "win32"
        else None,
        "power_source": ac_power_status(),
    }


# --------------------------------------------------------------------------------------------
# worker: one measurement in this process
# --------------------------------------------------------------------------------------------


def load_translator(model_dir: Path, precision: str, threads: str) -> Any:
    """CPU Translator from an exported model dir; int8 = `quantize_int8` applied after load."""
    from nmt.translate import Translator

    if threads != "default":
        torch.set_num_threads(int(threads))
    translator = Translator.from_pretrained(str(model_dir), device="cpu")
    if precision == "int8":
        quantize_int8(translator.model)
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")
    return translator


def load_e1(model_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from nmt.evaluate import load_split

    return load_split("e1")


def request_latencies_ms(
    translator: Any, texts: list[str], beam: int, seg: int | None
) -> list[float]:
    """One `translate([text])` call per sentence, wall time in ms each (end to end: normalize +
    tokenize + decode + detokenize). `perf_counter` around the whole call."""
    out: list[float] = []
    for text in texts:
        t0 = time.perf_counter()
        translator.translate([text], batch_size=1, beam=beam, alpha=ALPHA, segment_threshold=seg)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def phase_latency(translator: Any, split_texts: dict[str, list[str]], modes: list[str]) -> dict:
    result: dict[str, Any] = {}
    for mode in modes:
        beam, seg = MODES[mode]
        for text in split_texts["warmup"]:  # untimed warm-up in the SAME mode
            translator.translate(
                [text], batch_size=1, beam=beam, alpha=ALPHA, segment_threshold=seg
            )
        ms = request_latencies_ms(translator, split_texts["latency"], beam, seg)
        result[mode] = {"summary": summarize_latency(ms), "ms": [round(v, 3) for v in ms]}
    return result


def phase_throughput(translator: Any, split_texts: dict[str, list[str]]) -> dict:
    result: dict[str, Any] = {}
    texts = split_texts["throughput"]
    for mode in ("greedy", "beam5"):
        beam, seg = MODES[mode]
        for text in split_texts["warmup"]:
            translator.translate(
                [text], batch_size=1, beam=beam, alpha=ALPHA, segment_threshold=seg
            )
        t0 = time.perf_counter()
        outs = translator.translate(
            texts, batch_size=BATCH_SIZE, beam=beam, alpha=ALPHA, segment_threshold=seg
        )
        wall = time.perf_counter() - t0
        # output tokens: SentencePiece tokens of the detokenized hypotheses (as nmt.eval_l4's bench)
        out_tokens = sum(len(translator.sp.encode(o, out_type=int)) for o in outs)
        result[mode] = {
            "n_sentences": len(texts),
            "batch_size": BATCH_SIZE,
            "wall_seconds": wall,
            "sentences_per_second": len(texts) / wall,
            "output_tokens": out_tokens,
            "output_tokens_per_second": out_tokens / wall,
        }
    return result


def phase_quality(translator: Any, model_dir: Path, pred_dir: Path, precision: str) -> dict:
    inputs, _ = load_e1(model_dir)
    ids = [r["id"] for r in inputs]
    sources = [r["source"] for r in inputs]
    pred_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for mode in QUALITY_MODES:
        beam, seg = MODES[mode]
        before = (translator.stats.n_beam, translator.stats.n_greedy_fallback)
        t0 = time.perf_counter()
        outs = translator.translate(
            sources, batch_size=BATCH_SIZE, beam=beam, alpha=ALPHA, segment_threshold=seg
        )
        wall = time.perf_counter() - t0
        path = pred_dir / f"e1_{precision}_{mode}.json"
        data = json.dumps(dict(zip(ids, outs, strict=True)), ensure_ascii=False, indent=2) + "\n"
        path.write_bytes(data.encode("utf-8"))  # bytes: LF on every OS
        result[mode] = {
            "n": len(ids),
            "wall_seconds": wall,
            "predictions": path.name,
            "fallback_counts": {
                "beam": translator.stats.n_beam - before[0],
                "greedy": translator.stats.n_greedy_fallback - before[1],
            },
        }
    return result


def phase_size(model_dir: Path, precision: str, state_path: Path) -> dict:
    """Parameter counts and on-disk sizes. fp32 safetensors is the shipped file; the state_dict is
    additionally `torch.save`d (fp32 and int8 alike) for an apples-to-apples pair."""
    from nmt.translate import Translator

    state_path.parent.mkdir(parents=True, exist_ok=True)
    translator = Translator.from_pretrained(str(model_dir), device="cpu")
    selection = select_int8_modules(translator.model)
    sd = translator.model.state_dict()
    out: dict[str, Any] = {
        "param_count_fp32": selection["total_params"],
        "safetensors_bytes": (model_dir / "model.safetensors").stat().st_size,
        "module_selection": selection,
    }
    if precision == "int8":
        quantize_int8(translator.model)
        sd = translator.model.state_dict()
    torch.save(sd, state_path)
    out["state_dict_file"] = state_path.name
    out["state_dict_torch_save_bytes"] = state_path.stat().st_size
    out["state_dict_sha256"] = file_sha256(state_path)
    return out


class LoadMeter:
    """Machine-wide CPU utilisation while a phase runs, minus this process's own share, so a run
    can state how much OTHER work was on the machine during the measurement (the 5 s pre-run
    sample cannot). Needs psutil; without it `stop()` returns Nones."""

    def __init__(self) -> None:
        try:
            import psutil

            self._ps = psutil
            self._proc = psutil.Process()
        except ImportError:
            self._ps = None

    def start(self) -> None:
        if self._ps is None:
            return
        self._ps.cpu_percent(interval=None)  # primes the system-wide counter
        t = self._proc.cpu_times()
        self._cpu0, self._t0 = t.user + t.system, time.perf_counter()

    def stop(self) -> dict[str, float | None]:
        if self._ps is None:
            return {
                "system_cpu_pct_during": None,
                "own_cpu_pct_of_machine": None,
                "other_cpu_pct_during": None,
            }
        system = float(self._ps.cpu_percent(interval=None))
        t = self._proc.cpu_times()
        wall = time.perf_counter() - self._t0
        own = 100.0 * ((t.user + t.system) - self._cpu0) / wall / (os.cpu_count() or 1)
        return {
            "system_cpu_pct_during": system,
            "own_cpu_pct_of_machine": own,
            "other_cpu_pct_during": max(0.0, system - own),
        }


def worker_main(args: argparse.Namespace) -> int:
    t_start = time.perf_counter()
    model_dir = Path(args.model_dir)
    pre_threads = torch.get_num_threads()
    result: dict[str, Any] = {
        "phase": args.phase,
        "precision": args.precision,
        "threads_setting": args.threads,
    }
    if args.phase == "size":
        result["size"] = phase_size(
            model_dir, args.precision, Path(args.pred_dir) / f"state_{args.precision}.pt"
        )
    else:
        translator = load_translator(model_dir, args.precision, args.threads)
        result["rss_after_load_mb"] = current_rss_mb()
        inputs, _ = load_e1(model_dir)
        split = deterministic_split(len(inputs))
        texts = {k: [inputs[i]["source"] for i in idx] for k, idx in split.items()}
        meter = LoadMeter()
        meter.start()
        if args.phase == "latency":
            result["latency"] = phase_latency(translator, texts, args.modes.split(","))
        elif args.phase == "throughput":
            result["throughput"] = phase_throughput(translator, texts)
        elif args.phase == "quality":
            result["quality"] = phase_quality(
                translator, model_dir, Path(args.pred_dir), args.precision
            )
        else:
            raise ValueError(args.phase)
        result.update(meter.stop())
        result["torch_num_threads_used"] = torch.get_num_threads()
        result["torch_default_threads_in_worker"] = pre_threads
    result["rss_after_run_mb"] = current_rss_mb()
    result["peak_rss_mb"], result["peak_rss_source"] = peak_rss_mb()
    result["worker_seconds"] = time.perf_counter() - t_start
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------------------------
# orchestrator: launch jobs
# --------------------------------------------------------------------------------------------


def job_name(phase: str, threads: str, precision: str, rep: int) -> str:
    return f"{phase}_t{threads}_{precision}_rep{rep}"


def worker_env(threads: str) -> dict[str, str]:
    """Subprocess env: invalid HF tokens removed, hub offline (the model is a local dir), and for
    the 1-thread setting the OpenMP/MKL pools pinned too (torch.set_num_threads alone is set in
    the worker)."""
    env = dict(os.environ)
    for k in ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
        env.pop(k, None)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUTF8"] = "1"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(k, None)
    if threads != "default":
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = threads
    return env


def run_job(
    work_dir: Path,
    model_dir: Path,
    pred_dir: Path,
    phase: str,
    threads: str,
    precision: str,
    rep: int,
    modes: str,
) -> Path:
    """Run one worker in a fresh subprocess (skipped if its result file exists). Samples total
    CPU utilisation for BG_SAMPLE_SECONDS right before launching; stored next to the result."""
    name = job_name(phase, threads, precision, rep)
    out = work_dir / f"{name}.json"
    if out.is_file():
        print(f"skip {name} (exists)")
        return out
    bg, bg_src = cpu_utilisation_pct(BG_SAMPLE_SECONDS)
    cmd = [
        sys.executable, "-W", "ignore", "-m", "scripts.production_benchmark", "worker",
        "--phase", phase, "--precision", precision, "--threads", threads,
        "--model-dir", str(model_dir), "--pred-dir", str(pred_dir), "--modes", modes,
        "--out", str(work_dir / f"{name}.worker.json"),
    ]  # fmt: skip
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=worker_env(threads))
    wrapper = {
        "job": name,
        "code_sha": git_state()["code_sha"],
        "rep": rep,
        "bg_cpu_pct_before": bg,
        "bg_cpu_source": bg_src,
        "bg_sample_seconds": BG_SAMPLE_SECONDS,
        "wall_seconds_incl_load": time.perf_counter() - t0,
        "result": json.loads((work_dir / f"{name}.worker.json").read_text(encoding="utf-8")),
    }
    out.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done {name}: bg={bg} wall={wrapper['wall_seconds_incl_load']:.0f}s", flush=True)
    return out


def verify_model(model_dir: Path, pull_record: Path) -> dict[str, Any]:
    """Fail closed: the model on disk must match the manifest-verified pull record (HF revision +
    sha256) and the sha256 selection.json recorded for the `final` winner."""
    rec = json.loads(pull_record.read_text(encoding="utf-8"))
    sel = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    if sel["winner"]["candidate"] != "final":
        raise SystemExit("selection winner is not 'final'")
    actual = file_sha256(model_dir / "model.safetensors")
    want_manifest = rec["files"]["model/model.safetensors"]["sha256"]
    want_selection = sel["candidates"]["final"]["model_sha256"]
    if not (rec.get("verified") and rec.get("model_files_checked")):
        raise SystemExit("pull record is not a verified with-model pull")
    if rec["hf_revision"] != HF_REVISION or rec["hf_repo"] != HF_REPO:
        raise SystemExit("pull record repo/revision is not the expected one")
    if not (actual == want_manifest == want_selection):
        raise SystemExit(f"model sha mismatch: {actual} {want_manifest} {want_selection}")
    return {
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "model_safetensors_sha256": actual,
        "matches_manifest_and_selection_json": True,
        "pull_record_manifest_sha256": rec["manifest_sha256"],
        "selected_decode_config": sel["winner"],
    }


def run_main(args: argparse.Namespace) -> int:
    model_dir, work_dir = Path(args.model_dir), Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = work_dir / "predictions"
    verify_model(model_dir, Path(args.pull_record))
    phases = args.phases.split(",")
    for rep in range(args.rep_start, args.rep_start + args.reps):
        for threads in THREAD_SETTINGS:
            for precision in PRECISIONS:
                for phase in ("latency", "throughput"):
                    if phase not in phases:
                        continue
                    # beam-5-without-segmentation latency is single-run (first repetition only)
                    modes = "greedy,beam5,beam5_noseg" if rep == 1 else "greedy,beam5"
                    run_job(work_dir, model_dir, pred_dir, phase, threads, precision, rep, modes)
    if "quality" in phases:
        for precision in PRECISIONS:
            run_job(work_dir, model_dir, pred_dir, "quality", "default", precision, 1, "-")
    if "size" in phases:
        for precision in PRECISIONS:
            run_job(work_dir, model_dir, pred_dir, "size", "default", precision, 1, "-")
    return 0


# --------------------------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------------------------


def load_jobs(work_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for p in sorted(work_dir.glob("*_rep*.json")):
        if p.name.endswith(".worker.json"):
            continue
        jobs.append(json.loads(p.read_text(encoding="utf-8")))
    return jobs


def _group(jobs: list[dict[str, Any]], phase: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    g: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for j in jobs:
        r = j["result"]
        if r["phase"] == phase:
            g.setdefault((r["threads_setting"], r["precision"]), []).append(j)
    return g


def _bg(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [j["bg_cpu_pct_before"] for j in jobs if j["bg_cpu_pct_before"] is not None]
    return {
        "per_run_pct": [
            None if j["bg_cpu_pct_before"] is None else round(j["bg_cpu_pct_before"], 1)
            for j in jobs
        ],
        "during_other_cpu_pct_per_run": [
            None
            if j["result"].get("other_cpu_pct_during") is None
            else round(j["result"]["other_cpu_pct_during"], 1)
            for j in jobs
        ],
        "max_pct": max(vals) if vals else None,
        "mean_pct": statistics.fmean(vals) if vals else None,
        "any_above_flag": any(v > BG_FLAG_PCT for v in vals),
    }


def _low_bg(jobs: list[dict[str, Any]]) -> list[bool]:
    """Per job: was total CPU at or below BG_FLAG_PCT in the 5 s before it started."""
    return [
        j["bg_cpu_pct_before"] is not None and j["bg_cpu_pct_before"] <= BG_FLAG_PCT for j in jobs
    ]


def _restricted(per_run: list[dict], jobs: list[dict], keys: list[str]) -> dict[str, Any] | None:
    """`aggregate_runs` over only the runs whose pre-run background load was <= BG_FLAG_PCT."""
    keep = [r for r, ok in zip(per_run, _low_bg(jobs), strict=True) if ok]
    return {"n_runs": len(keep), **aggregate_runs(keep, keys)} if keep else None


def aggregate_latency(jobs: list[dict[str, Any]]) -> tuple[dict, dict]:
    summary: dict[str, Any] = {}
    raw: dict[str, Any] = {}
    for (threads, precision), js in _group(jobs, "latency").items():
        for mode in MODES:
            runs = [j for j in js if mode in j["result"]["latency"]]
            if not runs:
                continue
            per_run = [j["result"]["latency"][mode]["summary"] for j in runs]
            pooled_ms = [v for j in runs for v in j["result"]["latency"][mode]["ms"]]
            key = f"threads={threads}|{precision}|{mode}"
            summary[key] = {
                "threads_setting": threads,
                "precision": precision,
                "mode": mode,
                "n_runs": len(runs),
                "n_requests_per_run": per_run[0]["n"],
                "across_runs": aggregate_runs(per_run, ["p50", "p95", "p99", "mean"]),
                "across_runs_bg_le_flag": _restricted(per_run, runs, ["p50", "p95", "p99", "mean"]),
                "pooled_over_runs": summarize_latency(pooled_ms),
                "per_run": per_run,
                "background_cpu": _bg(runs),
                "peak_rss_mb_per_run": [j["result"]["peak_rss_mb"] for j in runs],
                "rss_after_load_mb_per_run": [j["result"]["rss_after_load_mb"] for j in runs],
                "rss_after_run_mb_per_run": [j["result"]["rss_after_run_mb"] for j in runs],
                "torch_num_threads_used": runs[0]["result"]["torch_num_threads_used"],
                "peak_rss_source": runs[0]["result"]["peak_rss_source"],
            }
            raw[key] = [j["result"]["latency"][mode]["ms"] for j in runs]
    return summary, raw


def aggregate_throughput(jobs: list[dict[str, Any]]) -> dict:
    summary: dict[str, Any] = {}
    for (threads, precision), js in _group(jobs, "throughput").items():
        for mode in ("greedy", "beam5"):
            per_run = [j["result"]["throughput"][mode] for j in js]
            summary[f"threads={threads}|{precision}|{mode}"] = {
                "threads_setting": threads,
                "precision": precision,
                "mode": mode,
                "n_runs": len(js),
                "n_sentences": per_run[0]["n_sentences"],
                "batch_size": per_run[0]["batch_size"],
                "across_runs": aggregate_runs(
                    per_run, ["sentences_per_second", "output_tokens_per_second"]
                ),
                "across_runs_bg_le_flag": _restricted(
                    per_run, js, ["sentences_per_second", "output_tokens_per_second"]
                ),
                "per_run": per_run,
                "background_cpu": _bg(js),
                "peak_rss_mb_per_run": [j["result"]["peak_rss_mb"] for j in js],
                "rss_after_run_mb_per_run": [j["result"]["rss_after_run_mb"] for j in js],
                "torch_num_threads_used": js[0]["result"]["torch_num_threads_used"],
            }
    return summary


def score_quality(work_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Official scorer (via nmt.evaluate.run_official_scorer) on every E1 predictions file, plus
    fp32-vs-int8 paired bootstrap (nmt.compare machinery, delta = int8 - fp32)."""
    from nmt.compare import _paired
    from nmt.evaluate import load_split, run_official_scorer

    inputs, labels = load_split("e1")
    gold = REPO_ROOT / "data" / "eval" / "e1" / "labels.jsonl"
    ids = [r["id"] for r in inputs]
    refs = {r["id"]: r["reference"] for r in labels}
    pred_src = work_dir / "predictions"
    pred_out = out_dir / "predictions"
    off_out = out_dir / "official"
    pred_out.mkdir(parents=True, exist_ok=True)
    off_out.mkdir(parents=True, exist_ok=True)
    preds: dict[tuple[str, str], dict[str, str]] = {}
    scores: dict[str, Any] = {}
    for precision in PRECISIONS:
        for mode in QUALITY_MODES:
            name = f"e1_{precision}_{mode}.json"
            data = (pred_src / name).read_bytes()
            (pred_out / name).write_bytes(data)
            preds[(precision, mode)] = json.loads(data.decode("utf-8"))
            rep = off_out / f"official_e1_{precision}_{mode}.json"
            run_official_scorer(gold, pred_out / name, rep)
            report = json.loads(rep.read_text(encoding="utf-8"))
            # official/score.py writes --out in text mode (CRLF on Windows); re-serialise the same
            # parsed report with LF so committed bytes are identical on every OS (content unchanged)
            rep.write_bytes(
                (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            )
            scores[f"{precision}|{mode}"] = {
                "bleu": report["all"]["bleu"],
                "chrf": report["all"]["chrf"],
                "n": len(ids),
                "predictions_sha256": file_sha256(pred_out / name),
                "official_report": f"official/{rep.name}",
            }
    comparison: dict[str, Any] = {}
    for mode in QUALITY_MODES:
        a = [preds[("int8", mode)][i] for i in ids]
        b = [preds[("fp32", mode)][i] for i in ids]
        r = [refs[i] for i in ids]
        paired = _paired(a, b, r, N_BOOTSTRAP, SEED)
        comparison[mode] = {
            "delta_definition": "int8 minus fp32",
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
            "n_sentences": len(ids),
            "n_outputs_differing": sum(x != y for x, y in zip(a, b, strict=True)),
            "official_cli_delta_bleu": scores[f"int8|{mode}"]["bleu"]
            - scores[f"fp32|{mode}"]["bleu"],
            "official_cli_delta_chrf": scores[f"int8|{mode}"]["chrf"]
            - scores[f"fp32|{mode}"]["chrf"],
            "bleu": paired["bleu"],
            "chrf": paired["chrf"],
        }
    parity: dict[str, Any] = {}
    if COLAB_E1_PREDICTIONS.is_file():
        colab = json.loads(COLAB_E1_PREDICTIONS.read_text(encoding="utf-8"))
        fp = preds[("fp32", "beam5")]
        parity = {
            "what": "CPU fp32 beam5 (T=192) E1 outputs vs the Colab L4 reports/final/main/"
            "seg_tuned/e1_predictions.json (sanity check of the CPU path, not a result)",
            "n_outputs_differing": sum(colab[i] != fp[i] for i in ids),
            "n": len(ids),
            "colab_predictions_sha256": file_sha256(COLAB_E1_PREDICTIONS),
        }
    return {"scores": scores, "paired_int8_vs_fp32": comparison, "cpu_fp32_vs_colab_l4": parity}


def aggregate_size(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (_, precision), js in _group(jobs, "size").items():
        out[precision] = js[0]["result"]["size"]
    return out


def sample_description(model_dir: Path) -> dict[str, Any]:
    import sentencepiece as spm

    from nmt.data.normalize import normalize_text

    inputs, _ = load_e1(model_dir)
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_dir / "spm.model"))

    def toks(i: int) -> int:
        return len(sp.encode(normalize_text(inputs[i]["source"]), out_type=int))

    split = deterministic_split(len(inputs))
    all_tok = [toks(i) for i in range(len(inputs))]
    out: dict[str, Any] = {
        "seed": SEED,
        "method": "random.Random(1234).sample(range(1940), 1940) permutation of E1 row indices; "
        "latency = first 200, warm-up = next 10, throughput = next 320 (disjoint)",
        "e1_n": len(inputs),
        "e1_source_tokens_over_segment_threshold": sum(t > SEG_THRESHOLD for t in all_tok),
        "e1_source_tokens": length_stats(all_tok),
    }
    for k, idx in split.items():
        t = [all_tok[i] for i in idx]
        c = [len(inputs[i]["source"]) for i in idx]
        out[k] = {
            "ids_sha256": ids_sha256([inputs[i]["id"] for i in idx]),
            "first_ids": [inputs[i]["id"] for i in idx[:3]],
            "source_tokens": length_stats(t),
            "source_chars": length_stats(c),
            "n_over_segment_threshold": sum(v > SEG_THRESHOLD for v in t),
        }
    return out


def _ratio(results: dict, key_a: str, key_b: str, stat: str = "p50") -> float | None:
    try:
        return (
            results[key_a]["across_runs"][stat]["median"]
            / results[key_b]["across_runs"][stat]["median"]
        )
    except (KeyError, ZeroDivisionError):
        return None


PRE_RECORD_CODE_SHA = "d3d4f01ee4a28f46b4df3e7e4d449b2d38e8c7cd"


def _job_code_shas(jobs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """{code sha: [job names]}. Jobs of repetitions 1-3 and the quality/size jobs predate the
    per-job `code_sha` field (added in a later, additive-instrumentation commit); they ran from
    the clean tree at PRE_RECORD_CODE_SHA, recorded here by hand (label in the key)."""
    out: dict[str, list[str]] = {}
    for j in jobs:
        sha = j.get("code_sha") or f"{PRE_RECORD_CODE_SHA} (not recorded per job; clean tree)"
        out.setdefault(sha, []).append(j["job"])
    return out


def aggregate_main(args: argparse.Namespace) -> int:
    work_dir, out_dir = Path(args.work_dir), Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(work_dir)
    lat, raw = aggregate_latency(jobs)
    results = {
        "title": "CPU production benchmark: main/final, fp32 vs int8 dynamic quantization",
        "provenance": {
            **git_state(),
            "model": verify_model(model_dir, Path(args.pull_record)),
            "script": "scripts/production_benchmark.py",
            "job_code_shas": _job_code_shas(jobs),
            "script_sha256": file_sha256(Path(__file__)),
            "scorer": "nmt.evaluate.run_official_scorer (official/score.py as shipped, UTF-8)",
            "official_score_py_sha256": file_sha256(REPO_ROOT / "official" / "score.py"),
        },
        "config": {
            "decode": {"beam_tuned": BEAM, "alpha": ALPHA, "segment_threshold": SEG_THRESHOLD},
            "greedy_definition": "beam=1 through Translator.translate (same segmentation T=192)",
            "latency": {
                "n_requests_per_run": N_LATENCY,
                "warmup_requests_per_mode": N_WARMUP,
                "per_request": "Translator.translate([text], batch_size=1): normalize + "
                "tokenize + decode + detokenize, perf_counter, one fresh subprocess per run",
            },
            "throughput": {"n_sentences": N_THROUGHPUT, "batch_size": BATCH_SIZE},
            "background_flag_pct": BG_FLAG_PCT,
            "background_sample_seconds": BG_SAMPLE_SECONDS,
            "spread_flag_fraction": SPREAD_FLAG_FRAC,
            "int8": "torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, torch.qint8, "
            "inplace=True); embedding, LayerNorms and the tied output projection "
            "(F.linear on embed.weight) stay fp32",
        },
        "hardware_software": hardware_context(),
        "sample": sample_description(model_dir),
        "latency_ms": lat,
        "throughput": aggregate_throughput(jobs),
        "quality_e1": score_quality(work_dir, out_dir),
        "size_and_memory": aggregate_size(jobs),
    }
    (out_dir / "results.json").write_bytes(
        (json.dumps(results, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    (out_dir / "raw_latencies.json").write_bytes(
        (json.dumps(raw, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print(f"wrote {out_dir / 'results.json'}")
    write_readme(out_dir)
    return 0


# --------------------------------------------------------------------------------------------
# README rendering (every number comes from results.json)
# --------------------------------------------------------------------------------------------


def _fmt_range(d: dict[str, float], nd: int = 1) -> str:
    return f"{d['median']:.{nd}f} ({d['min']:.{nd}f}-{d['max']:.{nd}f})"


def _ranges_overlap(results: dict, key_a: str, key_b: str, stat: str) -> bool:
    """Do the (min, max) ranges of the per-run `stat` of two cells intersect?"""
    a, b = results[key_a]["across_runs"][stat], results[key_b]["across_runs"][stat]
    return a["min"] <= b["max"] and b["min"] <= a["max"]


def _classify(ratio: float, overlap: bool, higher_is_better: bool) -> str:
    """'faster' | 'slower' | 'indistinguishable' for int8 vs fp32. `ratio` = int8/fp32 median of
    the statistic (latency: lower is better; throughput: higher is better). Overlapping
    between-run (min, max) ranges are never called faster or slower."""
    if overlap:
        return "indistinguishable"
    return "faster" if (ratio > 1.0) == higher_is_better else "slower"


def _verdict(
    ratio: float | None, what: str, overlap: bool = False, higher_is_better: bool = False
) -> str:
    """int8/fp32 ratio of medians in words (see `_classify`)."""
    if ratio is None:
        return f"{what}: not measured"
    base = f"ratio of medians {ratio:.2f}"
    kind = _classify(ratio, overlap, higher_is_better)
    if kind == "indistinguishable":
        return f"{what}: no clear difference, run-to-run ranges overlap ({base})"
    return (
        f"{what}: int8 {kind.upper() if kind == 'slower' else kind}, ranges do not overlap ({base})"
    )


def _low_cell(cell: dict[str, Any], key: str, nd: int = 1) -> str:
    low = cell.get("across_runs_bg_le_flag")
    return "none" if not low else f"{_fmt_range(low[key], nd)} (n={low['n_runs']})"


def render_readme(res: dict[str, Any], file_hashes: dict[str, str]) -> str:
    """Markdown report; every number is read from `res` (results.json)."""
    hw, prov = res["hardware_software"], res["provenance"]
    lat, tp = res["latency_ms"], res["throughput"]
    q, size, smp = res["quality_e1"], res["size_and_memory"], res["sample"]
    lines: list[str] = []
    a = lines.append
    a("# CPU production benchmark: main/final, fp32 vs int8 dynamic quantization")
    a("")
    a(
        "Local laptop CPU, shared machine: single-sentence latency and batch-32 throughput through "
        "`Translator.translate`, plus fp32-vs-int8 quality on the full E1 set. All numbers below "
        "are copied from `results.json` (same directory)."
    )
    a("")
    a("## Setup")
    a("")
    a(
        f"- CPU: {hw['cpu_model']}, {hw['physical_cores']} cores / {hw['logical_cpus']} logical; "
        f"torch {hw['torch']} (default intra-op threads {hw['torch_default_num_threads']}; "
        f"quantized engine `{hw['torch_quantized_engine']}`, supported "
        f"{hw['torch_supported_quantized_engines']}); Python {hw['python']}; {hw['os']}."
    )
    a(f"- Power: {hw['power_plan']}; power source at start: {hw['power_source']}.")
    m = prov["model"]
    a(
        f"- Model: `final` of the main run, HF `{m['hf_repo']}` @ `{m['hf_revision']}` "
        f"(manifest-verified pull; safetensors sha256 `{m['model_safetensors_sha256']}` equals the "
        "one in selection.json). Decode config (selection.json winner): beam 5, alpha 1.2, "
        'segmentation threshold 192; "greedy" = beam 1 through the same API, same segmentation.'
    )
    a(
        f"- Thread settings: `default` (torch default, {hw['torch_default_num_threads']} threads) "
        "and `1` (torch.set_num_threads(1), OMP/MKL pinned to 1; the relevant number for a small "
        "server instance)."
    )
    a(
        "- int8: `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, torch.qint8)`. "
        "Quantized: all nn.Linear (attention q/k/v/out, FFN fc1/fc2; weights int8, activations "
        "quantized per call, biases fp32). NOT quantized: the tied token embedding, all "
        "LayerNorms, and the tied output projection (the model computes it as "
        "`F.linear(x, embed.weight)`, a functional call that module-based dynamic quantization "
        "does not see). Exact counts are in the size section."
    )
    lt = smp["latency"]
    a(
        f"- Latency sample: {lt['source_tokens']['n']} E1 sentences, seed {smp['seed']} "
        f"({smp['method']}); source tokens mean {lt['source_tokens']['mean']:.1f}, median "
        f"{lt['source_tokens']['median']:.0f}, p95 {lt['source_tokens']['p95']:.0f}, max "
        f"{lt['source_tokens']['max']}; chars mean {lt['source_chars']['mean']:.1f}, max "
        f"{lt['source_chars']['max']}. Sentences over the segmentation threshold: "
        f"{lt['n_over_segment_threshold']} in the sample, "
        f"{smp['e1_source_tokens_over_segment_threshold']} of {smp['e1_n']} in E1."
    )
    a(
        f"- Warm-up: {res['config']['latency']['warmup_requests_per_mode']} untimed requests "
        "(disjoint sentences) in each mode before timing; every measurement run is a fresh process."
    )
    a("")
    a("## Single-sentence latency (ms; normalize + tokenize + decode + detokenize)")
    a("")
    a(
        "Per cell: median over repeated runs of the per-run statistic, with (min-max) across runs. "
        "p99 of 200 requests rests on the top two values and is noisy."
    )
    a("")
    a(
        "| threads | mode | precision | runs | p50 | p95 | p99 | mean | bg CPU % before (max) "
        "| p50, runs with bg <= 30% only (n) |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|")
    for th in THREAD_SETTINGS:
        for mode in MODES:
            for pr in PRECISIONS:
                c = lat.get(f"threads={th}|{pr}|{mode}")
                if not c:
                    continue
                ar = c["across_runs"]
                bgm = c["background_cpu"]["max_pct"]
                bgs = "n/a" if bgm is None else f"{bgm:.0f}"
                a(
                    f"| {th} | {mode} | {pr} | {c['n_runs']} | {_fmt_range(ar['p50'])} | "
                    f"{_fmt_range(ar['p95'])} | {_fmt_range(ar['p99'])} | "
                    f"{_fmt_range(ar['mean'])} | {bgs} | {_low_cell(c, 'p50')} |"
                )
    a("")
    a("`beam5_noseg` was measured in the first repetition only (single run, no spread).")
    a("")
    a("## Throughput, batches of 32 (320 E1 sentences per run)")
    a("")
    a(
        "| threads | mode | precision | runs | sentences/s | output tokens/s | bg CPU % (max) "
        "| sentences/s, runs with bg <= 30% only (n) |"
    )
    a("|---|---|---|---|---|---|---|---|")
    for th in THREAD_SETTINGS:
        for mode in ("greedy", "beam5"):
            for pr in PRECISIONS:
                c = tp.get(f"threads={th}|{pr}|{mode}")
                if not c:
                    continue
                ar = c["across_runs"]
                bgm = c["background_cpu"]["max_pct"]
                bgs = "n/a" if bgm is None else f"{bgm:.0f}"
                a(
                    f"| {th} | {mode} | {pr} | {c['n_runs']} | "
                    f"{_fmt_range(ar['sentences_per_second'], 2)} | "
                    f"{_fmt_range(ar['output_tokens_per_second'], 1)} | {bgs} | "
                    f"{_low_cell(c, 'sentences_per_second', 2)} |"
                )
    a("")
    a(
        "GPU reference (not measured here; from the main run's `bench.json`, L4, first 200 E2 "
        "sentences, batch 32): greedy 41.097 sent/s / 2305.93 out-tok/s, beam 5 21.345 sent/s / "
        "1198.86 out-tok/s. Different sentences and hardware: context only."
    )
    a("")
    a("## Quality, fp32 vs int8, full E1 (1,940 sentences, official scorer)")
    a("")
    a("| mode | precision | BLEU | chrF |")
    a("|---|---|---|---|")
    for mode in QUALITY_MODES:
        for pr in PRECISIONS:
            s = q["scores"][f"{pr}|{mode}"]
            a(f"| {mode} | {pr} | {s['bleu']:.3f} | {s['chrf']:.3f} |")
    a("")
    a(
        "| mode | metric | delta int8 - fp32 | 95% paired-bootstrap CI (1000 resamples, seed 1234) "
        "| outputs differing (of 1940) |"
    )
    a("|---|---|---|---|---|")
    for mode in QUALITY_MODES:
        c = q["paired_int8_vs_fp32"][mode]
        for met in ("bleu", "chrf"):
            a(
                f"| {mode} | {met} | {c[met]['delta']:+.3f} | "
                f"[{c[met]['ci_low']:+.3f}, {c[met]['ci_high']:+.3f}] | "
                f"{c['n_outputs_differing']} |"
            )
    par = q.get("cpu_fp32_vs_colab_l4") or {}
    if par:
        a("")
        a(
            "Sanity check of the CPU path: CPU fp32 beam-5 outputs differ from the Colab L4 "
            f"`seg_tuned/e1_predictions.json` in {par['n_outputs_differing']} of {par['n']} "
            "sentences."
        )
    a("")
    a("## Model size and memory")
    a("")
    a(
        "| precision | params (fp32 model) | safetensors on disk (bytes) | torch.save(state_dict) "
        "(bytes) | peak RSS, latency workers (MiB, median / max) | peak RSS, throughput workers "
        "(MiB, median / max) | working set after the run, latency workers (MiB, median) |"
    )
    a("|---|---|---|---|---|---|---|")
    for pr in PRECISIONS:
        sz = size.get(pr, {})
        rss_l = [v for k, c in lat.items() if f"|{pr}|" in k for v in c["peak_rss_mb_per_run"]]
        rss_t = [v for k, c in tp.items() if f"|{pr}|" in k for v in c["peak_rss_mb_per_run"]]
        rss_a = [v for k, c in lat.items() if f"|{pr}|" in k for v in c["rss_after_run_mb_per_run"]]
        a(
            f"| {pr} | {sz.get('param_count_fp32', 0):,} | {sz.get('safetensors_bytes', 0):,} | "
            f"{sz.get('state_dict_torch_save_bytes', 0):,} | "
            f"{statistics.median(rss_l):.0f} / {max(rss_l):.0f} | "
            f"{statistics.median(rss_t):.0f} / {max(rss_t):.0f} | {statistics.median(rss_a):.0f} |"
        )
    sel = size.get("fp32", {}).get("module_selection", {})
    if sel:
        types = ", ".join(sorted({x["type"] for x in sel["not_quantized_modules_with_params"]}))
        a("")
        a(
            f"Quantized: {sel['n_quantized_modules']} nn.Linear modules, "
            f"{sel['quantized_weight_params']:,} weight parameters (int8). Left fp32: "
            f"{sel['not_quantized_params']:,} parameters (Linear biases "
            f"{sel['linear_bias_params_kept_fp32']:,} included); other parameter-holding module "
            f"types: {types}."
        )
    a("")
    a(
        "Peak RSS = process peak working set in a fresh process per run "
        f"(measured with `{next(iter(lat.values()))['peak_rss_source']}`). An int8 process first "
        "loads the fp32 weights and quantizes in place, so its PEAK includes the transient fp32 "
        "copy; the working set after load and after the run are in `results.json` "
        "(`rss_after_load_mb_per_run`)."
    )
    a("")
    a("## Findings")
    a("")
    kinds: list[str] = []
    for th in THREAD_SETTINGS:
        for mode in ("greedy", "beam5"):
            ki, kf = f"threads={th}|int8|{mode}", f"threads={th}|fp32|{mode}"
            r = _ratio(lat, ki, kf)
            if r is not None:
                kinds.append(_classify(r, _ranges_overlap(lat, ki, kf, "p50"), False))
    fp_sz = size.get("fp32", {}).get("state_dict_torch_save_bytes")
    i8_sz = size.get("int8", {}).get("state_dict_torch_save_bytes")
    if fp_sz and i8_sz:
        a(
            f"- Plain summary: int8 dynamic quantization cuts the saved model state from "
            f"{fp_sz / 1e6:.1f} MB to {i8_sz / 1e6:.1f} MB ({fp_sz / i8_sz:.2f}x smaller, "
            "torch.save of the state_dict) and, on E1, changes no metric beyond bootstrap noise "
            "(see quality lines below; hundreds of individual outputs do differ). Single-request "
            f"p50 latency, int8 vs fp32 over the {len(kinds)} thread/mode settings: "
            + ", ".join(f"{kinds.count(k)} {k}" for k in ("faster", "slower", "indistinguishable"))
            + " (a difference is only called when the between-run ranges do not overlap)."
        )
        a(
            "- Hypothesis for the missing speed-up, NOT tested here: single-sentence decoding is "
            "dominated by many tiny (1 x 512) matmuls, the fp32 tied output projection and "
            "Python-loop overhead, so int8 matmuls (plus per-call activation quantization) have "
            "little to win."
        )
    for th in THREAD_SETTINGS:
        for mode in ("greedy", "beam5"):
            ki, kf = f"threads={th}|int8|{mode}", f"threads={th}|fp32|{mode}"
            r = _ratio(lat, ki, kf)
            ov = r is not None and _ranges_overlap(lat, ki, kf, "p50")
            a("- " + _verdict(r, f"latency p50, threads={th}, {mode}", ov))
    for th in THREAD_SETTINGS:
        for mode in ("greedy", "beam5"):
            key_i, key_f = f"threads={th}|int8|{mode}", f"threads={th}|fp32|{mode}"
            r = _ratio(tp, key_i, key_f, "sentences_per_second")
            if r is not None:
                ov = _ranges_overlap(tp, key_i, key_f, "sentences_per_second")
                a("- " + _verdict(r, f"throughput sent/s, threads={th}, {mode}", ov, True))
    for mode in QUALITY_MODES:
        c = q["paired_int8_vs_fp32"][mode]
        parts = []
        for met in ("bleu", "chrf"):
            inside = c[met]["ci_low"] <= 0.0 <= c[met]["ci_high"]
            parts.append(
                f"{met} {c[met]['delta']:+.3f} ({'CI includes 0' if inside else 'CI excludes 0'})"
            )
        a(
            f"- quality, {mode}: "
            + "; ".join(parts)
            + f"; {c['n_outputs_differing']} of 1940 differ."
        )
    cells = {**lat, **tp}
    flagged = [k for k, c in cells.items() if c["background_cpu"]["any_above_flag"]]
    spread = [k for k, c in lat.items() if c["across_runs"]["spread_large"]]
    a(
        f"- background load: {len(flagged)} of {len(cells)} timing cells had at least one run with "
        f"total CPU above {res['config']['background_flag_pct']:.0f}% in the 5 s before start; "
        f"{len(spread)} of {len(lat)} latency cells have a relative p50 spread above "
        f"{res['config']['spread_flag_fraction']:.0%}."
    )
    during = [
        v
        for c in cells.values()
        for v in c["background_cpu"]["during_other_cpu_pct_per_run"]
        if v is not None
    ]
    if during:
        a(
            "- other-process CPU (machine-wide minus the worker's own share) DURING the "
            f"{len(during)} runs that recorded it (the repeat repetition): min {min(during):.1f}%, "
            f"max {max(during):.1f}% of the machine; earlier runs have only the pre-run sample."
        )
    a(
        "- repetitions: 3 planned (1-3), then one more full pass (4) because several pre-run "
        "samples were above 30% or spreads were large; all runs are in the cells above, the "
        "`bg <= 30%` column restricts to runs whose pre-run sample was at or below 30%."
    )
    a("")
    a("## Limits")
    a("")
    a(
        "- One laptop CPU, shared with other jobs; background utilisation is sampled for 5 s "
        "BEFORE each run, not during it. Absolute latencies do not transfer to server CPUs."
    )
    a(
        "- Single-sentence latency includes a Python per-step decode loop (bookkeeping, n-gram "
        "blocking), not only matmuls; only nn.Linear matmuls are int8 and the output projection "
        "stays fp32. This torch build exposes only the `onednn` quantized engine; ratios may "
        "differ on other builds or CPUs."
    )
    a("- Quantization quality is measured on E1 only, at beam 5 (tuned) and greedy.")
    a("- No GPU numbers are produced here.")
    a("- p99 over 200 requests per run is a two-point estimate; pooled values: `results.json`.")
    a("")
    a("## Files")
    a("")
    a("- `results.json`: all numbers, per-run values, context, background load per run.")
    a("- `raw_latencies.json`: every timed request (ms) per cell and run.")
    a("- `predictions/`, `official/`: E1 outputs and official scorer reports per config.")
    a(
        "- Reproduce: pull with `scripts.eval_local.pull_run(..., with_model=True)`, then "
        "`python -m scripts.production_benchmark run` and `... aggregate`."
    )
    a("")
    a("## Provenance")
    a("")
    a(f"- code SHA: `{prov['code_sha']}` (tracked files dirty: {prov['tracked_files_dirty']})")
    a(f"- HF revision: `{m['hf_revision']}` (`{m['hf_repo']}`, private, read-only pull)")
    a(f"- official/score.py sha256: `{prov['official_score_py_sha256']}`")
    a("- code SHA the measurement jobs ran from (jobs per SHA):")
    for sha, names in prov["job_code_shas"].items():
        a(f"  - `{sha}`: {len(names)} jobs")
    a("- file sha256 (LF line endings verified at write time: no CR byte in any listed file):")
    for name, h in file_hashes.items():
        a(f"  - `{name}`: `{h}`")
    return "\n".join(lines) + "\n"


def write_readme(out_dir: Path) -> None:
    """Render README.md from results.json; hash every deliverable file and refuse CR bytes."""
    res = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    files = ["results.json", "raw_latencies.json"]
    for sub in ("predictions", "official"):
        files += sorted(f"{sub}/{p.name}" for p in (out_dir / sub).glob("*.json"))
    hashes: dict[str, str] = {}
    for f in files:
        data = (out_dir / f).read_bytes()
        if b"\r" in data:
            raise SystemExit(f"{f} contains CR bytes")
        hashes[f] = hashlib.sha256(data).hexdigest()
    (out_dir / "README.md").write_bytes(render_readme(res, hashes).encode("utf-8"))
    print(f"wrote {out_dir / 'README.md'}")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU production benchmark (fp32 vs int8 dynamic).")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "aggregate", "worker"):
        s = sub.add_parser(name)
        s.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
        s.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
        s.add_argument("--pull-record", type=Path, default=DEFAULT_PULL_RECORD)
        if name == "run":
            s.add_argument("--reps", type=int, default=3)
            s.add_argument("--rep-start", type=int, default=1)
            s.add_argument("--phases", default="latency,throughput,quality,size")
        if name == "aggregate":
            s.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
        if name == "worker":
            s.add_argument(
                "--phase", required=True, choices=("latency", "throughput", "quality", "size")
            )
            s.add_argument("--precision", required=True, choices=PRECISIONS)
            s.add_argument("--threads", default="default")
            s.add_argument("--modes", default="greedy,beam5")
            s.add_argument("--pred-dir", default=str(DEFAULT_WORK_DIR / "predictions"))
            s.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "worker":
        return worker_main(args)
    if args.cmd == "run":
        return run_main(args)
    return aggregate_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
