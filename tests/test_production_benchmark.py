from __future__ import annotations

# Offline tests for scripts/production_benchmark.py: percentile maths, the deterministic sample,
# which modules int8 dynamic quantization touches (on a tiny random model, CPU), the per-request
# timing loop, the subprocess peak-RSS helper, and README rendering. No model download, no HF.
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import scripts.production_benchmark as pb
from nmt.decode import DecodeConfig, beam_search_decode
from nmt.model.transformer import ModelConfig, Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_model() -> Transformer:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=4, enc_layers=2, dec_layers=1, d_ff=64)
    return Transformer(cfg).eval()


# ---- percentile maths ------------------------------------------------------------------------


def test_percentile_known_values() -> None:
    v = [5.0, 1.0, 3.0, 2.0, 4.0]  # unsorted on purpose
    assert pb.percentile(v, 0) == 1.0
    assert pb.percentile(v, 50) == 3.0
    assert pb.percentile(v, 100) == 5.0
    assert pb.percentile(v, 25) == 2.0
    assert pb.percentile(v, 90) == pytest.approx(4.6)  # 4 + 0.6 * (5 - 4)
    assert pb.percentile([7.0], 99) == 7.0


def test_percentile_matches_numpy_linear() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    vals = rng.exponential(100.0, size=200).tolist()
    for q in (50, 95, 99):
        assert pb.percentile(vals, q) == pytest.approx(float(np.percentile(vals, q)))


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        pb.percentile([], 50)
    with pytest.raises(ValueError):
        pb.percentile([1.0], 101)


def test_summarize_latency_fields() -> None:
    s = pb.summarize_latency([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["mean"] == 25.0
    assert (s["min"], s["max"]) == (10.0, 40.0)
    assert s["p50"] == 25.0
    assert s["min"] <= s["p50"] <= s["p95"] <= s["p99"] <= s["max"]


def test_aggregate_runs_median_and_spread_flag() -> None:
    tight = pb.aggregate_runs([{"p50": 100.0}, {"p50": 101.0}, {"p50": 99.0}], ["p50"])
    assert tight["p50"] == {"median": 100.0, "min": 99.0, "max": 101.0, "n_runs": 3}
    assert tight["spread_large"] is False
    loose = pb.aggregate_runs([{"p50": 100.0}, {"p50": 150.0}, {"p50": 100.0}], ["p50"])
    assert loose["relative_spread_of_p50"] == pytest.approx(0.5)
    assert loose["spread_large"] is True


# ---- deterministic sample ---------------------------------------------------------------------


def test_deterministic_split_is_pinned_and_disjoint() -> None:
    a = pb.deterministic_split(1940)
    b = pb.deterministic_split(1940)
    assert a == b
    assert a["latency"][:5] == [1593, 902, 239, 15, 185]  # fixed before any measurement
    assert a["warmup"][:3] == [812, 398, 1443]
    assert a["throughput"][:3] == [1891, 1438, 862]
    assert [len(a[k]) for k in ("latency", "warmup", "throughput")] == [200, 10, 320]
    allx = a["latency"] + a["warmup"] + a["throughput"]
    assert len(set(allx)) == len(allx)
    assert pb.deterministic_split(1940, seed=1)["latency"] != a["latency"]


def test_deterministic_split_too_few_rows() -> None:
    with pytest.raises(ValueError):
        pb.deterministic_split(100)


def test_length_stats_and_ids_hash() -> None:
    s = pb.length_stats([1, 2, 3, 4, 100])
    assert (s["n"], s["min"], s["max"], s["median"]) == (5, 1, 100, 3.0)
    assert pb.ids_sha256(["a", "b"]) == pb.ids_sha256(["a", "b"]) != pb.ids_sha256(["b", "a"])


# ---- int8 module selection --------------------------------------------------------------------


def test_select_int8_modules_lists_exactly_the_linears() -> None:
    model = _tiny_model()
    sel = pb.select_int8_modules(model)
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    assert sel["quantized_modules"] == linear_names
    # 2 enc layers x (4 self-attn + 2 ffn) + 1 dec layer x (4 self + 4 cross + 2 ffn)
    assert sel["n_quantized_modules"] == 2 * 6 + 1 * 10
    types = {m["type"] for m in sel["not_quantized_modules_with_params"]}
    assert "Embedding" in types and "LayerNorm" in types and "Linear" not in types
    assert sel["quantized_weight_params"] + sel["not_quantized_params"] == sel["total_params"]
    assert sel["total_params"] == model.param_count()


def test_quantize_int8_replaces_linears_but_not_embedding_and_still_decodes() -> None:
    model = _tiny_model()
    pb.quantize_int8(model)
    assert not any(type(m) is nn.Linear for m in model.modules())
    assert any("quantized" in type(m).__module__ for m in model.modules())
    assert type(model.embed) is nn.Embedding and model.embed.weight.dtype == torch.float32
    cfg = model.cfg
    src = torch.tensor([[5, 6, 7, cfg.eos_id, cfg.pad_id], [8, 9, 10, 11, cfg.eos_id]])
    hyps = beam_search_decode(
        model,
        src,
        src != cfg.pad_id,
        cfg.bos_id,
        cfg.eos_id,
        cfg.pad_id,
        DecodeConfig(beam_size=2, alpha=1.2),
    )
    assert len(hyps) == 2 and all(len(h.tokens) > 0 for h in hyps)


# ---- timing loop ------------------------------------------------------------------------------


class _FakeTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def translate(self, texts: list[str], **kw: object) -> list[str]:
        self.calls.append((texts, kw))
        return ["x"] * len(texts)


def test_request_latencies_one_request_per_sentence() -> None:
    tr = _FakeTranslator()
    ms = pb.request_latencies_ms(tr, ["a", "b", "c"], beam=5, seg=192)
    assert len(ms) == 3 and all(v >= 0.0 for v in ms)
    assert [c[0] for c in tr.calls] == [["a"], ["b"], ["c"]]
    assert tr.calls[0][1] == {
        "batch_size": 1,
        "beam": 5,
        "alpha": pb.ALPHA,
        "segment_threshold": 192,
    }


def test_phase_latency_warmup_not_timed() -> None:
    tr = _FakeTranslator()
    texts = {"warmup": ["w1", "w2"], "latency": ["a", "b", "c"]}
    out = pb.phase_latency(tr, texts, ["greedy"])
    assert out["greedy"]["summary"]["n"] == 3 and len(out["greedy"]["ms"]) == 3
    assert [c[0] for c in tr.calls][:2] == [["w1"], ["w2"]]
    assert len(tr.calls) == 5


# ---- RSS helper / environment -----------------------------------------------------------------


def test_peak_rss_helper_in_fresh_subprocess() -> None:
    code = (
        "import numpy as np;from scripts.production_benchmark import peak_rss_mb;"
        "a=np.ones(25_000_000);a+=1;mb,src=peak_rss_mb();print(mb,src)"  # 200 MB of float64
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    mb, src = r.stdout.split(maxsplit=1)
    assert float(mb) > 190.0
    assert src.strip() in {"psutil.peak_wset", "ctypes.PeakWorkingSetSize", "resource.ru_maxrss"}


def test_worker_env_strips_tokens_and_pins_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "bad")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    one = pb.worker_env("1")
    assert "HUGGINGFACEHUB_API_TOKEN" not in one
    assert one["OMP_NUM_THREADS"] == one["MKL_NUM_THREADS"] == "1"
    assert one["HF_HUB_OFFLINE"] == "1"
    assert "OMP_NUM_THREADS" not in pb.worker_env("default")


def test_cpu_model_and_system_probes_return_values() -> None:
    assert pb.cpu_model_name()
    pct, src = pb.cpu_utilisation_pct(0.2)
    assert src and (pct is None or 0.0 <= pct <= 100.0)


# ---- README rendering -------------------------------------------------------------------------


def _cell(p50: float, key: str = "p50") -> dict:
    stat = {"median": p50, "min": p50 - 1, "max": p50 + 1, "n_runs": 3}
    return {
        "threads_setting": "1",
        "n_runs": 3,
        "across_runs": {
            k: stat
            for k in (
                "p50",
                "p95",
                "p99",
                "mean",
                "sentences_per_second",
                "output_tokens_per_second",
            )
        }
        | {"spread_large": False},
        "background_cpu": {
            "max_pct": 12.0,
            "any_above_flag": False,
            "during_other_cpu_pct_per_run": [3.0, None, 5.0],
        },
        "peak_rss_mb_per_run": [500.0, 510.0, 505.0],
        "rss_after_run_mb_per_run": [400.0, 410.0, 405.0],
        "peak_rss_source": "psutil.peak_wset",
    }


def test_render_readme_from_synthetic_results() -> None:
    lat = {f"threads=1|{p}|{m}": _cell(100.0 if p == "fp32" else 130.0) for p in pb.PRECISIONS
           for m in pb.MODES}  # fmt: skip
    tp = {f"threads=1|{p}|{m}": _cell(10.0) for p in pb.PRECISIONS for m in ("greedy", "beam5")}
    stats = {"n": 200, "mean": 30.0, "median": 21.0, "p95": 80.0, "min": 1, "max": 150}
    pair = {"delta": -0.1, "ci_low": -0.3, "ci_high": 0.1}
    res = {
        "hardware_software": {
            "cpu_model": "Test CPU",
            "physical_cores": 8,
            "logical_cpus": 16,
            "torch": "x",
            "torch_default_num_threads": 8,
            "torch_quantized_engine": "onednn",
            "torch_supported_quantized_engines": ["onednn"],
            "python": "3",
            "os": "os",
            "power_plan": "plan",
            "power_source": "AC",
        },  # fmt: skip
        "provenance": {
            "code_sha": "abc",
            "tracked_files_dirty": False,
            "official_score_py_sha256": "d",
            "job_code_shas": {"abc": ["j1", "j2"]},
            "model": {"hf_repo": "r", "hf_revision": "v" * 40, "model_safetensors_sha256": "s"},
        },  # fmt: skip
        "config": {
            "latency": {"warmup_requests_per_mode": 10},
            "background_flag_pct": 30.0,
            "spread_flag_fraction": 0.1,
        },  # fmt: skip
        "sample": {
            "seed": 1234,
            "method": "m",
            "e1_n": 1940,
            "e1_source_tokens_over_segment_threshold": 6,
            "latency": {
                "source_tokens": stats,
                "source_chars": stats,
                "n_over_segment_threshold": 0,
            },
        },  # fmt: skip
        "latency_ms": lat,
        "throughput": tp,
        "quality_e1": {
            "scores": {
                f"{p}|{m}": {"bleu": 30.0, "chrf": 55.0}
                for p in pb.PRECISIONS
                for m in pb.QUALITY_MODES
            },  # fmt: skip
            "paired_int8_vs_fp32": {
                m: {"bleu": pair, "chrf": pair, "n_outputs_differing": 42} for m in pb.QUALITY_MODES
            },
            "cpu_fp32_vs_colab_l4": {"n_outputs_differing": 3, "n": 1940},
        },
        "size_and_memory": {
            "fp32": {
                "param_count_fp32": 50229248,
                "safetensors_bytes": 200944776,
                "state_dict_torch_save_bytes": 1,
                "module_selection": {
                    "n_quantized_modules": 60,
                    "quantized_weight_params": 5,
                    "not_quantized_params": 6,
                    "linear_bias_params_kept_fp32": 1,
                    "not_quantized_modules_with_params": [{"type": "Embedding"}],
                },
            },
            "int8": {
                "param_count_fp32": 50229248,
                "safetensors_bytes": 200944776,
                "state_dict_torch_save_bytes": 2,
            },
        },  # fmt: skip
    }
    text = pb.render_readme(res, {"results.json": "0" * 64})
    assert "\r" not in text
    assert "int8 SLOWER, ranges do not overlap (ratio of medians 1.30)" in text
    assert "Test CPU" in text and "`" + "0" * 64 + "`" in text
    json.dumps(res)  # synthetic results are JSON-serialisable like the real ones


def test_restricted_stats_use_only_low_background_runs() -> None:
    jobs = [{"bg_cpu_pct_before": 10.0}, {"bg_cpu_pct_before": 45.0}, {"bg_cpu_pct_before": None}]
    assert pb._low_bg(jobs) == [True, False, False]
    per_run = [{"p50": 100.0}, {"p50": 900.0}, {"p50": 800.0}]
    low = pb._restricted(per_run, jobs, ["p50"])
    assert low is not None and low["n_runs"] == 1 and low["p50"]["median"] == 100.0
    assert pb._restricted(per_run, [{"bg_cpu_pct_before": 99.0}] * 3, ["p50"]) is None


def test_load_meter_reports_own_and_other_share() -> None:
    meter = pb.LoadMeter()
    meter.start()
    sum(i * i for i in range(2_000_000))  # burn a little CPU in this process
    out = meter.stop()
    assert set(out) == {"system_cpu_pct_during", "own_cpu_pct_of_machine", "other_cpu_pct_during"}
    if out["system_cpu_pct_during"] is not None:  # psutil present
        assert out["own_cpu_pct_of_machine"] > 0.0
        assert 0.0 <= out["other_cpu_pct_during"] <= 100.0


def test_verdict_never_calls_overlapping_ranges_faster_or_slower() -> None:
    assert "no clear difference" in pb._verdict(0.9, "x", overlap=True)
    assert "faster" in pb._verdict(0.9, "x", overlap=False)
    assert "SLOWER" in pb._verdict(1.2, "x", overlap=False)
    assert "not measured" in pb._verdict(None, "x")
    cells = {
        "a": {"across_runs": {"p50": {"min": 1.0, "max": 3.0}}},
        "b": {"across_runs": {"p50": {"min": 2.0, "max": 4.0}}},
        "c": {"across_runs": {"p50": {"min": 3.5, "max": 4.0}}},
    }
    assert pb._ranges_overlap(cells, "a", "b", "p50") and not pb._ranges_overlap(
        cells, "a", "c", "p50"
    )
