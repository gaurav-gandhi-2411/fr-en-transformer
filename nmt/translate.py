from __future__ import annotations

# Public production API and CLI: `Translator.from_pretrained(path_or_repo_id).translate(...)`.
# Applies the same normalization as training, calls decode.py, and reports batched inference
# latency/throughput. Spec §7.
#
# CLI: `python -m nmt.translate --model DIR --input data/test/inputs.jsonl --output preds.json
# [--beam 5 --alpha 0.6 --batch-size 32 --segment-threshold T]`.
import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import sentencepiece as spm
import torch
from torch import Tensor

from nmt.data.normalize import normalize_text
from nmt.decode import DecodeConfig, beam_search_decode, greedy_decode, split_sentences
from nmt.hub import NMTModel, load_pretrained
from nmt.mbr import MBRConfig, beam_pool, mbr_select, sample_pool

FallbackKind = str  # "beam" | "greedy" | "copy"


@dataclass
class TranslatorStats:
    """Running counts of how each output was produced (spec §7: "never emit an empty string",
    counts of each fallback logged)."""

    n_total: int = 0
    n_beam: int = 0
    n_greedy_fallback: int = 0
    n_copy_fallback: int = 0

    def record(self, kind: FallbackKind) -> None:
        self.n_total += 1
        if kind == "beam":
            self.n_beam += 1
        elif kind == "greedy":
            self.n_greedy_fallback += 1
        elif kind == "copy":
            self.n_copy_fallback += 1
        else:
            raise ValueError(f"unknown fallback kind: {kind!r}")


def _pad_ids(
    ids_list: list[list[int]], pad_id: int, eos_id: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Append EOS to each id sequence and pad to a common length. Returns (src, src_mask)."""
    seqs = [ids + [eos_id] for ids in ids_list]
    max_len = max((len(s) for s in seqs), default=1)
    tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return tensor, tensor != pad_id


def _detok(sp: spm.SentencePieceProcessor, ids: list[int], eos_id: int) -> str:
    """Detokenize generated ids, dropping a trailing EOS (decode.py's Hypothesis.tokens includes
    it on natural termination; SentencePiece's vocabulary has no piece for it)."""
    content = [i for i in ids if i != eos_id]
    return sp.decode(content) if content else ""


class Translator:
    """The production translation API: `Translator.from_pretrained(dir_or_repo_id).translate(...)`.
    Applies `normalize_text` to every input (the same normalization used at train time),
    length-sorts for efficient batching, restores input order, and guarantees a non-empty output
    per input via the beam -> greedy -> normalized-source-copy fallback chain (spec §7).
    """

    def __init__(self, model: NMTModel, sp: spm.SentencePieceProcessor, device: torch.device):
        self.model = model.to(device).eval()
        self.sp = sp
        self.device = device
        self.stats = TranslatorStats()

    @classmethod
    def from_pretrained(
        cls, path_or_repo_id: str, device: str | torch.device | None = None
    ) -> Translator:
        """Load from a local export directory (`nmt.hub.export_checkpoint`'s output) or an HF
        Hub repo id -- read-only; never pushes anything.
        """
        model, spm_path = load_pretrained(path_or_repo_id)
        sp = spm.SentencePieceProcessor()
        sp.load(str(spm_path))
        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return cls(model, sp, resolved_device)

    @torch.no_grad()
    def _translate_batch(
        self,
        texts: list[str],
        beam: int,
        alpha: float,
        no_repeat_ngram_size: int,
        mbr: MBRConfig | None = None,
    ) -> tuple[list[str], list[FallbackKind]]:
        cfg = self.model.config
        ids_list = [self.sp.encode(t, out_type=int) for t in texts]
        src, src_mask = _pad_ids(ids_list, cfg.pad_id, cfg.eos_id, self.device)
        if mbr is None:
            decode_cfg = DecodeConfig(
                beam_size=beam, alpha=alpha, no_repeat_ngram_size=no_repeat_ngram_size
            )
            beam_hyps = beam_search_decode(
                self.model, src, src_mask, cfg.bos_id, cfg.eos_id, cfg.pad_id, decode_cfg
            )
            primary = [_detok(self.sp, h.tokens, cfg.eos_id) for h in beam_hyps]
        else:
            primary = self._mbr_outputs(texts, src, src_mask, alpha, no_repeat_ngram_size, mbr)

        outputs: list[str | None] = [None] * len(texts)
        kinds: list[FallbackKind | None] = [None] * len(texts)
        greedy_needed: list[int] = []
        for i, text in enumerate(primary):
            if text.strip():
                outputs[i], kinds[i] = text, "beam"
            else:
                greedy_needed.append(i)

        if greedy_needed:
            sub_src, sub_mask = src[greedy_needed], src_mask[greedy_needed]
            greedy_hyps = greedy_decode(
                self.model,
                sub_src,
                sub_mask,
                cfg.bos_id,
                cfg.eos_id,
                cfg.pad_id,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            for local_i, i in enumerate(greedy_needed):
                text = _detok(self.sp, greedy_hyps[local_i].tokens, cfg.eos_id)
                if text.strip():
                    outputs[i], kinds[i] = text, "greedy"
                else:
                    # Final fallback: the normalized source itself, flagged so callers can count
                    # it (spec §7: "output the normalized source with a copy flag ... logged and
                    # counted"). If even the normalized source is empty/whitespace (a pathological
                    # empty input), a single space is the only way to honor "never emit an empty
                    # string" -- documented here since spec §7 does not cover empty-input inputs.
                    outputs[i] = texts[i].strip() or " "
                    kinds[i] = "copy"

        assert all(o is not None for o in outputs)
        assert all(k is not None for k in kinds)
        return outputs, kinds  # type: ignore[return-value]

    def _mbr_outputs(
        self,
        texts: list[str],
        src: Tensor,
        src_mask: Tensor,
        alpha: float,
        no_repeat_ngram_size: int,
        mbr: MBRConfig,
    ) -> list[str]:
        """Build the MBR candidate pool of every source (`mbr.kind`) and return each pool's MBR
        pick (nmt.mbr: chrF utility, pool as pseudo-references). An empty result means the pool had
        no non-empty member; the caller's greedy/copy fallback then applies. These count as "beam"
        in `TranslatorStats` (the primary path)."""
        cfg = self.model.config
        if mbr.kind == "beam":
            pools = beam_pool(
                self.model,
                src,
                src_mask,
                cfg.bos_id,
                cfg.eos_id,
                cfg.pad_id,
                mbr.n,
                alpha,
                no_repeat_ngram_size,
            )
        else:
            pools = sample_pool(
                self.model,
                src,
                src_mask,
                cfg.bos_id,
                cfg.eos_id,
                texts,
                mbr.n,
                mbr.epsilon,
                mbr.seed,
            )
        picks = []
        for hyps in pools:
            cands = [_detok(self.sp, h.tokens, cfg.eos_id) for h in hyps]
            pick = cands[mbr_select(cands)]
            picks.append(pick)
        return picks

    def translate(
        self,
        texts: list[str],
        batch_size: int = 32,
        beam: int = 5,
        alpha: float = 0.6,
        segment_threshold: int | None = None,
        no_repeat_ngram_size: int = 3,
        mbr: MBRConfig | None = None,
    ) -> list[str]:
        """Translate `texts`, batched and length-sorted for efficiency, restoring input order.

        `segment_threshold`: sources whose subword token count exceeds this are split on
        sentence punctuation (`nmt.decode.split_sentences`), each segment translated
        independently, and the results joined with a space (spec §7). `None` (default) disables
        segmentation entirely. `mbr` (default None = plain beam search, unchanged) switches each
        segment to MBR decoding over the pool it describes; `beam` is then unused.
        """
        normalized = [normalize_text(t) for t in texts]

        work: list[tuple[int, int, str]] = []  # (orig_idx, seg_idx, seg_text)
        seg_counts: list[int] = []
        for orig_idx, text in enumerate(normalized):
            if segment_threshold is not None:
                n_tokens = len(self.sp.encode(text, out_type=int))
                segments = split_sentences(text) if n_tokens > segment_threshold else [text]
            else:
                segments = [text]
            seg_counts.append(len(segments))
            work.extend((orig_idx, seg_idx, seg) for seg_idx, seg in enumerate(segments))

        enc_lens = [len(self.sp.encode(w[2], out_type=int)) for w in work]
        order = sorted(range(len(work)), key=lambda i: enc_lens[i])

        seg_results: dict[tuple[int, int], str] = {}
        for start in range(0, len(order), batch_size):
            batch_positions = order[start : start + batch_size]
            batch_texts = [work[i][2] for i in batch_positions]
            translations, kinds = self._translate_batch(
                batch_texts, beam, alpha, no_repeat_ngram_size, mbr
            )
            for pos, txt, kind in zip(batch_positions, translations, kinds, strict=True):
                orig_idx, seg_idx, _ = work[pos]
                seg_results[(orig_idx, seg_idx)] = txt
                self.stats.record(kind)

        outputs = []
        for orig_idx, text in enumerate(texts):
            segs = [seg_results[(orig_idx, s)] for s in range(seg_counts[orig_idx])]
            joined = " ".join(s for s in segs if s.strip())
            if not joined.strip():
                joined = normalized[orig_idx].strip() or text.strip() or " "
            outputs.append(joined)
        return outputs


def model_size_mb(model: NMTModel) -> float:
    """Parameter-storage size in MB (sum of tensor byte sizes / 1024**2)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024**2)


@dataclass
class BenchmarkResult:
    """Production benchmark (spec §7): throughput, per-batch latency percentiles, peak memory
    and model size. `memory_metric` documents which proxy was used for peak memory: CUDA reports
    true `max_memory_allocated`; CPU falls back to `tracemalloc`'s peak Python-tracked allocation
    (not full process RSS -- no cross-platform, stdlib-only RSS reader exists on Windows without
    adding `psutil`, which is out of scope for this phase without an explicit ask).
    """

    device: str
    n_sentences: int
    batch_size: int
    beam: int
    alpha: float
    wall_seconds: float
    sentences_per_second: float
    batch_latency_p50_ms: float
    batch_latency_p95_ms: float
    peak_memory_mb: float
    memory_metric: str
    model_size_mb: float
    fallback_counts: dict[str, int] = field(default_factory=dict)


def benchmark_translator(
    translator: Translator,
    texts: list[str],
    batch_size: int = 32,
    beam: int = 5,
    alpha: float = 0.6,
) -> BenchmarkResult:
    """Run `translator.translate` in fixed-size batches and measure throughput/latency/memory."""
    model_mb = model_size_mb(translator.model)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    on_cuda = translator.device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(translator.device)
    else:
        tracemalloc.start()

    latencies_ms: list[float] = []
    t0 = time.perf_counter()
    for batch in batches:
        b0 = time.perf_counter()
        translator.translate(batch, batch_size=len(batch), beam=beam, alpha=alpha)
        latencies_ms.append((time.perf_counter() - b0) * 1000.0)
    wall = time.perf_counter() - t0

    if on_cuda:
        peak_mb = torch.cuda.max_memory_allocated(translator.device) / (1024**2)
        mem_metric = "cuda_max_memory_allocated_mb"
    else:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024**2)
        mem_metric = "tracemalloc_python_peak_mb"

    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95 = (
        statistics.quantiles(latencies_ms, n=100, method="inclusive")[94]
        if len(latencies_ms) >= 2
        else (latencies_ms[0] if latencies_ms else 0.0)
    )
    return BenchmarkResult(
        device=str(translator.device),
        n_sentences=len(texts),
        batch_size=batch_size,
        beam=beam,
        alpha=alpha,
        wall_seconds=wall,
        sentences_per_second=(len(texts) / wall) if wall > 0 else 0.0,
        batch_latency_p50_ms=p50,
        batch_latency_p95_ms=p95,
        peak_memory_mb=peak_mb,
        memory_metric=mem_metric,
        model_size_mb=model_mb,
        fallback_counts={
            "beam": translator.stats.n_beam,
            "greedy": translator.stats.n_greedy_fallback,
            "copy": translator.stats.n_copy_fallback,
        },
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate inputs.jsonl with a trained model (spec §7)."
    )
    parser.add_argument("--model", required=True, help="Local export dir or HF Hub repo id.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL with {id, source} rows.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output JSON: {id: translation}."
    )
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--segment-threshold", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    translator = Translator.from_pretrained(args.model, device=args.device)
    rows = _read_jsonl(args.input)
    ids = [r["id"] for r in rows]
    sources = [r["source"] for r in rows]
    translations = translator.translate(
        sources,
        batch_size=args.batch_size,
        beam=args.beam,
        alpha=args.alpha,
        segment_threshold=args.segment_threshold,
    )
    result = dict(zip(ids, translations, strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        f"translated {len(ids)} sentences: beam={translator.stats.n_beam} "
        f"greedy_fallback={translator.stats.n_greedy_fallback} "
        f"copy_fallback={translator.stats.n_copy_fallback}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
