# CPU production benchmark: main/final, fp32 vs int8 dynamic quantization

Local laptop CPU, shared machine: single-sentence latency and batch-32 throughput through `Translator.translate`, plus fp32-vs-int8 quality on the full E1 set. All numbers below are copied from `results.json` (same directory).

## Setup

- CPU: AMD Ryzen 7 6800H with Radeon Graphics, 8 cores / 16 logical; torch 2.14.0+cpu (default intra-op threads 8; quantized engine `onednn`, supported ['onednn']); Python 3.13.5; Windows-11-10.0.26300-SP0.
- Power: Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced); power source at start: AC.
- Model: `final` of the main run, HF `OWNER/fr-en-transformer-eval` @ `c3d8598252853fcd7df1ef4a00e8b0382b8f4351` (manifest-verified pull; safetensors sha256 `96ef3bb589b47c5c45ad6cf60a7b1c96ffca43ce3ea0377a28440648bf97757f` equals the one in selection.json). Decode config (selection.json winner): beam 5, alpha 1.2, segmentation threshold 192; "greedy" = beam 1 through the same API, same segmentation.
- Thread settings: `default` (torch default, 8 threads) and `1` (torch.set_num_threads(1), OMP/MKL pinned to 1; the relevant number for a small server instance).
- int8: `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, torch.qint8)`. Quantized: all nn.Linear (attention q/k/v/out, FFN fc1/fc2; weights int8, activations quantized per call, biases fp32). NOT quantized: the tied token embedding, all LayerNorms, and the tied output projection (the model computes it as `F.linear(x, embed.weight)`, a functional call that module-based dynamic quantization does not see). Exact counts are in the size section.
- Latency sample: 200 E1 sentences, seed 1234 (random.Random(1234).sample(range(1940), 1940) permutation of E1 row indices; latency = first 200, warm-up = next 10, throughput = next 320 (disjoint)); source tokens mean 29.7, median 19, p95 83, max 130; chars mean 119.3, max 676. Sentences over the segmentation threshold: 0 in the sample, 6 of 1940 in E1.
- Warm-up: 10 untimed requests (disjoint sentences) in each mode before timing; every measurement run is a fresh process.

## Single-sentence latency (ms; normalize + tokenize + decode + detokenize)

Per cell: median over repeated runs of the per-run statistic, with (min-max) across runs. p99 of 200 requests rests on the top two values and is noisy.

| threads | mode | precision | runs | p50 | p95 | p99 | mean | bg CPU % before (max) | p50, runs with bg <= 30% only (n) |
|---|---|---|---|---|---|---|---|---|---|
| default | greedy | fp32 | 4 | 196.5 (194.7-202.3) | 695.6 (661.6-728.8) | 1158.4 (1000.7-1235.0) | 279.5 (274.9-287.8) | 45 | 196.4 (194.7-198.1) (n=2) |
| default | greedy | int8 | 4 | 243.1 (236.7-253.1) | 871.9 (836.1-906.7) | 1244.5 (1158.5-1291.9) | 338.3 (325.0-350.5) | 67 | 237.4 (236.7-238.1) (n=2) |
| default | beam5 | fp32 | 4 | 287.8 (274.8-359.3) | 1171.3 (1081.3-1374.8) | 1673.4 (1544.6-1737.1) | 427.3 (398.8-507.4) | 45 | 317.1 (274.8-359.3) (n=2) |
| default | beam5 | int8 | 4 | 415.8 (410.5-534.3) | 1594.3 (1515.2-2312.4) | 2265.3 (2136.0-3088.0) | 592.9 (574.3-807.7) | 67 | 411.3 (410.5-412.1) (n=2) |
| default | beam5_noseg | fp32 | 1 | 307.1 (307.1-307.1) | 1154.8 (1154.8-1154.8) | 1722.9 (1722.9-1722.9) | 437.0 (437.0-437.0) | 45 | none |
| default | beam5_noseg | int8 | 1 | 531.8 (531.8-531.8) | 2457.4 (2457.4-2457.4) | 4769.5 (4769.5-4769.5) | 863.6 (863.6-863.6) | 32 | none |
| 1 | greedy | fp32 | 4 | 203.2 (185.2-224.3) | 769.1 (669.9-799.7) | 1042.4 (1017.2-1242.6) | 291.5 (264.6-316.6) | 40 | 198.9 (185.2-207.5) (n=3) |
| 1 | greedy | int8 | 4 | 256.0 (244.3-304.4) | 969.3 (935.6-1512.6) | 1358.6 (1320.5-1897.8) | 363.3 (354.3-469.5) | 22 | 256.0 (244.3-304.4) (n=4) |
| 1 | beam5 | fp32 | 4 | 481.4 (416.9-522.5) | 1950.6 (1625.4-2380.7) | 2598.5 (2400.3-3051.4) | 702.9 (606.2-816.9) | 40 | 467.8 (416.9-495.0) (n=3) |
| 1 | beam5 | int8 | 4 | 443.6 (418.7-465.4) | 1695.7 (1659.9-1793.9) | 2457.9 (2351.6-2614.5) | 638.9 (604.4-669.3) | 22 | 443.6 (418.7-465.4) (n=4) |
| 1 | beam5_noseg | fp32 | 1 | 563.0 (563.0-563.0) | 2169.2 (2169.2-2169.2) | 2950.7 (2950.7-2950.7) | 794.8 (794.8-794.8) | 40 | none |
| 1 | beam5_noseg | int8 | 1 | 442.0 (442.0-442.0) | 1769.5 (1769.5-1769.5) | 2506.7 (2506.7-2506.7) | 639.3 (639.3-639.3) | 22 | 442.0 (442.0-442.0) (n=1) |

`beam5_noseg` was measured in the first repetition only (single run, no spread).

## Throughput, batches of 32 (320 E1 sentences per run)

| threads | mode | precision | runs | sentences/s | output tokens/s | bg CPU % (max) | sentences/s, runs with bg <= 30% only (n) |
|---|---|---|---|---|---|---|---|
| default | greedy | fp32 | 4 | 21.66 (17.76-22.24) | 555.9 (455.7-570.7) | 28 | 21.66 (17.76-22.24) (n=4) |
| default | greedy | int8 | 4 | 18.08 (12.26-18.68) | 468.3 (317.6-483.8) | 50 | 18.55 (17.61-18.68) (n=3) |
| default | beam5 | fp32 | 4 | 6.82 (5.39-7.25) | 176.1 (139.2-187.4) | 28 | 6.82 (5.39-7.25) (n=4) |
| default | beam5 | int8 | 4 | 6.94 (4.46-7.14) | 180.8 (116.1-186.2) | 50 | 7.10 (6.78-7.14) (n=3) |
| 1 | greedy | fp32 | 4 | 10.02 (8.51-10.83) | 257.1 (218.4-277.9) | 65 | 9.87 (8.51-10.83) (n=3) |
| 1 | greedy | int8 | 4 | 11.72 (11.35-12.06) | 303.6 (293.9-312.4) | 23 | 11.72 (11.35-12.06) (n=4) |
| 1 | beam5 | fp32 | 4 | 3.33 (3.02-3.46) | 85.9 (77.9-89.4) | 65 | 3.33 (3.02-3.46) (n=3) |
| 1 | beam5 | int8 | 4 | 3.70 (3.42-3.89) | 96.5 (89.2-101.3) | 23 | 3.70 (3.42-3.89) (n=4) |

GPU reference (not measured here; from the main run's `bench.json`, L4, first 200 E2 sentences, batch 32): greedy 41.097 sent/s / 2305.93 out-tok/s, beam 5 21.345 sent/s / 1198.86 out-tok/s. Different sentences and hardware: context only.

## Quality, fp32 vs int8, full E1 (1,940 sentences, official scorer)

| mode | precision | BLEU | chrF |
|---|---|---|---|
| beam5 | fp32 | 35.977 | 54.982 |
| beam5 | int8 | 35.923 | 54.982 |
| greedy | fp32 | 35.176 | 54.406 |
| greedy | int8 | 35.211 | 54.402 |

| mode | metric | delta int8 - fp32 | 95% paired-bootstrap CI (1000 resamples, seed 1234) | outputs differing (of 1940) |
|---|---|---|---|---|
| beam5 | bleu | -0.054 | [-0.595, +0.267] | 660 |
| beam5 | chrf | +0.000 | [-0.158, +0.171] | 660 |
| greedy | bleu | +0.035 | [-0.281, +0.297] | 716 |
| greedy | chrf | -0.003 | [-0.195, +0.188] | 716 |

Sanity check of the CPU path: CPU fp32 beam-5 outputs differ from the Colab L4 `seg_tuned/e1_predictions.json` in 0 of 1940 sentences.

## Model size and memory

| precision | params (fp32 model) | safetensors on disk (bytes) | torch.save(state_dict) (bytes) | peak RSS, latency workers (MiB, median / max) | peak RSS, throughput workers (MiB, median / max) | working set after the run, latency workers (MiB, median) |
|---|---|---|---|---|---|---|
| fp32 | 50,229,248 | 200,944,776 | 201,008,318 | 596 / 597 | 1460 / 1728 | 481 |
| int8 | 50,229,248 | 200,944,776 | 75,249,518 | 596 / 597 | 1722 / 1933 | 526 |

Quantized: 88 nn.Linear modules, 41,943,040 weight parameters (int8). Left fp32: 8,286,208 parameters (Linear biases 63,488 included); other parameter-holding module types: Embedding, LayerNorm.

Peak RSS = process peak working set in a fresh process per run (measured with `psutil.peak_wset`). An int8 process first loads the fp32 weights and quantizes in place, so its PEAK includes the transient fp32 copy; the working set after load and after the run are in `results.json` (`rss_after_load_mb_per_run`).

## Findings

- Plain summary: int8 dynamic quantization cuts the saved model state from 201.0 MB to 75.2 MB (2.67x smaller, torch.save of the state_dict) and, on E1, changes no metric beyond bootstrap noise (see quality lines below; hundreds of individual outputs do differ). Single-request p50 latency, int8 vs fp32 over the 4 thread/mode settings: 0 faster, 3 slower, 1 indistinguishable (a difference is only called when the between-run ranges do not overlap).
- Hypothesis for the missing speed-up, NOT tested here: single-sentence decoding is dominated by many tiny (1 x 512) matmuls, the fp32 tied output projection and Python-loop overhead, so int8 matmuls (plus per-call activation quantization) have little to win.
- latency p50, threads=default, greedy: int8 SLOWER, ranges do not overlap (ratio of medians 1.24)
- latency p50, threads=default, beam5: int8 SLOWER, ranges do not overlap (ratio of medians 1.44)
- latency p50, threads=1, greedy: int8 SLOWER, ranges do not overlap (ratio of medians 1.26)
- latency p50, threads=1, beam5: no clear difference, run-to-run ranges overlap (ratio of medians 0.92)
- throughput sent/s, threads=default, greedy: no clear difference, run-to-run ranges overlap (ratio of medians 0.83)
- throughput sent/s, threads=default, beam5: no clear difference, run-to-run ranges overlap (ratio of medians 1.02)
- throughput sent/s, threads=1, greedy: int8 faster, ranges do not overlap (ratio of medians 1.17)
- throughput sent/s, threads=1, beam5: no clear difference, run-to-run ranges overlap (ratio of medians 1.11)
- quality, beam5: bleu -0.054 (CI includes 0); chrf +0.000 (CI includes 0); 660 of 1940 differ.
- quality, greedy: bleu +0.035 (CI includes 0); chrf -0.003 (CI includes 0); 716 of 1940 differ.
- background load: 7 of 12 timing cells had at least one run with total CPU above 30% in the 5 s before start; 6 of 12 latency cells have a relative p50 spread above 10%.
- other-process CPU (machine-wide minus the worker's own share) DURING the 8 runs that recorded it (the repeat repetition): min 10.6%, max 35.9% of the machine; earlier runs have only the pre-run sample.
- repetitions: 3 planned (1-3), then one more full pass (4) because several pre-run samples were above 30% or spreads were large; all runs are in the cells above, the `bg <= 30%` column restricts to runs whose pre-run sample was at or below 30%.

## Limits

- One laptop CPU, shared with other jobs; background utilisation is sampled for 5 s BEFORE each run, not during it. Absolute latencies do not transfer to server CPUs.
- Single-sentence latency includes a Python per-step decode loop (bookkeeping, n-gram blocking), not only matmuls; only nn.Linear matmuls are int8 and the output projection stays fp32. This torch build exposes only the `onednn` quantized engine; ratios may differ on other builds or CPUs.
- Quantization quality is measured on E1 only, at beam 5 (tuned) and greedy.
- No GPU numbers are produced here.
- p99 over 200 requests per run is a two-point estimate; pooled values: `results.json`.

## Files

- `results.json`: all numbers, per-run values, context, background load per run.
- `raw_latencies.json`: every timed request (ms) per cell and run.
- `predictions/`, `official/`: E1 outputs and official scorer reports per config.
- Reproduce: pull with `scripts.eval_local.pull_run(..., with_model=True)`, then `python -m scripts.production_benchmark run` and `... aggregate`.

## Provenance

- code SHA: `c93b86e47a4be44cea7028225b268cf0122776ed` (tracked files dirty: False)
- HF revision: `c3d8598252853fcd7df1ef4a00e8b0382b8f4351` (`OWNER/fr-en-transformer-eval`, private, read-only pull)
- official/score.py sha256: `0e023e486a2a0ca5a2d111b4bf43b0419479b23d4cb79b7789827f35783a8393`
- code SHA the measurement jobs ran from (jobs per SHA):
  - `d3d4f01ee4a28f46b4df3e7e4d449b2d38e8c7cd (not recorded per job; clean tree)`: 28 jobs
  - `510f8b9768d3ca1c64f5a457bf4d5322640d1232`: 8 jobs
- file sha256 (LF line endings verified at write time: no CR byte in any listed file):
  - `results.json`: `75ec36dc0b44ede75b8dfc8b576d154c4f91199c3a4628bce9d85d7c32236116`
  - `raw_latencies.json`: `78d76b7bdaec13857a636d91291d5f18c33eed84a17732864ee68aafa62e3141`
  - `predictions/e1_fp32_beam5.json`: `116b483cd76820b8278abc5dbdedaf2b66b8bb61acb427cdd07780b992b6be82`
  - `predictions/e1_fp32_greedy.json`: `49525bc6f6d3bcce94c9f505d244c4d529c391f129cd51f10d8c4bdf38e72c70`
  - `predictions/e1_int8_beam5.json`: `1a3ff26cd9be93a0b8a601794d4f1e8b117affdcef04cbcc02b11ad9683bcd00`
  - `predictions/e1_int8_greedy.json`: `d896381fdd9751362ecc866d6b8171e4668206eb0d1044e74b7f83ee45eed6d5`
  - `official/official_e1_fp32_beam5.json`: `86372ad5a8128d2f007a6a7429476041ceb47a3d54daf58ed2dd2d91a1f0b0b5`
  - `official/official_e1_fp32_greedy.json`: `564b3bc2ed6fed16c51ef7f4c4b0cf7154183dfd3d3e867ad48853466359276e`
  - `official/official_e1_int8_beam5.json`: `ab284282af29f5c455ec91981cb5dcd1402e7fe2109e3e0414a89c27b1aa5d2e`
  - `official/official_e1_int8_greedy.json`: `7ce967c07e706a20fabca8b647969f5a5157b2ee78da83399a57c33c3b3c781c`
