# Comparing Quantization Across Hardware

**Authors:** Bollapalli Vishnavi Abhishikta, Amrutha Natasha Kodali

How do INT8 and INT4 quantization affect the latency, throughput, memory, energy, thermal behavior, and
output quality of small LLMs on two edge devices?

| | Device A | Device B |
|---|---|---|
| Hardware | Apple M2 MacBook Air | Android Phone |
| Memory | 8 GB unified | 8 GB |
| Backend | llama.cpp (Metal) | llama.cpp (CPU) |

**Models:** Phi-3 Mini (~3.8B), Qwen2.5-3B, Llama-3.1-8B-Instruct
**Precisions:** FP16/BF16, INT8, INT4, and KV-cache quantization where supported

## Research questions

1. How much efficiency do INT8 and INT4 quantization provide?
2. Does INT4 sacrifice noticeable output quality?
3. How does context length affect these gains, and which operations become bottlenecks under more
   aggressive quantization?

## Repository layout

```
prompt_set/        150 fixed prompts (50 performance + 100 quality), generator and scorer
```

See [`prompt_set/README.md`](prompt_set/README.md) for the prompt set and the measurement protocol.

## Quick start

```bash
cd prompt_set
python3 build_prompt_set.py            # regenerate prompts + token-count estimates
python3 score_quality.py outputs.jsonl # score model outputs on the quality set
```

## Phi-3.1 Mini Mac experiment

### Latest completed analysis — November 16, 2026

Phi-3.1 Mini Q4 **CPU-only** llama-bench tests completed for 256/128,
256/512, 1024/128, and 1024/512. These are separate prompt-processing and
text-generation tests, not combined server requests. The newly completed
workloads used three repetitions, four threads, and zero GPU layers. Phi Q8
was not run under the memory constraint.

- [Consolidated report and three charts](results/profiling_analysis/profiling_report.html)
  (download and open the HTML file locally to view the embedded charts).
- [Phi CPU summary](results/phi_cpu/phi3_cpu_summary.md), raw CSVs, diagnostic logs,
  and the monitored session status are in `results/phi_cpu/`.
- [Consolidated statistics and source identities](results/profiling_analysis/consolidated_results.json).
- Standalone PNG/SVG charts and source snapshots are in `results/profiling_analysis/`.
- Analysis and CPU-run scripts are preserved in `scripts/profiling/`. They retain
  the original local paths and are not portable command-line tools.

Existing Qwen Q4/Q8 server results show Q4 generation throughput 20.5–46.3%
higher than Q8, with a 41.3% smaller model file. Q8 prompt throughput was higher.
Phi CPU and Qwen server measurements differ in backend and methodology, so
these results do not isolate a model-family speed advantage. Qwen llama-bench
files explicitly record 99 GPU layers; the exact server offload count was not
recorded. The report explains sample counts, variability, memory conditions,
and missing measurements.

The historical sections below describe the **separate, skipped Phi server
experiment**. Their unavailable server values do not supersede the completed
CPU results above. `benchmark_phi_mac.py` is the older server runner; it was
not executed as part of the completed CPU session.


### Earlier server preflight — November 16, 2026

**Execution is blocked by insufficient memory headroom.** The safety preflight ran on the
local Apple M2 MacBook Air with 8 GB unified memory. It skipped Q4 before loading the model;
Q8 was subsequently skipped because Q4 had not completed safely. There are **zero Phi
benchmark measurements**, so no Phi throughput or speedup claim is possible. This is an
incomplete experiment, not a model failure or a measured performance result.

See [`results/phi_mac/status.json`](results/phi_mac/status.json) for the actual memory
snapshot and required byte count. The conservative check requires model file size plus
2.5 GiB of headroom (approximately 1.5 GiB for the 4096-token FP16 KV cache and 1 GiB for
runtime buffers/reserve). Available headroom is free + inactive + speculative pages from
`vm_stat`; this is a safety estimate, not measured peak model memory. Compressor space
and swap are not treated as available RAM. This may decline a run the OS could otherwise
satisfy through compression or swapping; that is intentional on this 8 GB machine.

### Methodology and reproducibility

`benchmark_phi_mac.py` reuses `benchmark_mac.py`'s Qwen methodology:

- The exact existing `q4_prompt_256.txt` and `prompt_1024.txt` contents, with input targets
  256 and 1024. Tokenizers differ; **actual** Phi input counts would be saved, without
  trimming prompts to force an identical token count.
- Output lengths 128 and 512, all four input/output combinations per quantization.
- One unrecorded warm-up followed by three measured repetitions per combination
  (12 measured rows per model when complete).
- `llama-server`, 4 threads, context 4096, one parallel slot, localhost only. The Phi
  server uses port 8091 to avoid the original Qwen port 8090.
- `/completion` with temperature 0, `cache_prompt=false`, `ignore_eos=true`; throughput
  and durations come directly from response `timings`, not wall-clock estimates.
- The identical CSV columns: `model,input_target,output_target,run,input_tokens,output_tokens,
  prefill_tokens_per_sec,decode_tokens_per_sec,prompt_ms,generation_ms`.

Models are read from `~/quantization-project/models/phi3/`:
`Phi-3.1-mini-4k-instruct-Q4_K_M.gguf`, then
`Phi-3.1-mini-4k-instruct-Q8_0.gguf`. No model download or remote execution is performed.

From the repository directory:

```bash
python3 benchmark_phi_mac.py
python3 analyze_phi_mac.py
```

The runner verifies the authorized Mac hardware, checks memory before each model, and
refuses to load a model if another llama/Ollama/MLX process is detected or process inspection
is unavailable. An exclusive file lock prevents overlapping copies of this experiment.
Q4 always precedes Q8. Q8 requires a successful Q4 and no recorded Metal/allocation
error in Phi server logs. Startup exit, timeout, request failure, low runtime headroom,
invalid timings, incomplete output, and Metal/allocation errors stop the current model.
Every launched server is terminated and reaped before proceeding. Errors are recorded
in `status.json`; inspect it even when the script returns normally. No automatic retries
of failed measurements are performed.

Successful rows are saved atomically after every repetition. Resume validates the CSV
schema, row keys, numeric measurements, model file size/modification time, prompt hashes,
and settings. It keeps completed repetitions and warms up again before each incomplete
combination. Retain the metadata sidecar with the CSV. Model bytes are not cryptographically
hashed; the resume identity check uses size and modification time. Server logs append
across attempts, so a recorded Metal error continues to block Q8 pending investigation.
Before retrying, close memory-heavy applications and confirm adequate available RAM;
do not bypass the guard to force a run.

### Results and analysis

All original Qwen files and results are preserved. Phi artifacts are isolated under
`results/phi_mac/`:

| File | Purpose |
|---|---|
| `status.json` | Actual preflight/error outcome and memory evidence |
| `phi31_mini_q4_mac_benchmarks.csv` | Successful Q4 repetitions only; created when measured |
| `phi31_mini_q8_mac_benchmarks.csv` | Successful Q8 repetitions only; created when measured |
| `*.metadata.json` | Resume settings and input identity |
| `llama_server_q4.log`, `llama_server_q8.log` | Created only when the respective server starts |
| `comparison_summary.csv`, `comparison.md` | Per-combination sample count, means and sample SD |
| `comparison.png` | Prefill and decode means with sample SD error bars |

[`Comparison table`](results/phi_mac/comparison.md) and
[`plot`](results/phi_mac/comparison.png) use only actual rows from
`mac_automated_benchmarks.csv` and any successful Phi CSVs. Missing Phi values remain
blank/unavailable, never zero or estimated. The currently generated plot contains Qwen
measurements only and explicitly labels the missing Phi results. Summary CSVs also include
prompt and generation latency means and sample SD. Partial experiments display their
actual sample counts; only combinations with three repetitions are complete. The existing
Qwen runner records raw timings but contains no aggregation implementation; the new
analysis makes the mean/sample-SD convention explicit and applies it uniformly.
These results do not measure energy, thermal behavior, quality, or peak RAM.

### Retry after application cleanup

The November 16, 2026, 03:31 EST retry found 1.50 GiB of conservative available
headroom against 4.73 GiB required for Q4. Q4 was again skipped before loading;
Q8 was not attempted. No Phi measurements were produced. A restart followed by
opening only the benchmark tools may release additional memory; the same guards
must pass before running.

### Latest preflight retry

At 2026-11-16T03:32:35-0500, available conservative headroom was 1.86 GiB
versus 4.73 GiB required. Q4 was not loaded and Q8 was not attempted.
No Phi measurements were produced. Repeated cleanup retries have not cleared
the memory guard; further execution remains blocked on sufficient RAM.

### Preflight after user cache removal

At 2026-11-16T03:36:05-0500, the three update/package cache directories were confirmed absent.
The system reported 21 GiB available disk space. Conservative available RAM was
1.65 GiB versus 4.73 GiB required. Q4 was skipped before loading and Q8 was
not attempted. No Phi measurements were collected.
