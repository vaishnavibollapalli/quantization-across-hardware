# Phi-3.1 Mini Q4 CPU benchmark results

Hardware: MacBook Air, Apple M2, 8 GB RAM. Model: Phi-3.1-mini-4k-instruct-Q4_K_M.gguf (2,392,493,568 bytes; 3,821,079,552 parameters). llama-bench build 11669, commit 620d2225c.

Settings: 4 CPU threads, 0 GPU layers, batch 2048, microbatch 512, FP16 K/V cache. Newly launched runs use three repetitions with the default warm-up. The already-running 1024/128 command was observed using three repetitions. Repetition counts for the previously completed files are not stored in their CSVs.

These are separate prompt-processing (pp) and generation (tg) microbenchmarks. A filename such as 1024_512 contains pp1024 and tg512, not generation conditioned on a 1024-token prompt. Reported values are llama-bench mean throughput ± standard deviation across repetitions.

| File/workload | Prompt tokens/s | Generation tokens/s |
|---|---:|---:|
| phi3_q4_cpu_repeated.csv (pp256/tg64) | 15.90 ± 0.65 | 8.33 ± 0.19 |
| phi3_q4_cpu_256_128.csv (pp256/tg128) | 15.42 ± 1.00 | 8.13 ± 0.11 |
| phi3_q4_cpu_1024_128.csv (pp1024/tg128) | 11.49 ± 1.61 | 7.55 ± 0.45 |
| phi3_q4_cpu_256_512.csv (pp256/tg512) | 12.05 ± 3.09 | 4.84 ± 0.25 |
| phi3_q4_cpu_1024_512.csv (pp1024/tg512) | 11.40 ± 0.27 | 4.46 ± 0.20 |

Original results remain in `/Users/amruthanatashakodali/quantization-project`. Copies are included beside this report. Existing files were preserved. New-run diagnostic logs remain beside their CSVs in the project.

Memory pressure was initially elevated (warning level 2), later returning to normal (level 1). Swap was already in use. This can influence timings; these results represent the machine’s observed state rather than an isolated idle-system comparison. The runner checked memory every 10 seconds and was configured to terminate its own run on critical pressure (level 4) or allocation errors. It did not terminate the pre-existing benchmark. Q8 was not launched.

Qwen result files inspected used GPU offload (`n_gpu_layers=99`), so matching token counts do not make these Phi CPU timings a hardware-equivalent model comparison.

Runner outcome: complete.
