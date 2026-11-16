# Mac benchmark comparison

Actual server timings only. Sample standard deviation (ddof=1); warm-ups excluded.

| Model | Quantization | Input target | Output target | n | Prefill tokens/s | Decode tokens/s |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B | Q4 | 256 | 128 | 3 | 160.21 ± 16.09 | 13.54 ± 0.80 |
| Qwen2.5-3B | Q4 | 256 | 512 | 3 | 122.00 ± 23.88 | 12.09 ± 1.70 |
| Qwen2.5-3B | Q4 | 1024 | 128 | 3 | 172.89 ± 3.47 | 14.45 ± 0.04 |
| Qwen2.5-3B | Q4 | 1024 | 512 | 3 | 172.84 ± 4.26 | 14.27 ± 0.18 |
| Qwen2.5-3B | Q8 | 256 | 128 | 3 | 184.86 ± 5.02 | 9.26 ± 0.65 |
| Qwen2.5-3B | Q8 | 256 | 512 | 3 | 173.28 ± 21.31 | 10.03 ± 0.90 |
| Qwen2.5-3B | Q8 | 1024 | 128 | 3 | 194.38 ± 7.94 | 10.22 ± 0.87 |
| Qwen2.5-3B | Q8 | 1024 | 512 | 3 | 191.47 ± 5.92 | 9.83 ± 0.61 |
| Phi-3.1 Mini | Q4 | 256 | 128 | 0 | Unavailable | Unavailable |
| Phi-3.1 Mini | Q4 | 256 | 512 | 0 | Unavailable | Unavailable |
| Phi-3.1 Mini | Q4 | 1024 | 128 | 0 | Unavailable | Unavailable |
| Phi-3.1 Mini | Q4 | 1024 | 512 | 0 | Unavailable | Unavailable |
| Phi-3.1 Mini | Q8 | 256 | 128 | 0 | Unavailable | Unavailable |
| Phi-3.1 Mini | Q8 | 256 | 512 | 0 | Unavailable | Unavailable |
| Phi-3.1 Mini | Q8 | 1024 | 128 | 0 | Unavailable | Unavailable |
| Phi-3.1 Mini | Q8 | 1024 | 512 | 0 | Unavailable | Unavailable |
