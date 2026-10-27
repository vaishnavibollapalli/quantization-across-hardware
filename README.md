# Comparing Quantization Across Hardware

**Authors:** Bollapalli Vishnavi Abhishikta, Amrutha Natasha Kodali

How do INT8 and INT4 quantization affect the latency, throughput, memory, energy, thermal behavior, and
output quality of small LLMs on two edge devices?

| | Device A | Device B |
|---|---|---|
| Hardware | Apple M2 MacBook Air | Raspberry Pi 5 |
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
