# Prompt Set — Comparing Quantization Across Hardware

150 fixed prompts for the Raspberry Pi 5 (8 GB) vs. M2 MacBook Air (8 GB) quantization study, in two
separate sets:

| Set | Prompts | File | Used for | Decoding |
|---|---|---|---|---|
| **Performance** | 50 (12 core) | `perf_prompts.jsonl`, `perf/*.txt` | Prefill/decode latency, tokens/s, memory, energy/token, thermals (Weeks 2–5) | Greedy, `ignore_eos`, fixed `n_predict` |
| **Quality** (held out) | 100 | `quality_prompts.jsonl` | FP16 vs INT8 vs INT4 output quality (Week 6) | Greedy, stop at EOS |

Regenerate everything with `python3 build_prompt_set.py`.

## Performance set (50)

| IDs | Category | Count | Prefill | Decode tokens | Core |
|---|---|---|---|---|---|
| P01–P08 | Factual QA | 8 | short (~20–30 tok) | 128 | P01–P02 |
| P09–P14 | Reasoning | 6 | short | 256 | P09–P10 |
| P15–P20 | Few-shot (sentiment, JSON extraction, topic, translation, city extraction, tone rewrite) | 6 | medium (~150–250) | 64 | P15–P16 |
| P21–P25 | Code | 5 | short | 256 | P21 |
| P26–P30 | Long generation (300 words) | 5 | short | 512 | P26 |
| P31–P48 | Context sweep: 2 reports × 3 questions × 256/1024/2048 | 18 | 256 / 1024 / 2048 | 128 | P31–P33 |
| P49–P50 | Summarize a full report | 2 | ~2048 | 256 | P49 |

**Core subset (12, `"core": true`):** run these with the full protocol (warm-up + 3 measured runs) for
every model × quant × device. Run the other 38 once each to show the results hold across task types.
Running all 50 × 4 runs × every configuration on the Pi would take days.

**Context sweep:** each question is asked over 1, 4 and 8 blocks of the same report. The answer is in
block 1, so all three lengths are answerable and only the prefill length changes.

## Quality set (100)

| IDs | Category | Count | Scoring |
|---|---|---|---|
| M01–M20 | Math word problems | 20 | Exact number on the `Answer:` line |
| L01–L08 | Logic | 8 | yes/no, letter, or number |
| F01–F15 | Factual | 15 | Keyword match |
| R01–R15 | Reading comprehension (1–2 report blocks) | 15 | Regex on the facts |
| X01–X12 | Long context (full ~2k-token report) | 12 | Hidden facts late in the report, multi-hop questions |
| I01–I12 | Instruction following | 12 | Bullet/number/line/sentence counts, exact output, JSON, lowercase, prefixes |
| S01–S06 | Few-shot classification | 6 | Exact label |
| C01–C06 | Code | 6 | Runs unit tests |
| O01–O06 | Open-ended | 6 | Manual 1–5 rubric (`rubric.json`); score blind to quant level |

94 of 100 are scored automatically. Every answer key was tested against a correct and a wrong answer.

```bash
python3 score_quality.py outputs.jsonl                      # auto items
python3 score_quality.py outputs.jsonl --manual manual.csv  # + rubric scores (id,model,quant,score)
```

`outputs.jsonl` has one line per run: `{"id","model","quant","device","output"}`.

**Comparing INT8 vs INT4:** both see the same 100 items, so use a paired test (McNemar's test on the
pass/fail pairs) rather than comparing two accuracies independently. Pair it with llama.cpp's perplexity
and KL-divergence against FP16 for an objective measure:

```bash
./llama-perplexity -m model-f16.gguf -f wiki.test.raw --kl-divergence-base model-f16.kld
./llama-perplexity -m model-q4_k_m.gguf --kl-divergence-base model-f16.kld --kl-divergence
```

Output quality for a given GGUF is essentially device-independent, so the quality set only needs to run
once per model × quant, on the faster device (the Mac).

## Run settings (identical across devices)

- `temperature 0`, `seed 42`, system prompt: *"You are a helpful, concise assistant."*
- **`cache_prompt: false`.** The long-context prompts share the same report text, and llama-server's
  prompt cache would otherwise reuse it and make prefill look artificially fast.
- Apply each model's chat template (`llama-server --jinja`, then POST to `/v1/chat/completions`). The
  response's `timings` field gives `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_per_second`.
- Context size `-c 4096` (≈2k-token prompt + up to 512 generated tokens).
- 1–2 warm-up runs, then 3 measured runs per core configuration; report mean ± SD. Restart the server
  between quant levels so memory numbers are clean.
- For an exact-length synthetic sweep, `llama-bench -m model.gguf -p 256,1024,2048 -n 128 -r 3 -o csv`
  is independent of tokenizer; use it alongside the real prompts.

## Token counts

`token_counts_estimate.csv` uses a 4-characters-per-token estimate. Tokenizers differ per model family,
so get exact counts once per model:

```bash
python3 build_prompt_set.py --model models/qwen2.5-3b-q8_0.gguf \
        --tokenizer-bin ~/llama.cpp/build/bin/llama-tokenize
```

Bucket names (ctx256/1024/2048) are nominal; report the **actual** `prompt_n` from each run.

## The two reports

*Kestrel Valley Community Microgrid* (report A) and *Ashgrove County Public Library System* (report B)
are fictional and written for this benchmark, so no model has memorized them and there are no copyright
issues. Every reading and long-context answer is in them.
