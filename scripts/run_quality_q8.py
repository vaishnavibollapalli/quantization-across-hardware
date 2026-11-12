import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
prompts_file = root / "prompt_set" / "quality_prompts.jsonl"
output_file = root / "q8_pilot_outputs.jsonl"

with prompts_file.open(encoding="utf-8") as f:
    prompts = [json.loads(line) for line in f if line.strip()]

with output_file.open("w", encoding="utf-8") as f:
    for item in prompts[:2]:
        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
            ],
            "temperature": item["temperature"],
            "seed": item["seed"],
            "max_tokens": item["n_predict"],
            "cache_prompt": False,
            "stream": False,
        }

        request = urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)

        answer = result["choices"][0]["message"]["content"]

        record = {
            "id": item["id"],
            "model": "qwen2.5-3b",
            "quant": "Q8_0",
            "device": "mac-m2",
            "output": answer,
        }
        f.write(json.dumps(record) + "\n")
        print(f"Saved {item['id']}: {answer[-100:]}")

print(f"Done! Results saved to {output_file.name}")
