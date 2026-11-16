import csv
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

BASE = Path.home() / ".cache/huggingface/hub/models--bartowski--Qwen2.5-3B-Instruct-GGUF/snapshots"
MODELS = {
    "Q4": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "Q8": "Qwen2.5-3B-Instruct-Q8_0.gguf",
}
PORT = 8090
URL = f"http://127.0.0.1:{PORT}"
RESULTS = []

def request(endpoint, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        URL + endpoint, data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def wait_ready(proc):
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError("Server exited during startup")
        try:
            if request("/health", timeout=2).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Server did not become ready")

def benchmark(model, prompt, input_size, output_size):
    payload = {
        "prompt": prompt,
        "n_predict": output_size,
        "temperature": 0,
        "cache_prompt": False,
        "ignore_eos": True
    }

    for run in range(4):
        t = request("/completion", payload)["timings"]
        if run == 0:
            continue  # Warm-up
        RESULTS.append({
            "model": model,
            "input_target": input_size,
            "output_target": output_size,
            "run": run,
            "input_tokens": t["prompt_n"],
            "output_tokens": t["predicted_n"],
            "prefill_tokens_per_sec": t["prompt_per_second"],
            "decode_tokens_per_sec": t["predicted_per_second"],
            "prompt_ms": t["prompt_ms"],
            "generation_ms": t["predicted_ms"]
        })
        print(f"  Run {run}: prefill={t['prompt_per_second']:.2f}, "
              f"decode={t['predicted_per_second']:.2f} tokens/s", flush=True)

        save_results()

def save_results():
    with open("mac_automated_benchmarks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "input_target", "output_target", "run",
            "input_tokens", "output_tokens",
            "prefill_tokens_per_sec", "decode_tokens_per_sec",
            "prompt_ms", "generation_ms"
        ])
        writer.writeheader()
        writer.writerows(RESULTS)

def make_prompt(target):
    if target == 256:
        return Path("q4_prompt_256.txt").read_text()
    if target == 1024:
        return Path("prompt_1024.txt").read_text()
    return None

snapshots = list(BASE.glob("*"))
if not snapshots:
    raise RuntimeError("No downloaded model snapshots found")

for model, filename in MODELS.items():
    matches = list(BASE.glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(filename)

    print(f"\nStarting {model}...", flush=True)
    with open(f"llama_server_{model.lower()}.log", "w") as log:
        proc = subprocess.Popen([
            "llama-server", "-m", str(matches[0]),
            "-t", "4", "-c", "4096", "-np", "1",
            "--host", "127.0.0.1", "--port", str(PORT)
        ], stdout=log, stderr=subprocess.STDOUT)

        try:
            wait_ready(proc)

            for input_size in [256, 1024]:
                prompt = make_prompt(input_size)

                for output_size in [128, 512]:
                    print(f"\n{model}: input={input_size}, output={output_size}",
                          flush=True)
                    benchmark(model, prompt, input_size, output_size)

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

print("\nCOMPLETE! Saved mac_automated_benchmarks.csv")
