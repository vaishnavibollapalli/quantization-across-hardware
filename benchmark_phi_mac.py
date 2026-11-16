"""Sequential, resumable Phi experiment; original Qwen inputs and CSV schema."""
import csv
import fcntl
import hashlib
import json
import math
import platform
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'phi_mac'
MODELS = Path.home() / 'quantization-project/models/phi3'
FIELDS = ['model', 'input_target', 'output_target', 'run', 'input_tokens',
          'output_tokens', 'prefill_tokens_per_sec', 'decode_tokens_per_sec',
          'prompt_ms', 'generation_ms']
URL = 'http://127.0.0.1:8091'


def memory():
    raw = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+)', raw)[1])
    def pages(name):
        return int(re.search(re.escape(name) + r':\s+(\d+)', raw)[1])
    # Conservative reclaimable headroom, excluding compressor and active pages.
    available = page * sum(pages(k) for k in ['Pages free', 'Pages inactive', 'Pages speculative'])
    return {'available_bytes': available, 'vm_stat': raw}


def request(endpoint, payload=None, timeout=180):
    req = urllib.request.Request(URL + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def persist(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        import os
        os.fsync(f.fileno())
    temp.replace(path)


def metal_error(path):
    return bool(re.search(r'(metal.*(?:error|failed|abort)|(?:error|failed).*metal|out of memory|failed to allocate)',
                          path.read_text(errors='replace'), re.I))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hardware = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True)
        if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not all(
                s in hardware for s in ['MacBook Air', 'Apple M2', 'Memory: 8 GB']):
            raise RuntimeError('Only the MacBook Air M2 with 8 GB is authorized')
        status = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'hardware': 'MacBook Air M2, 8 GB', 'models': {}}
        def save():
            temp = OUT / 'status.tmp'
            temp.write_text(json.dumps(status, indent=2))
            temp.replace(OUT / 'status.json')
        for quant, suffix in [('Q4', 'Q4_K_M'), ('Q8', 'Q8_0')]:
            proc = None
            entry = status['models'][quant] = {}
            try:
                if quant == 'Q8' and status['models']['Q4'].get('status') != 'complete':
                    raise RuntimeError('Q8 skipped: Q4 did not complete safely')
                if quant == 'Q8' and any(metal_error(p) for p in OUT.glob('llama_server_*.log')):
                    raise RuntimeError('Q8 blocked by recorded Metal/allocation error')
                model = MODELS / f'Phi-3.1-mini-4k-instruct-{suffix}.gguf'
                # Model + 4096-token FP16 KV cache (~1.5 GiB) + 1 GiB runtime reserve.
                required = model.stat().st_size + int(2.5 * 1024**3)
                entry.update(memory=memory(), required_bytes=required, model_path=str(model))
                if entry['memory']['available_bytes'] < required:
                    raise RuntimeError('Insufficient conservative memory headroom; no model loaded')
                processes = subprocess.run(['pgrep', '-lf', 'llama-server|llama-cli|ollama|mlx_lm'], capture_output=True, text=True)
                if processes.returncode != 1:
                    raise RuntimeError('Another model process exists or process inspection is unavailable')
                path = OUT / f'phi31_mini_{quant.lower()}_mac_benchmarks.csv'
                prompts = {n: (ROOT / name).read_text() for n, name in [(256, 'q4_prompt_256.txt'), (1024, 'prompt_1024.txt')]}
                fingerprint = {'model_bytes': model.stat().st_size, 'model_mtime_ns': model.stat().st_mtime_ns,
                    'prompts': {str(n): hashlib.sha256(p.encode()).hexdigest() for n, p in prompts.items()},
                    'settings': 'threads=4,context=4096,parallel=1,temp=0,cache=false,ignore_eos=true'}
                meta = path.with_suffix('.metadata.json')
                if path.exists() and (not meta.exists() or json.loads(meta.read_text()) != fingerprint):
                    raise RuntimeError('Resume metadata mismatch; existing results preserved')
                rows = []
                if path.exists():
                    with path.open(newline='') as f:
                        reader = csv.DictReader(f)
                        if reader.fieldnames != FIELDS:
                            raise RuntimeError('Resume CSV schema mismatch')
                        rows = list(reader)
                keys = [(int(r['input_target']), int(r['output_target']), int(r['run'])) for r in rows]
                valid = {(i, o, r) for i in prompts for o in [128, 512] for r in [1, 2, 3]}
                if len(set(keys)) != len(keys) or not set(keys) <= valid:
                    raise RuntimeError('Invalid or duplicate resume rows')
                for row in rows:
                    if row['model'] != quant or int(row['output_tokens']) != int(row['output_target']):
                        raise RuntimeError('Invalid resume model or output count')
                    if not all(math.isfinite(float(row[k])) and float(row[k]) > 0 for k in FIELDS[4:]):
                        raise RuntimeError('Invalid resume measurement')
                meta.write_text(json.dumps(fingerprint, indent=2))
                logpath = OUT / f'llama_server_{quant.lower()}.log'
                with logpath.open('a') as log:
                    proc = subprocess.Popen(['llama-server', '-m', str(model), '-t', '4', '-c', '4096', '-np', '1', '--host', '127.0.0.1', '--port', '8091'], stdout=log, stderr=subprocess.STDOUT)
                    for _ in range(120):
                        if proc.poll() is not None or metal_error(logpath):
                            raise RuntimeError('Startup failure or Metal error; inspect server log')
                        try:
                            if request('/health', timeout=2).get('status') == 'ok':
                                break
                        except Exception:
                            pass
                        time.sleep(1)
                    else:
                        raise RuntimeError('Startup timeout')
                    for i, prompt in prompts.items():
                        for o in [128, 512]:
                            missing = [r for r in [1, 2, 3] if (i, o, r) not in keys]
                            if not missing:
                                continue
                            # A new warm-up on every resumed combination, never recorded.
                            for rep in [0] + missing:
                                snap = memory()
                                if snap['available_bytes'] < 256 * 1024**2 or metal_error(logpath):
                                    raise RuntimeError('Low runtime headroom or Metal error')
                                t = request('/completion', {'prompt': prompt, 'n_predict': o, 'temperature': 0, 'cache_prompt': False, 'ignore_eos': True})['timings']
                                values = [t[k] for k in ['prompt_n', 'predicted_n', 'prompt_per_second', 'predicted_per_second', 'prompt_ms', 'predicted_ms']]
                                if not all(math.isfinite(v) and v > 0 for v in values) or t['predicted_n'] != o:
                                    raise RuntimeError('Invalid or incomplete measurement')
                                if metal_error(logpath):
                                    raise RuntimeError('Metal error after request')
                                if rep:
                                    rows.append(dict(zip(FIELDS, [quant, i, o, rep] + values)))
                                    persist(path, rows)
                                    print(quant, i, o, rep, flush=True)
                entry['status'] = 'complete'
            except Exception as exc:
                entry.update(status='skipped_or_stopped', reason=str(exc))
                print(quant, str(exc), flush=True)
            finally:
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                save()
        return status


if __name__ == '__main__':
    run()
