import csv,json,pathlib,shutil
root=pathlib.Path('/Users/amruthanatashakodali/quantization-project')
out=pathlib.Path('/Users/amruthanatashakodali/Documents/Codex/2026-11-16/referenced-chatgpt-conversation-this-is-an/outputs')
status=json.loads((out/'phi3_cpu_run_status.json').read_text())
files=[root/'phi3_q4_cpu_repeated.csv',root/'phi3_q4_cpu_256_128.csv',root/'phi3_q4_cpu_1024_128.csv']+[pathlib.Path(r['csv']) for r in status['runs'] if r['status']=='complete']
lines=['# Phi-3.1 Mini Q4 CPU benchmark results','','Hardware: MacBook Air, Apple M2, 8 GB RAM. Model: Phi-3.1-mini-4k-instruct-Q4_K_M.gguf (2,392,493,568 bytes; 3,821,079,552 parameters). llama-bench build 11669, commit 620d2225c.','','Settings: 4 CPU threads, 0 GPU layers, batch 2048, microbatch 512, FP16 K/V cache. Newly launched runs use three repetitions with the default warm-up. The already-running 1024/128 command was observed using three repetitions. Repetition counts for the previously completed files are not stored in their CSVs.','','These are separate prompt-processing (pp) and generation (tg) microbenchmarks. A filename such as 1024_512 contains pp1024 and tg512, not generation conditioned on a 1024-token prompt. Reported values are llama-bench mean throughput ± standard deviation across repetitions.','','| File/workload | Prompt tokens/s | Generation tokens/s |','|---|---:|---:|']
for p in files:
 rows=list(csv.DictReader(p.open()))
 assert len(rows)==2
 assert all(r['n_gpu_layers']=='0' and r['n_threads']=='4' and r['build_commit']=='620d2225c' for r in rows)
 pp=next(r for r in rows if int(r['n_prompt'])>0);tg=next(r for r in rows if int(r['n_gen'])>0)
 lines.append(f"| {p.name} (pp{pp['n_prompt']}/tg{tg['n_gen']}) | {float(pp['avg_ts']):.2f} ± {float(pp['stddev_ts']):.2f} | {float(tg['avg_ts']):.2f} ± {float(tg['stddev_ts']):.2f} |")
 shutil.copy2(p,out/p.name)
lines+=['','Original results remain in `/Users/amruthanatashakodali/quantization-project`. Copies are included beside this report. Existing files were preserved. New-run diagnostic logs remain beside their CSVs in the project.','','Memory pressure was initially elevated (warning level 2), later returning to normal (level 1). Swap was already in use. This can influence timings; these results represent the machine’s observed state rather than an isolated idle-system comparison. The runner checked memory every 10 seconds and was configured to terminate its own run on critical pressure (level 4) or allocation errors. It did not terminate the pre-existing benchmark. Q8 was not launched.','','Qwen result files inspected used GPU offload (`n_gpu_layers=99`), so matching token counts do not make these Phi CPU timings a hardware-equivalent model comparison.','','Runner outcome: '+status.get('status','running')+'.']
(out/'phi3_cpu_summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
