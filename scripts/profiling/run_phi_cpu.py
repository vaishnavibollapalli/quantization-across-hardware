import csv,json,pathlib,re,subprocess,time,datetime
root=pathlib.Path('/Users/amruthanatashakodali/quantization-project')
out=pathlib.Path('/Users/amruthanatashakodali/Documents/Codex/2026-11-16/referenced-chatgpt-conversation-this-is-an/outputs')
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
record={'repetitions':3,'runs':[],'memory':[]}
def memory():
 s=subprocess.check_output(['sysctl','kern.memorystatus_vm_pressure_level','vm.swapusage'],text=True)
 record['memory'].append({'time':datetime.datetime.now().isoformat(),'snapshot':s})
 return int(re.search(r'pressure_level: (\d+)',s)[1])
def save():
 (out/'phi3_cpu_run_status.json').write_text(json.dumps(record,indent=2))
def valid(p):
 try:
  rows=list(csv.DictReader(p.open()))
  return len(rows)==2 and all(int(r['n_gpu_layers'])==0 and float(r['avg_ts'])>0 for r in rows)
 except Exception:return False
try:
 while subprocess.run(['pgrep','-x','llama-bench'],capture_output=True).returncode==0:
  level=memory();save();print('Waiting for existing benchmark; memory pressure',level,flush=True)
  if level>=4: raise RuntimeError('Critical memory pressure during existing run; no additional workloads launched')
  time.sleep(20)
 p=root/'phi3_q4_cpu_1024_128.csv'
 if valid(p): print('Existing 1024/128 finished successfully',flush=True)
 else: raise RuntimeError('Existing 1024/128 did not produce a complete CSV; stopping for inspection')
 for prompt,gen in [(256,512),(1024,512)]:
  if memory()>=4:raise RuntimeError('Critical memory pressure before next workload')
  p=root/f'phi3_q4_cpu_{prompt}_{gen}.csv'
  if p.exists():p=root/f'phi3_q4_cpu_{prompt}_{gen}_{stamp}.csv'
  log=p.with_suffix('.stderr.log')
  cmd=['/opt/homebrew/bin/llama-bench','-m','models/phi3/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf','-p',str(prompt),'-n',str(gen),'-r','3','-t','4','-ngl','0','-o','csv']
  entry={'prompt':prompt,'generation':gen,'csv':str(p),'log':str(log),'command':cmd,'status':'running'}
  record['runs'].append(entry);save();print('Starting',prompt,gen,flush=True)
  with p.open('x') as f,log.open('x') as err:
   proc=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=err)
   reason=None
   while proc.poll() is None:
    time.sleep(10)
    text=log.read_text(errors='replace')
    if re.search(r'out of memory|failed to allocate|cannot allocate|bad_alloc|allocation failed',text,re.I):reason='Memory allocation error'
    if memory()>=4:reason='Critical memory pressure'
    if reason:
     proc.terminate()
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     break
    save()
   entry.update(exit_code=proc.returncode,status='complete' if not reason and proc.returncode==0 and valid(p) else 'failed',reason=reason)
  save();print(json.dumps(entry),flush=True)
  if entry['status']!='complete':raise RuntimeError('Benchmark stopped; no further workloads launched')
 record['status']='complete'
except Exception as e:
 record.update(status='stopped',reason=str(e));print(str(e),flush=True)
finally:save()
