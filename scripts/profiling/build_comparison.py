import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path(__file__).resolve().parent / 'mpl_cache')
import csv, json, statistics as st, hashlib, shutil, html, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/Users/amruthanatashakodali/quantization-project')
OUT = Path(__file__).resolve().parents[1] / 'outputs' / 'profiling_analysis'
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'sources').mkdir(exist_ok=True)
sources = []
def read(rel):
    p=ROOT/rel
    data=p.read_bytes()
    dest=OUT/'sources'/p.name
    if dest.exists(): assert dest.read_bytes()==data
    else: shutil.copy2(p,dest)
    sources.append({'file':rel,'snapshot':'sources/'+p.name,'sha256':hashlib.sha256(data).hexdigest()})
    return list(csv.DictReader(data.decode().splitlines()))

workloads=[(256,128),(256,512),(1024,128),(1024,512)]
phi=[]
for p,g in workloads:
    rows=read(f'phi3_q4_cpu_{p}_{g}.csv')
    assert len(rows)==2 and all(r['n_gpu_layers']=='0' and r['n_threads']=='4' for r in rows)
    pp=next(r for r in rows if int(r['n_prompt'])==p)
    tg=next(r for r in rows if int(r['n_gen'])==g)
    phi.append(dict(prompt=p,output=g,pp=float(pp['avg_ts']),pp_sd=float(pp['stddev_ts']),tg=float(tg['avg_ts']),tg_sd=float(tg['stddev_ts']),pp_s=float(pp['avg_ns'])/1e9,tg_s=float(tg['avg_ns'])/1e9,source=f'phi3_q4_cpu_{p}_{g}.csv'))
phi_model_bytes=int(pp['model_size'])
qraw=read('quantization-across-hardware/mac_automated_benchmarks.csv')
qwen=[]
assert len(qraw)==24
for quant in ['Q4','Q8']:
    for p,g in workloads:
        rows=[r for r in qraw if r['model']==quant and int(r['input_target'])==p and int(r['output_target'])==g]
        assert len(rows)==3 and {int(r['run']) for r in rows}=={1,2,3}
        assert all(int(r['input_tokens'])==p and int(r['output_tokens'])==g for r in rows)
        result=dict(quant=quant,prompt=p,output=g,n=3)
        for key,col in [('pp','prefill_tokens_per_sec'),('tg','decode_tokens_per_sec'),('prompt_ms','prompt_ms'),('generation_ms','generation_ms')]:
            values=[float(r[col]) for r in rows]
            result[key]=st.mean(values);result[key+'_sd']=st.stdev(values)
        result['timing_sum_s']=st.mean((float(r['prompt_ms'])+float(r['generation_ms']))/1000 for r in rows)
        qwen.append(result)
micro=[]
for quant in ['q4','q8']:
    for r in read(f'results/mac_m2_{quant}_full_bench.csv'):
        assert r['n_gpu_layers']=='99'
        micro.append(dict(quant=quant.upper(),test='pp'+r['n_prompt'] if int(r['n_prompt']) else 'tg'+r['n_gen'],mean=float(r['avg_ts']),sd=float(r['stddev_ts']),size_bytes=int(r['model_size']),gpu_layers=int(r['n_gpu_layers'])))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#162438','axes.edgecolor':'#cbd5e1','figure.facecolor':'white','savefig.facecolor':'white'})
teal='#087e8b'; orange='#d97928'; blue='#3565a9'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11.8,4.8))
labels=[f'pp{p} / tg{g}' for p,g in workloads]
for ax,key,title in zip(axs,['pp','tg'],['Prompt processing','Token generation']):
    vals=[r[key] for r in phi];sd=[r[key+'_sd'] for r in phi]
    ax.bar(np.arange(4),vals,yerr=sd,capsize=4,color=teal,width=.58)
    ax.set_xticks(range(4),labels,rotation=15);ax.set_ylabel('Tokens per second');ax.set_title(title,loc='left',pad=14)
    ax.set_ylim(0,max(v+s for v,s in zip(vals,sd))*1.23)
    for i,(v,s) in enumerate(zip(vals,sd)):ax.text(i,v+s+.25,f'{v:.2f}',ha='center',fontsize=10)
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
fig.suptitle('Phi-3.1 Mini Q4 · CPU only',x=.08,ha='left',fontsize=17,fontweight='bold')
fig.text(.08,.01,'Separate pp and tg tests; means ± reported standard deviations. Three repetitions for newly completed workloads.',fontsize=9,color='#526174')
fig.tight_layout(rect=[0,.06,1,.92]);save(fig,'phi_cpu_workloads')

fig,axs=plt.subplots(1,2,figsize=(11.8,5.1));x=np.arange(4);w=.34
for ax,key,title in zip(axs,['pp','tg'],['Prompt processing','Token generation']):
    for quant,color,offset in [('Q4',teal,-w/2),('Q8',orange,w/2)]:
        rr=[r for r in qwen if r['quant']==quant]
        ax.bar(x+offset,[r[key] for r in rr],w,yerr=[r[key+'_sd'] for r in rr],capsize=3,color=color,label=quant)
    ax.set_xticks(x,[f'{p} → {g}' for p,g in workloads]);ax.set_ylabel('Tokens per second');ax.set_title(title,loc='left',pad=14)
    ax.set_ylim(0,max(r[key]+r[key+'_sd'] for r in qwen)*1.24)
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);ax.legend(frameon=False)
fig.suptitle('Qwen2.5-3B · existing server workloads',x=.08,ha='left',fontsize=17,fontweight='bold')
fig.text(.08,.02,'Three requests per condition; mean ± sample SD. Input → output token counts. Backend offload count was not recorded.',fontsize=9,color='#526174')
fig.tight_layout(rect=[0,.07,1,.92]);save(fig,'qwen_quantization')

fig,axs=plt.subplots(1,3,figsize=(12.5,4.8),gridspec_kw={'width_ratios':[1.15,1.15,1]})
names=['Phi Q4\nCPU (0 layers)','Qwen Q4\nGPU offload (99)','Qwen Q8\nGPU offload (99)']
for ax,key,test,title in [(axs[0],'pp','pp256','pp256 throughput'),(axs[1],'tg','tg128','tg128 throughput')]:
    rr=[r for r in micro if r['test']==test]
    values=[phi[0][key]]+[r['mean'] for r in rr]
    sd=[phi[0][key+'_sd']]+[r['sd'] for r in rr]
    ax.bar(range(3),values,yerr=sd,capsize=3,color=[blue,teal,orange],width=.6)
    ax.set_xticks(range(3),names,fontsize=8);ax.set_ylabel('Tokens per second');ax.set_title(title,loc='left');ax.set_ylim(0,max(v+s for v,s in zip(values,sd))*1.2)
    for i,(v,s) in enumerate(zip(values,sd)):ax.text(i,v+s+max(values)*.035,f'{v:.2f}',ha='center',fontsize=9)
sizes=[phi_model_bytes]+[next(r['size_bytes'] for r in micro if r['quant']==q) for q in ['Q4','Q8']]
axs[2].bar(range(3),[v/1024**3 for v in sizes],color=[blue,teal,orange],width=.6)
axs[2].set_xticks(range(3),['Phi Q4','Qwen Q4','Qwen Q8'],fontsize=9);axs[2].set_ylabel('GiB');axs[2].set_title('Model file size',loc='left');axs[2].set_ylim(0,3.7)
for i,v in enumerate(sizes):axs[2].text(i,v/1024**3+.08,f'{v/1024**3:.2f}',ha='center')
for ax in axs:ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
fig.suptitle('Observed configurations · different compute backends',x=.07,ha='left',fontsize=17,fontweight='bold')
fig.text(.07,.02,'llama-bench microbenchmarks. Throughput includes backend and run-condition differences; it does not isolate model effects.',fontsize=9,color='#526174')
fig.tight_layout(rect=[0,.07,1,.9]);save(fig,'configuration_comparison')

def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def pm(r,k):return f"{r[k]:.2f} ± {r[k+'_sd']:.2f}"
def image(name,alt):
    return '<figure><img alt="'+alt+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'"></figure>'
q4=[r for r in qwen if r['quant']=='Q4'];q8=[r for r in qwen if r['quant']=='Q8']
decode_gain=[(a['tg']/b['tg']-1)*100 for a,b in zip(q4,q8)]
prefill_gain=[(b['pp']/a['pp']-1)*100 for a,b in zip(q4,q8)]
size_reduction=(1-sizes[1]/sizes[2])*100
speed_drop=[(1-phi[1]['tg']/phi[0]['tg'])*100,(1-phi[3]['tg']/phi[2]['tg'])*100]

body='''<header><div class="eyebrow">EDGE LLM PROFILING · MACBOOK AIR M2 · 8 GB</div><h1>Quantization, workload length,<br>and the limits of comparison</h1><p class="subtitle">Consolidated local results · November 16, 2026</p></header>
<section><h2>Findings</h2><p>Phi-3.1 Mini Q4 completed all four requested CPU workloads. Its measured generation throughput was 7.55–8.13 tokens/s for tg128 and 4.46–4.84 tokens/s for tg512. Memory pressure reached the warning level during parts of the session and later returned to normal. No allocation errors were found in the two newly launched runs. Phi Q8 remains unmeasured.</p>'''
body+=f'<p>The existing Qwen server results show a quantization trade-off: Q4 generated tokens {min(decode_gain):.1f}–{max(decode_gain):.1f}% faster than Q8 across the four conditions, while Q8 processed prompts {min(prefill_gain):.1f}–{max(prefill_gain):.1f}% faster. The Q4 model file is {size_reduction:.1f}% smaller than Q8. These are descriptive observations from three requests per condition.</p>'
body+='''<div class="note"><strong>Interpretation boundary.</strong> Phi uses CPU-only llama-bench tests. Qwen’s full-workload results use llama-server requests, with a different prompt and generation procedure. Separate Qwen llama-bench files explicitly record 99 GPU layers. Server logs do not record the exact offload count, so their backend is described as default server configuration, rather than assumed to be identical to those microbenchmarks. This dataset cannot establish an architecture-only Phi versus Qwen speed ranking.</div></section>
<section><h2>1. Phi Q4 CPU results</h2><p>Each filename encodes two independent tests: prompt processing at the first length and generation at the second length. For example, 1024/512 means pp1024 plus tg512. It does not measure 512-token generation conditioned on a 1024-token prompt. All four rows use four CPU threads, zero GPU layers, batch 2048, microbatch 512, and FP16 key/value caches.</p>'''
body+=table(['File workload','Prompt tokens/s','Generation tokens/s','Prompt time (s)','Generation time (s)'],[[f"{r['prompt']}/{r['output']}",pm(r,'pp'),pm(r,'tg'),f"{r['pp_s']:.2f}",f"{r['tg_s']:.2f}"] for r in phi])
body+=image('phi_cpu_workloads','Phi CPU prompt processing and generation throughput with standard deviations')
body+=f'<p>Generation throughput in the two tg512 runs was {speed_drop[0]:.1f}% and {speed_drop[1]:.1f}% lower than the corresponding tg128 runs. Since these tests were sequential, this difference may include generated-context growth, memory state, and thermal effects. There are no temperature or power traces to isolate the cause. The label’s prompt length is not an input to the separate tg test.</p>'
body+='''<p>The main 256/128 result is retained as the baseline. The earlier repeated file contains pp256/tg64 (15.90 ± 0.65 and 8.33 ± 0.19 tokens/s), a different generation length; it is not pooled with tg128. The one-shot test file is excluded from the main analysis.</p></section>
<section><h2>2. Qwen Q4 versus Q8: existing full requests</h2><p>These 24 source rows cover two quantizations × four workloads × three measured requests. Recorded input and output counts match all targets. The saved runner specifies four CPU threads, context 4096, one request slot, temperature zero, prompt caching disabled, and forced output length. It excludes one warm-up request per condition. Means and sample standard deviations below were recomputed from the raw CSV.</p>'''
body+=table(['Quantization','Input → output','n','Prompt tokens/s','Generation tokens/s','Timing sum (s)'],[[r['quant'],f"{r['prompt']} → {r['output']}",r['n'],pm(r,'pp'),pm(r,'tg'),f"{r['timing_sum_s']:.2f}"] for r in qwen])
body+=image('qwen_quantization','Qwen Q4 and Q8 server throughput across four request workloads')
body+='''<p>“Timing sum” is mean recorded prompt time plus generation time. It excludes HTTP overhead and is not a client-side wall-clock latency or a time-to-first-token measurement. No rates were recomputed from rounded timings, and rates were not pooled across different workload lengths.</p>'''
body+=table(['Input → output','Q4 decode advantage','Q8 prefill advantage'],[[f'{p} → {g}',f'{d:.1f}%',f'{f:.1f}%'] for (p,g),d,f in zip(workloads,decode_gain,prefill_gain)])
body+='''<p>For this recorded Qwen workload set, Q4 combines a smaller file with faster generation. Q8 has higher prompt throughput. This does not measure an accuracy trade-off, prove the cause of either performance difference, or establish significance from three samples.</p></section>
<section><h2>3. Configuration comparison</h2><p>The closest available common test format is llama-bench pp256/tg128. Phi was recorded on November 16 using zero GPU layers; Qwen was recorded on November 12 using 99 GPU layers. All record the same build commit, four CPU threads, batch 2048, and microbatch 512. Model family, parameter count, backend, and session state differ.</p>'''
body+=image('configuration_comparison','Three observed model configurations showing throughput and model file sizes with backend labels')
body+=table(['Configuration','Parameters (billions)','File size (GiB)','pp256 tokens/s','tg128 tokens/s'],[
    ['Phi Q4 · CPU only','3.821',f'{sizes[0]/1024**3:.2f}',pm(phi[0],'pp'),pm(phi[0],'tg')],
    *[[f'Qwen {q} · 99 GPU layers','3.086',f"{next(r['size_bytes'] for r in micro if r['quant']==q)/1024**3:.2f}",f"{next(r['mean'] for r in micro if r['quant']==q and r['test']=='pp256'):.2f} ± {next(r['sd'] for r in micro if r['quant']==q and r['test']=='pp256'):.2f}",f"{next(r['mean'] for r in micro if r['quant']==q and r['test']=='tg128'):.2f} ± {next(r['sd'] for r in micro if r['quant']==q and r['test']=='tg128'):.2f}"] for q in ['Q4','Q8']]])
body+='''<p>Model file size is not runtime memory consumption. It excludes the key/value cache, compute buffers, runtime state, and other applications. Phi also contains more parameters than Qwen, adding another reason to avoid attributing the observed difference solely to architecture.</p></section>
<section><h2>4. Memory, missing measurements, and scope</h2><ul><li><strong>Phi Q8:</strong> deliberately not run under the memory constraint. Its missing results are unavailable, never zero. Existing Qwen Q8 results are historical measurements and do not establish that Phi Q8 is safe.</li><li><strong>Phi memory state:</strong> the monitored session moved between warning level 2 and normal level 1, with swap already active. The runner never observed critical level 4. Only the two newly launched workloads have captured error logs; the pre-existing 1024/128 run’s stderr was attached to its terminal.</li><li><strong>Uncertainty:</strong> error bars are standard deviations, not confidence intervals. Phi CSVs contain aggregate statistics rather than individual repetitions. Three repetitions were observed or explicitly requested for the newly completed workloads; the earlier 256/128 CSV does not itself encode its repetition count.</li><li><strong>Energy and quality:</strong> this performance report makes no accuracy or energy-efficiency claim. The files named “energy_response” contain completion timings, not joules or watt readings. Existing quality scores require their own rubric audit before a defensible quality comparison.</li><li><strong>Duplicates:</strong> identical Qwen benchmark copies in the nested results folder are not counted as extra experiments. Pilot and ad hoc server CSVs are excluded from the main four-workload analysis. Earlier Phi server-status files describe a skipped server experiment; the completed CPU microbenchmarks are a separate experiment.</li></ul></section>
<section><h2>5. Conclusion</h2><p>On this 8 GB M2 system, Phi-3.1 Mini Q4 completed the requested CPU benchmark matrix, with lower observed generation throughput in the longer tg512 tests. The existing Qwen server measurements favor Q4 for generation speed and file size, while Q8 shows higher prompt throughput. These results support configuration-specific conclusions. A controlled model-family comparison remains unmeasured because the available runs differ in compute backend and test method.</p><p>A future matched study would use the same backend, benchmark method, workload, cache settings, and recorded memory/thermal state for both model families. That follow-up has not been run here; no additional models were loaded for this analysis.</p></section>
<section><h2>Sources and reproducibility</h2><p>Source snapshots are included in the sources folder. The consolidated JSON preserves computed statistics, file identities, and SHA-256 fingerprints. All original project files remain unchanged.</p><ul>'''
for source in sources:
    body+='<li><a href="'+source['snapshot']+'">'+html.escape(source['file'])+'</a></li>'
body+='''</ul><p>Additional inspected evidence: benchmark_mac.py, llama_server_q4.log, llama_server_q8.log, the earlier Phi server status.json, and phi3_cpu_run_status.json from the completed CPU session. No external data were used.</p></section>'''
css='''body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#223248;background:#f1f5f9;margin:0}main{max-width:1080px;margin:32px auto;background:white;padding:52px 60px}header{border-bottom:3px solid #087e8b;padding-bottom:24px;margin-bottom:30px}.eyebrow{font-size:12px;letter-spacing:1.6px;font-weight:700;color:#087e8b}h1{font-size:40px;line-height:1.18;letter-spacing:-1px;color:#162438;margin:18px 0}h2{font-size:24px;color:#162438;margin-top:36px}.subtitle{color:#64748b}.note{padding:18px 22px;background:#edf6f7;border-left:4px solid #087e8b;margin:24px 0}img{width:100%;height:auto}figure{margin:28px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th{background:#edf2f7;text-align:left;color:#334155}td,th{padding:11px 12px;border-bottom:1px solid #e2e8f0}td:not(:first-child){text-align:right}li{margin-bottom:9px}a{color:#087e8b}.table-wrap{overflow-x:auto}@media(max-width:700px){main{padding:24px 18px;margin:0}h1{font-size:30px}h2{font-size:22px}}@media print{body{background:white}main{margin:0;padding:0;max-width:none}figure,table,.note{break-inside:avoid}h2{break-after:avoid}a{color:inherit;text-decoration:none}}'''
(OUT/'profiling_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Edge LLM profiling results</title><style>'+css+'</style><main>'+body+'</main></html>')
(OUT/'consolidated_results.json').write_text(json.dumps({'hardware':'MacBook Air M2, 8 GB','phi_cpu_microbenchmarks':phi,'qwen_server_workloads':qwen,'qwen_gpu_microbenchmarks':micro,'sources':sources,'q4_decode_advantage_percent':decode_gain,'q8_prefill_advantage_percent':prefill_gain,'qwen_q4_file_reduction_percent':size_reduction},indent=2))
evidence=[]
for rel in ['quantization-across-hardware/benchmark_mac.py','quantization-across-hardware/llama_server_q4.log','quantization-across-hardware/llama_server_q8.log','quantization-across-hardware/results/phi_mac/status.json','phi3_q4_cpu_repeated.csv','phi3_q4_cpu_256_512.stderr.log','phi3_q4_cpu_1024_512.stderr.log']:
    p=ROOT/rel
    dest=OUT/'sources'/p.name
    shutil.copy2(p,dest)
    evidence.append({'original':rel,'snapshot':'sources/'+dest.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
cpu_status=OUT.parent/'phi3_cpu_run_status.json'
shutil.copy2(cpu_status,OUT/'sources'/cpu_status.name)
for quant in ['q4','q8']:
    name=f'mac_m2_{quant}_full_bench.csv'
    assert (ROOT/'results'/name).read_bytes()==(ROOT/'quantization-across-hardware/results/mac_m2'/name).read_bytes()
(OUT/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
shutil.make_archive(str(OUT.parent/'profiling_analysis_bundle'),'zip',OUT.parent,OUT.name)
print(json.dumps({'output':str(OUT),'q4_decode_advantage_percent':decode_gain,'q8_prefill_advantage_percent':prefill_gain,'qwen_q4_file_reduction_percent':size_reduction,'phi_long_generation_drop_percent':speed_drop},indent=2))
