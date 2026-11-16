"""Summarize actual measurements; missing Phi runs remain missing."""
import csv
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/phi_mac'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}
    sources = [('Qwen2.5-3B', ROOT / 'mac_automated_benchmarks.csv')]
    sources += [('Phi-3.1 Mini', OUT / f'phi31_mini_{q}_mac_benchmarks.csv') for q in ['q4', 'q8']]
    for family, path in sources:
        if not path.exists():
            continue
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                key = (family, row['model'], int(row['input_target']), int(row['output_target']))
                groups.setdefault(key, []).append(row)
    metrics = ['prefill_tokens_per_sec', 'decode_tokens_per_sec', 'prompt_ms', 'generation_ms']
    summary = []
    lines = ['# Mac benchmark comparison', '', 'Actual server timings only. Sample standard deviation (ddof=1); warm-ups excluded.', '',
             '| Model | Quantization | Input target | Output target | n | Prefill tokens/s | Decode tokens/s |',
             '|---|---|---:|---:|---:|---:|---:|']
    for family in ['Qwen2.5-3B', 'Phi-3.1 Mini']:
        for q in ['Q4', 'Q8']:
            for i in [256, 1024]:
                for o in [128, 512]:
                    rows = groups.get((family, q, i, o), [])
                    item = dict(family=family, model=q, input_target=i, output_target=o, n=len(rows))
                    for metric in metrics:
                        values = [float(r[metric]) for r in rows]
                        item[metric + '_mean'] = statistics.mean(values) if values else ''
                        item[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else ''
                    summary.append(item)
                    def fmt(metric):
                        mean, sd = item[metric + '_mean'], item[metric + '_sd']
                        return 'Unavailable' if mean == '' else f'{mean:.2f}' + (f' ± {sd:.2f}' if sd != '' else '')
                    lines.append(f'| {family} | {q} | {i} | {o} | {len(rows)} | {fmt(metrics[0])} | {fmt(metrics[1])} |')
    with (OUT / 'comparison_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    combos = [(256, 128), (256, 512), (1024, 128), (1024, 512)]
    for ax, metric, label in zip(axes, metrics[:2], ['Prefill', 'Decode']):
        for family in ['Qwen2.5-3B', 'Phi-3.1 Mini']:
            for q in ['Q4', 'Q8']:
                items = [next(s for s in summary if (s['family'], s['model'], s['input_target'], s['output_target']) == (family, q, i, o)) for i, o in combos]
                present = [(x, s) for x, s in enumerate(items) if s['n']]
                if present:
                    ax.errorbar([x for x, s in present], [s[metric + '_mean'] for x, s in present],
                                yerr=[s[metric + '_sd'] or 0 for x, s in present], marker='o', capsize=4, label=f'{family} {q}')
        ax.set(title=label, ylabel='Tokens / second', xlabel='Input target / output target')
        ax.set_xticks(range(4), [f'{i}/{o}' for i, o in combos])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('MacBook Air M2 · actual measurements (mean ± sample SD)')
    fig.text(.5, .01, 'Phi Q4 and Q8: no measurements — skipped by memory safety preflight.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .95])
    fig.savefig(OUT / 'comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
