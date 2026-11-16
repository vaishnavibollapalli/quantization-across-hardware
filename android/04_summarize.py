#!/usr/bin/env python3
"""Turn the Android llama-bench + energy outputs into tables that line up with the Mac ones.
Handles any number of models (Qwen2.5-3B, Phi-3.5-mini, ...).

Writes:
  results/android/android_pilot_comparison.csv   core pp256/tg128 rows, same columns as the Mac file
  results/android/android_sweeps.csv             context / thread / KV-cache sweeps (if run)
  results/android/android_energy_mean_sd.csv     energy runs, mean +- SD over run1..runN
  results/mac_vs_android.csv                     side-by-side per model x quant
"""
import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
AND = RES / "android"
MAC = RES / "mac_m2"
GIB = 1024 ** 3
QUANT_RE = re.compile(r"-(Q\d\w*|f16|F16|bf16|BF16)$")


def split_name(path_str):
    stem = Path(path_str).stem                    # Phi-3.5-mini-instruct-Q4_K_M
    m = QUANT_RE.search(stem)
    return (stem[:m.start()], m.group(1)) if m else (stem, "?")


def bench_rows(path, device):
    out = []
    if not path.exists() or path.stat().st_size == 0:
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            model, q = split_name(r["model_filename"])
            npp, ntg = int(r["n_prompt"]), int(r["n_gen"])
            test = f"pp{npp}" if ntg == 0 else (f"tg{ntg}" if npp == 0 else f"pp{npp}+tg{ntg}")
            if r.get("type_k", "f16") != "f16":
                test += f"_kv{r['type_k']}"
            out.append({
                "device": device, "model": model, "quantization": q,
                "model_size_gib": round(int(r["model_size"]) / GIB, 2), "test": test,
                "threads": r["n_threads"],
                "tokens_per_second": round(float(r["avg_ts"]), 2),
                "std_dev": round(float(r["stddev_ts"]), 2),
                "repeat_count": 3,
            })
    return out


def print_table(title, rows, cols):
    if not rows:
        return
    print(f"== {title}")
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))
    print()


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def msd(vals, nd=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{st.mean(vals):.{nd}f} ± {st.stdev(vals):.{nd}f}"


def energy_groups():
    """{(model, quant): [summary, ...]} using run files when present, else the single summary."""
    groups = {}
    for p in sorted(AND.glob("android_*energy_summary*.json")):
        d = json.loads(p.read_text())
        key = (d.get("model", "Qwen2.5-3B-Instruct"), d.get("quant"), d.get("kv_cache", "f16"))
        groups.setdefault(key, {"runs": [], "single": None})
        if "_run" in p.stem:
            groups[key]["runs"].append(d)
        else:
            groups[key]["single"] = d
    return {k: (v["runs"] or [v["single"]]) for k, v in groups.items()}


def main():
    # ---- Android core bench
    a_rows = []
    for p in sorted(AND.glob("android_*full_bench.csv")):
        a_rows += bench_rows(p, "Android")
    cols = ["device", "model", "quantization", "model_size_gib", "test",
            "tokens_per_second", "std_dev", "repeat_count"]
    write_csv(AND / "android_pilot_comparison.csv", a_rows, cols)
    print_table("Android core bench (pp256 / tg128)", a_rows, cols)

    # ---- sweeps
    extra = []
    for pat in ("android_*ctx_sweep.csv", "android_*thread_sweep.csv", "android_*kvq_*.csv"):
        for p in sorted(AND.glob(pat)):
            extra += bench_rows(p, "Android")
    if extra:
        write_csv(AND / "android_sweeps.csv", extra, list(extra[0].keys()))
        print_table("Android sweeps / ablations", extra,
                    ["model", "quantization", "test", "threads", "tokens_per_second", "std_dev"])

    # ---- energy mean +- SD
    e_rows = []
    for (model, q, kv), runs in sorted(energy_groups().items()):
        e_rows.append({
            "model": model, "quant": q, "kv_cache": kv, "n_runs": len(runs),
            "decode_tok_s": msd([r.get("predicted_per_second") for r in runs], 2),
            "avg_power_w": msd([r.get("avg_power_w") for r in runs], 2),
            "idle_power_w": msd([r.get("idle_power_w") for r in runs], 2),
            "j_per_token": msd([r.get("j_per_token") for r in runs]),
            "net_j_per_token": msd([r.get("net_j_per_token") for r in runs]),
            "peak_rss_mib": msd([r.get("peak_rss_mib") for r in runs], 0),
            "peak_temp_c": msd([r.get("peak_temp_c") for r in runs], 1),
        })
    if e_rows:
        ecols = list(e_rows[0].keys())
        write_csv(AND / "android_energy_mean_sd.csv", e_rows, ecols)
        print_table("Android energy (mean ± SD over runs)", e_rows, ecols)

    # ---- Mac vs Android
    m_rows = []
    for p in sorted(MAC.glob("mac_m2_*full_bench.csv")):
        m_rows += bench_rows(p, "MacBook_Air_M2")
    look = {(r["device"], r["model"], r["quantization"], r["test"]): r["tokens_per_second"]
            for r in m_rows + a_rows}
    models = sorted({r["model"] for r in m_rows + a_rows})
    cmp_rows = []
    for model in models:
        for test in ("pp256", "tg128"):
            for q in ("Q8_0", "Q4_K_M"):
                mac = look.get(("MacBook_Air_M2", model, q, test))
                andr = look.get(("Android", model, q, test))
                if mac is None and andr is None:
                    continue
                cmp_rows.append({"model": model, "test": test, "quant": q,
                                 "mac_tok_s": mac if mac is not None else "not run",
                                 "android_tok_s": andr if andr is not None else "not run / OOM",
                                 "mac/android": round(mac / andr, 2) if mac and andr else ""})
        for dev in ("MacBook_Air_M2", "Android"):
            for test in ("pp256", "tg128"):
                q8, q4 = look.get((dev, model, "Q8_0", test)), look.get((dev, model, "Q4_K_M", test))
                if q8 and q4:
                    cmp_rows.append({"model": model, "test": f"{test} Q4/Q8 speedup", "quant": dev,
                                     "mac_tok_s": "", "android_tok_s": "",
                                     "mac/android": round(q4 / q8, 2)})
    ccols = ["model", "test", "quant", "mac_tok_s", "android_tok_s", "mac/android"]
    write_csv(RES / "mac_vs_android.csv", cmp_rows, ccols)
    print_table("Mac M2 vs Android", cmp_rows, ccols)


if __name__ == "__main__":
    main()
