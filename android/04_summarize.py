#!/usr/bin/env python3
"""Turn the Android llama-bench + energy outputs into tables that line up with the Mac ones.

Writes:
  results/android/android_pilot_comparison.csv   same columns as results/mac_m2/mac_m2_pilot_comparison.csv
  results/mac_vs_android.csv                     side-by-side throughput, speedups, energy, memory
  (and prints both as tables)

Runs on the phone or on a laptop after copying results/android/ back.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
AND = RES / "android"
MAC = RES / "mac_m2"
GIB = 1024 ** 3


def bench_rows(path, device):
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            fn = Path(r["model_filename"]).stem       # ...-Q4_K_M.gguf -> Q4_K_M
            q = fn.split("-")[-1] if "-" in fn else r["model_type"].split()[-1]
            npp, ntg = int(r["n_prompt"]), int(r["n_gen"])
            test = f"pp{npp}" if ntg == 0 else (f"tg{ntg}" if npp == 0 else f"pp{npp}+tg{ntg}")
            if r.get("type_k", "f16") != "f16":
                test += f"_kv{r['type_k']}"
            out.append({
                "device": device, "model": "Qwen2.5-3B-Instruct", "quantization": q,
                "model_size_gib": round(int(r["model_size"]) / GIB, 2), "test": test,
                "threads": r["n_threads"],
                "tokens_per_second": round(float(r["avg_ts"]), 2),
                "std_dev": round(float(r["stddev_ts"]), 2),
                "repeat_count": 3,
            })
    return out


def print_table(rows, cols):
    if not rows:
        return
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))
    print()


def energy(path):
    return json.loads(path.read_text()) if path.exists() else {}


def mac_energy(q):
    """Mac energy: tokens/s from the saved response; watts must come from powermetrics."""
    p = MAC / f"mac_m2_{q}_energy_response.json"
    if not p.exists():
        return {}
    t = json.loads(p.read_text()).get("timings", {})
    mem = MAC / f"mac_m2_{q}_memory_samples.txt"
    rss = [int(x) for x in mem.read_text().split() if x.strip().isdigit()] if mem.exists() else []
    return {"predicted_per_second": t.get("predicted_per_second"),
            "peak_rss_mib": round(max(rss) / 1024, 1) if rss else None}


def main():
    # ---- Android pilot table (same format as Mac)
    a_rows = []
    for q in ("q8", "q4"):
        a_rows += bench_rows(AND / f"android_{q}_full_bench.csv", "Android")
    cols = ["device", "model", "quantization", "model_size_gib", "test",
            "tokens_per_second", "std_dev", "repeat_count"]
    with open(AND / "android_pilot_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(a_rows)
    print("== Android core bench (pp256 / tg128)")
    print_table(a_rows, cols)

    # ---- extra sweeps if present
    extra = []
    for p in sorted(AND.glob("android_*_ctx_sweep.csv")) + sorted(AND.glob("android_*_thread_sweep.csv")) \
            + sorted(AND.glob("android_*_kvq_*.csv")):
        extra += bench_rows(p, "Android")
    if extra:
        print("== Android sweeps / ablations")
        print_table(extra, ["quantization", "test", "threads", "tokens_per_second", "std_dev"])
        with open(AND / "android_sweeps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(extra[0].keys()))
            w.writeheader()
            w.writerows(extra)

    # ---- Mac vs Android
    m_rows = []
    for q in ("q8", "q4"):
        m_rows += bench_rows(MAC / f"mac_m2_{q}_full_bench.csv", "MacBook_Air_M2")
    look = {(r["device"], r["quantization"], r["test"]): r["tokens_per_second"]
            for r in m_rows + a_rows}
    cmp_rows = []
    for test in ("pp256", "tg128"):
        for q in ("Q8_0", "Q4_K_M"):
            mac = look.get(("MacBook_Air_M2", q, test))
            andr = look.get(("Android", q, test))
            cmp_rows.append({"test": test, "quant": q, "mac_tok_s": mac, "android_tok_s": andr,
                             "mac/android": round(mac / andr, 2) if mac and andr else ""})
    # INT4 speedup over INT8 on each device
    for dev in ("MacBook_Air_M2", "Android"):
        for test in ("pp256", "tg128"):
            q8, q4 = look.get((dev, "Q8_0", test)), look.get((dev, "Q4_K_M", test))
            if q8 and q4:
                cmp_rows.append({"test": f"{test} Q4/Q8 speedup", "quant": dev,
                                 "mac_tok_s": "", "android_tok_s": "",
                                 "mac/android": round(q4 / q8, 2)})

    for q in ("q8", "q4"):
        a, m = energy(AND / f"android_{q}_energy_summary.json"), mac_energy(q)
        cmp_rows.append({"test": "energy run tok/s (512 tok)", "quant": q.upper(),
                         "mac_tok_s": round(m["predicted_per_second"], 2) if m.get("predicted_per_second") else "",
                         "android_tok_s": round(a["predicted_per_second"], 2) if a.get("predicted_per_second") else "",
                         "mac/android": ""})
        cmp_rows.append({"test": "peak RSS MiB", "quant": q.upper(),
                         "mac_tok_s": m.get("peak_rss_mib", ""), "android_tok_s": a.get("peak_rss_mib", ""),
                         "mac/android": ""})
        if a:
            cmp_rows.append({"test": "android J/token (gross / net of idle)", "quant": q.upper(),
                             "mac_tok_s": "",
                             "android_tok_s": f"{a['j_per_token']:.3f} / {a['net_j_per_token']:.3f}"
                             if a.get("j_per_token") and a.get("net_j_per_token") else "",
                             "mac/android": ""})
            cmp_rows.append({"test": "android avg W / peak temp C", "quant": q.upper(),
                             "mac_tok_s": "",
                             "android_tok_s": f"{a['avg_power_w']:.2f} / {a['peak_temp_c']}"
                             if a.get("avg_power_w") else "",
                             "mac/android": ""})

    ccols = ["test", "quant", "mac_tok_s", "android_tok_s", "mac/android"]
    with open(RES / "mac_vs_android.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ccols)
        w.writeheader()
        w.writerows(cmp_rows)
    print("== Mac M2 vs Android")
    print_table(cmp_rows, ccols)


if __name__ == "__main__":
    main()
