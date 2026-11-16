#!/usr/bin/env python3
"""Energy / memory / thermal run on Android (Termux), matched to the Mac energy run.

Starts llama-server, measures an idle baseline, sends the same 512-token request the
Mac run used, and samples battery power, server RSS, temperatures and CPU clocks
every 0.5 s. Writes, per quant level:

  results/android/android_<q>_energy_response.json   server response (with timings)
  results/android/android_<q>_power_samples.csv      raw samples (time, W, RSS, temps, clocks)
  results/android/android_<q>_memory_samples.txt     RSS in KB, one per line (same as Mac file)
  results/android/android_<q>_energy_summary.json    J, J/token, avg/peak W, peak RSS, temps

Usage (phone UNPLUGGED — battery current only means anything while discharging):
  python3 android/03_energy.py --probe                 # check which sensors are readable
  python3 android/03_energy.py --quant Q8_0
  python3 android/03_energy.py --quant Q4_K_M
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
MODELS = ROOT / "models"
OUT = ROOT / "results" / "android"
PORT = 8080
# Exactly the prompt used for results/mac_m2/mac_m2_*_energy_response.json
PROMPT = ("Explain quantization, model compression, memory bandwidth, "
          "and inference optimization in detail.")

PS = "/sys/class/power_supply/battery"
SAMPLE_S = 0.5


# ---------------------------------------------------------------- sensors
def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def termux_battery():
    """Fallback via Termux:API app (slower, ~0.3-1 s per call)."""
    if not shutil.which("termux-battery-status"):
        return None
    try:
        out = subprocess.run(["termux-battery-status"], capture_output=True,
                             text=True, timeout=5).stdout
        return json.loads(out)
    except Exception:
        return None


class Battery:
    """Returns instantaneous battery power draw in watts (positive = discharging)."""

    def __init__(self):
        self.mode = None
        cur, volt = read_int(f"{PS}/current_now"), read_int(f"{PS}/voltage_now")
        if cur is not None and volt is not None and volt != 0:
            self.mode = "sysfs"
        elif termux_battery() and termux_battery().get("current") is not None:
            self.mode = "termux-api"

    @staticmethod
    def _amps(raw):       # most kernels report uA; some (older Samsung) report mA
        return abs(raw) / (1e3 if abs(raw) < 20000 else 1e6)

    @staticmethod
    def _volts(raw):      # usually uV (~3.8e6); some report mV
        return raw / (1e3 if raw < 20000 else 1e6)

    def watts(self):
        if self.mode == "sysfs":
            c, v = read_int(f"{PS}/current_now"), read_int(f"{PS}/voltage_now")
            if c is None or v is None:
                return None
            return self._amps(c) * self._volts(v)
        if self.mode == "termux-api":
            d = termux_battery() or {}
            c = d.get("current")
            v = d.get("voltage")              # mV on newer Termux:API; may be missing
            if c is None:
                return None
            volts = (v / 1000) if v else 3.85  # nominal Li-ion if voltage is missing
            return self._amps(c) * volts
        return None

    def status(self):
        try:
            with open(f"{PS}/status") as f:
                return f.read().strip()
        except OSError:
            d = termux_battery() or {}
            return d.get("status", "unknown")


def thermal_zones():
    zones = []
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t = read_int(f"{z}/temp")
        if t is None:
            continue
        try:
            name = open(f"{z}/type").read().strip()
        except OSError:
            name = os.path.basename(z)
        zones.append((name, z))
    return zones


def temps(zones):
    """Returns (max cpu-ish zone C, max any zone C, battery C)."""
    cpu, allz = [], []
    for name, z in zones:
        t = read_int(f"{z}/temp")
        if t is None:
            continue
        c = t / 1000 if abs(t) > 1000 else float(t)
        if c < -30 or c > 150:
            continue
        allz.append(c)
        if any(k in name.lower() for k in ("cpu", "soc", "tsens", "cluster", "big", "mtktscpu")):
            cpu.append(c)
    bt = read_int(f"{PS}/temp")
    return (max(cpu) if cpu else None, max(allz) if allz else None,
            bt / 10 if bt is not None else None)


def cpu_freqs_mhz():
    out = {}
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")):
        v = read_int(p)
        if v is not None:
            out[p.split("/")[5]] = v // 1000
    return out


def rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- server
def post(path, payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_healthy(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit("llama-server exited during load (out of memory?). See server log.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    sys.exit("llama-server did not become healthy in time")


# ---------------------------------------------------------------- main
def probe():
    b = Battery()
    zones = thermal_zones()
    print(f"battery power source : {b.mode or 'NOT AVAILABLE'}")
    print(f"battery status       : {b.status()}")
    w = b.watts()
    print(f"power now            : {w:.3f} W" if w is not None else "power now            : n/a")
    print(f"thermal zones        : {len(zones)} readable")
    for name, _ in zones[:12]:
        print(f"   - {name}")
    print(f"temps (cpu/max/batt) : {temps(zones)}")
    f = cpu_freqs_mhz()
    print(f"cpu clocks (MHz)     : {f if f else 'not readable'}")
    if b.mode is None:
        print("\nNo battery power reading. Install the Termux:API app (F-Droid/GitHub, same "
              "source as Termux) and `pkg install termux-api`, or use a USB power meter.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--quant", default="Q8_0", help="Q8_0, Q4_K_M or f16")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--idle-s", type=int, default=20, help="idle baseline seconds")
    ap.add_argument("--ctk", default="f16", help="KV cache type (f16, q8_0, q4_0)")
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    model = MODELS / f"Qwen2.5-3B-Instruct-{args.quant}.gguf"
    if not model.exists():
        sys.exit(f"missing {model} — run android/01_setup.sh")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.quant.lower().split("_")[0] + ("" if args.ctk == "f16" else f"_kv{args.ctk}")

    bat, zones = Battery(), thermal_zones()
    st = bat.status()
    if bat.mode is None:
        print("WARNING: no power source — energy will be blank (memory/thermals still recorded)")
    elif st.lower() not in ("discharging", "not charging"):
        print(f"WARNING: battery status is '{st}'. Unplug the phone for valid energy numbers.")

    cmd = [str(SERVER), "-m", str(model), "-c", "4096", "-t", str(args.threads),
           "--port", str(PORT), "--host", "127.0.0.1", "-ngl", "0"]
    if args.ctk != "f16":
        cmd += ["-fa", "on", "-ctk", args.ctk, "-ctv", args.ctk]
    log = open(OUT / f"android_{tag}_server.log", "w")
    print("starting:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    rows, phase = [], {"name": "load"}
    stop = threading.Event()

    def sampler():
        t0 = time.time()
        while not stop.is_set():
            c, mx, bt = temps(zones)
            fr = cpu_freqs_mhz()
            rows.append({
                "t_s": round(time.time() - t0, 2), "phase": phase["name"],
                "power_w": bat.watts(), "rss_kb": rss_kb(proc.pid),
                "cpu_temp_c": c, "max_temp_c": mx, "batt_temp_c": bt,
                "max_cpu_mhz": max(fr.values()) if fr else None,
            })
            stop.wait(SAMPLE_S)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        wait_healthy(proc)
        phase["name"] = "idle"
        print(f"model loaded; measuring {args.idle_s}s idle baseline (don't touch the phone)")
        time.sleep(args.idle_s)

        phase["name"] = "infer"
        print(f"generating {args.n_predict} tokens ...")
        t_start = time.time()
        resp = post("/completion", {"prompt": PROMPT, "n_predict": args.n_predict,
                                    "temperature": 0.0, "seed": 42,
                                    "cache_prompt": False, "stream": False})
        wall = time.time() - t_start
        phase["name"] = "done"
        time.sleep(1)
    finally:
        stop.set()
        th.join()
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    # ---- write raw files
    (OUT / f"android_{tag}_energy_response.json").write_text(json.dumps(resp))
    with open(OUT / f"android_{tag}_power_samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / f"android_{tag}_memory_samples.txt", "w") as f:
        f.writelines(f"{r['rss_kb']}\n" for r in rows if r["rss_kb"])

    # ---- summarize
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def pick(ph, key):
        return [r[key] for r in rows if r["phase"] == ph]

    idle_w = mean(pick("idle", "power_w"))
    infer_w = mean(pick("infer", "power_w"))
    tm = resp.get("timings", {})
    n_tok = tm.get("predicted_n") or resp.get("tokens_predicted")
    energy_j = infer_w * wall if infer_w is not None else None
    net_j = (infer_w - idle_w) * wall if None not in (infer_w, idle_w) else None
    temps_inf = [x for x in pick("infer", "cpu_temp_c") if x is not None] or \
                [x for x in pick("infer", "max_temp_c") if x is not None]
    clocks = [x for x in pick("infer", "max_cpu_mhz") if x is not None]
    rss = [r["rss_kb"] for r in rows if r["rss_kb"]]

    summary = {
        "device": "android", "phone": subprocess.run(["getprop", "ro.product.model"],
                                                       capture_output=True, text=True).stdout.strip()
        if shutil.which("getprop") else None,
        "model": "Qwen2.5-3B-Instruct", "quant": args.quant, "kv_cache": args.ctk,
        "threads": args.threads, "power_source": bat.mode,
        "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
        "predicted_n": n_tok, "predicted_per_second": tm.get("predicted_per_second"),
        "wall_s": round(wall, 2),
        "idle_power_w": idle_w, "avg_power_w": infer_w,
        "peak_power_w": max([x for x in pick("infer", "power_w") if x is not None], default=None),
        "energy_j": energy_j, "net_energy_j": net_j,
        "j_per_token": energy_j / n_tok if energy_j and n_tok else None,
        "net_j_per_token": net_j / n_tok if net_j and n_tok else None,
        "peak_rss_mib": round(max(rss) / 1024, 1) if rss else None,
        "start_temp_c": temps_inf[0] if temps_inf else None,
        "peak_temp_c": max(temps_inf) if temps_inf else None,
        "max_cpu_mhz_start": clocks[0] if clocks else None,
        "max_cpu_mhz_end": clocks[-1] if clocks else None,
        "n_samples": len(rows),
    }
    (OUT / f"android_{tag}_energy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote results/android/android_{tag}_*  — let the phone cool before the next run")


if __name__ == "__main__":
    main()
