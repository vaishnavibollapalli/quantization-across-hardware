# Android runs (Device B)

These scripts run the same Qwen2.5-3B-Instruct Q8_0 / Q4_K_M experiments on an Android phone that we
ran on the M2 MacBook Air. They use the same llama.cpp commit, the same `llama-bench` settings
(pp256 / tg128, 4 threads, 3 reps) and the same 512-token energy prompt. The only difference is the
backend: CPU (ARM NEON) on the phone instead of Metal.

## 0. Install Termux (one time)

1. Install **Termux** and **Termux:API** from **F-Droid** or the Termux GitHub releases. Get both from
   the same source. Don't use the Play Store build, which is outdated and can't run Termux:API.
2. Open Termux once. Then go to Android Settings → Apps → Termux → Battery and set it to
   **Unrestricted**, so Android doesn't kill long runs.
3. Make sure there is about **6 GB free storage**. The two models are about 5 GB, and the llama.cpp
   build is about 0.5 GB.

## 1. Setup (~15–25 min, mostly the build and downloads)

```bash
pkg install -y git
git clone https://github.com/vaishnavibollapalli/quantization-across-hardware
cd quantization-across-hardware
bash android/01_setup.sh
```

This prints the phone model, SoC, RAM and core count (put these in the hardware table). At the end it
checks which sensors Termux can read.

## 2. Throughput benchmark: matches `results/mac_m2/*_full_bench.csv`

Before you start: turn on airplane mode, close other apps, use fixed brightness, and let the phone
cool to room temperature. Keep Termux in the foreground.

```bash
bash android/02_bench.sh                    # core run, ~10 min
SWEEP=1 KVQ=1 THREADS_SWEEP=1 bash android/02_bench.sh   # + context sweep, KV-cache quant, threads (~1–2 h)
```

The script waits 90 s between configurations (`COOLDOWN=...` changes this) so throttling from one
run doesn't carry over into the next.

## 3. Energy / memory / thermals: matches `mac_m2_*_energy_response.json` + memory samples

**Unplug the phone.** Battery current is only meaningful while it's discharging. Keep it above ~30 %.

```bash
python3 android/03_energy.py --quant Q8_0
# let it cool ~2 min
python3 android/03_energy.py --quant Q4_K_M
# optional KV-cache-quant ablation:
python3 android/03_energy.py --quant Q4_K_M --ctk q8_0
```

The script records a 20 s idle baseline and then the 512-token generation. It samples power (W),
llama-server RSS, CPU/battery temperatures and the highest CPU clock every 0.5 s. It reports gross
and **net-of-idle** J/token. A falling CPU clock during the run is your throttling evidence.

## 4. Summarize and compare with the Mac

```bash
python3 android/04_summarize.py
```

This writes `results/android/android_pilot_comparison.csv` (same columns as the Mac file) and
`results/mac_vs_android.csv`.

## 5. Get results off the phone

```bash
# option A: push from Termux (needs a GitHub personal access token as the password)
git add results/android results/mac_vs_android.csv && git commit -m "Android results" && git push
# option B: copy to Downloads, then move it with USB / Drive
termux-setup-storage && cp -r results/android ~/storage/downloads/
```

## Notes for the paper

- **FP16** (6.2 GB) usually gets killed on an 8 GB phone, because Android leaves apps only about
  4–5 GB. Try `WITH_F16=1` in both scripts. An out-of-memory kill is a valid result: "FP16 does not
  fit".
- **Battery-based power** includes the screen and the rest of the system, so report net-of-idle
  energy. Battery fuel gauges also update slowly (~1 s), so use the 512-token run rather than short
  prompts.
- **RSS is not comparable across OSes.** On the Mac the Metal-resident weights don't appear in RSS.
  On Android the mmap'd weights do, once they've been touched. Compare model file size + KV cache
  instead, and only use RSS within one device.
- **Threads:** 4 threads matches the Mac. Phones have big.LITTLE cores, though, so report the
  thread-sweep optimum too.
- **Quality:** a GGUF gives essentially the same outputs on every device, so the 100-item quality set
  stays Mac-only (see `prompt_set/README.md`).
