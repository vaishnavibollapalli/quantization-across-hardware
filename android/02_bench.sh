#!/data/data/com.termux/files/usr/bin/bash
# llama-bench runs on Android, matched to the Mac protocol.
#
#   bash android/02_bench.sh            # core: pp256 + tg128, 4 threads, 3 reps (same as Mac)
#   SWEEP=1   bash android/02_bench.sh  # + context sweep pp256/1024/2048, tg128/512
#   THREADS_SWEEP=1 bash android/02_bench.sh  # + tg128 at 1/2/4/6/8 threads (big.LITTLE effects)
#   KVQ=1     bash android/02_bench.sh  # + KV-cache quant ablation (f16 vs q8_0 vs q4_0, flash-attn on)
#
# Before running: phone unplugged or plugged (bench doesn't measure power), screen on,
# airplane mode, close other apps, let it cool to room temperature.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$REPO_ROOT/llama.cpp/build/bin/llama-bench"
MODEL_DIR="$REPO_ROOT/models"
OUT="$REPO_ROOT/results/android"
THREADS="${THREADS:-4}"
REPS="${REPS:-3}"
COOLDOWN="${COOLDOWN:-90}"   # seconds between configs so throttling doesn't carry over
mkdir -p "$OUT"

termux-wake-lock 2>/dev/null || true

max_temp () {   # hottest thermal zone in deg C (blank if unreadable)
  cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 | awk '{ printf "%.1f", ($1>1000)?$1/1000:$1 }'
}
cool () {
  echo "   cooling ${COOLDOWN}s (hottest zone now: $(max_temp) C)"
  sleep "$COOLDOWN"
}
run () {   # run <tag> <model-file> <extra llama-bench args...>
  local tag="$1" model="$2"; shift 2
  local csv="$OUT/android_${tag}.csv"
  echo "==> $tag  ($model)  start temp $(max_temp) C"
  "$BENCH" -m "$MODEL_DIR/$model" -r "$REPS" -o csv "$@" > "$csv"
  echo "   wrote $csv  end temp $(max_temp) C"
  cool
}

for Q in Q8_0 Q4_K_M; do
  M="Qwen2.5-3B-Instruct-$Q.gguf"
  q=$(echo "$Q" | tr 'A-Z' 'a-z' | cut -d_ -f1)    # q8 / q4

  # 1) Core config — identical to results/mac_m2/mac_m2_${q}_full_bench.csv
  run "${q}_full_bench" "$M" -p 256 -n 128 -t "$THREADS"

  # 2) Context sweep (prefill knee points)
  if [ "${SWEEP:-0}" = "1" ]; then
    run "${q}_ctx_sweep" "$M" -p 256,1024,2048 -n 128,512 -t "$THREADS"
  fi

  # 3) Thread sweep — decode only
  if [ "${THREADS_SWEEP:-0}" = "1" ]; then
    run "${q}_thread_sweep" "$M" -p 0 -n 128 -t 1,2,4,6,8
  fi

  # 4) KV-cache quantization ablation at long context
  if [ "${KVQ:-0}" = "1" ]; then
    # matched K/V pairs (llama-bench would otherwise run the full 3x3 cross product)
    for KV in f16 q8_0 q4_0; do
      run "${q}_kvq_${KV}" "$M" -p 2048 -n 128 -t "$THREADS" -fa 1 -ctk "$KV" -ctv "$KV"
    done
  fi
done

if [ "${WITH_F16:-0}" = "1" ] && [ -s "$MODEL_DIR/Qwen2.5-3B-Instruct-f16.gguf" ]; then
  run "f16_full_bench" "Qwen2.5-3B-Instruct-f16.gguf" -p 256 -n 128 -t "$THREADS" \
    || echo "F16 failed (likely out of memory) — note it as an OOM result"
fi

termux-wake-unlock 2>/dev/null || true
echo "Done. Next: python3 android/03_energy.py --quant Q8_0  (phone UNPLUGGED)"
