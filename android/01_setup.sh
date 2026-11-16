#!/data/data/com.termux/files/usr/bin/bash
# Android (Termux) setup for "Comparing Quantization Across Hardware".
# Builds llama.cpp from source (CPU backend, ARM NEON/dotprod auto-detected) and
# downloads the same Qwen2.5-3B-Instruct GGUFs used on the Mac.
#
# Usage (inside Termux, from the repo root):
#   bash android/01_setup.sh
#   WITH_F16=1 bash android/01_setup.sh     # also fetch the FP16 reference (6.2 GB, may not fit)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"
MODEL_DIR="$REPO_ROOT/models"
# Same llama.cpp commit as the Mac runs (results/mac_m2/*_bench.csv -> build_commit)
LLAMA_COMMIT="${LLAMA_COMMIT:-620d2225c}"
HF="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main"

echo "==> Installing Termux packages"
pkg update -y
pkg install -y git cmake clang make python curl termux-api

echo "==> Device info"
echo "model:   $(getprop ro.product.model)"
echo "soc:     $(getprop ro.soc.model 2>/dev/null || true) $(getprop ro.board.platform)"
echo "android: $(getprop ro.build.version.release)"
echo "cores:   $(nproc)"
grep MemTotal /proc/meminfo

echo "==> Building llama.cpp @ $LLAMA_COMMIT"
if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
git fetch origin --quiet || true
if git checkout --quiet "$LLAMA_COMMIT" 2>/dev/null; then
  echo "checked out $LLAMA_COMMIT (matches Mac)"
else
  echo "WARNING: commit $LLAMA_COMMIT not found, building master. Record this in the paper."
  git checkout --quiet master && git pull --quiet
fi
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build --config Release -j "$(nproc)" \
  --target llama-bench llama-server llama-cli llama-perplexity
ls -la build/bin/llama-bench build/bin/llama-server

echo "==> Downloading models to $MODEL_DIR"
mkdir -p "$MODEL_DIR"
fetch () {
  local f="$1"
  if [ -s "$MODEL_DIR/$f" ]; then echo "have $f"; return; fi
  curl -L --fail --retry 3 -C - -o "$MODEL_DIR/$f.part" "$HF/$f"
  mv "$MODEL_DIR/$f.part" "$MODEL_DIR/$f"
}
fetch Qwen2.5-3B-Instruct-Q8_0.gguf
fetch Qwen2.5-3B-Instruct-Q4_K_M.gguf
if [ "${WITH_F16:-0}" = "1" ]; then fetch Qwen2.5-3B-Instruct-f16.gguf; fi
ls -lh "$MODEL_DIR"

echo "==> Checking sensor access (energy/thermal)"
python3 "$REPO_ROOT/android/03_energy.py" --probe

echo "Done. Next: bash android/02_bench.sh"
