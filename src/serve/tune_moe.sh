#!/bin/bash
# Tune vLLM's fused-MoE Triton kernel tile configs for GB10 (sm_121) on our expert shape
# (E=128, N=768, top-8) with vLLM's own tuner for the vLLM version we serve, then re-run the
# serving benchmark with the tuned configs loaded (VLLM_TUNED_CONFIG_FOLDER).
# The vLLM server log reports "Using default MoE config ... NVIDIA_GB10" without this.
set -uo pipefail
OUT=/runs/serve/moe-configs; mkdir -p $OUT; cd $OUT
V=$(python3 -c "import vllm; print(vllm.__version__)"); echo "vllm=$V"
for f in benchmark_moe.py utils.py; do
  curl -sfL -o /tmp/$f "https://raw.githubusercontent.com/vllm-project/vllm/v${V}/benchmarks/kernels/$f" || echo "fetch $f failed"
done
# vLLM's tuner imports ray, which the vLLM image does not ship; install it per-user (no root needed).
export PYTHONUSERBASE=/tmp/pyuser PATH=/tmp/pyuser/bin:$PATH
python3 -c "import ray" 2>/dev/null || pip install --quiet --user ray 2>&1 | tail -3
python3 -c "import ray; print('RESULT {\"kind\": \"moe_tune_ray\", \"ok\": true}')" || echo 'RESULT {"kind": "moe_tune_ray", "ok": false}'

M=/models/Qwen/Qwen3-Coder-30B-A3B-Instruct
for DT in fp8_w8a8 auto; do
  t=$(date +%s)
  (cd /tmp && python3 benchmark_moe.py --model $M --tp-size 1 --dtype $DT --tune \
      --batch-size 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 --trust-remote-code) 2>&1 | tail -30
  echo "RESULT {\"kind\": \"moe_tune\", \"dtype\": \"$DT\", \"seconds\": $(( $(date +%s) - t ))}"
done
cp /tmp/E=*.json $OUT/ 2>/dev/null; ls -la $OUT
echo "RESULT {\"kind\": \"moe_tune_files\", \"files\": \"$(ls $OUT | tr '\n' ' ')\"}"
export VLLM_TUNED_CONFIG_FOLDER=$OUT
python3 /opt/serve_bench.py --configs /opt/serve-ft-tuned.json --out /runs/serve/ft-tuned --concurrency 1,16,64
