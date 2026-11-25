#!/usr/bin/env bash
# Start the vLLM servers on a 2xT4 Kaggle box, plus a GPU power sampler.
#   serve.sh <vllm-python> <llm-model> <embed-model> <logdir>
# LLM: tensor-parallel over both GPUs on :8000. Embeddings: small model on GPU 1 on :8001.
set -u
PY=$1; LLM=$2; EMB=$3; LOG=$4
mkdir -p "$LOG"

nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv,nounits -l 1 > "$LOG/power.csv" &
echo $! > "$LOG/power.pid"

wait_ready() {  # port, seconds, pid
  for _ in $(seq 1 "$2"); do
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0
    kill -0 "$3" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

"$PY" -m vllm.entrypoints.openai.api_server --model "$LLM" --served-model-name "$LLM" --port 8000 \
  --tensor-parallel-size 2 --dtype half --max-model-len 16384 --gpu-memory-utilization 0.80 \
  --max-num-seqs 64 --disable-log-requests > "$LOG/vllm_llm.log" 2>&1 &
LLM_PID=$!
wait_ready 8000 1500 $LLM_PID || { echo "LLM server failed"; tail -60 "$LOG/vllm_llm.log"; exit 1; }
echo "LLM ready"

start_embed() {
  CUDA_VISIBLE_DEVICES=1 "$PY" -m vllm.entrypoints.openai.api_server --model "$EMB" --served-model-name "$EMB" \
    --port 8001 --dtype half --gpu-memory-utilization 0.12 --max-model-len 8192 "$@" \
    > "$LOG/vllm_embed.log" 2>&1 &
  EMB_PID=$!
}
if [ "$EMB" = "none" ]; then echo "no embed server (judge mode)"; exit 0; fi
start_embed --runner pooling
if ! wait_ready 8001 600 $EMB_PID; then
  echo "embed server failed with --runner pooling, retrying with --task embed"; tail -20 "$LOG/vllm_embed.log"
  start_embed --task embed
  wait_ready 8001 600 $EMB_PID || { echo "embed server failed"; tail -60 "$LOG/vllm_embed.log"; exit 1; }
fi
echo "EMBED ready"
