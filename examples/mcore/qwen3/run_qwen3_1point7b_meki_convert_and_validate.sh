#!/bin/bash
set -euo pipefail

# One-click pipeline:
#   1) Convert MG(MeKi) -> HF
#   2) Validate logits alignment
#   3) Validate layerwise alignment

# -----------------------------
# Environment
# -----------------------------
# Please update this for your environment.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# -----------------------------
# Paths (edit these)
# -----------------------------
MG_CKPT_DIR="/path/to/mcore_mg_ckpt"
HF_WORK_DIR="/path/to/hf_work_dir"
HF_OUTPUT_DIR="${HF_WORK_DIR}/mg2hf"
TOKENIZER_DIR="${HF_OUTPUT_DIR}"

# -----------------------------
# Runtime options
# -----------------------------
PROMPT="Hello, please introduce yourself."
RUN_CONVERT=1
RUN_LOGITS_CHECK=1
RUN_LAYERWISE_CHECK=1

# -----------------------------
# Model / feature config (must match training)
# -----------------------------
NUM_LAYERS=28
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=6144
NUM_ATTENTION_HEADS=16
NUM_QUERY_GROUPS=8
KV_CHANNELS=128
SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=40960
PADDED_VOCAB_SIZE=151936

MEKI_DIM=256
MEKI_ALPHA=1.0
MEKI_BETA=1.0

# Optional PLE fields (set >0 only if enabled in training).
HIDDEN_SIZE_PER_LAYER_INPUT=0
VOCAB_SIZE_PER_LAYER_INPUT=0

# Parallel settings
TARGET_TP=1
TARGET_PP=1
TARGET_EP=1

# Check ports
LOGITS_MASTER_PORT=6001
LAYERWISE_MASTER_PORT=6002

COMMON_MG_ARGS="
  --use-mcore-models \
  --model-type GPT \
  --tensor-model-parallel-size ${TARGET_TP} \
  --pipeline-model-parallel-size ${TARGET_PP} \
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTENTION_HEADS} \
  --qk-layernorm \
  --group-query-attention \
  --num-query-groups ${NUM_QUERY_GROUPS} \
  --kv-channels ${KV_CHANNELS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
  --position-embedding-type rope \
  --use-rotary-position-embeddings \
  --rotary-base 1000000 \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --disable-bias-linear \
  --swiglu \
  --attention-dropout 0 \
  --hidden-dropout 0 \
  --micro-batch-size 1 \
  --make-vocab-size-divisible-by 1 \
  --padded-vocab-size ${PADDED_VOCAB_SIZE} \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path ${TOKENIZER_DIR} \
  --bf16 \
  --transformer-impl local \
  --ckpt-format torch \
  --meki-dim ${MEKI_DIM} \
  --meki-alpha ${MEKI_ALPHA} \
  --meki-beta ${MEKI_BETA}
"

if [ "${HIDDEN_SIZE_PER_LAYER_INPUT}" -gt 0 ]; then
  COMMON_MG_ARGS="${COMMON_MG_ARGS} --hidden-size-per-layer-input ${HIDDEN_SIZE_PER_LAYER_INPUT}"
fi
if [ "${VOCAB_SIZE_PER_LAYER_INPUT}" -gt 0 ]; then
  COMMON_MG_ARGS="${COMMON_MG_ARGS} --vocab-size-per-layer-input ${VOCAB_SIZE_PER_LAYER_INPUT}"
fi

EXTRA_CONVERT_ARGS=""
if [ "${HIDDEN_SIZE_PER_LAYER_INPUT}" -gt 0 ]; then
  EXTRA_CONVERT_ARGS="${EXTRA_CONVERT_ARGS} --hidden-size-per-layer-input ${HIDDEN_SIZE_PER_LAYER_INPUT}"
fi
if [ "${VOCAB_SIZE_PER_LAYER_INPUT}" -gt 0 ]; then
  EXTRA_CONVERT_ARGS="${EXTRA_CONVERT_ARGS} --vocab-size-per-layer-input ${VOCAB_SIZE_PER_LAYER_INPUT}"
fi

if [ "${RUN_CONVERT}" -eq 1 ]; then
  echo "[STEP 1/3] Converting MG(MeKi) -> HF ..."
  mkdir -p "${HF_WORK_DIR}"
  python convert_ckpt.py \
      --use-mcore-models \
      --model-type GPT \
      --model-type-hf qwen3 \
      --load-model-type mg \
      --save-model-type hf \
      --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
      --transformer-impl local \
      --target-tensor-parallel-size "${TARGET_TP}" \
      --target-pipeline-parallel-size "${TARGET_PP}" \
      --target-expert-parallel-size "${TARGET_EP}" \
      --load-hf-from-config \
      --meki-dim "${MEKI_DIM}" \
      --meki-alpha "${MEKI_ALPHA}" \
      --meki-beta "${MEKI_BETA}" \
      ${EXTRA_CONVERT_ARGS} \
      --load-dir "${MG_CKPT_DIR}" \
      --save-dir "${HF_WORK_DIR}"

  python - <<PY
import json
from pathlib import Path

cfg_path = Path("${HF_OUTPUT_DIR}") / "config.json"
if not cfg_path.exists():
    raise FileNotFoundError(f"Missing config.json: {cfg_path}")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
cfg["meki_dim"] = int(${MEKI_DIM})
cfg["meki_alpha"] = float(${MEKI_ALPHA})
cfg["meki_beta"] = float(${MEKI_BETA})
if int(${HIDDEN_SIZE_PER_LAYER_INPUT}) > 0:
    cfg["hidden_size_per_layer_input"] = int(${HIDDEN_SIZE_PER_LAYER_INPUT})
if int(${VOCAB_SIZE_PER_LAYER_INPUT}) > 0:
    cfg["vocab_size_per_layer_input"] = int(${VOCAB_SIZE_PER_LAYER_INPUT})
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Updated config:", cfg_path)
PY
fi

if [ "${RUN_LOGITS_CHECK}" -eq 1 ]; then
  echo "[STEP 2/3] Running logits-level validation ..."
  torchrun \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "${LOGITS_MASTER_PORT}" \
    compare_mg_hf_logits.py \
    --mg-load-dir "${MG_CKPT_DIR}" \
    --hf-dir "${HF_OUTPUT_DIR}" \
    --tokenizer-dir "${TOKENIZER_DIR}" \
    --prompt "${PROMPT}" \
    --topk 10 \
    -- \
    ${COMMON_MG_ARGS}
fi

if [ "${RUN_LAYERWISE_CHECK}" -eq 1 ]; then
  echo "[STEP 3/3] Running layerwise validation ..."
  torchrun \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port "${LAYERWISE_MASTER_PORT}" \
    compare_mg_hf_layerwise.py \
    --mg-load-dir "${MG_CKPT_DIR}" \
    --hf-dir "${HF_OUTPUT_DIR}" \
    --tokenizer-dir "${TOKENIZER_DIR}" \
    --prompt "${PROMPT}" \
    --dtype float32 \
    --hf-cpu \
    -- \
    ${COMMON_MG_ARGS}
fi

echo "[DONE] MeKi conversion + validation pipeline finished."
