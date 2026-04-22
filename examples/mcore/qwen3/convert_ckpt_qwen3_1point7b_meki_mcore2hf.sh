#!/bin/bash
set -euo pipefail

# Please update this path for your environment.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

# -----------------------------
# Required paths
# -----------------------------
CKPT_LOAD_DIR="/path/to/mcore_mg_ckpt"            # e.g. /data/ckpt/qwen3_1p7b_meki
HF_WORK_DIR="/path/to/hf_work_dir"                # must contain config.json + tokenizer files + modeling_qwen3.py + configuration_qwen3.py

# -----------------------------
# Parallel settings (match source ckpt topology)
# -----------------------------
TARGET_TP=1
TARGET_PP=1
TARGET_EP=1

# -----------------------------
# MeKi config (must match trained checkpoint)
# -----------------------------
MEKI_DIM=256
MEKI_ALPHA=1.0
MEKI_BETA=1.0

# Optional: if you enabled PLE in training, set these as well.
HIDDEN_SIZE_PER_LAYER_INPUT=0
VOCAB_SIZE_PER_LAYER_INPUT=0

mkdir -p "${HF_WORK_DIR}"

EXTRA_PLE_ARGS=""
if [ "${HIDDEN_SIZE_PER_LAYER_INPUT}" -gt 0 ]; then
  EXTRA_PLE_ARGS="${EXTRA_PLE_ARGS} --hidden-size-per-layer-input ${HIDDEN_SIZE_PER_LAYER_INPUT}"
fi
if [ "${VOCAB_SIZE_PER_LAYER_INPUT}" -gt 0 ]; then
  EXTRA_PLE_ARGS="${EXTRA_PLE_ARGS} --vocab-size-per-layer-input ${VOCAB_SIZE_PER_LAYER_INPUT}"
fi

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf qwen3 \
    --load-model-type mg \
    --save-model-type hf \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --target-tensor-parallel-size "${TARGET_TP}" \
    --target-pipeline-parallel-size "${TARGET_PP}" \
    --target-expert-parallel-size "${TARGET_EP}" \
    --load-hf-from-config \
    --transformer-impl local \
    --meki-dim "${MEKI_DIM}" \
    --meki-alpha "${MEKI_ALPHA}" \
    --meki-beta "${MEKI_BETA}" \
    ${EXTRA_PLE_ARGS} \
    --load-dir "${CKPT_LOAD_DIR}" \
    --save-dir "${HF_WORK_DIR}"

# Ensure config.json explicitly carries MeKi fields for HF loading.
python - <<PY
import json
from pathlib import Path

cfg_path = Path("${HF_WORK_DIR}") / "mg2hf" / "config.json"
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
print("Updated:", cfg_path)
print("meki_dim =", cfg["meki_dim"], "meki_alpha =", cfg["meki_alpha"], "meki_beta =", cfg["meki_beta"])
PY

echo "Done: MG(MeKi) -> HF conversion finished."
echo "HF output dir: ${HF_WORK_DIR}/mg2hf"
