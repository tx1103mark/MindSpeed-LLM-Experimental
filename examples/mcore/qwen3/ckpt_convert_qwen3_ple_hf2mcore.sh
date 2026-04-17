#!/bin/bash
# Qwen3 + PLE checkpoint convert: hf -> mcore
# NOTE: Requires PLE mappings for qwen3 in configs/checkpoint/model_cfg.json.

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf qwen3 \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --hidden-size-per-layer-input 256 \
    --vocab-size-per-layer-input 151936 \
    --load-dir ./model_from_hf/qwen3_ple_hf/ \
    --save-dir ./model_weights/qwen3_ple_mcore/
