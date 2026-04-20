#!/bin/bash

# Launch MG-vs-HF logits compare with torchrun (single process by default).
# Fill these paths before running.
MG_CKPT="./Qwen3/qwen3-ple-test/"
HF_DIR="./Qwen3/qwen3-ple-test-hf/mg2hf"
TOKENIZER_DIR="${HF_DIR}"
PROMPT="你好，介绍一下你自己。"

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

torchrun ${DISTRIBUTED_ARGS} compare_mg_hf_logits.py \
  --mg-load-dir ${MG_CKPT} \
  --hf-dir ${HF_DIR} \
  --tokenizer-dir ${TOKENIZER_DIR} \
  --prompt "${PROMPT}" \
  --topk 10 \
  -- \
  --use-mcore-models \
  --model-type GPT \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
  --num-layers 28 \
  --hidden-size 1024 \
  --ffn-hidden-size 3072 \
  --num-attention-heads 16 \
  --group-query-attention \
  --num-query-groups 8 \
  --kv-channels 128 \
  --seq-length 4096 \
  --max-position-embeddings 40960 \
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
  --padded-vocab-size 151936 \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path ${TOKENIZER_DIR} \
  --bf16 \
  --transformer-impl local \
  --ckpt-format torch \
  --hidden-size-per-layer-input 256 \
  --vocab-size-per-layer-input 151936
