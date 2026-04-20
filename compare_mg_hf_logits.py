#!/usr/bin/env python
# coding=utf-8
"""Compare Megatron (MG) and HF logits for the same prompt.

Usage example (append your normal Megatron runtime args after `--`):
python compare_mg_hf_logits.py ^
  --mg-load-dir ./Qwen3/qwen3-ple-test/ ^
  --hf-dir ./Qwen3/qwen3-ple-test-hf/mg2hf ^
  --tokenizer-dir ./Qwen3/qwen3-ple-test-hf/mg2hf ^
  --prompt "你好，介绍一下你自己。" ^
  --topk 10 ^
  -- ^
  --use-mcore-models ^
  --model-type GPT ^
  --num-layers 28 ^
  --hidden-size 1024 ^
  --ffn-hidden-size 3072 ^
  --num-attention-heads 16 ^
  --seq-length 4096 ^
  --max-position-embeddings 40960 ^
  --position-embedding-type rope ^
  --tokenizer-type HuggingFaceTokenizer ^
  --tokenizer-model ./Qwen3/qwen3-ple-test-hf/mg2hf ^
  --load ./Qwen3/qwen3-ple-test/ ^
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec ^
  --hidden-size-per-layer-input 256 ^
  --vocab-size-per-layer-input 151936
"""

import argparse
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindspeed_llm import megatron_adaptor  # noqa: F401
from inference import model_provider
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.inference.module import MegatronModuleForCausalLM
from megatron.core import InferenceParams


def _split_args() -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(description="Compare MG and HF logits.")
    parser.add_argument("--mg-load-dir", required=True, help="Megatron checkpoint directory.")
    parser.add_argument("--hf-dir", required=True, help="HF model directory.")
    parser.add_argument("--tokenizer-dir", default=None, help="HF tokenizer directory.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k for overlap checks.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--cpu", action="store_true", help="Force run HF side on CPU.")
    parser.add_argument(
        "megatron_args",
        nargs=argparse.REMAINDER,
        help="Megatron args after '--'.",
    )
    args = parser.parse_args()
    extra = args.megatron_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def _resolve_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _torch_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    cli_args, megatron_extra = _split_args()
    if not megatron_extra:
        raise ValueError("Please provide Megatron runtime args after '--'.")

    # Let Megatron parse only its own CLI options.
    sys.argv = [sys.argv[0]] + megatron_extra
    initialize_megatron(args_defaults={"no_load_rng": True, "no_load_optim": True})
    mg_args = get_args()

    mg_model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=cli_args.mg_load_dir,
    )
    mg_model.eval()
    if not hasattr(mg_model, "infer_model"):
        raise RuntimeError("Loaded model does not expose infer_model; expected GPTModelInfer.")

    tokenizer_dir = cli_args.tokenizer_dir or cli_args.hf_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    batch = tokenizer(cli_args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = batch["input_ids"]

    device = _resolve_device(cli_args.cpu)
    input_ids = input_ids.to(device)
    print(f"[INFO] device={device}, prompt_tokens={input_ids.shape[1]}")

    with torch.no_grad():
        infer = mg_model.infer_model
        attention_mask, position_ids = infer.build_attention_mask_and_position_ids(input_ids)
        model = get_args().model[0]
        try:
            # Some branches keep old ForwardStep(model, batch, seq) signature.
            forward_step = infer.ForwardStep(model, input_ids.size(0), input_ids.size(1))
        except TypeError:
            # Newer branches use ForwardStep(model, inference_context).
            inference_context = InferenceParams(
                max_batch_size=input_ids.size(0),
                max_sequence_length=input_ids.size(1),
            )
            forward_step = infer.ForwardStep(model, inference_context)
        mg_logits = forward_step(input_ids, position_ids, attention_mask)

    hf_model = AutoModelForCausalLM.from_pretrained(
        cli_args.hf_dir,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(cli_args.dtype),
    ).to(device)
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits

    if mg_logits.shape != hf_logits.shape:
        print(f"[WARN] shape mismatch: MG={tuple(mg_logits.shape)} HF={tuple(hf_logits.shape)}")

    mg_last = mg_logits[:, -1, :].float().cpu()
    hf_last = hf_logits[:, -1, :].float().cpu()

    diff = (mg_last - hf_last).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    mse = F.mse_loss(mg_last, hf_last).item()
    cos = F.cosine_similarity(mg_last, hf_last, dim=-1).mean().item()

    topk = min(cli_args.topk, mg_last.shape[-1], hf_last.shape[-1])
    mg_topk_ids = torch.topk(mg_last, k=topk, dim=-1).indices[0].tolist()
    hf_topk_ids = torch.topk(hf_last, k=topk, dim=-1).indices[0].tolist()
    overlap = len(set(mg_topk_ids) & set(hf_topk_ids))

    mg_next = int(torch.argmax(mg_last, dim=-1)[0].item())
    hf_next = int(torch.argmax(hf_last, dim=-1)[0].item())

    print(f"[RESULT] max_abs={max_abs:.6e}")
    print(f"[RESULT] mean_abs={mean_abs:.6e}")
    print(f"[RESULT] mse={mse:.6e}")
    print(f"[RESULT] cosine={cos:.6f}")
    print(f"[RESULT] greedy_next_token MG={mg_next} HF={hf_next} same={mg_next == hf_next}")
    print(f"[RESULT] top{topk}_overlap={overlap}/{topk}")
    print(f"[DETAIL] MG top{topk} ids: {mg_topk_ids}")
    print(f"[DETAIL] HF top{topk} ids: {hf_topk_ids}")

    # Keep a clear signal for CI/manual checks.
    if cos < 0.999 or overlap < max(1, int(topk * 0.7)):
        print("[CHECK] WARNING: MG/HF logits diverge more than expected.")
    else:
        print("[CHECK] PASS: MG/HF logits are closely aligned.")


if __name__ == "__main__":
    main()
