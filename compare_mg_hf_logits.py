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
    parser.add_argument("--hf-cpu", action="store_true", help="Run HF model on CPU while keeping MG on accelerator.")
    parser.add_argument("--print-generate", action="store_true", help="Print short greedy generation from MG and HF.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens for generation print.")
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


def _infer_accel_device() -> torch.device:
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
    attention_mask = batch.get("attention_mask", None)

    mg_device = _infer_accel_device()
    hf_device = torch.device("cpu") if cli_args.hf_cpu else _resolve_device(cli_args.cpu)

    mg_input_ids = input_ids.to(mg_device)
    print(f"[INFO] mg_device={mg_device}, hf_device={hf_device}, prompt_tokens={mg_input_ids.shape[1]}")

    with torch.no_grad():
        infer = mg_model.infer_model
        mg_attention_mask, position_ids = infer.build_attention_mask_and_position_ids(mg_input_ids)
        model = get_args().model[0]
        try:
            # Some branches keep old ForwardStep(model, batch, seq) signature.
            forward_step = infer.ForwardStep(model, mg_input_ids.size(0), mg_input_ids.size(1))
        except TypeError:
            # Newer branches use ForwardStep(model, inference_context).
            inference_context = InferenceParams(
                max_batch_size=mg_input_ids.size(0),
                max_sequence_length=mg_input_ids.size(1),
            )
            forward_step = infer.ForwardStep(model, inference_context)
        mg_logits = forward_step(mg_input_ids, position_ids, mg_attention_mask)

    hf_model = AutoModelForCausalLM.from_pretrained(
        cli_args.hf_dir,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(cli_args.dtype),
    ).to(hf_device)
    hf_model.eval()

    hf_cfg = hf_model.config
    print(
        "[INFO] HF config:",
        f"model_type={getattr(hf_cfg, 'model_type', None)}",
        f"architectures={getattr(hf_cfg, 'architectures', None)}",
        f"vocab_size={getattr(hf_cfg, 'vocab_size', None)}",
        f"hidden_size={getattr(hf_cfg, 'hidden_size', None)}",
    )
    if hasattr(hf_model, 'lm_head') and hasattr(hf_model.lm_head, 'weight'):
        print(f"[INFO] HF lm_head.weight shape={tuple(hf_model.lm_head.weight.shape)}")
    else:
        print("[WARN] HF model has no lm_head.weight")

    bad_params = []
    for n, w in hf_model.named_parameters():
        if not torch.is_floating_point(w):
            continue
        wn = w.detach()
        if torch.isnan(wn).any() or torch.isinf(wn).any():
            bad_params.append(n)
            if len(bad_params) >= 10:
                break
    if bad_params:
        print(f"[WARN] HF has NaN/Inf parameters (showing up to 10): {bad_params}")
    else:
        print("[INFO] HF parameters finite check: no NaN/Inf found.")
    hf_input_ids = input_ids.to(hf_device)
    hf_attention_mask = attention_mask.to(hf_device) if attention_mask is not None else None

    with torch.no_grad():
        hf_logits = hf_model(input_ids=hf_input_ids, attention_mask=hf_attention_mask).logits

    print(f"[INFO] MG logits shape={tuple(mg_logits.shape)} dtype={mg_logits.dtype}")
    print(f"[INFO] HF logits shape={tuple(hf_logits.shape)} dtype={hf_logits.dtype}")
    if mg_logits.ndim == 3 and hf_logits.ndim == 3:
        print(f"[INFO] vocab dim MG={mg_logits.shape[-1]} HF={hf_logits.shape[-1]}")
    if mg_logits.shape != hf_logits.shape:
        print(f"[WARN] shape mismatch: MG={tuple(mg_logits.shape)} HF={tuple(hf_logits.shape)}")

    mg_last = mg_logits[:, -1, :].float().cpu()
    hf_last = hf_logits[:, -1, :].float().cpu()

    mg_nan = torch.isnan(mg_last).sum().item()
    hf_nan = torch.isnan(hf_last).sum().item()
    mg_inf = torch.isinf(mg_last).sum().item()
    hf_inf = torch.isinf(hf_last).sum().item()
    print(f"[INFO] nan/inf counts MG nan={mg_nan} inf={mg_inf}; HF nan={hf_nan} inf={hf_inf}")
    if mg_nan or hf_nan or mg_inf or hf_inf:
        print("[WARN] Found NaN/Inf in logits; numeric compare may be invalid.")
        if hf_nan > 0 and cli_args.dtype != "float32":
            print("[HINT] HF has NaN in non-fp32 mode; retry with --dtype float32.")

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

    if cli_args.print_generate:
        print("[INFO] Running short generation preview ...")
        # HF greedy generate
        with torch.no_grad():
            hf_gen = hf_model.generate(
                input_ids=hf_input_ids,
                attention_mask=hf_attention_mask,
                do_sample=False,
                max_new_tokens=cli_args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        hf_text = tokenizer.decode(hf_gen[0], skip_special_tokens=False)

        # MG greedy generate (through MindSpeed infer wrapper)
        mg_input_for_gen = mg_input_ids.detach().cpu()
        mg_out = mg_model.generate(
            input_ids=mg_input_for_gen,
            do_sample=False,
            max_new_tokens=cli_args.max_new_tokens,
            detokenize=False,
            include_input=True,
            tokenizer=tokenizer,
        )

        mg_tokens = None
        if isinstance(mg_out, list) and len(mg_out) > 0:
            first = mg_out[0]
            if torch.is_tensor(first):
                mg_tokens = first.tolist()
            elif isinstance(first, list):
                mg_tokens = first
        elif torch.is_tensor(mg_out):
            mg_tokens = mg_out[0].tolist() if mg_out.ndim > 1 else mg_out.tolist()

        if mg_tokens is not None:
            mg_text = tokenizer.decode(mg_tokens, skip_special_tokens=False)
            print(f"[GEN][MG ] {mg_text}")
        else:
            print(f"[GEN][MG ] raw={mg_out}")

        print(f"[GEN][HF ] {hf_text}")

    # Keep a clear signal for CI/manual checks.
    if cos < 0.999 or overlap < max(1, int(topk * 0.7)):
        print("[CHECK] WARNING: MG/HF logits diverge more than expected.")
    else:
        print("[CHECK] PASS: MG/HF logits are closely aligned.")


if __name__ == "__main__":
    main()
