#!/usr/bin/env python
# coding=utf-8
"""Step-wise greedy decode comparison between Megatron (MG) and HF.

This script runs both sides with the same autoregressive loop:
  1) forward on current tokens
  2) pick argmax token
  3) append and continue

It reports first divergence step and local top-k/gap diagnostics.
Run with Megatron args after `--`.
"""

import argparse
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindspeed_llm import megatron_adaptor  # noqa: F401
from inference import model_provider
from megatron.core import InferenceParams
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.inference.module import MegatronModuleForCausalLM


def _split_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Step-wise greedy decode MG vs HF.")
    parser.add_argument("--mg-load-dir", required=True, help="Megatron checkpoint directory.")
    parser.add_argument("--hf-dir", required=True, help="HF model directory.")
    parser.add_argument("--tokenizer-dir", default=None, help="Tokenizer directory (defaults to hf-dir).")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Decode steps.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k shown at first divergence step.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--cpu", action="store_true", help="Force HF to run on CPU.")
    parser.add_argument("--hf-cpu", action="store_true", help="Run HF on CPU while keeping MG on accelerator.")
    parser.add_argument("--stop-when-both-eos", action=argparse.BooleanOptionalAction, default=True)
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


def _infer_accel_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_hf_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return _infer_accel_device()


def _torch_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _run_mg_forward(mg_wrap, tokens: torch.Tensor) -> torch.Tensor:
    infer = mg_wrap.infer_model
    model = get_args().model[0]
    attention_mask, position_ids = infer.build_attention_mask_and_position_ids(tokens)
    try:
        forward_step = infer.ForwardStep(model, tokens.size(0), tokens.size(1))
    except TypeError:
        inference_context = InferenceParams(
            max_batch_size=tokens.size(0),
            max_sequence_length=tokens.size(1),
        )
        forward_step = infer.ForwardStep(model, inference_context)
    return forward_step(tokens, position_ids, attention_mask)


def _top2_gap(logits_1d: torch.Tensor) -> float:
    if logits_1d.numel() < 2:
        return 0.0
    top2 = torch.topk(logits_1d, k=2, dim=-1).values
    return float((top2[0] - top2[1]).item())


def _decode(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)


def _safe_topk_ids(logits_1d: torch.Tensor, topk: int) -> List[int]:
    k = min(topk, logits_1d.numel())
    return torch.topk(logits_1d, k=k, dim=-1).indices.tolist()


def _eos_id(tokenizer, hf_model) -> Optional[int]:
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    cfg_eos = getattr(hf_model.config, "eos_token_id", None)
    return int(cfg_eos) if cfg_eos is not None else None


def main():
    cli_args, megatron_extra = _split_args()
    if not megatron_extra:
        raise ValueError("Please provide Megatron runtime args after '--'.")

    sys.argv = [sys.argv[0]] + megatron_extra
    initialize_megatron(args_defaults={"no_load_rng": True, "no_load_optim": True})

    mg_wrap = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=cli_args.mg_load_dir,
    )
    mg_wrap.eval()
    if not hasattr(mg_wrap, "infer_model"):
        raise RuntimeError("Loaded model does not expose infer_model; expected GPTModelInfer.")

    tokenizer_dir = cli_args.tokenizer_dir or cli_args.hf_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    batch = tokenizer(cli_args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = batch["input_ids"]

    mg_device = _infer_accel_device()
    hf_device = torch.device("cpu") if cli_args.hf_cpu else _resolve_hf_device(cli_args.cpu)

    hf_model = AutoModelForCausalLM.from_pretrained(
        cli_args.hf_dir,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(cli_args.dtype),
    ).to(hf_device)
    hf_model.eval()

    eos_id = _eos_id(tokenizer, hf_model)
    print(
        "[INFO]",
        f"mg_device={mg_device}",
        f"hf_device={hf_device}",
        f"prompt_tokens={input_ids.shape[1]}",
        f"eos_id={eos_id}",
        f"steps={cli_args.max_new_tokens}",
    )
    print(
        "[INFO] decode_mode=manual_greedy",
        "do_sample=False",
        "top_k=0",
        "top_p=0.0",
        "temperature=1.0",
    )

    mg_ids = input_ids.to(mg_device)
    hf_ids = input_ids.to(hf_device)

    first_diverge_step = -1
    first_diverge_payload = None

    with torch.no_grad():
        for step in range(1, cli_args.max_new_tokens + 1):
            mg_logits = _run_mg_forward(mg_wrap, mg_ids)
            hf_logits = hf_model(input_ids=hf_ids).logits

            mg_last = mg_logits[:, -1, :].float().cpu()
            hf_last = hf_logits[:, -1, :].float().cpu()

            mg_next = int(torch.argmax(mg_last, dim=-1)[0].item())
            hf_next = int(torch.argmax(hf_last, dim=-1)[0].item())
            same = mg_next == hf_next

            mg_gap = _top2_gap(mg_last[0])
            hf_gap = _top2_gap(hf_last[0])
            cos = float(F.cosine_similarity(mg_last, hf_last, dim=-1).mean().item())

            print(
                f"[STEP {step:03d}]",
                f"same={same}",
                f"mg_next={mg_next}",
                f"hf_next={hf_next}",
                f"cos={cos:.6f}",
                f"mg_gap={mg_gap:.6e}",
                f"hf_gap={hf_gap:.6e}",
            )

            if (not same) and first_diverge_step < 0:
                first_diverge_step = step
                first_diverge_payload = {
                    "mg_topk": _safe_topk_ids(mg_last[0], cli_args.topk),
                    "hf_topk": _safe_topk_ids(hf_last[0], cli_args.topk),
                    "mg_gap": mg_gap,
                    "hf_gap": hf_gap,
                    "cos": cos,
                }

            mg_next_t = torch.tensor([[mg_next]], dtype=mg_ids.dtype, device=mg_ids.device)
            hf_next_t = torch.tensor([[hf_next]], dtype=hf_ids.dtype, device=hf_ids.device)
            mg_ids = torch.cat([mg_ids, mg_next_t], dim=1)
            hf_ids = torch.cat([hf_ids, hf_next_t], dim=1)

            if cli_args.stop_when_both_eos and eos_id is not None and mg_next == eos_id and hf_next == eos_id:
                print(f"[INFO] both sides reached EOS at step={step}, early stop.")
                break

    mg_text = _decode(mg_ids.detach().cpu(), tokenizer)
    hf_text = _decode(hf_ids.detach().cpu(), tokenizer)
    print(f"[GEN][MG ] {mg_text}")
    print(f"[GEN][HF ] {hf_text}")

    if first_diverge_step < 0:
        print("[CHECK] PASS: no divergence within tested steps.")
        return

    print(f"[CHECK] first_diverge_step={first_diverge_step}")
    if first_diverge_payload is not None:
        print(
            "[DIVERGE]",
            f"cos={first_diverge_payload['cos']:.6f}",
            f"mg_gap={first_diverge_payload['mg_gap']:.6e}",
            f"hf_gap={first_diverge_payload['hf_gap']:.6e}",
        )
        print(f"[DIVERGE] mg_top{cli_args.topk}={first_diverge_payload['mg_topk']}")
        print(f"[DIVERGE] hf_top{cli_args.topk}={first_diverge_payload['hf_topk']}")


if __name__ == "__main__":
    main()
