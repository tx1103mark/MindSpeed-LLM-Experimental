#!/usr/bin/env python
# coding=utf-8
"""Layer-wise MG vs HF activation comparison for Nebula-PLE/Qwen3-style models.

Run with torchrun and pass Megatron args after `--`.
"""

import argparse
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindspeed_llm import megatron_adaptor  # noqa: F401
from inference import model_provider
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.inference.module import MegatronModuleForCausalLM


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(description="Compare MG and HF layer activations.")
    p.add_argument("--mg-load-dir", required=True)
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--hf-cpu", action="store_true")
    p.add_argument("megatron_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    extra = args.megatron_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def to_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def accel_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_from_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    return None


def main():
    args, megatron_extra = parse_args()
    if not megatron_extra:
        raise ValueError("Please append Megatron runtime args after '--'.")

    sys.argv = [sys.argv[0]] + megatron_extra
    initialize_megatron(args_defaults={"no_load_rng": True, "no_load_optim": True})

    mg_wrap = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.mg_load_dir,
    )
    mg_wrap.eval()
    if not hasattr(mg_wrap, "infer_model"):
        raise RuntimeError("Expected GPTModelInfer with infer_model.")

    tokenizer_dir = args.tokenizer_dir or args.hf_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    batch = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)

    mg_dev = accel_device()
    hf_dev = torch.device("cpu") if args.hf_cpu else mg_dev

    input_ids_cpu = batch["input_ids"]
    attn_2d_cpu = batch.get("attention_mask", None)

    mg_input_ids = input_ids_cpu.to(mg_dev)

    infer = mg_wrap.infer_model
    mg_attn_mask, position_ids = infer.build_attention_mask_and_position_ids(mg_input_ids)

    mg_model = get_args().model[0]
    mg_model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=to_dtype(args.dtype),
    ).to(hf_dev)
    hf_model.eval()

    hf_input_ids = input_ids_cpu.to(hf_dev)
    hf_attn_2d = attn_2d_cpu.to(hf_dev) if attn_2d_cpu is not None else None

    mg_layers = getattr(mg_model.decoder, "layers", None)
    hf_layers = getattr(hf_model.model, "layers", None)
    if mg_layers is None or hf_layers is None:
        raise RuntimeError("Cannot find decoder layers in MG or HF model.")

    mg_hidden: Dict[int, torch.Tensor] = {}
    hf_hidden: Dict[int, torch.Tensor] = {}
    hooks = []

    for i, layer in enumerate(mg_layers):
        def _mk_mg_hook(idx):
            def _hook(_m, _inp, out):
                t = tensor_from_output(out)
                if t is not None:
                    mg_hidden[idx] = t.detach().float().cpu()
            return _hook
        hooks.append(layer.register_forward_hook(_mk_mg_hook(i)))

    for i, layer in enumerate(hf_layers):
        def _mk_hf_hook(idx):
            def _hook(_m, _inp, out):
                t = tensor_from_output(out)
                if t is not None:
                    hf_hidden[idx] = t.detach().float().cpu()
            return _hook
        hooks.append(layer.register_forward_hook(_mk_hf_hook(i)))

    with torch.no_grad():
        mg_logits = mg_model(
            input_ids=mg_input_ids,
            position_ids=position_ids,
            attention_mask=mg_attn_mask,
        )
        hf_logits = hf_model(input_ids=hf_input_ids, attention_mask=hf_attn_2d).logits

    for h in hooks:
        h.remove()

    # Convert MG layer outputs [s,b,h] -> [b,s,h] for comparison.
    common = min(len(mg_hidden), len(hf_hidden))
    print(f"[INFO] devices mg={mg_dev} hf={hf_dev}, captured_layers mg={len(mg_hidden)} hf={len(hf_hidden)}")

    first_bad = None
    for i in range(common):
        mg_t = mg_hidden[i]
        hf_t = hf_hidden[i]
        if mg_t.ndim == 3 and mg_t.shape[0] == hf_t.shape[1] and mg_t.shape[1] == hf_t.shape[0]:
            mg_t = mg_t.permute(1, 0, 2).contiguous()
        if mg_t.shape != hf_t.shape:
            print(f"[LAYER {i:02d}] SHAPE MISMATCH MG={tuple(mg_t.shape)} HF={tuple(hf_t.shape)}")
            if first_bad is None:
                first_bad = i
            continue

        diff = (mg_t - hf_t).abs()
        mean_abs = diff.mean().item()
        max_abs = diff.max().item()
        cos = F.cosine_similarity(
            mg_t.reshape(-1, mg_t.shape[-1]),
            hf_t.reshape(-1, hf_t.shape[-1]),
            dim=-1,
        ).mean().item()
        print(f"[LAYER {i:02d}] mean_abs={mean_abs:.6e} max_abs={max_abs:.6e} cos={cos:.6f}")
        if first_bad is None and (cos < 0.99 or mean_abs > 1e-2):
            first_bad = i

    mg_last = mg_logits[:, -1, :].float().cpu()
    hf_last = hf_logits[:, -1, :].float().cpu()
    logit_cos = F.cosine_similarity(mg_last, hf_last, dim=-1).mean().item()
    print(f"[LOGITS] cos={logit_cos:.6f}, mg_next={int(mg_last.argmax(dim=-1)[0])}, hf_next={int(hf_last.argmax(dim=-1)[0])}")

    if first_bad is None:
        print("[CHECK] PASS: no obvious divergent layer found.")
    else:
        print(f"[CHECK] first divergent layer appears around index {first_bad}.")


if __name__ == "__main__":
    main()
