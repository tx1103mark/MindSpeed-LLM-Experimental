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


def unwrap_module(m):
    # Megatron may wrap GPTModel with Float16Module/DDP wrappers.
    seen = set()
    cur = m
    while hasattr(cur, "module") and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.module
    return cur


def split_grouped_qkv_weight(qkv_weight: torch.Tensor, num_attention_heads: int, num_key_value_heads: int):
    """Split Megatron grouped QKV packed weight into (q, k, v)."""
    if qkv_weight.ndim != 2:
        raise ValueError(f"Expected 2D qkv weight, got shape={tuple(qkv_weight.shape)}")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads({num_attention_heads}) must be divisible by "
            f"num_key_value_heads({num_key_value_heads})"
        )

    repeats = num_attention_heads // num_key_value_heads
    groups = num_key_value_heads
    rows, hidden = qkv_weight.shape
    per_group = repeats + 2
    if rows % (groups * per_group) != 0:
        raise ValueError(
            f"QKV rows({rows}) is not divisible by groups*per_group ({groups}*{per_group})"
        )

    head_dim = rows // (groups * per_group)
    packed = qkv_weight.reshape(groups, per_group, head_dim, hidden)
    q = packed[:, :repeats, :, :].reshape(-1, hidden)
    k = packed[:, repeats:repeats + 1, :, :].reshape(-1, hidden)
    v = packed[:, repeats + 1:, :, :].reshape(-1, hidden)
    return q, k, v


def get_hf_fc1_weight(hf_layer):
    mlp = hf_layer.mlp
    if hasattr(mlp, "gate_up_proj") and hasattr(mlp.gate_up_proj, "weight"):
        return mlp.gate_up_proj.weight.detach().float().cpu()
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
        gate = mlp.gate_proj.weight.detach().float().cpu()
        up = mlp.up_proj.weight.detach().float().cpu()
        return torch.cat([gate, up], dim=0)
    if hasattr(mlp, "linear_fc1") and hasattr(mlp.linear_fc1, "weight"):
        return mlp.linear_fc1.weight.detach().float().cpu()
    raise AttributeError("Cannot find HF fc1 projection (gate_up_proj or gate_proj/up_proj or linear_fc1).")


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

    mg_model = unwrap_module(get_args().model[0])
    mg_model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=to_dtype(args.dtype),
    ).to(hf_dev)
    hf_model.eval()

    hf_input_ids = input_ids_cpu.to(hf_dev)
    hf_attn_2d = attn_2d_cpu.to(hf_dev) if attn_2d_cpu is not None else None

    # Quick param-alignment diagnostics for likely mismatch hotspots.
    try:
        mg_emb = mg_model.embedding.word_embeddings.weight.detach().float().cpu()
        hf_emb = hf_model.model.embed_tokens.weight.detach().float().cpu()
        emb_cos = F.cosine_similarity(mg_emb.reshape(1, -1), hf_emb.reshape(1, -1), dim=-1).item()
        emb_mae = (mg_emb - hf_emb).abs().mean().item()
        print(f"[PARAM] embed_tokens cos={emb_cos:.6f} mae={emb_mae:.6e}")
    except Exception as e:
        print(f"[PARAM] embed_tokens compare skipped: {e}")

    try:
        mg_l0 = mg_model.decoder.layers[0]
        hf_l0 = hf_model.model.layers[0]

        mg_qkv = mg_l0.self_attention.linear_qkv.weight.detach().float().cpu()
        cfg = hf_model.config
        nh = int(getattr(cfg, "num_attention_heads"))
        ng = int(getattr(cfg, "num_key_value_heads", nh))
        mg_q, mg_k, mg_v = split_grouped_qkv_weight(mg_qkv, nh, ng)

        hf_q = hf_l0.self_attn.q_proj.weight.detach().float().cpu()
        hf_k = hf_l0.self_attn.k_proj.weight.detach().float().cpu()
        hf_v = hf_l0.self_attn.v_proj.weight.detach().float().cpu()

        def _cos(a, b):
            return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1).item()

        print(
            "[PARAM] l0 qkv cos:",
            f"q={_cos(mg_q, hf_q):.6f}",
            f"k={_cos(mg_k, hf_k):.6f}",
            f"v={_cos(mg_v, hf_v):.6f}",
        )
        print(
            "[PARAM] l0 qkv cross-cos:",
            f"q~k={_cos(mg_q, hf_k):.6f}",
            f"q~v={_cos(mg_q, hf_v):.6f}",
            f"k~q={_cos(mg_k, hf_q):.6f}",
            f"v~q={_cos(mg_v, hf_q):.6f}",
        )

        mg_fc1 = mg_l0.mlp.linear_fc1.weight.detach().float().cpu()
        hf_fc1 = get_hf_fc1_weight(hf_l0)
        fc1_cos = _cos(mg_fc1, hf_fc1)
        half = hf_fc1.shape[0] // 2
        hf_fc1_swapped = torch.cat([hf_fc1[half:], hf_fc1[:half]], dim=0)
        fc1_swap_cos = _cos(mg_fc1, hf_fc1_swapped)
        print(f"[PARAM] l0 fc1 cos={fc1_cos:.6f} swapped_half_cos={fc1_swap_cos:.6f}")
    except Exception as e:
        print(f"[PARAM] l0 qkv/fc1 compare skipped: {e}")

    print(f"[INFO] MG model class={mg_model.__class__.__name__}")
    mg_decoder = getattr(mg_model, "decoder", None)
    if mg_decoder is None and hasattr(mg_model, "language_model"):
        mg_decoder = getattr(mg_model.language_model, "encoder", None) or getattr(mg_model.language_model, "decoder", None)
    mg_layers = getattr(mg_decoder, "layers", None) if mg_decoder is not None else None

    hf_layers = getattr(hf_model.model, "layers", None)
    if mg_layers is None or hf_layers is None:
        raise RuntimeError(
            f"Cannot find decoder layers in MG or HF model. "
            f"MG decoder={type(mg_decoder).__name__ if mg_decoder is not None else None}, "
            f"HF model has layers={hasattr(getattr(hf_model, 'model', None), 'layers')}"
        )

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
