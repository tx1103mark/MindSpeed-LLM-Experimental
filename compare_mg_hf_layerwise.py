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
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--print-generate", action=argparse.BooleanOptionalAction, default=True)
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


def safe_cos(a: torch.Tensor, b: torch.Tensor):
    def _flat_cos(x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.detach().double().reshape(-1)
        y = y.detach().double().reshape(-1)
        denom = x.norm() * y.norm()
        if denom.item() == 0.0:
            return float("nan")
        v = (x @ y / denom).item()
        return max(-1.0, min(1.0, v))

    if a.shape == b.shape:
        return _flat_cos(a, b), "direct"
    if a.numel() == b.numel():
        return _flat_cos(a, b), "reshape_only"
    if a.t().shape == b.shape:
        return _flat_cos(a.t(), b), "a_transposed"
    if b.t().shape == a.shape:
        return _flat_cos(a, b.t()), "b_transposed"
    raise ValueError(f"shape mismatch: a={tuple(a.shape)} b={tuple(b.shape)}")


def mean_feature_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.detach().double().reshape(-1, a.shape[-1])
    b2 = b.detach().double().reshape(-1, b.shape[-1])
    a2 = F.normalize(a2, dim=-1)
    b2 = F.normalize(b2, dim=-1)
    v = (a2 * b2).sum(dim=-1).clamp(min=-1.0, max=1.0).mean().item()
    return float(v)


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
    try:
        cfg = hf_model.config
        print(
            "[INFO] HF config:",
            f"class={hf_model.__class__.__name__}",
            f"model_type={getattr(cfg, 'model_type', None)}",
            f"architectures={getattr(cfg, 'architectures', None)}",
            f"hidden_size={getattr(cfg, 'hidden_size', None)}",
            f"intermediate_size={getattr(cfg, 'intermediate_size', None)}",
            f"num_attention_heads={getattr(cfg, 'num_attention_heads', None)}",
            f"num_key_value_heads={getattr(cfg, 'num_key_value_heads', None)}",
        )
    except Exception as e:
        print(f"[INFO] HF config print skipped: {e}")

    hf_input_ids = input_ids_cpu.to(hf_dev)
    hf_attn_2d = attn_2d_cpu.to(hf_dev) if attn_2d_cpu is not None else None

    # Quick param-alignment diagnostics for likely mismatch hotspots.
    try:
        mg_emb = mg_model.embedding.word_embeddings.weight.detach().float().cpu()
        hf_emb = hf_model.model.embed_tokens.weight.detach().float().cpu()
        emb_cos, _ = safe_cos(mg_emb, hf_emb)
        emb_mae = (mg_emb - hf_emb).abs().mean().item()
        print(f"[PARAM] embed_tokens cos={emb_cos:.6f} mae={emb_mae:.6e}")
    except Exception as e:
        print(f"[PARAM] embed_tokens compare skipped: {e}")

    try:
        mg_l0 = mg_model.decoder.layers[0]
        hf_l0 = hf_model.model.layers[0]
        print(
            "[INFO] L0 module classes:",
            f"mg_layer={mg_l0.__class__.__name__}",
            f"hf_layer={hf_l0.__class__.__name__}",
            f"hf_mlp={getattr(hf_l0, 'mlp', None).__class__.__name__ if hasattr(hf_l0, 'mlp') else None}",
        )
        if hasattr(hf_l0, "mlp"):
            mlp = hf_l0.mlp
            for name in ["gate_up_proj", "gate_proj", "up_proj", "down_proj", "linear_fc1", "linear_fc2"]:
                if hasattr(mlp, name):
                    mod = getattr(mlp, name)
                    if hasattr(mod, "weight"):
                        print(f"[INFO] HF l0 mlp.{name}.weight shape={tuple(mod.weight.shape)}")

        mg_qkv = mg_l0.self_attention.linear_qkv.weight.detach().float().cpu()
        cfg = hf_model.config
        nh = int(getattr(cfg, "num_attention_heads"))
        ng = int(getattr(cfg, "num_key_value_heads", nh))
        mg_q, mg_k, mg_v = split_grouped_qkv_weight(mg_qkv, nh, ng)

        hf_q = hf_l0.self_attn.q_proj.weight.detach().float().cpu()
        hf_k = hf_l0.self_attn.k_proj.weight.detach().float().cpu()
        hf_v = hf_l0.self_attn.v_proj.weight.detach().float().cpu()

        def _cos(a, b):
            c, _ = safe_cos(a, b)
            return c

        print(
            "[PARAM] l0 qkv cos:",
            f"q={_cos(mg_q, hf_q):.6f}",
            f"k={_cos(mg_k, hf_k):.6f}",
            f"v={_cos(mg_v, hf_v):.6f}",
        )
        cross_terms = []
        for name, a, b in [
            ("q~k", mg_q, hf_k),
            ("q~v", mg_q, hf_v),
            ("k~q", mg_k, hf_q),
            ("v~q", mg_v, hf_q),
        ]:
            try:
                cross_terms.append(f"{name}={_cos(a, b):.6f}")
            except Exception:
                cross_terms.append(f"{name}=SKIP(shape)")
        print("[PARAM] l0 qkv cross-cos:", " ".join(cross_terms))

        try:
            mg_fc1 = mg_l0.mlp.linear_fc1.weight.detach().float().cpu()
            hf_fc1 = get_hf_fc1_weight(hf_l0)
            print(f"[PARAM] l0 fc1 shapes: mg={tuple(mg_fc1.shape)} hf={tuple(hf_fc1.shape)}")
            fc1_cos, fc1_mode = safe_cos(mg_fc1, hf_fc1)
            print(f"[PARAM] l0 fc1 cos={fc1_cos:.6f} mode={fc1_mode}")
            if hf_fc1.ndim == 2 and hf_fc1.shape[0] % 2 == 0:
                half = hf_fc1.shape[0] // 2
                hf_fc1_swapped = torch.cat([hf_fc1[half:], hf_fc1[:half]], dim=0)
                try:
                    fc1_swap_cos, fc1_swap_mode = safe_cos(mg_fc1, hf_fc1_swapped)
                    print(f"[PARAM] l0 fc1 swapped_half_cos={fc1_swap_cos:.6f} mode={fc1_swap_mode}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[PARAM] l0 fc1 compare skipped: {e}")

        try:
            mg_o = mg_l0.self_attention.linear_proj.weight.detach().float().cpu()
            hf_o = hf_l0.self_attn.o_proj.weight.detach().float().cpu()
            o_cos, o_mode = safe_cos(mg_o, hf_o)
            print(f"[PARAM] l0 o_proj cos={o_cos:.6f} mode={o_mode}")
        except Exception as e:
            print(f"[PARAM] l0 o_proj compare skipped: {e}")

        try:
            mg_in_ln = mg_l0.input_layernorm.weight.detach().float().cpu()
            hf_in_ln = hf_l0.input_layernorm.weight.detach().float().cpu()
            in_ln_cos, in_ln_mode = safe_cos(mg_in_ln, hf_in_ln)
            print(f"[PARAM] l0 input_ln cos={in_ln_cos:.6f} mode={in_ln_mode}")
        except Exception as e:
            print(f"[PARAM] l0 input_ln compare skipped: {e}")

        try:
            mg_post_ln = mg_l0.pre_mlp_layernorm.weight.detach().float().cpu()
            hf_post_ln = hf_l0.post_attention_layernorm.weight.detach().float().cpu()
            post_ln_cos, post_ln_mode = safe_cos(mg_post_ln, hf_post_ln)
            print(f"[PARAM] l0 post_attn_ln cos={post_ln_cos:.6f} mode={post_ln_mode}")
        except Exception as e:
            print(f"[PARAM] l0 post_attn_ln compare skipped: {e}")

        has_hf_ple = hasattr(hf_l0, "per_layer_input_gate") and hasattr(hf_l0, "per_layer_projection")
        has_mg_ple = hasattr(mg_l0, "per_layer_input_gate") and hasattr(mg_l0, "per_layer_projection")
        print(f"[PARAM] l0 ple_modules: mg={has_mg_ple} hf={has_hf_ple}")
        if has_hf_ple and has_mg_ple:
            mg_ple_gate = mg_l0.per_layer_input_gate.weight.detach().float().cpu()
            hf_ple_gate = hf_l0.per_layer_input_gate.weight.detach().float().cpu()
            pg_cos, pg_mode = safe_cos(mg_ple_gate, hf_ple_gate)
            print(f"[PARAM] l0 ple_gate cos={pg_cos:.6f} mode={pg_mode}")

            mg_ple_proj = mg_l0.per_layer_projection.weight.detach().float().cpu()
            hf_ple_proj = hf_l0.per_layer_projection.weight.detach().float().cpu()
            pp_cos, pp_mode = safe_cos(mg_ple_proj, hf_ple_proj)
            print(f"[PARAM] l0 ple_proj cos={pp_cos:.6f} mode={pp_mode}")

            mg_ple_ln = mg_l0.post_per_layer_input_norm.weight.detach().float().cpu()
            hf_ple_ln = hf_l0.post_per_layer_input_norm.weight.detach().float().cpu()
            pln_cos, pln_mode = safe_cos(mg_ple_ln, hf_ple_ln)
            print(f"[PARAM] l0 ple_ln cos={pln_cos:.6f} mode={pln_mode}")

        try:
            has_hf_global_ple = hasattr(hf_model.model, "embed_tokens_per_layer") and \
                hasattr(hf_model.model, "per_layer_model_projection")
            has_mg_global_ple = hasattr(mg_model, "embedding") and \
                hasattr(mg_model.embedding, "embed_tokens_per_layer") and \
                hasattr(mg_model, "per_layer_model_projection")
            print(f"[PARAM] global ple_modules: mg={has_mg_global_ple} hf={has_hf_global_ple}")
            if has_hf_global_ple and has_mg_global_ple:
                mg_ple_emb = mg_model.embedding.embed_tokens_per_layer.weight.detach().float().cpu()
                hf_ple_emb = hf_model.model.embed_tokens_per_layer.weight.detach().float().cpu()
                pe_cos, pe_mode = safe_cos(mg_ple_emb, hf_ple_emb)
                print(f"[PARAM] ple_embed_tokens_per_layer cos={pe_cos:.6f} mode={pe_mode}")

                mg_ple_proj = mg_model.per_layer_model_projection.weight.detach().float().cpu()
                hf_ple_proj = hf_model.model.per_layer_model_projection.weight.detach().float().cpu()
                pp_cos, pp_mode = safe_cos(mg_ple_proj, hf_ple_proj)
                print(f"[PARAM] ple_model_projection cos={pp_cos:.6f} mode={pp_mode}")
        except Exception as e:
            print(f"[PARAM] global ple compare skipped: {e}")
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
    layer_warn_cnt = 0
    layer_records = []
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
        cos = mean_feature_cos(mg_t, hf_t)
        print(f"[LAYER {i:02d}] mean_abs={mean_abs:.6e} max_abs={max_abs:.6e} cos={cos:.6f}")
        layer_records.append((i, mean_abs, cos))
        if cos < 0.80 or mean_abs > 0.50:
            layer_warn_cnt += 1
        if first_bad is None and (cos < 0.80 or mean_abs > 0.50):
            first_bad = i

    mg_last = mg_logits[:, -1, :].float().cpu()
    hf_last = hf_logits[:, -1, :].float().cpu()
    logit_cos = mean_feature_cos(mg_last, hf_last)
    mg_next = int(mg_last.argmax(dim=-1)[0])
    hf_next = int(hf_last.argmax(dim=-1)[0])
    print(f"[LOGITS] cos={logit_cos:.6f}, mg_next={mg_next}, hf_next={hf_next}")

    if args.print_generate:
        try:
            with torch.no_grad():
                hf_gen = hf_model.generate(
                    input_ids=hf_input_ids,
                    attention_mask=hf_attn_2d,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            hf_text = tokenizer.decode(hf_gen[0], skip_special_tokens=False)
        except Exception as e:
            hf_text = f"<HF generation failed: {e}>"

        try:
            mg_out = mg_wrap.generate(
                input_ids=mg_input_ids.detach().cpu(),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
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
            mg_text = tokenizer.decode(mg_tokens, skip_special_tokens=False) if mg_tokens is not None else str(mg_out)
        except Exception as e:
            mg_text = f"<MG generation failed: {e}>"
        print(f"[GEN][MG ] {mg_text}")
        print(f"[GEN][HF ] {hf_text}")

    # Practical check policy: prioritize logits and greedy next-token match.
    if logit_cos >= 0.95 and mg_next == hf_next:
        if layer_warn_cnt > 0:
            print(
                f"[CHECK] PASS(logits): logits aligned; "
                f"layer warnings={layer_warn_cnt}/{common}, first_warn_layer={first_bad}."
            )
        else:
            print("[CHECK] PASS: logits and layers are aligned.")
    else:
        print(
            f"[CHECK] WARNING: logits mismatch (cos={logit_cos:.6f}, next_same={mg_next == hf_next}), "
            f"layer warnings={layer_warn_cnt}/{common}, first_warn_layer={first_bad}."
        )


if __name__ == "__main__":
    main()
