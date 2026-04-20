# save as: verify_nebula_ple_weights.py
# usage:
#   python verify_nebula_ple_weights.py --hf-dir "Qwen3/qwen3-ple-test-hf/mg2hf"

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    sd = model.state_dict()
    cfg = model.config

    required_global = [
        "model.embed_tokens_per_layer.weight",
        "model.per_layer_model_projection.weight",
    ]

    required_layer_suffix = [
        "per_layer_input_gate.weight",
        "per_layer_projection.weight",
        "post_per_layer_input_norm.weight",
    ]

    print("== Config ==")
    print("model_type:", getattr(cfg, "model_type", None))
    print("hidden_size_per_layer_input:", getattr(cfg, "hidden_size_per_layer_input", None))
    print("vocab_size_per_layer_input:", getattr(cfg, "vocab_size_per_layer_input", None))
    print("num_hidden_layers:", getattr(cfg, "num_hidden_layers", None))

    missing = []
    for k in required_global:
        if k not in sd:
            missing.append(k)

    n_layers = int(getattr(cfg, "num_hidden_layers", 0))
    for i in range(n_layers):
        for suffix in required_layer_suffix:
            k = f"model.layers.{i}.{suffix}"
            if k not in sd:
                missing.append(k)

    # norm can be LayerNorm(weight+bias) or RMSNorm(weight only), accept either
    norm_weight_key = "model.per_layer_projection_norm.weight"
    norm_bias_key = "model.per_layer_projection_norm.bias"
    has_norm = (norm_weight_key in sd) or (norm_bias_key in sd)
    if not has_norm:
        missing.append("model.per_layer_projection_norm.(weight|bias)")

    print("\n== Result ==")
    if missing:
        print(f"[FAIL] Missing {len(missing)} PLE keys. First 20:")
        for k in missing[:20]:
            print(" -", k)
        raise SystemExit(1)

    print("[OK] All required PLE keys are present.")

    # quick shape print
    print("\n== Key Shapes ==")
    print("model.embed_tokens_per_layer.weight:", tuple(sd["model.embed_tokens_per_layer.weight"].shape))
    print("model.per_layer_model_projection.weight:", tuple(sd["model.per_layer_model_projection.weight"].shape))
    if norm_weight_key in sd:
        print("model.per_layer_projection_norm.weight:", tuple(sd[norm_weight_key].shape))
    if norm_bias_key in sd:
        print("model.per_layer_projection_norm.bias:", tuple(sd[norm_bias_key].shape))
    print("model.layers.0.per_layer_input_gate.weight:", tuple(sd["model.layers.0.per_layer_input_gate.weight"].shape))


if __name__ == "__main__":
    main()
