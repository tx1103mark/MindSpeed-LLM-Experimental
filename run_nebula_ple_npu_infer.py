# coding=utf-8
# usage:
# python run_nebula_ple_npu_infer.py \
#   --hf-dir "Qwen3/qwen3-ple-test-hf/mg2hf" \
#   --prompt "你好，简单介绍一下你自己。" \
#   --max-new-tokens 64

import argparse
import time
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception as e:
    raise RuntimeError("torch_npu is required for NPU inference.") from e

from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True, help="Exported nebula_ple HF directory")
    parser.add_argument("--prompt", default="你好，简单介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--npu", type=int, default=0, help="NPU device index")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    if not hf_dir.exists():
        raise FileNotFoundError(f"hf-dir not found: {hf_dir}")

    device = f"npu:{args.npu}"
    torch.npu.set_device(device)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] loading tokenizer from: {hf_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)

    print(f"[INFO] loading model on {device}, dtype={args.dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # PLE quick checks
    sd = model.state_dict()
    ple_keys = [
        "model.embed_tokens_per_layer.weight",
        "model.per_layer_model_projection.weight",
        "model.layers.0.per_layer_input_gate.weight",
        "model.layers.0.per_layer_projection.weight",
    ]
    missing = [k for k in ple_keys if k not in sd]
    if missing:
        print("[WARN] Missing some PLE keys:")
        for k in missing:
            print(" -", k)
    else:
        print("[INFO] PLE keys detected.")

    cfg = model.config
    print("[INFO] model_type:", getattr(cfg, "model_type", None))
    print("[INFO] hidden_size_per_layer_input:", getattr(cfg, "hidden_size_per_layer_input", None))
    print("[INFO] vocab_size_per_layer_input:", getattr(cfg, "vocab_size_per_layer_input", None))

    # Prepare input
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Warmup + timing
    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
        )
    torch.npu.synchronize()

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        )
    torch.npu.synchronize()
    t1 = time.time()

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    new_tokens = out.shape[1] - input_ids.shape[1]
    latency = t1 - t0
    tps = new_tokens / latency if latency > 0 else 0.0

    print("\n===== OUTPUT =====")
    print(text)
    print("==================")
    print(f"[INFO] new_tokens={new_tokens}, latency={latency:.3f}s, tok/s={tps:.2f}")


if __name__ == "__main__":
    main()
