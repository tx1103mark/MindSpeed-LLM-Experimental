# save as: patch_nebula_ple_export.py
# usage:
#   python patch_nebula_ple_export.py --hf-dir "Qwen3/qwen3-ple-test-hf/mg2hf"

import argparse
import json
import shutil
from pathlib import Path


def patch_export(hf_dir: Path):
    project_root = Path(__file__).resolve().parent
    src_cfg = project_root / "configuration_nebula_ple.py"
    src_model = project_root / "modeling_nebula_ple.py"
    dst_cfg = hf_dir / "configuration_nebula_ple.py"
    dst_model = hf_dir / "modeling_nebula_ple.py"
    cfg_json = hf_dir / "config.json"

    if not src_cfg.exists() or not src_model.exists():
        raise FileNotFoundError(
            "Missing source files in project root: configuration_nebula_ple.py / modeling_nebula_ple.py"
        )
    if not cfg_json.exists():
        raise FileNotFoundError(f"Missing {cfg_json}")

    hf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_cfg, dst_cfg)
    shutil.copy2(src_model, dst_model)

    with cfg_json.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["model_type"] = "nebula_ple"
    cfg["architectures"] = ["NebulaPLEForCausalLM"]

    auto_map = cfg.get("auto_map", {})
    auto_map.update(
        {
            "AutoConfig": "configuration_nebula_ple.NebulaPLEConfig",
            "AutoModel": "modeling_nebula_ple.NebulaPLEModel",
            "AutoModelForCausalLM": "modeling_nebula_ple.NebulaPLEForCausalLM",
        }
    )
    cfg["auto_map"] = auto_map

    # Ensure PLE fields exist
    cfg.setdefault("hidden_size_per_layer_input", 256)
    cfg.setdefault("vocab_size_per_layer_input", cfg.get("vocab_size", 151936))

    with cfg_json.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[OK] Patched export directory:")
    print(" -", dst_cfg)
    print(" -", dst_model)
    print(" -", cfg_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True, help="Path to exported HF dir, e.g. Qwen3/qwen3-ple-test-hf/mg2hf")
    args = parser.parse_args()
    patch_export(Path(args.hf_dir))


if __name__ == "__main__":
    main()
