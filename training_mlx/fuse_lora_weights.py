#!/usr/bin/env python3
"""
fuse_lora_weights.py

NogaiLLM Training Pipeline - Bridge Operations
Fuses the parameter-efficient LoRA weights (Phase 1) directly into the 
Multi-Layer Perceptron (MLP) weights of the underlying base architecture.
This standalone fused model is required before Phase 2 SFT can commence.
"""

import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_tensor_map(path: Path) -> Dict[str, torch.Tensor]:
    """Extracts weights from safetensors format into memory safely."""
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors

def load_lora_scale(adapter_dir: Path) -> float:
    """Reads the mlx-lm LoRA scale from adapter_config.json (delta = scale * B @ A)."""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        logger.warning("adapter_config.json not found; assuming mlx-lm default scale 20.0")
        return 20.0
    with open(config_path, "r") as f:
        config = json.load(f)
    return float(config.get("lora_parameters", {}).get("scale", 20.0))

def fuse_weights(base: Dict[str, torch.Tensor], adapter: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    """Mathematically projects low-rank matrices (A and B) back onto the primary weight tensor."""
    fused = dict(base)
    adapter_bases = {key.rsplit(".", 1)[0] for key in adapter if key.endswith(".lora_a")}
    
    for base_key in sorted(adapter_bases):
        a_key = f"{base_key}.lora_a"
        b_key = f"{base_key}.lora_b"
        weight_key = f"{base_key}.weight"
        
        if a_key not in adapter or b_key not in adapter:
            raise KeyError(f"Missing LoRA constraint matrices for {base_key}")

        if weight_key not in base:
            raise KeyError(f"Target base architecture is missing tensor {weight_key}")

        lora_a = adapter[a_key]
        lora_b = adapter[b_key]
        weight = base[weight_key]

        # Execute matrix multiplication: W' = W + scale * (B @ A)
        delta = (scale * (lora_b.transpose(0, 1) @ lora_a.transpose(0, 1))).to(weight.dtype)
        fused[weight_key] = weight + delta

    return fused

def execute_fusion(base_dir: str, adapter_dir: str, output_dir: str):
    base_model_path = Path(base_dir) / "model.safetensors"
    adapter_path = Path(adapter_dir) / "adapters.safetensors"
    out_path = Path(output_dir)

    if not base_model_path.exists():
        logger.error(f"Base tensors missing at {base_model_path}")
        return
    if not adapter_path.exists():
        logger.error(f"Adapter tensors missing at {adapter_path}")
        return

    logger.info("Loading Base Model Weights into memory...")
    base_tensors = load_tensor_map(base_model_path)
    
    logger.info("Loading Phase 1 LoRA Matrix mappings...")
    adapter_tensors = load_tensor_map(adapter_path)
    
    scale = load_lora_scale(Path(adapter_dir))
    logger.info(f"Executing mathematical fusion operation (LoRA scale: {scale})...")
    fused_tensors = fuse_weights(base_tensors, adapter_tensors, scale)

    if out_path.exists():
        logger.warning(f"Overwriting existing directory at {out_path}")
        shutil.rmtree(out_path)
        
    shutil.copytree(Path(base_dir), out_path)
    
    # Save the newly fused, standalone architecture
    save_file(fused_tensors, out_path / "model.safetensors")
    logger.info(f"SUCCESS: Fused {len(adapter_tensors)} adapter tensors into a new standalone model at {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Weight Fusion Engine")
    parser.add_argument("--base-model", type=str, default="local_qwen_1.5B", help="Directory of downloaded base model")
    parser.add_argument("--adapter-path", type=str, default="adapters/qwen_1.5b_nogai_phase1", help="Phase 1 LoRA weights directory")
    parser.add_argument("--output-dir", type=str, default="local_qwen_1.5B_Nogai_Base", help="Output directory for the fused model")
    args = parser.parse_args()
    
    execute_fusion(args.base_model, args.adapter_path, args.output_dir)