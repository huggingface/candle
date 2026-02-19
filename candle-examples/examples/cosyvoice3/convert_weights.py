#!/usr/bin/env python3
"""
CosyVoice3 PyTorch to Safetensors Converter

Converts CosyVoice3 model weights from PyTorch (.pt) format to safetensors format
for use with the Candle framework.

Usage:
    python convert_weights.py --input <input_dir> --output <output_dir>

Example:
    python convert_weights.py \
        --input weights/Fun-CosyVoice3-0.5B-2512 \
        --output weights/CosyVoice3-0.5B-2512-Candle
"""

import argparse
import os
import shutil
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def fuse_weight_norm(state_dict: dict) -> dict:
    """
    Fuse PyTorch weight_norm parametrizations into standard weights.
    
    HiFT uses weight_norm for conv layers:
      conv_pre.parametrizations.weight.original0: (out, 1, 1)  # g - direction
      conv_pre.parametrizations.weight.original1: (out, in, k) # v - magnitude
    
    Fusion formula: weight = g * v / ||v||
    """
    new_dict = {}
    processed = set()
    
    for key, value in state_dict.items():
        if '.parametrizations.weight.original0' in key:
            # Found weight_norm g parameter
            base_key = key.replace('.parametrizations.weight.original0', '')
            v_key = key.replace('original0', 'original1')
            
            if v_key not in state_dict:
                print(f"  Warning: Missing {v_key}, skipping {key}")
                continue
            
            g = value  # [out, 1, 1] or [out, 1]
            v = state_dict[v_key]  # [out, in, kernel] or [out, in]
            
            # Compute L2 norm and fuse
            v_flat = v.reshape(v.shape[0], -1)
            v_norm = torch.linalg.norm(v_flat, dim=1, keepdim=True)
            
            # Reshape v_norm to match g's shape
            v_norm = v_norm.reshape(g.shape)
            
            # Fuse: weight = g * v / ||v||
            weight = g * v / (v_norm + 1e-12)
            
            new_dict[base_key + '.weight'] = weight
            processed.add(key)
            processed.add(v_key)
            print(f"  Fused weight_norm: {base_key}.weight {tuple(weight.shape)}")
            
        elif key not in processed and '.parametrizations.' not in key:
            new_dict[key] = value
    
    return new_dict


def convert_llm_weights(llm_state: dict) -> dict:
    """Convert LLM (CosyVoice3LM) weights."""
    converted = {}
    
    for key, value in llm_state.items():
        # Rename keys for Candle compatibility
        new_key = key
        
        # llm.model.model.* -> llm.*
        if new_key.startswith('llm.model.model.'):
            new_key = 'llm.' + new_key[len('llm.model.model.'):]
        elif new_key.startswith('llm.model.'):
            new_key = 'llm.' + new_key[len('llm.model.'):]
        
        # Convert to float32 for compatibility (can be quantized later)
        # Use clone() to handle shared tensors (tie_word_embeddings)
        converted[new_key] = value.to(torch.float32).clone().contiguous()
    
    return converted


def convert_flow_weights(flow_state: dict) -> dict:
    """Convert Flow (DiT + CFM) weights."""
    converted = {}
    
    for key, value in flow_state.items():
        new_key = key
        
        # decoder.estimator.* -> dit.*
        if 'decoder.estimator.' in new_key:
            new_key = new_key.replace('decoder.estimator.', 'dit.')
        
        converted[new_key] = value.to(torch.float32).contiguous()
    
    return converted


def convert_hift_weights(hift_state: dict) -> dict:
    """Convert HiFT (Vocoder) weights with weight_norm fusion."""
    # First fuse weight_norm
    print("  Fusing weight_norm parameters...")
    fused = fuse_weight_norm(hift_state)
    
    converted = {}
    for key, value in fused.items():
        # HiFT keys are already in correct format
        converted[key] = value.to(torch.float32).contiguous()
    
    return converted


def create_config_json(input_dir: Path, output_dir: Path):
    """Create a unified config.json for the Candle model."""
    # Read cosyvoice3.yaml for reference values
    yaml_path = input_dir / "cosyvoice3.yaml"
    
    config = {
        "model_type": "cosyvoice3",
        "sample_rate": 24000,
        "llm_input_size": 896,
        "llm_output_size": 896,
        "speech_token_size": 6561,
        "spk_embed_dim": 192,
        "token_frame_rate": 25,
        "token_mel_ratio": 2,
        "chunk_size": 25,
        "pre_lookahead_len": 3,
        
        "dit": {
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "dim_head": 64,
            "ff_mult": 2,
            "mel_dim": 80,
            "spk_dim": 80
        },
        
        "hift": {
            "in_channels": 80,
            "base_channels": 512,
            "nb_harmonics": 8,
            "upsample_rates": [8, 5, 3],
            "upsample_kernel_sizes": [16, 11, 7],
            "istft_n_fft": 16,
            "istft_hop_len": 4,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "source_resblock_kernel_sizes": [7, 7, 11],
            "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "conv_pre_look_right": 4,
            "nsf_alpha": 0.1,
            "nsf_sigma": 0.003
        },
        
        "qwen2": {
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "vocab_size": 151936
        }
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Created: config.json")


def convert_cosyvoice3(input_dir: str, output_dir: str):
    """Main conversion function."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"\n{'='*60}")
    print(f"CosyVoice3 Weight Converter")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Validate input
    required_files = ['llm.pt', 'flow.pt', 'hift.pt']
    for f in required_files:
        if not (input_path / f).exists():
            raise FileNotFoundError(f"Missing required file: {input_path / f}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert LLM weights
    print("[1/6] Converting LLM weights...")
    llm_state = torch.load(input_path / "llm.pt", map_location="cpu", weights_only=True)
    llm_converted = convert_llm_weights(llm_state)
    save_file(llm_converted, output_path / "llm.safetensors")
    print(f"  Saved: llm.safetensors ({len(llm_converted)} tensors)")
    del llm_state, llm_converted
    
    # 2. Convert Flow weights
    print("\n[2/6] Converting Flow weights...")
    flow_state = torch.load(input_path / "flow.pt", map_location="cpu", weights_only=True)
    flow_converted = convert_flow_weights(flow_state)
    save_file(flow_converted, output_path / "flow.safetensors")
    print(f"  Saved: flow.safetensors ({len(flow_converted)} tensors)")
    del flow_state, flow_converted
    
    # 3. Convert HiFT weights
    print("\n[3/6] Converting HiFT weights...")
    hift_state = torch.load(input_path / "hift.pt", map_location="cpu", weights_only=True)
    hift_converted = convert_hift_weights(hift_state)
    save_file(hift_converted, output_path / "hift.safetensors")
    print(f"  Saved: hift.safetensors ({len(hift_converted)} tensors)")
    del hift_state, hift_converted
    
    # 4. Copy ONNX files
    print("\n[4/6] Copying ONNX files...")
    onnx_files = ['campplus.onnx', 'speech_tokenizer_v3.onnx']
    for onnx_file in onnx_files:
        src = input_path / onnx_file
        if src.exists():
            shutil.copy2(src, output_path / onnx_file)
            print(f"  Copied: {onnx_file}")
        else:
            print(f"  Warning: {onnx_file} not found, skipping")
    
    # 5. Copy tokenizer files (from CosyVoice-BlankEN)
    print("\n[5/6] Copying tokenizer files...")
    tokenizer_dir = input_path / "CosyVoice-BlankEN"
    tokenizer_output = output_path / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer_output.mkdir(exist_ok=True)
        tokenizer_files = [
            'config.json', 'generation_config.json', 
            'tokenizer_config.json', 'vocab.json', 'merges.txt'
        ]
        for tf in tokenizer_files:
            src = tokenizer_dir / tf
            if src.exists():
                shutil.copy2(src, tokenizer_output / tf)
                print(f"  Copied: tokenizer/{tf}")
    else:
        print("  Warning: CosyVoice-BlankEN directory not found")
    
    # 6. Create config.json
    print("\n[6/6] Creating config.json...")
    create_config_json(input_path, output_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print("\nFiles created:")
    for f in sorted(output_path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            rel_path = f.relative_to(output_path)
            print(f"  {rel_path}: {size_mb:.1f} MB")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CosyVoice3 PyTorch weights to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python convert_weights.py \\
        --input weights/Fun-CosyVoice3-0.5B-2512 \\
        --output weights/CosyVoice3-0.5B-2512-Candle
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the original CosyVoice3 model directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output directory for converted weights"
    )
    
    args = parser.parse_args()
    convert_cosyvoice3(args.input, args.output)


if __name__ == "__main__":
    main()
