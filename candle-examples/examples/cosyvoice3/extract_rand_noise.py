#!/usr/bin/env python3
"""
Extract Pre-computed Random Noise for CosyVoice3

Extracts the deterministic random noise used by CosyVoice3's Flow Matching module
for exact reproducibility with the Python implementation.

This file is OPTIONAL - the Candle implementation will generate its own noise
if this file is not provided. However, for exact numerical matching with the
Python implementation, this pre-extracted noise is required.

Usage:
    python extract_rand_noise.py --output <output_path> [--cosyvoice-path <path>]

Example:
    python extract_rand_noise.py \
        --output weights/CosyVoice3-0.5B-2512-Candle/rand_noise.safetensors

    # With custom CosyVoice path
    python extract_rand_noise.py \
        --output rand_noise.safetensors \
        --cosyvoice-path /path/to/CosyVoice

Requirements:
    - PyTorch
    - safetensors
    - CosyVoice repository (for set_all_random_seed function)
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def set_all_random_seed_fallback(seed: int):
    """
    Fallback implementation of set_all_random_seed if CosyVoice is not available.
    This matches the behavior of cosyvoice.utils.common.set_all_random_seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_set_all_random_seed(cosyvoice_path: str = None):
    """
    Try to import set_all_random_seed from CosyVoice, fall back to local implementation.
    """
    if cosyvoice_path:
        sys.path.insert(0, cosyvoice_path)
    
    try:
        from cosyvoice.utils.common import set_all_random_seed
        print("Using CosyVoice's set_all_random_seed")
        return set_all_random_seed
    except ImportError:
        print("CosyVoice not found, using fallback implementation")
        return set_all_random_seed_fallback


def extract_rand_noise(output_path: str, cosyvoice_path: str = None, save_numpy: bool = False):
    """
    Extract the pre-computed random noise used in CausalConditionalCFM.
    
    The noise is generated with:
    - seed = 0
    - shape = [1, 80, 15000] (50 seconds * 300 frames/second)
    - dtype = float32
    """
    print(f"\n{'='*60}")
    print("CosyVoice3 Random Noise Extractor")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Get the seed function
    set_all_random_seed = get_set_all_random_seed(cosyvoice_path)
    
    # Generate noise with fixed seed (matches Python CausalConditionalCFM)
    print("[1/3] Generating random noise with seed=0...")
    set_all_random_seed(0)
    
    # Shape: [1, 80, 50*300] = [1, 80, 15000]
    # This supports up to 50 seconds of audio at 300 mel frames/second
    rand_noise = torch.randn([1, 80, 50 * 300], dtype=torch.float32)
    
    # Print statistics for verification
    print(f"\n[2/3] Noise statistics:")
    print(f"  Shape: {tuple(rand_noise.shape)}")
    print(f"  Dtype: {rand_noise.dtype}")
    print(f"  Mean:  {rand_noise.mean().item():.6f}")
    print(f"  Std:   {rand_noise.std().item():.6f}")
    print(f"  Min:   {rand_noise.min().item():.6f}")
    print(f"  Max:   {rand_noise.max().item():.6f}")
    
    # Print first few values for verification
    flat = rand_noise.flatten()
    print(f"\n  First 10 values: {[f'{v:.6f}' for v in flat[:10].tolist()]}")
    
    # Save to safetensors
    print(f"\n[3/3] Saving to {output_path}...")
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created directory: {output_dir}")
    
    from safetensors.torch import save_file
    save_file({"rand_noise": rand_noise}, output_path)
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.2f} MB)")
    
    # Optionally save as numpy for inspection
    if save_numpy:
        import numpy as np
        np_path = str(output_path).replace('.safetensors', '.npy')
        np.save(np_path, rand_noise.numpy())
        print(f"  Saved numpy: {np_path}")
    
    print(f"\n{'='*60}")
    print("Extraction Complete!")
    print(f"{'='*60}")
    print("\nUsage in Candle:")
    print("  Place this file in your model weights directory.")
    print("  The Candle implementation will automatically load it")
    print("  for exact reproducibility with Python.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-computed random noise for CosyVoice3 Flow Matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script extracts the deterministic random noise used by CosyVoice3's
Flow Matching module. The noise is generated with a fixed seed (0) to ensure
reproducibility.

Note: This file is OPTIONAL for the Candle implementation. Without it,
the model will generate its own noise, which produces valid but slightly
different audio output.

Example:
    python extract_rand_noise.py \\
        --output weights/CosyVoice3-0.5B-2512-Candle/rand_noise.safetensors
        """
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for rand_noise.safetensors"
    )
    parser.add_argument(
        "--cosyvoice-path",
        default=None,
        help="Path to CosyVoice repository (optional, for exact seed behavior)"
    )
    parser.add_argument(
        "--save-numpy",
        action="store_true",
        help="Also save as .npy file for inspection"
    )
    
    args = parser.parse_args()
    extract_rand_noise(args.output, args.cosyvoice_path, args.save_numpy)


if __name__ == "__main__":
    main()
