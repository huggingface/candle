#!/usr/bin/env python3
"""
Reference script: run Granite 4.0 3B Vision in Python (transformers)
and dump intermediate activations for comparison with Candle.

Usage:
  python3 compare_ref.py --image /path/to/image.png --task tables_json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

MODEL_ID = "ibm-granite/granite-4.0-3b-vision"


def dump(name, t):
    """Print tensor stats and first few values."""
    if isinstance(t, torch.Tensor):
        t_f = t.float()
        flat = t_f.flatten()
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype} "
              f"min={flat.min().item():.6f} max={flat.max().item():.6f} "
              f"mean={flat.mean().item():.6f} first5={flat[:5].tolist()}")
    else:
        print(f"  {name}: {type(t)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", default="tables_json")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    model.eval()
    print("Model loaded.")

    # Build the chat prompt the same way the processor expects
    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")

    task_tag = f"<{args.task}>"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": task_tag},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(f"Prompt: {repr(prompt)}")

    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")

    print(f"\n=== INPUT ===")
    dump("input_ids", input_ids)
    print(f"  input_ids list: {input_ids[0].tolist()}")
    if pixel_values is not None:
        dump("pixel_values", pixel_values)

    # Hook into the model to capture intermediate states
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach()
            else:
                captured[name] = output.detach()
        return hook

    # Register hooks on key components
    hooks = []

    # Vision encoder output
    if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'vision_model'):
        vt = model.vision_tower.vision_model
        if hasattr(vt, 'encoder'):
            hooks.append(vt.encoder.register_forward_hook(make_hook("vision_encoder_out")))
        if hasattr(vt, 'embeddings'):
            hooks.append(vt.embeddings.register_forward_hook(make_hook("vision_embeddings")))

    # Language model embedding
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        lm = model.model.language_model
        if hasattr(lm, 'embed_tokens'):
            hooks.append(lm.embed_tokens.register_forward_hook(make_hook("text_embeddings")))
        # First few LM layers
        if hasattr(lm, 'layers'):
            for i in range(min(4, len(lm.layers))):
                hooks.append(lm.layers[i].register_forward_hook(make_hook(f"lm_layer_{i}")))
            # Also capture layers around injection points
            for i in [9, 12, 15, 18, 21, 39]:
                if i < len(lm.layers):
                    hooks.append(lm.layers[i].register_forward_hook(make_hook(f"lm_layer_{i}")))

    # Projectors
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layerwise_projectors'):
            for i, proj in enumerate(model.model.layerwise_projectors):
                hooks.append(proj.register_forward_hook(make_hook(f"layerwise_proj_{i}")))
        if hasattr(model.model, 'spatial_projectors'):
            for i, proj in enumerate(model.model.spatial_projectors):
                hooks.append(proj.register_forward_hook(make_hook(f"spatial_proj_{i}")))

    print(f"\n=== FORWARD PASS ===")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"\n=== CAPTURED ACTIVATIONS ===")
    for name in sorted(captured.keys()):
        dump(name, captured[name])

    # Decode output
    generated_ids = outputs[0][input_ids.shape[1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True)
    print(f"\n=== OUTPUT ===")
    print(f"Generated token IDs: {generated_ids.tolist()[:30]}")
    print(f"Generated text (first 500 chars): {text[:500]}")


if __name__ == "__main__":
    main()
