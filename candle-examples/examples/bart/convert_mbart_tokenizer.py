#!/usr/bin/env python3
"""Convert mBART SentencePiece tokenizer to tokenizer.json format.

mBART models use SentencePiece tokenization which isn't directly supported
by the Rust tokenizers crate. This script converts the tokenizer to the
tokenizer.json format that can be loaded by the Rust example.

The script removes the language-specific post-processor so that source
language tokens can be handled dynamically at runtime.

Usage:
    python convert_mbart_tokenizer.py

    # Or specify a custom model:
    python convert_mbart_tokenizer.py --model-id facebook/mbart-large-50-many-to-many-mmt

    # Then run the BART example:
    cargo run --example bart --release -- \
        --model-id facebook/mbart-large-50-many-to-many-mmt \
        --prompt "Hello, how are you?" \
        --source-lang en_XX \
        --target-lang fr_XX

Requirements:
    pip install transformers sentencepiece protobuf
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert mBART tokenizer to tokenizer.json format"
    )
    parser.add_argument(
        "--model-id",
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: HuggingFace cache)",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers not installed.")
        print("Install with: pip install transformers sentencepiece protobuf")
        return 1

    print(f"Loading tokenizer from: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Is fast: {tokenizer.is_fast}")

    if not tokenizer.is_fast:
        print("\nWarning: Slow tokenizer, conversion may not produce tokenizer.json")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Save to HuggingFace cache directory
        from huggingface_hub import hf_hub_download

        cache_path = hf_hub_download(args.model_id, "config.json")
        output_dir = Path(cache_path).parent

    print(f"\nSaving tokenizer to: {output_dir}")
    tokenizer.save_pretrained(str(output_dir))

    # Modify tokenizer.json to remove the hardcoded language token from post-processor
    # This allows the Rust code to handle source language dynamically
    tokenizer_json = output_dir / "tokenizer.json"
    if tokenizer_json.exists():
        print("\nModifying post-processor for dynamic language handling...")
        with open(tokenizer_json) as f:
            data = json.load(f)

        # Replace the TemplateProcessing post-processor with a simpler one
        # that only adds </s> at the end (no language token)
        if data.get("post_processor", {}).get("type") == "TemplateProcessing":
            data["post_processor"] = {
                "type": "TemplateProcessing",
                "single": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "</s>", "type_id": 0}},
                ],
                "pair": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 0}},
                    {"SpecialToken": {"id": "</s>", "type_id": 0}},
                ],
                "special_tokens": {
                    "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
                },
            }
            with open(tokenizer_json, "w") as f:
                json.dump(data, f)
            print("  Removed hardcoded language token from post-processor")

        size_kb = tokenizer_json.stat().st_size / 1024
        print(f"\nSuccess! Created tokenizer.json ({size_kb:.1f} KB)")

        # Show language codes
        print("\nAvailable language codes:")
        lang_codes = list(tokenizer.lang_code_to_id.keys())
        for i in range(0, len(lang_codes), 10):
            print(f"  {', '.join(lang_codes[i:i+10])}")

        print(f"\nYou can now run:")
        print(f"  cargo run --example bart --release -- \\")
        print(f"      --model-id {args.model_id} \\")
        print(f"      --prompt \"Hello, how are you?\" \\")
        print(f"      --source-lang en_XX \\")
        print(f"      --target-lang fr_XX")
    else:
        print("\nError: tokenizer.json was not created")
        print("Files created:", list(output_dir.glob("*")))
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
