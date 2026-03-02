#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gpt_oss.torch.model import ModelConfig, Transformer
from gpt_oss.torch.weights import Checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Rust GPT-OSS logits dump against PyTorch reference logits."
    )
    p.add_argument(
        "--checkpoint",
        default="/tmp/gpt-oss-20b",
        help="Checkpoint directory used by both Rust and PyTorch.",
    )
    p.add_argument(
        "--rust-dump-dir",
        default="/tmp/gptoss_full_parity",
        help="Directory containing rust_logits_step*.bin and metadata.json.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu, mps, cuda:0, ...).",
    )
    p.add_argument(
        "--max-abs-thr",
        type=float,
        default=2e-1,
        help="Maximum allowed absolute difference per step.",
    )
    p.add_argument(
        "--mean-abs-thr",
        type=float,
        default=2e-2,
        help="Maximum allowed mean absolute difference per step.",
    )
    p.add_argument(
        "--summary-out",
        default=None,
        help="Optional JSON output path for parity summary.",
    )
    return p.parse_args()


def load_rust_logits(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    if arr.size == 0:
        raise ValueError(f"empty logits dump: {path}")
    return arr


def diff_stats(py_logits: np.ndarray, rust_logits: np.ndarray) -> tuple[float, float]:
    if py_logits.shape != rust_logits.shape:
        raise ValueError(f"shape mismatch: py={py_logits.shape} rust={rust_logits.shape}")
    diff = np.abs(py_logits - rust_logits)
    return float(diff.max()), float(diff.mean())


def load_transformer_filtered_config(checkpoint: Path, device: torch.device) -> Transformer:
    config_json = json.loads((checkpoint / "config.json").read_text())
    fields = set(ModelConfig.__dataclass_fields__.keys())
    cfg = {k: v for k, v in config_json.items() if k in fields}
    if "num_experts" not in cfg and "num_local_experts" in config_json:
        cfg["num_experts"] = config_json["num_local_experts"]
    if "experts_per_token" not in cfg and "num_experts_per_tok" in config_json:
        cfg["experts_per_token"] = config_json["num_experts_per_tok"]
    config = ModelConfig(**cfg)

    model = Transformer(config=config, device=device)
    model.eval()

    checkpoint_reader = Checkpoint(str(checkpoint), device)

    def load_param(name: str) -> torch.Tensor:
        if name == "embedding.weight":
            return checkpoint_reader._get_tensor("model.embed_tokens.weight")
        if name == "norm.scale":
            return checkpoint_reader._get_tensor("model.norm.weight")
        if name == "unembedding.weight":
            return checkpoint_reader._get_tensor("lm_head.weight")

        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "block":
            raise KeyError(f"unsupported model parameter name: {name}")
        layer = int(parts[1])
        prefix = f"model.layers.{layer}"

        if parts[2:5] == ["attn", "norm", "scale"]:
            return checkpoint_reader._get_tensor(f"{prefix}.input_layernorm.weight")
        if parts[2:5] == ["attn", "sinks"]:
            return checkpoint_reader._get_tensor(f"{prefix}.self_attn.sinks")
        if parts[2:5] == ["attn", "out", "weight"]:
            return checkpoint_reader._get_tensor(f"{prefix}.self_attn.o_proj.weight")
        if parts[2:5] == ["attn", "out", "bias"]:
            return checkpoint_reader._get_tensor(f"{prefix}.self_attn.o_proj.bias")
        if parts[2:5] == ["attn", "qkv", "weight"]:
            q = checkpoint_reader._get_tensor(f"{prefix}.self_attn.q_proj.weight")
            k = checkpoint_reader._get_tensor(f"{prefix}.self_attn.k_proj.weight")
            v = checkpoint_reader._get_tensor(f"{prefix}.self_attn.v_proj.weight")
            return torch.cat([q, k, v], dim=0)
        if parts[2:5] == ["attn", "qkv", "bias"]:
            q = checkpoint_reader._get_tensor(f"{prefix}.self_attn.q_proj.bias")
            k = checkpoint_reader._get_tensor(f"{prefix}.self_attn.k_proj.bias")
            v = checkpoint_reader._get_tensor(f"{prefix}.self_attn.v_proj.bias")
            return torch.cat([q, k, v], dim=0)

        if parts[2:5] == ["mlp", "norm", "scale"]:
            return checkpoint_reader._get_tensor(f"{prefix}.post_attention_layernorm.weight")
        if parts[2:5] == ["mlp", "gate", "weight"]:
            return checkpoint_reader._get_tensor(f"{prefix}.mlp.router.weight")
        if parts[2:5] == ["mlp", "gate", "bias"]:
            return checkpoint_reader._get_tensor(f"{prefix}.mlp.router.bias")
        if parts[2:5] == ["mlp", "mlp1_weight"]:
            return checkpoint_reader._get_mxfp4_tensor(
                f"{prefix}.mlp.experts.gate_up_proj_blocks",
                f"{prefix}.mlp.experts.gate_up_proj_scales",
                dtype=torch.bfloat16,
            )
        if parts[2:5] == ["mlp", "mlp1_bias"]:
            return checkpoint_reader._get_tensor(f"{prefix}.mlp.experts.gate_up_proj_bias")
        if parts[2:5] == ["mlp", "mlp2_weight"]:
            return checkpoint_reader._get_mxfp4_tensor(
                f"{prefix}.mlp.experts.down_proj_blocks",
                f"{prefix}.mlp.experts.down_proj_scales",
                dtype=torch.bfloat16,
            )
        if parts[2:5] == ["mlp", "mlp2_bias"]:
            return checkpoint_reader._get_tensor(f"{prefix}.mlp.experts.down_proj_bias")

        raise KeyError(f"unsupported model parameter name: {name}")

    for name, param in model.named_parameters():
        loaded = load_param(name)
        if loaded.dtype != param.dtype:
            loaded = loaded.to(param.dtype)
        if loaded.shape != param.shape:
            raise ValueError(
                f"shape mismatch for {name}: model {tuple(param.shape)} != checkpoint {tuple(loaded.shape)}"
            )
        param.data.copy_(loaded)
    return model


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    rust_dump = Path(args.rust_dump_dir)

    metadata_path = rust_dump / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata.json in {rust_dump}")
    metadata = json.loads(metadata_path.read_text())
    prompt_tokens = list(metadata["prompt_tokens"])
    fed_tokens = list(metadata["fed_tokens"])
    decode_steps = int(metadata["decode_steps"])

    if decode_steps != len(fed_tokens):
        raise ValueError(
            f"decode_steps ({decode_steps}) must equal len(fed_tokens) ({len(fed_tokens)})"
        )

    device = torch.device(args.device)
    print(f"loading PyTorch GPT-OSS model on {device} from {checkpoint}")
    model = load_transformer_filtered_config(checkpoint, device)

    seq = list(prompt_tokens)
    step_rows = []
    failures = []

    for step in range(decode_steps + 1):
        rust_logits_path = rust_dump / f"rust_logits_step{step}.bin"
        if not rust_logits_path.exists():
            raise FileNotFoundError(f"missing {rust_logits_path}")

        inp = torch.as_tensor(seq, dtype=torch.int32, device=device)
        py_logits = model(inp)[-1].to(torch.float32).cpu().numpy()
        rust_logits = load_rust_logits(rust_logits_path)

        max_abs, mean_abs = diff_stats(py_logits, rust_logits)
        py_top = int(np.argmax(py_logits))
        rust_top = int(np.argmax(rust_logits))
        top_match = py_top == rust_top

        row = {
            "step": step,
            "seq_len": len(seq),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "py_top": py_top,
            "rust_top": rust_top,
            "top_match": top_match,
        }
        step_rows.append(row)
        print(
            f"step={step} seq_len={len(seq)} max_abs={max_abs:.6f} "
            f"mean_abs={mean_abs:.6f} py_top={py_top} rust_top={rust_top}"
        )

        if (not top_match) or max_abs > args.max_abs_thr or mean_abs > args.mean_abs_thr:
            failures.append(row)

        if step < decode_steps:
            seq.append(int(fed_tokens[step]))

    summary = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "thresholds": {"max_abs": args.max_abs_thr, "mean_abs": args.mean_abs_thr},
        "steps": step_rows,
        "ok": len(failures) == 0,
        "failures": failures,
    }
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    if failures:
        print(f"parity FAILED with {len(failures)} failing step(s)")
        return 1
    print("parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
