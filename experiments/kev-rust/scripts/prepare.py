"""Download a pinned Kev-0.8B and export text-only merged fp32 safetensors for Rust.

Python/PyTorch is used only for checkpoint conversion and the reference comparison.
The Rust inference executable has no Python dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

KEV_REVISION = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    args = ap.parse_args()
    torch.set_num_threads(2)
    ck = Path(snapshot_download("jaredpalmer/kev-0.8b", revision=KEV_REVISION,
                              allow_patterns=["*.json", "*.safetensors", "head.pt"]))
    meta = torch.load(ck / "head.pt", map_location="cpu", weights_only=True)
    assert meta["base"] == "Qwen/Qwen3.5-0.8B-Base", meta["base"]
    assert not meta.get("option_isolation") and not meta.get("special_embeddings")
    base = Path(snapshot_download(meta["base"], revision=meta["base_revision"],
                                allow_patterns=["config.json", "tokenizer.json", "model*.safetensors"]))
    adapter_cfg = json.loads((ck / "adapter_config.json").read_text())
    assert not adapter_cfg.get("use_dora") and not adapter_cfg.get("use_rslora")
    assert not adapter_cfg.get("fan_in_fan_out") and not adapter_cfg.get("rank_pattern")
    adapter = load_file(ck / "adapter_model.safetensors")
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    weights = {}
    used = set()
    for shard in sorted(base.glob("model*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if not name.startswith("model.language_model."):
                    continue  # no vision tower, MTP, or vocabulary readout
                weight = f.get_tensor(name).float()
                # PEFT saves keys relative to the bare text backbone.
                prefix = "base_model.model." + name.removeprefix("model.language_model.").removesuffix(".weight")
                a, b = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
                if a in adapter:
                    assert b in adapter
                    weight += (adapter[b].float() @ adapter[a].float()) * scale
                    used.update([a, b])
                weights[name] = weight.contiguous()
    assert used == set(adapter), f"unmerged adapter keys: {set(adapter) - used}"
    assert weights, "text backbone not found"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(weights)} backbone tensors", flush=True)
    save_file(weights, args.out / "model.safetensors")
    save_file({k: v.float().contiguous() for k, v in meta["head"].items()}, args.out / "head.safetensors")
    for name in ["config.json", "tokenizer.json"]:
        (args.out / name).write_bytes((base / name).read_bytes())
    info = {k: meta.get(k) for k in ["base", "base_revision", "head_dim", "temperature", "lora", "option_isolation"]}
    info.update(kev_revision=KEV_REVISION, adapter_scale=scale, dtype="float32",
                kev_path=str(ck), base_path=str(base), merged_tensors=len(used) // 2,
                adapter_sha256=hashlib.sha256((ck / "adapter_model.safetensors").read_bytes()).hexdigest())
    (args.out / "metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
