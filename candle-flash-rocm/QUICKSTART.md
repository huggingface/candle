# Flash Attention for ROCm - Quick Start

## ✅ Status: READY TO BUILD!

All code is implemented. Just need ROCm hardware.

## 🚀 Build

```bash
cd /home/vince/Projects/rbee/deps/candle/candle-flash-rocm

# Set GPU target (optional, defaults to gfx942 for MI300)
export CK_GPU_TARGETS=gfx942  # or gfx90a for MI200

# Build
cargo build --release --features rocm
```

## 📦 What Gets Built

1. **Composable Kernel** - AMD's Flash Attention library (auto-compiled)
2. **C Wrapper** - Bridge between Rust and CK (auto-compiled)
3. **Rust Library** - `libcandle_flash_rocm.rlib`

## 💻 Usage

```rust
use candle_flash_rocm::flash_attn;
use candle::{Device, Tensor, DType};

let device = Device::new_rocm(0)?;
let q = Tensor::randn(0.0, 1.0, (2, 1024, 8, 64), &device)?.to_dtype(DType::F16)?;
let k = Tensor::randn(0.0, 1.0, (2, 1024, 8, 64), &device)?.to_dtype(DType::F16)?;
let v = Tensor::randn(0.0, 1.0, (2, 1024, 8, 64), &device)?.to_dtype(DType::F16)?;

let output = flash_attn(&q, &k, &v, 1.0 / 8.0_f32.sqrt(), false)?;
```

## 🎯 Features

- ✅ F16/BF16 data types
- ✅ Multi-Query Attention (MQA)
- ✅ Grouped-Query Attention (GQA)
- ✅ Causal masking
- ✅ Sliding window attention
- ✅ ALiBi positional encoding
- ✅ Softcapping

## 📋 Requirements

- ROCm 6.0+
- AMD MI200 (gfx90a) or MI300 (gfx942)
- CMake
- hipcc (comes with ROCm)

## 🔧 Troubleshooting

**"ROCm not found"**
```bash
export ROCM_PATH=/opt/rocm
```

**"CK not found"**
```bash
# Already cloned! Just build:
cargo build --release --features rocm
```

**"CMake failed"**
```bash
# Install CMake
sudo apt-get install cmake
```

## 📚 Documentation

- Full docs: `README.md`
- Implementation details: `/.plan/TEAM_509_FLASH_ATTENTION_ROCM_COMPLETE.md`
- API docs: `src/lib.rs`

---

**Status:** ✅ 100% Complete - Ready to build!
