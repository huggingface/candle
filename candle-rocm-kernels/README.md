# candle-rocm-kernels

ROCm/HIP kernel support for the Candle deep learning framework.

## Overview

This crate provides ROCm (AMD GPU) kernel support for Candle. Unlike CUDA which can embed PTX directly, ROCm/HIP requires ahead-of-time (AOT) compilation for specific GPU architectures.

## Architecture

### AOT Cache System

We use an **Ahead-of-Time (AOT) compilation cache** approach:

1. **Source Code**: HIP kernels are shipped as source code (`.hip` files)
2. **On-Demand Compilation**: First time a kernel is needed, it's compiled using `hipcc`
3. **Caching**: Compiled binaries are cached for reuse
4. **Future Runs**: Load cached binaries directly (no recompilation)

### Cache Location

Compiled binaries are stored at:

```
~/.cache/candle-rocm/{arch}-{rocm_version}/
```

For example:
- `~/.cache/candle-rocm/gfx908-6.1/binary_a1b2c3d4.cso`
- `~/.cache/candle-rocm/gfx942-6.2/binary_a1b2c3d4.cso`

Where:
- `{arch}` = GPU architecture (gfx908, gfx90a, gfx942, etc.)
- `{rocm_version}` = ROCm version (6.0, 6.1, 6.2, etc.)
- `{hash}` = SHA256 hash of source code (first 16 chars)

### Key Components

#### CacheManager (`src/cache.rs`)

Manages the AOT compilation cache:

- **GPU Detection**: Automatically detects GPU architecture using `rocminfo` or falls back to environment variable `CANDLE_ROCM_ARCH`
- **Version Detection**: Detects ROCm version using `hipcc --version` or environment variable `CANDLE_ROCM_VERSION`
- **Compilation**: Invokes `hipcc` with appropriate flags:
  ```bash
  hipcc --offload-arch={arch} -O3 -fPIC -c -o output.o input.hip
  ```
- **Caching**: Stores compiled `.cso` (code object) files with source hash versioning

Usage:
```rust
use candle_rocm_kernels::CacheManager;
use rocm_rs::hip::Device;

let device = Device::new(0)?;
let cache = CacheManager::new(&device)?;
let binary = cache.get_or_compile("binary", source_code)?;
```

#### KernelManager (`src/manager.rs`)

Higher-level manager that:

- Wraps CacheManager
- Loads compiled binaries as `rocm_rs::hip::Module`
- Returns `Arc<Module>` for thread-safe sharing
- Maintains in-memory module cache

Usage:
```rust
use candle_rocm_kernels::KernelManager;
use candle_rocm_kernels::source::Source;

let device = Device::new(0)?;
let manager = KernelManager::new(&device)?;
let module = manager.get_or_compile_module(Source::Binary)?;
```

### Environment Variables

- `CANDLE_ROCM_ARCH` - Override GPU architecture detection (e.g., "gfx908")
- `CANDLE_ROCM_VERSION` - Override ROCm version detection (e.g., "6.1")

## Requirements

- ROCm/HIP installed (provides `hipcc`)
- AMD GPU with supported architecture

## Kernel Types

Currently supports:
- **Binary operations**: Add, Sub, Mul, Div, Minimum, Maximum

## Building

```bash
cd candle-rocm-kernels
cargo build
```

Note: First build will compile dependencies. No GPU required for building, but `hipcc` must be in PATH if you want to compile kernels.

## Testing

```bash
cargo test
```

## Implementation Notes

### Why AOT instead of JIT?

The `rocm-rs` crate (v0.5) doesn't support runtime compilation. It only supports:
- `Module::load(path)` - Load from file
- `Module::load_data(bytes)` - Load from bytes

This makes JIT compilation (via hiprtc) unavailable, so we compile ahead-of-time on first run.

### Supported GPU Architectures

Common AMD GPU architectures:
- CDNA2: gfx90a (MI200 series)
- CDNA3: gfx942 (MI300 series)
- RDNA3: gfx1100, gfx1101, gfx1102 (RX 7000 series)

The system will try to auto-detect, but you can override with `CANDLE_ROCM_ARCH`.

## License

MIT OR Apache-2.0
