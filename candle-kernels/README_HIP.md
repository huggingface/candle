# HIP Kernel Translation Guide

**Created by:** TEAM-488 (Phase 2)  
**Purpose:** Translate Candle's CUDA kernels to HIP for AMD GPU support

---

## Overview

This directory contains scripts to translate CUDA kernels to HIP and compile them to `.hsaco` binaries.

**Translation Tool:** `hipify-clang` (AMD's official CUDA→HIP translator)  
**Compilation Tool:** `hipcc` (HIP compiler)  
**Output Format:** `.hsaco` (HIP Code Object)

---

## Quick Start

### Prerequisites

1. **ROCm installed** (provides hipify-clang and hipcc)
   ```bash
   # Ubuntu/Debian
   sudo apt install rocm-hip-sdk
   
   # Or download from: https://rocm.docs.amd.com/
   ```

2. **Verify installation**
   ```bash
   hipify-clang --version
   hipcc --version
   ```

### Translation Workflow

```bash
# 1. Translate CUDA → HIP
./translate_to_hip.sh

# 2. Compile HIP → HSACO
./compile_kernels.sh

# 3. (Optional) Specify target GPU architecture
./compile_kernels.sh --arch gfx90a
```

---

## Directory Structure

```
candle-kernels/
├── src/
│   ├── *.cu              # Original CUDA kernels
│   ├── *.cuh             # CUDA headers
│   └── hip/              # Translated HIP kernels (generated)
│       ├── *.hip         # HIP kernel files
│       └── *.h           # HIP headers
├── hsaco/                # Compiled binaries (generated)
│   └── *.hsaco           # HIP code objects
├── translate_to_hip.sh   # CUDA → HIP translation script
├── compile_kernels.sh    # HIP → HSACO compilation script
└── README_HIP.md         # This file
```

---

## Kernel List

| Kernel | Size | Complexity | Status |
|--------|------|------------|--------|
| affine.cu | 1.7KB | Simple | ⏳ |
| fill.cu | 3.3KB | Simple | ⏳ |
| sort.cu | 2.6KB | Simple | ⏳ |
| ternary.cu | 2.6KB | Simple | ⏳ |
| binary.cu | 5.0KB | Medium | ⏳ |
| cast.cu | 7.9KB | Medium | ⏳ |
| unary.cu | 8.7KB | Medium | ⏳ |
| indexing.cu | 15KB | Complex | ⏳ |
| conv.cu | 24KB | Complex | ⏳ |
| reduce.cu | 25KB | Complex | ⏳ |
| quantized.cu | 158KB | Huge | ⏳ |

**Total:** 11 kernels, ~259KB of CUDA code

---

## AMD GPU Architectures

Common target architectures for `--arch` flag:

| Architecture | GPUs | Notes |
|--------------|------|-------|
| `gfx900` | Vega 10 (RX Vega 56/64) | Consumer |
| `gfx906` | Vega 20 (Radeon VII, MI50/60) | Prosumer |
| `gfx908` | CDNA1 (MI100) | Data center |
| `gfx90a` | CDNA2 (MI200 series) | Data center, BF16 |
| `gfx940` | CDNA3 (MI300 series) | Data center, FP8 |
| `gfx1030` | RDNA2 (RX 6000 series) | Consumer |
| `gfx1100` | RDNA3 (RX 7000 series) | Consumer |

**Default:** `gfx90a` (MI200 series - AWS g4ad instances)

---

## Translation Details

### What hipify-clang Does

1. **API Translation**
   - `cudaMalloc` → `hipMalloc`
   - `cudaMemcpy` → `hipMemcpy`
   - `cudaDeviceSynchronize` → `hipDeviceSynchronize`

2. **Type Translation**
   - `__nv_bfloat16` → `__hip_bfloat16`
   - `__half` → `_Float16`
   - CUDA intrinsics → HIP equivalents

3. **Header Translation**
   - `<cuda_runtime.h>` → `<hip/hip_runtime.h>`
   - `<cuda_fp16.h>` → `<hip/hip_fp16.h>`

### What Stays the Same

- `__global__` kernel syntax
- `blockIdx`, `threadIdx`, `blockDim`, `gridDim`
- Most device code (math, control flow)
- Kernel launch syntax (mostly)

---

## Compilation Flags

The `compile_kernels.sh` script uses:

```bash
hipcc --genco \                    # Generate code object
      --offload-arch=gfx90a \      # Target GPU
      -O3 \                        # Optimization
      -I src/hip \                 # Include path
      -ffast-math \                # Fast math
      -fgpu-rdc \                  # Relocatable device code
      kernel.hip -o kernel.hsaco
```

---

## Testing on Cloud

Since you don't have a local AMD GPU, use the cloud testing scripts:

```bash
cd /home/vince/Projects/rbee/scripts/rocm-cloud/aws

# 1. Setup AWS credentials
./setup.sh

# 2. Provision AMD GPU instance
./provision.sh

# 3. Deploy code and translate/compile kernels
./deploy.sh

# 4. SSH to instance and run translation
./ssh.sh
# On instance:
cd rbee/deps/candle/candle-kernels
./translate_to_hip.sh
./compile_kernels.sh

# 5. Cleanup when done
exit
./cleanup.sh
```

---

## Troubleshooting

### hipify-clang not found
```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# Or install ROCm
sudo apt install rocm-hip-sdk
```

### Compilation errors
- Check `/tmp/hipcc_*.log` for detailed errors
- Verify target architecture matches your GPU
- Some CUDA features may not have HIP equivalents (e.g., FP8 on older GPUs)

### Missing headers
```bash
# Install ROCm development headers
sudo apt install rocm-dev
```

---

## Next Steps

After successful compilation:

1. **Update build.rs** - Embed .hsaco files in Rust binary
2. **Create KernelCache** - Runtime kernel loading system
3. **Test kernels** - Verify execution on AMD GPU
4. **Integrate with Candle** - Connect to rocm_backend

---

## References

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [hipify-clang Documentation](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/)
- [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)

---

**Status:** Scripts ready, awaiting AMD GPU for execution  
**Next:** Run on AWS g4ad.xlarge instance
