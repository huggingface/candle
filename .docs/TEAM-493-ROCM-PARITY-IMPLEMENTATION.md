# TEAM-493: ROCm CUDA Parity Implementation

**Date:** 2025-11-13  
**Status:** 🚧 IN PROGRESS - Core infrastructure complete, BackendStorage implementation needed  
**Critical Review:** External developer requesting these features for rbee project

## Critical Analysis

### ✅ What's Been Done (Phases 1-2)
1. **Basic Infrastructure** (TEAM-488, TEAM-489, TEAM-492)
   - `RocmDevice` - Device management
   - `RocmStorageSlice` - Storage enum matching CUDA pattern
   - `RocmError` - Error handling
   - `kernels.rs` - Kernel loading infrastructure with Candle CUDA signature parity

2. **Kernel Operations** (TEAM-492)
   - `launch_unary` - Matches Candle CUDA signature `(numel, num_dims, info, inp, out)`
   - `launch_affine` - Matches Candle CUDA signature `(numel, num_dims, info, inp, out, mul, add)`
   - `launch_ternary` - Matches Candle CUDA signature `(numel, num_dims, info, ids, t, f, out)`
   - `SlicePtrOrNull` - Layout info handling (null for contiguous, pointer for strided)

### ❌ CRITICAL GAPS - Missing CUDA Parity

#### 1. **NO ROCm variant in Storage enum** ⚠️ BLOCKER
```rust
// candle-core/src/storage.rs
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    // MISSING: Rocm(RocmStorage),  ← NEEDS THIS!
}
```

#### 2. **NO RocmStorage struct** ⚠️ BLOCKER
CUDA has:
```rust
pub struct CudaStorage {
    slice: CudaStorageSlice,
    device: CudaDevice,
}
```

ROCm needs equivalent structure.

#### 3. **NO BackendStorage implementation** ⚠️ BLOCKER
CUDA implements ~30+ operations:
- `try_clone` - Clone tensors
- `dtype` - Get data type
- `const_set` - Fill with constant
- `to_dtype` - Cast operations
- `affine` - Affine transforms (y = mx + b)
- `powf` - Power operations
- `elu` - ELU activation
- `cmp` - Comparison operations
- `reduce_op` - Reductions (sum, max, min)
- `unary` - Unary ops (exp, log, sin, cos, sqrt, gelu, silu)
- `binary` - Binary ops (add, sub, mul, div)
- `where_cond` - Ternary conditional
- `conv1d`, `conv2d` - Convolutions
- `avg_pool2d`, `max_pool2d` - Pooling
- `upsample_nearest1d`, `upsample_nearest2d` - Upsampling
- `index_select`, `gather`, `scatter_add` - Indexing
- `index_add` - Index addition
- `matmul` - Matrix multiplication
- `copy2d`, `copy_strided_src` - Memory operations

**ROCm has NONE of these implemented yet.**

#### 4. **NO Map traits usage** ⚠️ BLOCKER
Created `utils.rs` with Map1, Map2, Map3 traits, but they're not used anywhere yet.

### 📊 Parity Status

| Component | CUDA | ROCm | Status |
|-----------|------|------|--------|
| Device management | ✅ | ✅ | **DONE** |
| Storage slice enum | ✅ | ✅ | **DONE** |
| Error handling | ✅ | ✅ | **DONE** |
| Kernel infrastructure | ✅ | ✅ | **DONE** |
| Map traits | ✅ | ✅ | **DONE** |
| Storage enum variant | ✅ | ❌ | **MISSING** |
| RocmStorage struct | ✅ | ❌ | **MISSING** |
| BackendStorage impl | ✅ | ❌ | **MISSING** |
| Unary operations | ✅ | ❌ | **MISSING** |
| Binary operations | ✅ | ❌ | **MISSING** |
| Ternary operations | ✅ | ❌ | **MISSING** |
| Reduction operations | ✅ | ❌ | **MISSING** |
| Convolution operations | ✅ | ❌ | **MISSING** |
| Pooling operations | ✅ | ❌ | **MISSING** |
| Indexing operations | ✅ | ❌ | **MISSING** |
| Matrix operations | ✅ | ❌ | **MISSING** |

**Parity Score: 5/17 (29%)** ⚠️

## What TEAM-493 Added

### 1. `utils.rs` - Map Traits for Type Dispatch
**File:** `candle-core/src/rocm_backend/utils.rs`  
**Purpose:** EXACT parity with `cuda_backend/utils.rs`

**Traits:**
- `Map1` - Unary operations (exp, log, sin, cos, sqrt, gelu, silu)
- `Map2` - Binary operations (add, sub, mul, div)
- `Map3` - Ternary operations (where/select)
- `Map2InPlace` - In-place binary operations
- `Map1Any` - Unary ops that change dtype (cast)
- `Map2Any` - Binary ops that change dtype

**Pattern:**
```rust
pub trait Map1 {
    fn f<T: WithDType>(&self, src: &DeviceMemory<T>, dev: &RocmDevice, layout: &Layout) -> Result<DeviceMemory<T>>;
    fn map(&self, s: &S, d: &RocmDevice, l: &Layout) -> Result<S> {
        // Dispatch to correct dtype variant
    }
}
```

### 2. Updated `mod.rs`
- Added `pub mod utils;`
- Added `pub type Result<T> = std::result::Result<T, RocmError>;`

### 3. Updated `error.rs`
- Added `KernelError(String)` variant for kernel launch failures

## Next Steps (Priority Order)

### Phase 3: Core Storage Implementation (CRITICAL)

1. **Add ROCm to Storage enum** (`storage.rs`)
   ```rust
   pub enum Storage {
       Cpu(CpuStorage),
       Cuda(CudaStorage),
       Metal(MetalStorage),
       Rocm(RocmStorage),  // ← ADD THIS
   }
   ```

2. **Create RocmStorage struct** (new file or in `mod.rs`)
   ```rust
   pub struct RocmStorage {
       slice: RocmStorageSlice,
       device: RocmDevice,
   }
   ```

3. **Implement BackendStorage trait**
   - Start with essential operations:
     - `try_clone`
     - `dtype`
     - `device`
     - `const_set`
     - `to_dtype` (cast)
   - Then add mathematical operations:
     - `affine` (y = mx + b)
     - `unary` (exp, log, sin, cos, sqrt, gelu, silu)
     - `binary` (add, sub, mul, div)
     - `where_cond` (ternary conditional)
     - `reduce_op` (sum, max, min)
   - Finally add advanced operations:
     - `matmul`
     - `conv1d`, `conv2d`
     - `avg_pool2d`, `max_pool2d`
     - Indexing operations

4. **Update all Storage match arms**
   - Every `match self` in `storage.rs` needs ROCm arm
   - ~50+ locations to update

### Phase 4: Kernel Implementations

1. **Verify rocm-rs kernels** (`deps/rocm-rs/src/rocarray/kernels.hip`)
   - ✅ Cast operations (FP16, FP32, FP64)
   - ✅ Where operations (ternary conditional)
   - ✅ Affine operations (y = mx + b)
   - ✅ Unary operations (exp, log, sin, cos, sqrt, gelu, silu)
   - ❌ Binary operations (add, sub, mul, div) - NEED WRAPPERS
   - ❌ Reduction operations - NEED WRAPPERS
   - ❌ Matrix operations - NEED WRAPPERS

2. **Add missing kernel wrappers** in `kernels.rs`
   - `launch_binary` - Binary operations
   - `launch_reduce` - Reduction operations
   - `launch_matmul` - Matrix multiplication

### Phase 5: Testing & Validation

1. **Unit tests** for each operation
2. **Integration tests** with Candle tensor operations
3. **Performance benchmarks** vs CUDA
4. **Memory leak checks**

## Files Modified (TEAM-493)

1. ✅ `candle-core/src/rocm_backend/utils.rs` - Created (Map traits)
2. ✅ `candle-core/src/rocm_backend/mod.rs` - Updated (added utils module)
3. ✅ `candle-core/src/rocm_backend/error.rs` - Updated (added KernelError)

## Files That NEED Modification (Next Team)

1. ❌ `candle-core/src/storage.rs` - Add ROCm variant to Storage enum
2. ❌ `candle-core/src/rocm_backend/mod.rs` - Create RocmStorage struct
3. ❌ `candle-core/src/rocm_backend/mod.rs` - Implement BackendStorage trait
4. ❌ `candle-core/src/lib.rs` - Export RocmStorage
5. ❌ `candle-core/src/device.rs` - Add ROCm device support

## Why This Matters

**For rbee project:**
- Enables AMD ROCm GPU support for tensor operations
- Maintains API parity with CUDA (no code changes needed)
- Allows heterogeneous hardware (NVIDIA + AMD + Apple Metal)
- Critical for "no vendor lock-in" positioning

**For Candle:**
- Expands hardware support to AMD GPUs
- Maintains consistent API across backends
- Enables ML workload portability

## Estimated Effort

- **Phase 3 (Core Storage):** 4-6 hours
- **Phase 4 (Kernel Wrappers):** 2-3 hours
- **Phase 5 (Testing):** 3-4 hours
- **Total:** 9-13 hours

## References

- CUDA backend: `candle-core/src/cuda_backend/mod.rs`
- Metal backend: `candle-core/src/metal_backend/mod.rs`
- ROCm kernels: `deps/rocm-rs/src/rocarray/kernels.hip`
- Candle CUDA kernels: `candle-kernels/src/cuda/*.cu`

---

**TEAM-493 Status:** Infrastructure complete, ready for BackendStorage implementation.  
**Next Team:** Implement RocmStorage struct and BackendStorage trait (Phase 3).
