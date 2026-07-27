# Vendored Marlin kernel

`marlin_cuda_kernel.cu` is vendored verbatim from the Marlin project:

- Source: https://github.com/IST-DASLab/marlin (`marlin/marlin_cuda_kernel.cu`)
- Author: Elias Frantar (IST-DASLab)
- License: Apache License 2.0 (see `LICENSE` in this directory)

It is the real Marlin FP16xINT4 tensor-core GEMM kernel; this crate compiles it and drives it
over FFI (`marlin_shim.cu` -> `run_marlin_gemm`, bound in `src/ffi.rs` / `src/marlin.rs`) rather
than reimplementing it.

Not vendored: upstream's `marlin/marlin_cuda.cpp` (a torch/PyBind wrapper) and `marlin/__init__.py`
(the Python repack). The repack is reimplemented host-side in Rust in `src/marlin.rs`, ported from
`__init__.py`'s `Layer.pack` / `_get_perms`.

Constraints inherited from the upstream kernel: 4-bit, symmetric (no per-output zero point), FP16
operands, group size 128 or -1 (per-channel), no act-order (`g_idx` must be sequential), and
`infeatures % 128 == 0`, `outfeatures % 256 == 0`.
