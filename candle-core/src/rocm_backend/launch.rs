//! Kernel naming, launch geometry, and the launch itself.
//!
//! Everything a launcher needs between having a slice and having queued work:
//! what the kernel is called for dtype `T`, how many blocks it should get, and
//! the call into the driver. There is one loading path ([`RocmDevice::get_or_load_func`])
//! and one launch-failure message, so a kernel that cannot be found or cannot be
//! launched reports the same way whichever entry point was used.
//!
//! Two entry points remain because the grids genuinely differ:
//! [`launch_kernel`] takes a grid the caller sized (usually from
//! [`launch_config`], which may cap it — sound only for the grid-stride
//! elementwise kernels), while [`launch_dense`] sizes its own from the output
//! element count and never caps.

use super::{kernels, RocmDevice, RocmError};
use crate::Result;

/// Roots whose F8E4M3 instantiation is spelled `fp8_e4m3` rather than
/// `f8_e4m3`.
///
/// candle-kernels is not self-consistent here: `unary.cu` and `ternary.cu` use
/// `fp8_e4m3` for everything except `ucopy_f8_e4m3`, while `binary.cu`,
/// `affine.cu`, `fill.cu`, `indexing.cu` and `cast.cu` use `f8_e4m3`. Nothing
/// in a root name says which, so the exceptions are listed. They are not
/// renamed in place: those strings are the CUDA backend's symbol table too, and
/// there is no NVIDIA toolchain here to verify a rename against.
///
/// `f8e4m3_spelling_matches_the_kernel_sources` below checks every entry (and
/// the absence of any other) against the shipped `.cu` text.
const FP8_SPELLED_ROOTS: &[&str] = &[
    "uabs",
    "uceil",
    "ucos",
    "uelu",
    "uerf",
    "uexp",
    "ufloor",
    "ugelu",
    "ugelu_erf",
    "ulog",
    "uneg",
    "unormcdf",
    "upowf",
    "urecip",
    "urelu",
    "uround",
    "usigmoid",
    "usign",
    "usilu",
    "usin",
    "usqr",
    "usqrt",
    "utanh",
    "where_i16",
    "where_i32",
    "where_i64",
    "where_u32",
    "where_u8",
];

/// Kernel-name suffix candle-kernels uses for the Rust type `T` in `kernel`.
///
/// `bf16` has to be probed before `f16` — `half::bf16`'s type name contains
/// both. F8E4M3 is the only dtype whose suffix depends on the kernel; see
/// [`FP8_SPELLED_ROOTS`].
fn dtype_suffix<T: Copy + Send + Sync + 'static>(kernel: &str) -> Option<&'static str> {
    let type_name = std::any::type_name::<T>();
    let suffix = if type_name.contains("F8E4M3") {
        if FP8_SPELLED_ROOTS.contains(&kernel) {
            "fp8_e4m3"
        } else {
            "f8_e4m3"
        }
    } else if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f64") {
        "f64"
    } else if type_name.contains("u8") {
        "u8"
    } else if type_name.contains("u32") {
        "u32"
    } else if type_name.contains("i64") {
        "i64"
    } else if type_name.contains("bf16") {
        "bf16"
    } else if type_name.contains("f16") {
        "f16"
    } else if type_name.contains("i16") {
        "i16"
    } else if type_name.contains("i32") {
        "i32"
    } else {
        return None;
    };
    Some(suffix)
}

/// Name of the `kernel` variant compiled for `T`, or an error when candle
/// ships no kernels for that dtype.
pub fn try_kernel_name<T: Copy + Send + Sync + 'static>(kernel: &str) -> Result<String> {
    match dtype_suffix::<T>(kernel) {
        Some(suffix) => Ok(format!("{}_{}", kernel, suffix)),
        None => Err(RocmError::Internal(format!(
            "unsupported dtype {} for kernel {}",
            std::any::type_name::<T>(),
            kernel
        ))
        .into()),
    }
}

/// Infallible variant kept for the public API used by candle-nn.
///
/// An unsupported dtype yields a name no module defines, so the caller gets a
/// "kernel not found" error at launch instead of aborting the process.
pub fn kernel_name<T: Copy + Send + Sync + 'static>(kernel: &str) -> String {
    let suffix = dtype_suffix::<T>(kernel).unwrap_or("unsupported_dtype");
    format!("{}_{}", kernel, suffix)
}

/// Grid and block dimensions for a grid-stride elementwise kernel.
///
/// Every elementwise kernel in `candle-kernels` is written as
///
/// ```text
/// for (i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x)
/// ```
///
/// so the grid may launch *fewer* threads than there are elements. A kernel
/// that instead maps one thread to one element and returns early would silently
/// leave the tail of its output untouched; size those by hand — `rope` in
/// `reduce.cu` is one, and everything in `conv.cu` goes through
/// [`utils::dense_grid`] for the same reason.
///
/// `contiguous` picks between two grids that measure very differently, and the
/// discriminator really is contiguity rather than size. On gfx1101 (60 CUs,
/// ms/iter, lower is better):
///
/// ```text
///                         16 blk/CU   256 blk/CU   one block per 256 elems
///   sqrt, 64M contiguous     1.76        1.53        0.96
///   relu, 64M transposed    10.6         6.97       55.9
///   relu,  4M transposed     0.31        0.69        1.10
///   sqrt,  4M contiguous     0.031       0.032       0.038
/// ```
///
/// A contiguous kernel streams, and wants every block it can get. A strided one
/// re-reads scattered cache lines, and a persistent grid keeps its stride small
/// enough that the lines a thread touches on consecutive iterations stay in L2 —
/// worth 3.5x at 4M elements and far more at 64M. `MAX_BLOCKS_PER_CU` sits at
/// the flat bottom of that curve.
///
/// The grid is never zero: `hipModuleLaunchKernel` rejects `gridDim.x == 0`, so
/// an elementwise op on an empty tensor would error instead of being a no-op.
/// The block count is also computed *before* narrowing to `u32` — casting first
/// makes any element count that is an exact multiple of 2^32 round to zero.
///
/// Nothing here caps the grid at 65535 any more. That was a CUDA
/// compute-capability-1.x limit inherited from `cuda_backend`; HIP's `gridDim.x`
/// limit is 2^32-1 on every supported AMD architecture, and the cap cost 16% on
/// a 64M-element contiguous unary.
pub fn launch_config_for(
    num_elems: usize,
    multiprocessor_count: u32,
    contiguous: bool,
) -> (rocm_rs::hip::Dim3, rocm_rs::hip::Dim3) {
    const BLOCK_SIZE: u32 = 256;
    /// 16 blocks x 256 threads = 4096 threads per CU, i.e. 4 waves per SIMD on
    /// RDNA3.
    const MAX_BLOCKS_PER_CU: u32 = 16;

    let num_blocks = num_elems.div_ceil(BLOCK_SIZE as usize);
    let max_blocks = if contiguous {
        u32::MAX
    } else {
        multiprocessor_count
            .saturating_mul(MAX_BLOCKS_PER_CU)
            .max(1)
    };
    let grid_dim = u32::try_from(num_blocks)
        .unwrap_or(u32::MAX)
        .clamp(1, max_blocks);
    (
        rocm_rs::hip::Dim3::from(grid_dim),
        rocm_rs::hip::Dim3::from(BLOCK_SIZE),
    )
}

/// [`launch_config_for`] sized to `dev`, for a kernel reading contiguously.
pub fn launch_config(
    dev: &RocmDevice,
    num_elems: usize,
) -> (rocm_rs::hip::Dim3, rocm_rs::hip::Dim3) {
    launch_config_for(num_elems, dev.multiprocessor_count(), true)
}

/// [`launch_config_for`] sized to `dev`, for a kernel reading through strides.
pub fn launch_config_layout(
    dev: &RocmDevice,
    num_elems: usize,
    contiguous: bool,
) -> (rocm_rs::hip::Dim3, rocm_rs::hip::Dim3) {
    launch_config_for(num_elems, dev.multiprocessor_count(), contiguous)
}

/// Grid and block for a kernel that maps one thread to one output element.
///
/// [`launch_config`] may cap the grid, which is only sound for the grid-stride
/// elementwise kernels. Everything in `conv.cu` instead returns early past the
/// end of its output, so a capped grid would silently leave the tail untouched;
/// this sizes the grid in full and errors rather than clamping.
pub(crate) fn dense_grid(dst_el: usize) -> Result<(rocm_rs::hip::Dim3, rocm_rs::hip::Dim3)> {
    const BLOCK: usize = 256;
    let blocks = u32::try_from(dst_el.div_ceil(BLOCK).max(1))
        .map_err(|_| RocmError::Internal(format!("output of {dst_el} elements is too large")))?;
    Ok((
        rocm_rs::hip::Dim3::from(blocks),
        rocm_rs::hip::Dim3::from(BLOCK as u32),
    ))
}

/// Launches `func_name` from `module` on `dev`'s stream.
///
/// # Safety
/// `args` must match the kernel's parameter list exactly, and `grid` must cover
/// every element the kernel writes unless the kernel is a grid-stride loop.
pub(super) unsafe fn launch_kernel(
    dev: &RocmDevice,
    module: &kernels::Module,
    func_name: &str,
    grid: rocm_rs::hip::Dim3,
    block: rocm_rs::hip::Dim3,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    // No lock is held across the launch: `get_or_load_func` releases the kernel
    // cache's own locks and hands back a plain `hipFunction_t`.
    let func = dev.get_or_load_func(func_name, module)?;
    func.launch(grid, block, 0, Some(dev.stream()), args)
        .map_err(|e| RocmError::Internal(format!("{func_name} launch failed: {e}")).into())
}

/// [`launch_kernel`] with the grid sized by [`dense_grid`] instead of by the
/// caller.
///
/// # Safety
/// As [`launch_kernel`].
pub(crate) unsafe fn launch_dense(
    dev: &RocmDevice,
    module: &kernels::Module,
    func_name: &str,
    dst_el: usize,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    let (grid, block) = dense_grid(dst_el)?;
    launch_kernel(dev, module, func_name, grid, block, args)
}

#[cfg(test)]
mod tests {
    use super::{launch_config_for, FP8_SPELLED_ROOTS};

    const CUS: u32 = 60;
    const BLOCK: u32 = 256;

    /// Roots that `sources` instantiates as `{root}{suffix}`, ignoring
    /// commented-out lines and the `__nv_fp8_e4m3` type name itself.
    fn roots_ending_in(suffix: &str) -> Vec<String> {
        use super::kernels as k;
        // Every shared translation unit; `Id` has no `Module` lookup.
        let modules = [
            &k::AFFINE,
            &k::BINARY,
            &k::CAST,
            &k::CONV,
            &k::FILL,
            &k::INDEXING,
            &k::QUANTIZED,
            &k::REDUCE,
            &k::SORT,
            &k::TERNARY,
            &k::UNARY,
        ];
        let mut found = vec![];
        for module in modules {
            for line in module.source().lines() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for (at, _) in line.match_indices(suffix) {
                    let head = &line[..at];
                    let start = head
                        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
                        .map_or(0, |i| i + 1);
                    let root = &head[start..];
                    // `__nv_fp8_e4m3` / `__hip_fp8_e4m3` are the type, not a kernel.
                    if root.is_empty() || root.starts_with("__") {
                        continue;
                    }
                    // Require a kernel name, not a longer identifier that merely
                    // contains the suffix (`cast_f8_e4m3_f32`).
                    if line[at + suffix.len()..]
                        .starts_with(|c: char| c.is_alphanumeric() || c == '_')
                    {
                        continue;
                    }
                    found.push(root.to_string());
                }
            }
        }
        found.sort();
        found.dedup();
        found
    }

    /// The `fp8_e4m3`/`f8_e4m3` split is an upstream inconsistency matched by
    /// hand, so it has to be checked against the sources rather than trusted.
    #[test]
    fn f8e4m3_spelling_matches_the_kernel_sources() {
        let irregular = roots_ending_in("_fp8_e4m3");
        assert_eq!(
            irregular,
            FP8_SPELLED_ROOTS
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            "FP8_SPELLED_ROOTS has drifted from candle-kernels"
        );
        // Nothing may be spelled both ways: the suffix has to be a function of
        // the root alone.
        for root in roots_ending_in("_f8_e4m3") {
            assert!(
                !FP8_SPELLED_ROOTS.contains(&root.as_str()),
                "{root} exists with both spellings"
            );
        }
    }

    /// `hipModuleLaunchKernel` rejects `gridDim.x == 0`, so an elementwise op on
    /// an empty tensor would error instead of being a no-op.
    #[test]
    fn the_grid_is_never_zero() {
        for contiguous in [true, false] {
            let (grid, block) = launch_config_for(0, CUS, contiguous);
            assert_eq!(grid.x, 1);
            assert_eq!(block.x, BLOCK);
        }
    }

    /// A contiguous kernel gets one block per `BLOCK` elements, with no cap.
    /// The 65535 this used to clamp to was a CUDA compute-capability-1.x limit;
    /// HIP allows 2^32-1, and the clamp cost 16% on a 64M-element unary.
    #[test]
    fn a_contiguous_grid_is_uncapped() {
        assert_eq!(launch_config_for(1, CUS, true).0.x, 1);
        assert_eq!(launch_config_for(1024, CUS, true).0.x, 4);
        // 64M elements: far past both 65535 blocks and the strided cap.
        assert_eq!(launch_config_for(64 << 20, CUS, true).0.x, (64 << 20) / 256);
    }

    /// A strided kernel gets a persistent-style grid instead, so that the
    /// scattered lines a thread revisits on consecutive grid-stride iterations
    /// stay close enough together to hit in L2.
    #[test]
    fn a_strided_grid_is_capped_to_the_machine() {
        // Below the cap the two agree.
        assert_eq!(launch_config_for(1024, CUS, false).0.x, 4);
        // Above it the strided grid saturates while the contiguous one does not.
        let big = 64 << 20;
        let strided = launch_config_for(big, CUS, false).0.x;
        assert_eq!(strided, 16 * CUS);
        assert!(strided < launch_config_for(big, CUS, true).0.x);
    }

    /// The block count is computed before narrowing to `u32`; casting first
    /// makes any element count that is an exact multiple of 2^32 round to zero.
    #[test]
    fn an_enormous_element_count_saturates_rather_than_wrapping() {
        let elems = (1usize << 32) * 256;
        assert_eq!(launch_config_for(elems, CUS, true).0.x, u32::MAX);
    }

    /// A device reporting no compute units must still get a launchable grid.
    #[test]
    fn a_zero_cu_device_still_gets_one_block() {
        assert_eq!(launch_config_for(1 << 20, 0, false).0.x, 1);
    }
}
