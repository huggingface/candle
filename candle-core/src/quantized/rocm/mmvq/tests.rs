//! Unit tests for the MMVQ launch tables.
//!
//! These are the pieces that have to agree with `quantized.cu` and with the
//! measurements in [`super`]; the on-device correctness tests live in
//! `quantized::rocm::tests_mmvq`.

use super::*;

/// The suffix is appended to the stem, so a stem that does not match
/// `quantized.cu` byte for byte resolves to no symbol at all — or, worse,
/// to a differently shaped one.
#[test]
fn kernel_stems_match_the_source() {
    assert_eq!(
        kernel_stem(GgmlDType::Q4_0),
        Some("mul_mat_vec_q4_0_q8_1_cuda")
    );
    // Upper-case K for the K-quants, lower case for the legacy quants.
    assert_eq!(
        kernel_stem(GgmlDType::Q4K),
        Some("mul_mat_vec_q4_K_q8_1_cuda")
    );
    assert_eq!(
        kernel_stem(GgmlDType::Q6K),
        Some("mul_mat_vec_q6_K_q8_1_cuda")
    );
    // No mmvq kernel is compiled for q8_K or the float dtypes.
    assert_eq!(kernel_stem(GgmlDType::Q8K), None);
    assert_eq!(kernel_stem(GgmlDType::F16), None);
}

/// Grid and block have to track the kernel's compile-time `nwarps` /
/// `rows_per_cuda_block` choice.
#[test]
fn launch_shape_matches_the_kernel_constants() {
    assert_eq!(launch_shape(4096, 1), Some((4096, 4)));
    assert_eq!(launch_shape(4096, 2), Some((2048, 4)));
    assert_eq!(launch_shape(4096, 4), Some((2048, 4)));
    assert_eq!(launch_shape(4096, 5), Some((2048, 2)));
    assert_eq!(launch_shape(4096, 8), Some((2048, 2)));
    assert_eq!(launch_shape(4096, 9), None);
    assert_eq!(launch_shape(4096, 0), None);
}

/// At a batch of one, DMMV is the cheaper path for the K-quants at any
/// size and for the legacy quants below [`MMVQ_MIN_WORK`]; see
/// [`dmmv_is_faster`] for the numbers.
#[test]
fn dmmv_wins_the_small_single_row_shapes() {
    // SmolLM2-360M's projections: far below the threshold.
    assert!(dmmv_is_faster(GgmlDType::Q8_0, 960 * 2560));
    assert!(dmmv_is_faster(GgmlDType::Q4_0, 2560 * 960));
    // A 7B-shaped projection and lm_head: MMVQ, for the legacy quants only.
    assert!(!dmmv_is_faster(GgmlDType::Q4_0, 4096 * 4096));
    assert!(!dmmv_is_faster(GgmlDType::Q8_0, 4096 * 32000));
    // The K-quant DMMV kernels are good enough to keep at every size.
    assert!(dmmv_is_faster(GgmlDType::Q4K, 4096 * 32000));
    assert!(dmmv_is_faster(GgmlDType::Q6K, 4096 * 32000));
    assert!(dmmv_is_faster(GgmlDType::Q2K, 960 * 960));
}

/// An odd `nrows` with a batch would have the last block write into the
/// next batch row's output; see [`supports`].
#[test]
fn odd_row_counts_are_rejected_only_for_batches() {
    assert!(supports(GgmlDType::Q4K, 4097, 4096, 1));
    assert!(!supports(GgmlDType::Q4K, 4097, 4096, 2));
    assert!(supports(GgmlDType::Q4K, 4096, 4096, 8));
    assert!(!supports(GgmlDType::Q4K, 4096, 4096, 9));
    // `ncols` must be whole blocks: the kernel divides by `qk`.
    assert!(!supports(GgmlDType::Q4K, 4096, 128, 1));
    assert!(supports(GgmlDType::Q4_0, 4096, 128, 1));
    // No kernel for q8_K.
    assert!(!supports(GgmlDType::Q8K, 4096, 4096, 1));
}
