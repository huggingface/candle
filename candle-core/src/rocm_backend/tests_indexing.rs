//! Unit tests for the indexing launchers in `ops_indexing.rs`.
//!
//! The index dtype and the value dtype are dispatched separately, so every test
//! that resolves a kernel name walks both axes of that matrix. The offset tests
//! guard the other recurring failure mode: `Layout::start_offset` counts
//! elements, while raw pointer arithmetic on a `*mut c_void` counts bytes.

use super::tests::rocm_device;
use crate::{DType, Result, Tensor};

/// Index dtypes candle instantiates indexing kernels for.
const ID_DTYPES: [DType; 3] = [DType::U8, DType::U32, DType::I64];
/// Value dtypes the indexing kernels are instantiated for and `to_dtype`
/// supports on ROCm.
const VALUE_DTYPES: [DType; 7] = [
    DType::U8,
    DType::U32,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
];

/// `IS_OP` names its kernels `is_{index}_{value}`: the two dtypes are dispatched
/// independently, so the whole matrix has to resolve.
#[test]
fn index_select_dispatches_index_and_value_dtypes() -> Result<()> {
    let dev = rocm_device!();
    let values = Tensor::new(&[[0f64, 1., 2.], [3., 4., 5.], [6., 7., 8.]], &dev)?;
    let ids = Tensor::new(&[2u32, 0, 1, 0], &dev)?;
    for id_dtype in ID_DTYPES {
        let ids = ids.to_dtype(id_dtype)?;
        for value_dtype in VALUE_DTYPES {
            let out = values.to_dtype(value_dtype)?.index_select(&ids, 0)?;
            assert_eq!(out.dtype(), value_dtype);
            assert_eq!(
                out.to_dtype(DType::F64)?.to_vec2::<f64>()?,
                [[6., 7., 8.], [0., 1., 2.], [3., 4., 5.], [0., 1., 2.]],
                "ids {id_dtype:?} values {value_dtype:?}"
            );
        }
    }
    // Selecting along the last dim makes `left_size` and `right_size` both
    // non-trivial, which a dim-0 select never exercises.
    let ids = Tensor::new(&[2u32, 0], &dev)?;
    assert_eq!(
        values.index_select(&ids, 1)?.to_vec2::<f64>()?,
        [[2., 0.], [5., 3.], [8., 6.]]
    );
    Ok(())
}

/// Both the source and the index view carry element offsets that have to be
/// scaled by the dtype width, not applied as raw byte arithmetic.
#[test]
fn index_select_handles_start_offsets() -> Result<()> {
    let dev = rocm_device!();
    let base = Tensor::new(
        &[[0f64, 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 10., 11.]],
        &dev,
    )?;
    // Contiguous view starting at element 3.
    let src = base.narrow(0, 1, 3)?;
    let ids = Tensor::new(&[7u32, 0, 2], &dev)?.narrow(0, 1, 2)?;
    assert_eq!(
        src.index_select(&ids, 0)?.to_vec2::<f64>()?,
        [[3., 4., 5.], [9., 10., 11.]]
    );
    // Same, one byte per element, so a byte/element mix-up would still land
    // inside the allocation and read the wrong rows rather than fault.
    let src = base.to_dtype(DType::U8)?.narrow(0, 1, 3)?;
    assert_eq!(
        src.index_select(&ids, 0)?.to_vec2::<u8>()?,
        [[3, 4, 5], [9, 10, 11]]
    );
    Ok(())
}

/// `GATHER_OP` also crosses the index dtype with the value dtype.
#[test]
fn gather_dispatches_index_and_value_dtypes() -> Result<()> {
    let dev = rocm_device!();
    let values = Tensor::new(&[[0f64, 1., 2.], [3., 4., 5.]], &dev)?;
    let ids = Tensor::new(&[[2u32, 1, 0], [0, 0, 2]], &dev)?;
    for id_dtype in ID_DTYPES {
        let ids = ids.to_dtype(id_dtype)?;
        for value_dtype in VALUE_DTYPES {
            let out = values.to_dtype(value_dtype)?.gather(&ids, 1)?;
            assert_eq!(out.dtype(), value_dtype);
            assert_eq!(
                out.to_dtype(DType::F64)?.to_vec2::<f64>()?,
                [[2., 1., 0.], [3., 3., 5.]],
                "ids {id_dtype:?} values {value_dtype:?}"
            );
        }
    }
    // Gathering along dim 0 gives `right_size == 3` instead of 1.
    let ids = Tensor::new(&[[1u32, 0, 1]], &dev)?;
    assert_eq!(values.gather(&ids, 0)?.to_vec2::<f64>()?, [[3., 1., 5.]]);
    Ok(())
}

#[test]
fn gather_handles_start_offsets() -> Result<()> {
    let dev = rocm_device!();
    let base = Tensor::new(&[[9f64, 9., 9.], [0., 1., 2.], [3., 4., 5.]], &dev)?;
    let values = base.narrow(0, 1, 2)?;
    let ids = Tensor::new(&[[9u32, 9, 9], [2, 1, 0], [0, 0, 2]], &dev)?.narrow(0, 1, 2)?;
    assert_eq!(
        values.gather(&ids, 1)?.to_vec2::<f64>()?,
        [[2., 1., 0.], [3., 3., 5.]]
    );
    Ok(())
}

/// `S_OP` overwrites and `SA_OP` accumulates; both take the index dtype in the
/// kernel-name prefix.
#[test]
fn scatter_dispatches_index_and_value_dtypes() -> Result<()> {
    let dev = rocm_device!();
    let ids = Tensor::new(&[[0u32, 2], [1, 0]], &dev)?;
    let src = Tensor::new(&[[1f64, 2.], [3., 4.]], &dev)?;
    for id_dtype in ID_DTYPES {
        let ids = ids.to_dtype(id_dtype)?;
        for value_dtype in VALUE_DTYPES {
            let dst = Tensor::ones((2, 3), value_dtype, &dev)?;
            let src = src.to_dtype(value_dtype)?;
            assert_eq!(
                dst.scatter(&ids, &src, 1)?
                    .to_dtype(DType::F64)?
                    .to_vec2::<f64>()?,
                [[1., 1., 2.], [4., 3., 1.]],
                "scatter ids {id_dtype:?} values {value_dtype:?}"
            );
            assert_eq!(
                dst.scatter_add(&ids, &src, 1)?
                    .to_dtype(DType::F64)?
                    .to_vec2::<f64>()?,
                [[2., 1., 3.], [5., 4., 1.]],
                "scatter-add ids {id_dtype:?} values {value_dtype:?}"
            );
        }
    }
    Ok(())
}

/// `scatter_set` writes through the destination *view*, so its start offset has
/// to reach the kernel.
#[test]
fn scatter_set_honours_the_destination_offset() -> Result<()> {
    let dev = rocm_device!();
    let base = Tensor::zeros((3, 3), DType::F32, &dev)?;
    let dst = base.narrow(0, 1, 2)?;
    let ids = Tensor::new(&[[0u32, 2], [1, 0]], &dev)?;
    let src = Tensor::new(&[[1f32, 2.], [3., 4.]], &dev)?;
    dst.scatter_set(&ids, &src, 1)?;
    assert_eq!(
        base.to_vec2::<f32>()?,
        [[0., 0., 0.], [1., 0., 2.], [4., 3., 0.]]
    );

    let base = Tensor::ones((3, 3), DType::F32, &dev)?;
    let dst = base.narrow(0, 1, 2)?;
    dst.scatter_add_set(&ids, &src, 1)?;
    assert_eq!(
        base.to_vec2::<f32>()?,
        [[1., 1., 1.], [2., 1., 3.], [5., 4., 1.]]
    );
    Ok(())
}

/// `IA_OP` takes `ids_dim_size` right after `ids`, unlike `S_OP`/`SA_OP`.
#[test]
fn index_add_dispatches_index_and_value_dtypes() -> Result<()> {
    let dev = rocm_device!();
    let values = Tensor::new(&[[1f64, 2., 3.], [4., 5., 6.]], &dev)?;
    let ids = Tensor::new(&[0u32, 2], &dev)?;
    let src = Tensor::new(&[[10f64, 20.], [30., 40.]], &dev)?;
    for id_dtype in ID_DTYPES {
        let ids = ids.to_dtype(id_dtype)?;
        for value_dtype in VALUE_DTYPES {
            // f16 has 10 bits of mantissa; keep the operands exactly
            // representable so the comparison stays a correctness check.
            let out =
                values
                    .to_dtype(value_dtype)?
                    .index_add(&ids, &src.to_dtype(value_dtype)?, 1)?;
            assert_eq!(out.dtype(), value_dtype);
            assert_eq!(
                out.to_dtype(DType::F64)?.to_vec2::<f64>()?,
                [[11., 2., 23.], [34., 5., 46.]],
                "ids {id_dtype:?} values {value_dtype:?}"
            );
        }
    }
    Ok(())
}

/// `index_add` copies `self` into a fresh accumulator before launching, so the
/// launcher must be told the accumulator is contiguous from element 0 — passing
/// the original layout would re-apply this view's start offset to a buffer that
/// does not have one.
#[test]
fn index_add_handles_a_source_with_a_start_offset() -> Result<()> {
    let dev = rocm_device!();
    let base = Tensor::new(&[[9f32, 9., 9.], [1., 2., 3.], [4., 5., 6.]], &dev)?;
    let values = base.narrow(0, 1, 2)?;
    let ids = Tensor::new(&[0u32, 2], &dev)?;
    let src = Tensor::new(&[[10f32, 20.], [30., 40.]], &dev)?;
    assert_eq!(
        values.index_add(&ids, &src, 1)?.to_vec2::<f32>()?,
        [[11., 2., 23.], [34., 5., 46.]]
    );
    Ok(())
}

/// The indexing kernels all assume contiguous buffers, so a strided operand has
/// to be rejected rather than silently read along the wrong strides.
#[test]
fn indexing_rejects_non_contiguous_operands() -> Result<()> {
    let dev = rocm_device!();
    let values = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.]], &dev)?;
    let strided = values.t()?;
    let ids = Tensor::new(&[1u32, 0], &dev)?;
    assert!(strided.index_select(&ids, 0).is_err());

    let gather_ids = Tensor::new(&[[1u32, 0], [0, 1], [1, 1]], &dev)?;
    assert!(strided.gather(&gather_ids, 1).is_err());

    // A strided *index* buffer is rejected too, over a contiguous source.
    let src = Tensor::zeros((3, 3), DType::F32, &dev)?;
    let strided_ids = Tensor::new(&[[1u32, 0, 1], [0, 1, 0]], &dev)?.t()?;
    assert!(src.gather(&strided_ids, 1).is_err());
    Ok(())
}
