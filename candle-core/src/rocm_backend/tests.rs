//! Unit tests for the ROCm launcher glue.
//!
//! Every test resolves the device itself and returns early when the machine has
//! no ROCm GPU, so `cargo test -p candle-core --features rocm` stays runnable on
//! CPU-only hosts.

use crate::{DType, Device, IndexOp, Result, Tensor, D};

/// `None` when no ROCm device is available.
pub(super) fn device() -> Option<Device> {
    Device::new_rocm(0).ok()
}

macro_rules! rocm_device {
    () => {
        match crate::rocm_backend::tests::device() {
            Some(dev) => dev,
            None => return Ok(()),
        }
    };
}
pub(super) use rocm_device;

/// `fast_argmin`/`fast_argmax` write `uint32_t`, never the input dtype. With an
/// f64 input the old code allocated `dst_el` f64s and tagged the result F64,
/// which both mis-reported the dtype and read 8 bytes per index.
#[test]
fn argmin_argmax_are_u32_for_f64() -> Result<()> {
    let dev = rocm_device!();
    let t = Tensor::new(&[[3f64, 1., 4., 1.], [5., 9., 2., 6.]], &dev)?;

    let argmax = t.argmax(D::Minus1)?;
    assert_eq!(argmax.dtype(), DType::U32);
    assert_eq!(argmax.to_vec1::<u32>()?, [2, 1]);

    let argmin = t.argmin(D::Minus1)?;
    assert_eq!(argmin.dtype(), DType::U32);
    assert_eq!(argmin.to_vec1::<u32>()?, [1, 2]);
    Ok(())
}

/// The value-returning reductions keep the input dtype.
#[test]
fn min_max_sum_keep_the_input_dtype() -> Result<()> {
    let dev = rocm_device!();
    let t = Tensor::new(&[[3f64, 1., 4., 1.], [5., 9., 2., 6.]], &dev)?;
    assert_eq!(t.min(D::Minus1)?.dtype(), DType::F64);
    assert_eq!(t.min(D::Minus1)?.to_vec1::<f64>()?, [1., 2.]);
    assert_eq!(t.max(D::Minus1)?.to_vec1::<f64>()?, [4., 9.]);
    assert_eq!(t.sum(D::Minus1)?.to_vec1::<f64>()?, [9., 22.]);
    Ok(())
}

/// Reducing an empty tensor with min/max/argmin/argmax has no answer; the guard
/// has to reject it rather than launch a kernel over zero blocks.
#[test]
fn reducing_an_empty_tensor_errors() -> Result<()> {
    let dev = rocm_device!();
    let t = Tensor::zeros((0, 3), DType::F32, &dev)?;
    assert!(t.min(0).is_err());
    assert!(t.max(0).is_err());
    assert!(t.argmin(0).is_err());
    assert!(t.argmax(0).is_err());
    // Summing nothing is well defined, so that one still works.
    assert_eq!(t.sum(0)?.to_vec1::<f32>()?, [0., 0., 0.]);
    Ok(())
}

/// `BINARY_OP_OUT` yields u8 for every input dtype.
#[test]
fn cmp_output_is_u8_for_every_input_dtype() -> Result<()> {
    let dev = rocm_device!();
    for dtype in [DType::U8, DType::U32, DType::I64, DType::F32, DType::F64] {
        let lhs = Tensor::new(&[1f64, 2., 3.], &dev)?.to_dtype(dtype)?;
        let rhs = Tensor::new(&[3f64, 2., 1.], &dev)?.to_dtype(dtype)?;
        let out = lhs.lt(&rhs)?;
        assert_eq!(out.dtype(), DType::U8, "lt on {dtype:?}");
        assert_eq!(out.to_vec1::<u8>()?, [1, 0, 0], "lt on {dtype:?}");
        assert_eq!(lhs.ge(&rhs)?.to_vec1::<u8>()?, [0, 1, 1], "ge on {dtype:?}");
    }
    Ok(())
}

/// The strided branch of `BINARY_OP_OUT` needs dims + both stride vectors, and
/// the start offsets have to be applied in elements.
#[test]
fn cmp_handles_non_contiguous_operands() -> Result<()> {
    let dev = rocm_device!();
    let base = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.]], &dev)?;
    // `t()` leaves both operands strided; the narrow adds a start offset.
    let lhs = base.t()?;
    let rhs = Tensor::new(&[[0f32, 3.], [9., 4.], [2., 9.]], &dev)?;
    assert_eq!(
        lhs.eq(&rhs)?.to_vec2::<u8>()?,
        [[1, 1], [0, 1], [1, 0]],
        "strided lhs"
    );

    let offset = base.flatten_all()?.narrow(0, 2, 3)?;
    let plain = Tensor::new(&[2f32, 9., 4.], &dev)?;
    assert_eq!(offset.eq(&plain)?.to_vec1::<u8>()?, [1, 0, 1]);
    Ok(())
}

/// The predicate dtype and the value dtype are dispatched independently, which
/// is what the `where_{ids}_{dtype}` kernel names encode.
#[test]
fn where_cond_dispatches_predicate_and_value_dtypes() -> Result<()> {
    let dev = rocm_device!();
    let pred = Tensor::new(&[1u8, 0, 1, 0], &dev)?;
    // Kept positive so the same expectation holds for the unsigned dtypes too.
    let t = Tensor::new(&[1f64, 2., 3., 4.], &dev)?;
    let f = Tensor::new(&[10f64, 20., 30., 40.], &dev)?;

    for pred_dtype in [DType::U8, DType::U32, DType::I64] {
        let pred = pred.to_dtype(pred_dtype)?;
        for value_dtype in [DType::U8, DType::U32, DType::I64, DType::F32, DType::F64] {
            let out = pred.where_cond(&t.to_dtype(value_dtype)?, &f.to_dtype(value_dtype)?)?;
            assert_eq!(out.dtype(), value_dtype);
            assert_eq!(
                out.to_dtype(DType::F64)?.to_vec1::<f64>()?,
                [1., 20., 3., 40.],
                "pred {pred_dtype:?} values {value_dtype:?}"
            );
        }
    }
    Ok(())
}

/// `WHERE_OP` dereferences its info buffer unconditionally, and takes a separate
/// stride vector for the predicate, `t` and `f`.
#[test]
fn where_cond_handles_non_contiguous_operands() -> Result<()> {
    let dev = rocm_device!();
    let pred = Tensor::new(&[[1u8, 0], [0, 1], [1, 1]], &dev)?;
    let t = Tensor::new(&[[10f32, 30., 50.], [20., 40., 60.]], &dev)?.t()?;
    let f = Tensor::new(&[[-1f32, -3., -5.], [-2., -4., -6.]], &dev)?.t()?;
    assert_eq!(
        pred.where_cond(&t, &f)?.to_vec2::<f32>()?,
        [[10., -2.], [-3., 40.], [50., 60.]]
    );

    // Strided predicate over contiguous values.
    let pred = Tensor::new(&[[1u8, 0, 1], [0, 1, 0]], &dev)?.t()?;
    let t = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &dev)?;
    let f = Tensor::new(&[[-1f32, -2.], [-3., -4.], [-5., -6.]], &dev)?;
    assert_eq!(
        pred.where_cond(&t, &f)?.to_vec2::<f32>()?,
        [[1., -2.], [-3., 4.], [5., -6.]]
    );
    Ok(())
}

/// `const_set_f8_e4m3` and the F8E4M3 host-to-device path both treat the storage
/// as raw bytes; a round trip proves the byte view is not shifted or truncated.
#[test]
fn f8e4m3_const_set_and_host_roundtrip() -> Result<()> {
    use float8::F8E4M3;

    let dev = rocm_device!();
    let ones = Tensor::ones((2, 3), DType::F8E4M3, &dev)?;
    assert_eq!(
        ones.to_vec2::<F8E4M3>()?,
        [[F8E4M3::from_f32(1.); 3], [F8E4M3::from_f32(1.); 3]]
    );

    let values: Vec<F8E4M3> = [0.5f32, 1., 2., 4.]
        .iter()
        .map(|v| F8E4M3::from_f32(*v))
        .collect();
    let t = Tensor::new(values.as_slice(), &dev)?;
    assert_eq!(t.dtype(), DType::F8E4M3);
    assert_eq!(t.to_vec1::<F8E4M3>()?, values);

    // A strided const_set has to reach exactly the elements the view covers, so
    // it exercises the `get_strided_index` branch of `CONST_SET_OP`.
    let full = Tensor::zeros((2, 3), DType::F8E4M3, &dev)?;
    full.i((.., 1..2))?.const_set(F8E4M3::from_f32(7.).into())?;
    let seven = F8E4M3::from_f32(7.);
    let zero = F8E4M3::from_f32(0.);
    assert_eq!(
        full.to_vec2::<F8E4M3>()?,
        [[zero, seven, zero], [zero, seven, zero]]
    );
    Ok(())
}
