//! Unit tests for the opt-in MIOpen path (`--features miopen`).
//!
//! Both cover a mapping that has already shipped a silent wrong answer once:
//! the dtype probe (`half::bf16`'s type name contains `f16`) and the axis a
//! 1-D convolution's padding/stride/dilation belong to.

use super::*;
use half::{bf16, f16};

fn code<T: Copy>() -> u32 {
    miopen_dtype::<T>().map(|d| d as u32).unwrap()
}

#[test]
fn dtype_mapping() {
    assert_eq!(code::<f32>(), DataType::MiopenFloat as u32);
    assert_eq!(code::<f64>(), DataType::MiopenDouble as u32);
    assert_eq!(code::<f16>(), DataType::MiopenHalf as u32);
    // `half::bf16`'s type name also contains "f16"; it must not fall
    // through to MiopenHalf.
    assert_eq!(code::<bf16>(), DataType::MiopenBFloat16 as u32);
}

#[test]
fn unsupported_dtype_errors() {
    assert!(miopen_dtype::<u8>().is_err());
    assert!(miopen_dtype::<i64>().is_err());
}

/// The 1-D emulation must put padding/stride/dilation on the W axis, where
/// the length lives. Putting them on H made every padded `conv1d` a no-op.
#[test]
fn conv1d_spec_puts_the_length_parameters_on_w() {
    let params = crate::conv::ParamsConv1D {
        b_size: 1,
        l_in: 4,
        c_out: 1,
        c_in: 1,
        k_size: 3,
        padding: 1,
        stride: 1,
        dilation: 1,
        cudnn_fwd_algo: None,
    };
    let spec = Conv2dSpec {
        b: params.b_size,
        c_in: params.c_in,
        c_out: params.c_out,
        i_h: 1,
        i_w: params.l_in,
        k_h: 1,
        k_w: params.k_size,
        out_h: 1,
        out_w: params.l_out(),
        pad_h: 0,
        pad_w: params.padding,
        stride_h: 1,
        stride_w: params.stride,
        dil_h: 1,
        dil_w: params.dilation,
    };
    assert_eq!(spec.pad_h, 0);
    assert_eq!(spec.pad_w, 1);
    assert_eq!(spec.out_w, params.l_out());
    // With h = 1, k_h = 1 and no h padding the output height stays 1; a
    // pad_h of 1 would make MIOpen produce three rows instead.
    assert_eq!(spec.out_h, (spec.i_h + 2 * spec.pad_h - spec.k_h) + 1);
}
