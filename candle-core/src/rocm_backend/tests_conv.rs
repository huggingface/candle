//! Convolution, pooling and upsampling launchers, checked against the CPU
//! backend.
//!
//! The integration suites (`conv_tests`, `pool_tests`, `bilinear_tests`) cover
//! the contiguous happy path. What they barely touch — and where every silent
//! wrong-answer bug in this backend has lived — is a source whose layout is not
//! a packed buffer starting at element 0. Each test below therefore feeds a
//! transposed, narrowed or offset view and demands the CPU's answer.

use super::tests::rocm_device;
use crate::{Device, Result, Tensor};

/// Same data on the CPU and on the GPU.
fn pair(
    data: &[f32],
    shape: (usize, usize, usize, usize),
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    Ok((
        Tensor::new(data, &Device::Cpu)?.reshape(shape)?,
        Tensor::new(data, dev)?.reshape(shape)?,
    ))
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.37).sin()).collect()
}

fn assert_close(gpu: &Tensor, cpu: &Tensor) -> Result<()> {
    assert_eq!(gpu.dims(), cpu.dims());
    let g = gpu.flatten_all()?.to_vec1::<f32>()?;
    let c = cpu.flatten_all()?.to_vec1::<f32>()?;
    for (i, (g, c)) in g.iter().zip(c.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "element {i}: gpu {g} vs cpu {c}\ngpu {g:?}\ncpu {c:?}"
        );
    }
    Ok(())
}

/// `im2col` reads the input through the strides in `info`, so a transposed
/// view has to give the same answer as its contiguous copy.
#[test]
fn conv2d_with_a_non_contiguous_input() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(2 * 3 * 5 * 4);
    let (c, g) = pair(&data, (2, 3, 5, 4), &dev)?;
    let (c, g) = (c.transpose(2, 3)?, g.transpose(2, 3)?);
    let kd = ramp(4 * 3 * 2 * 2);
    let (ck, gk) = pair(&kd, (4, 3, 2, 2), &dev)?;
    for padding in [0usize, 1] {
        assert_close(
            &g.conv2d(&gk, padding, 1, 1, 1)?,
            &c.conv2d(&ck, padding, 1, 1, 1)?,
        )?;
    }
    Ok(())
}

/// A kernel that is a channel slice of a bigger weight is both strided and
/// offset. cuda_backend copies it and then uses the original buffer anyway;
/// [`super::ops_conv`] deliberately uses the copy.
#[test]
fn conv2d_with_a_non_contiguous_kernel() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(2 * 4 * 4);
    let (c, g) = pair(&data, (1, 2, 4, 4), &dev)?;
    let kd = ramp(3 * 4);
    let (ck, gk) = pair(&kd, (3, 4, 1, 1), &dev)?;
    let ck = ck.narrow(1, 1, 2)?;
    let gk = gk.narrow(1, 1, 2)?;
    assert!(!gk.is_contiguous());
    assert_close(&g.conv2d(&gk, 0, 2, 1, 1)?, &c.conv2d(&ck, 0, 2, 1, 1)?)?;
    Ok(())
}

/// `im2col1d` also strides through `info`; the batch narrow additionally moves
/// the start offset off zero.
#[test]
fn conv1d_with_an_offset_input() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(3 * 4 * 7);
    let cpu = Tensor::new(data.as_slice(), &Device::Cpu)?.reshape((3, 4, 7))?;
    let gpu = Tensor::new(data.as_slice(), &dev)?.reshape((3, 4, 7))?;
    let cpu = cpu.narrow(0, 1, 2)?;
    let gpu = gpu.narrow(0, 1, 2)?;
    assert_ne!(gpu.layout().start_offset(), 0);
    let kd = ramp(2 * 4 * 3);
    let ck = Tensor::new(kd.as_slice(), &Device::Cpu)?.reshape((2, 4, 3))?;
    let gk = Tensor::new(kd.as_slice(), &dev)?.reshape((2, 4, 3))?;
    for (padding, stride, dilation) in [(0, 1, 1), (1, 1, 1), (2, 2, 1), (0, 1, 2)] {
        assert_close(
            &gpu.conv1d(&gk, padding, stride, dilation, 1)?,
            &cpu.conv1d(&ck, padding, stride, dilation, 1)?,
        )?;
    }
    Ok(())
}

/// The col2im shortcut only applies with a contiguous kernel and no
/// padding/dilation/output padding; every other combination has to land on the
/// naive `conv_transpose1d` kernel.
///
/// The two paths are compared against each other rather than against the CPU
/// for `b_size > 1`: `cpu_backend`'s `MatMul::f` collapses a batched GEMM whose
/// rhs is broadcast (`b_skip == 0 && a_skip == m * k`) into one `(1, b*m, n, k)`
/// call without also requiring `lhs_rs == k`, so a transposed lhs — exactly what
/// col2im passes — makes the CPU read the wrong rows from batch 1 on. Verified
/// against a from-the-definition reference: the GPU is the one that is right.
/// The CPU comparisons below therefore use a single batch.
#[test]
fn conv_transpose1d_covers_both_paths() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(3 * 5);
    let cpu = Tensor::new(data.as_slice(), &Device::Cpu)?.reshape((1, 3, 5))?;
    let gpu = Tensor::new(data.as_slice(), &dev)?.reshape((1, 3, 5))?;
    let kd = ramp(3 * 4 * 3);
    let ck = Tensor::new(kd.as_slice(), &Device::Cpu)?.reshape((3, 4, 3))?;
    let gk = Tensor::new(kd.as_slice(), &dev)?.reshape((3, 4, 3))?;

    // col2im path: contiguous kernel, no padding/dilation/output padding.
    assert_close(
        &gpu.conv_transpose1d(&gk, 0, 0, 1, 1, 1)?,
        &cpu.conv_transpose1d(&ck, 0, 0, 1, 1, 1)?,
    )?;
    // Padding and a stride force the direct kernel.
    assert_close(
        &gpu.conv_transpose1d(&gk, 1, 0, 2, 1, 1)?,
        &cpu.conv_transpose1d(&ck, 1, 0, 2, 1, 1)?,
    )?;
    // So does a dilation.
    assert_close(
        &gpu.conv_transpose1d(&gk, 0, 0, 1, 2, 1)?,
        &cpu.conv_transpose1d(&ck, 0, 0, 1, 2, 1)?,
    )?;

    // Batched: the same kernel, once contiguous (col2im) and once as a
    // narrowed view of a wider buffer (non-contiguous, so the direct kernel).
    let batched = Tensor::new(ramp(2 * 3 * 5).as_slice(), &dev)?.reshape((2, 3, 5))?;
    let wide = Tensor::cat(&[&gk, &gk.zeros_like()?], 2)?;
    let gk_nc = wide.narrow(2, 0, 3)?;
    assert!(!gk_nc.is_contiguous());
    assert_close(
        &batched.conv_transpose1d(&gk, 0, 0, 1, 1, 1)?,
        &batched.conv_transpose1d(&gk_nc, 0, 0, 1, 1, 1)?,
    )?;
    Ok(())
}

#[test]
fn conv_transpose2d_with_an_offset_input() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(3 * 2 * 4 * 4);
    let (c, g) = pair(&data, (3, 2, 4, 4), &dev)?;
    let (c, g) = (c.narrow(0, 1, 2)?, g.narrow(0, 1, 2)?);
    let kd = ramp(2 * 3 * 2 * 2);
    let (ck, gk) = pair(&kd, (2, 3, 2, 2), &dev)?;
    for (padding, stride) in [(0, 1), (0, 2), (1, 1)] {
        assert_close(
            &g.conv_transpose2d(&gk, padding, 0, stride, 1)?,
            &c.conv_transpose2d(&ck, padding, 0, stride, 1)?,
        )?;
    }
    Ok(())
}

#[test]
fn pool2d_on_a_strided_input() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(2 * 3 * 6 * 6);
    let (c, g) = pair(&data, (2, 3, 6, 6), &dev)?;
    let (c, g) = (c.transpose(2, 3)?, g.transpose(2, 3)?);
    assert_close(&g.avg_pool2d(2)?, &c.avg_pool2d(2)?)?;
    assert_close(&g.max_pool2d(2)?, &c.max_pool2d(2)?)?;
    assert_close(
        &g.avg_pool2d_with_stride(3, 2)?,
        &c.avg_pool2d_with_stride(3, 2)?,
    )?;
    assert_close(
        &g.max_pool2d_with_stride(3, 2)?,
        &c.max_pool2d_with_stride(3, 2)?,
    )?;
    Ok(())
}

#[test]
fn upsample_on_an_offset_input() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(4 * 2 * 3 * 5);
    let (c, g) = pair(&data, (4, 2, 3, 5), &dev)?;
    let (c, g) = (c.narrow(0, 2, 2)?, g.narrow(0, 2, 2)?);
    assert_ne!(g.layout().start_offset(), 0);
    assert_close(&g.upsample_nearest2d(6, 10)?, &c.upsample_nearest2d(6, 10)?)?;
    assert_close(
        &g.upsample_bilinear2d(6, 10, false)?,
        &c.upsample_bilinear2d(6, 10, false)?,
    )?;
    assert_close(
        &g.upsample_bilinear2d(6, 10, true)?,
        &c.upsample_bilinear2d(6, 10, true)?,
    )?;
    Ok(())
}

/// `conv.cu` has no 1-D upsample kernel; parity with cuda_backend is to refuse.
#[test]
fn upsample_nearest1d_is_refused() -> Result<()> {
    let dev = rocm_device!();
    let t = Tensor::new(&[[[1f32, 2., 3.]]], &dev)?;
    let err = t.upsample_nearest1d(6).unwrap_err().to_string();
    assert!(err.contains("upsample-nearest1d"), "{err}");
    Ok(())
}
