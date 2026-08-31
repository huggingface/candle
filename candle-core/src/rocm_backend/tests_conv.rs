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

/// Runs the chunked im2col conv2d directly with a deliberately tiny buffer
/// cap, so multi-chunk assembly is exercised on shapes small enough to test.
fn conv2d_chunked_to_vec(
    inp: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    cap: usize,
) -> Result<Vec<f32>> {
    use crate::backend::BackendStorage;
    let (b_size, c_in, i_h, i_w) = inp.dims4()?;
    let (c_out, _, k_h, k_w) = kernel.dims4()?;
    let params = crate::conv::ParamsConv2D {
        b_size,
        i_h,
        i_w,
        k_h,
        k_w,
        c_out,
        c_in,
        padding,
        stride,
        dilation,
        cudnn_fwd_algo: None,
    };
    let (inp_s, inp_l) = inp.storage_and_layout();
    let (k_s, k_l) = kernel.storage_and_layout();
    let (crate::Storage::Rocm(inp_s), crate::Storage::Rocm(k_s)) = (&*inp_s, &*k_s) else {
        crate::bail!("not rocm storage")
    };
    let res = super::ops_conv_chunked::conv2d(inp_s, inp_l, k_s, k_l, &params, cap)?;
    match res.to_cpu_storage()? {
        crate::CpuStorage::F32(v) => Ok(v),
        _ => crate::bail!("not f32"),
    }
}

fn conv1d_chunked_to_vec(
    inp: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    cap: usize,
) -> Result<Vec<f32>> {
    use crate::backend::BackendStorage;
    let (b_size, c_in, l_in) = inp.dims3()?;
    let (c_out, _, k_size) = kernel.dims3()?;
    let params = crate::conv::ParamsConv1D {
        b_size,
        l_in,
        c_out,
        c_in,
        k_size,
        padding,
        stride,
        dilation,
        cudnn_fwd_algo: None,
    };
    let (inp_s, inp_l) = inp.storage_and_layout();
    let (k_s, k_l) = kernel.storage_and_layout();
    let (crate::Storage::Rocm(inp_s), crate::Storage::Rocm(k_s)) = (&*inp_s, &*k_s) else {
        crate::bail!("not rocm storage")
    };
    let res = super::ops_conv_chunked::conv1d(inp_s, inp_l, k_s, k_l, &params, cap)?;
    match res.to_cpu_storage()? {
        crate::CpuStorage::F32(v) => Ok(v),
        _ => crate::bail!("not f32"),
    }
}

fn assert_vec_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "element {i}: chunked {g} vs full {w}");
    }
}

/// Every cap has to reproduce the single-shot answer: one row per chunk, a
/// step that does not divide the row count, and a cap large enough for a
/// single chunk (the degenerate case that mirrors the un-chunked path).
#[test]
fn conv2d_chunked_matches_the_single_shot_path() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(2 * 3 * 10 * 9);
    let (_, g) = pair(&data, (2, 3, 10, 9), &dev)?;
    let kd = ramp(4 * 3 * 3 * 3);
    let (_, gk) = pair(&kd, (4, 3, 3, 3), &dev)?;
    for (padding, stride, dilation) in [(1, 1, 1), (0, 2, 1), (1, 1, 2)] {
        let want = g
            .conv2d(&gk, padding, stride, dilation, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // k = 27 f32s per row = 108 bytes; 756 → 7 rows per chunk.
        for cap in [1, 756, usize::MAX] {
            let got = conv2d_chunked_to_vec(&g, &gk, padding, stride, dilation, cap)?;
            assert_vec_close(&got, &want);
        }
    }
    Ok(())
}

/// The chunk kernel reads through the strides in `info` like the un-chunked
/// one; a transposed view has to give the same answer.
#[test]
fn conv2d_chunked_on_a_non_contiguous_input() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(2 * 3 * 8 * 6);
    let (_, g) = pair(&data, (2, 3, 8, 6), &dev)?;
    let g = g.transpose(2, 3)?;
    let kd = ramp(4 * 3 * 2 * 2);
    let (_, gk) = pair(&kd, (4, 3, 2, 2), &dev)?;
    let want = g.conv2d(&gk, 1, 1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let got = conv2d_chunked_to_vec(&g, &gk, 1, 1, 1, 500)?;
    assert_vec_close(&got, &want);
    Ok(())
}

#[test]
fn conv1d_chunked_matches_the_single_shot_path() -> Result<()> {
    let dev = rocm_device!();
    let data = ramp(2 * 3 * 50);
    let g = Tensor::new(data.as_slice(), &dev)?.reshape((2, 3, 50))?;
    let kd = ramp(4 * 3 * 5);
    let gk = Tensor::new(kd.as_slice(), &dev)?.reshape((4, 3, 5))?;
    for (padding, stride, dilation) in [(2, 1, 1), (0, 2, 1), (1, 1, 2)] {
        let want = g
            .conv1d(&gk, padding, stride, dilation, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // k = 15 f32s per row = 60 bytes; 420 → 7 rows per chunk.
        for cap in [1, 420, usize::MAX] {
            let got = conv1d_chunked_to_vec(&g, &gk, padding, stride, dilation, cap)?;
            assert_vec_close(&got, &want);
        }
    }
    Ok(())
}

/// Above [`super::ops_conv_chunked::IM2COL_MAX_BYTES`] the public entry point
/// has to route to the chunked path: the un-chunked answer at a
/// bigger-than-cap shape must match. Also the regression test for the OOM
/// class reported on PR #3801 — before chunking, this shape materialized a
/// ~786 MB im2col buffer in one piece.
#[test]
fn conv2d_dispatches_to_chunking_above_the_cap() -> Result<()> {
    let dev = rocm_device!();
    let (b, c_in, hw, c_out) = (1, 256, 292, 8);
    let g = Tensor::rand(-1f32, 1f32, (b, c_in, hw, hw), &dev)?;
    let gk = Tensor::rand(-1f32, 1f32, (c_out, c_in, 3, 3), &dev)?;
    let col_bytes = hw * hw * c_in * 3 * 3 * 4;
    assert!(col_bytes > super::ops_conv_chunked::IM2COL_MAX_BYTES);
    let got = g.conv2d(&gk, 1, 1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let want = conv2d_chunked_to_vec(&g, &gk, 1, 1, 1, usize::MAX)?;
    assert_vec_close(&got, &want);
    Ok(())
}
