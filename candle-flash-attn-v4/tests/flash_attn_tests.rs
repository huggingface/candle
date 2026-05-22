use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};
use rstest::rstest;

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

fn fa_acausal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[test]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 2 * 64, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 64))?;
    let k = (&q / 400.)?;
    let v = (&q / 500.)?;
    let q = (&q / 300.)?;

    let ys1 = fa_acausal(&q, &k, &v, 0.5)?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        candle_flash_attn_v4::flash_attn(&q, &k, &v, 0.5, false, 8.0)?.transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys2.dims(), &[3, 2, 64]);
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-5);
    Ok(())
}

#[test]
fn flash_attn_acausal_gqa() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let n_h = 4usize;
    let n_h_k = 1usize;

    let q = Tensor::arange(0u32, (n_h * 2 * 64) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, n_h, 2, 64))?;
    let gqa = q.clone().i((.., ..n_h_k, .., ..))?;
    assert_eq!(gqa.dims(), &[1, n_h_k, 2, 64]);

    let q = (q.clone() / 1000.)?;
    let k_gqa = (&gqa / 400.)?;
    let v_gqa = (&gqa / 500.)?;

    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k_gqa = k_gqa.transpose(1, 2)?;
        let v_gqa = v_gqa.transpose(1, 2)?;
        candle_flash_attn_v4::flash_attn_with_options(
            &q, &k_gqa, &v_gqa, 0.125, false, true, 8.0, false, false,
        )?
        .transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    assert_eq!(ys2.dims(), &[n_h, 2, 64]);
    Ok(())
}

#[test]
fn flash_attn_deterministic() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 2 * 64, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 64))?;
    let k = (&q / 400.)?;
    let v = (&q / 500.)?;
    let q = (&q / 300.)?;

    let ys1 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        candle_flash_attn_v4::flash_attn_deterministic(&q, &k, &v, 0.5, false, 8.0)?
            .transpose(1, 2)?
    };
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;

    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        candle_flash_attn_v4::flash_attn_deterministic(&q, &k, &v, 0.5, false, 8.0)?
            .transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;

    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;
    assert_eq!(ys1.dims(), &[3, 2, 64]);
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-5);
    Ok(())
}

#[test]
fn flash_attn_varlen() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 2 * 64, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 64))?;
    let k = (&q / 400.)?;
    let v = (&q / 500.)?;
    let q = (&q / 300.)?;

    let seqlens_q = Tensor::new(&[0u32, 2u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        candle_flash_attn_v4::flash_attn_varlen(
            &q, &k, &v, &seqlens_q, &seqlens_q, 2, 2, 0.5, false, false, 8.0,
        )?
        .transpose(0, 1)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, 2, 64]);
    Ok(())
}

#[rstest(
    head_dim => [64, 128, 256],
    seq_len => [2, 4, 9],
)]
fn flash_attn_varlen_param(head_dim: usize, seq_len: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;

    let q = Tensor::arange(0u32, (3 * seq_len * head_dim) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, seq_len, head_dim))?;
    let k = (&q / ((head_dim * seq_len) as f64 * 4.))?;
    let v = (&q / ((head_dim * seq_len) as f64 * 2.))?;
    let q = (&q / ((head_dim * seq_len) as f64 * 3.))?;

    let seqlens_q = Tensor::new(&[0u32, seq_len as u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, seq_len as u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        candle_flash_attn_v4::flash_attn_varlen(
            &q,
            &k,
            &v,
            &seqlens_q,
            &seqlens_k,
            seq_len,
            seq_len,
            0.5,
            false,
            false,
            8.0,
        )?
        .transpose(0, 1)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, seq_len, head_dim]);
    let ys2 = {
        let q = q.unsqueeze(0)?;
        let k = k.unsqueeze(0)?;
        let v = v.unsqueeze(0)?;
        let y = fa_acausal(&q, &k, &v, 0.5)?;
        y.i(0)?.to_dtype(DType::F32)?
    };

    let diff = ys.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;
    assert!(diff.to_vec0::<f32>()?.abs() < 5e-3);
    Ok(())
}

#[test]
fn rescale_threshold_default() {
    assert_eq!(candle_flash_attn_v4::default_rescale_threshold(), 8.0);
}
