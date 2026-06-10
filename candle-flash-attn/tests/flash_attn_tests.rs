use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};

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
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

fn fa_acausal_softcap(q: &Tensor, k: &Tensor, v: &Tensor, softcap: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    // let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = q.matmul(&k.t()?)?;
    let att = (softcap as f64 * ((att / softcap as f64)?.tanh())?)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[test]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let ys1 = fa_acausal(&q, &k, &v, 0.5)?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        candle_flash_attn::flash_attn(&q, &k, &v, 0.5, false)?.transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys1.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys1, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );

    assert_eq!(ys2.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys2, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-5);
    Ok(())
}

#[test]
fn flash_attn_acausal_softcap() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 5 * 8, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 5, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;
    let softcap = 5.0f32;

    let ys1 = fa_acausal_softcap(&q, &k, &v, softcap.clone())?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        candle_flash_attn::flash_attn_alibi_windowed_softcap(
            &q,
            &k,
            &v,
            None,            //  alibi_slopes //
            1.0,             // softmax //
            None,            // window_size_left //
            None,            // window_size_right //
            softcap.clone(), // softcap //
        )?
        .transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys1.dims(), &[3, 5, 8]);
    assert_eq!(ys2.dims(), &[3, 5, 8]);
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-3);
    Ok(())
}

#[test]
fn flash_attn_acausal_splitkv() -> Result<()> {
    // Shape designed to enter the splitkv dispatch path on any modern CUDA
    // GPU (sm80+) per the heuristic ported from upstream FA v2.8.3 in
    // PR-FA-3: batch * heads * m_blocks = 1 * 2 * 1 = 2 leaves headroom
    // under the 0.8 * num_SMs short-circuit on A6000 / 4090 / A100, and
    // seqlen_k = 512 with head_dim = 64 produces num_n_blocks = 2 (split
    // tile is 256 in K for hdim <= 64) — so the heuristic returns >= 2
    // and the kernel takes the splitkv path. If the splitkv combine
    // kernel is wrong, this test diverges from the fp32 attention
    // reference.
    let device = Device::new_cuda(0)?;
    let (b, h, sq, sk, d) = (1usize, 2, 8, 512, 64);
    let scale = 1.0f32 / (d as f32).sqrt();

    // Provenance check: assert the dispatcher actually picks splitkv for this
    // shape on this device, so the test fails (instead of silently passing
    // via the dense path) if the heuristic ever regresses. Mirrors the
    // computation done inside `set_params_splitkv` in src/lib.rs.
    {
        let cuda_dev = device.as_cuda_device()?;
        let num_sm = cuda_dev
            .cuda_stream()
            .context()
            .attribute(
                candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| anyhow::anyhow!("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT): {e}"))?
            as usize;
        let block_n = candle_flash_attn::splitkv_block_n(d);
        let num_n_blocks = (sk + block_n - 1) / block_n;
        let num_m_blocks = (sq + 64 - 1) / 64;
        let num_splits = candle_flash_attn::num_splits_heuristic(
            b * h * num_m_blocks,
            num_sm * 2,
            num_n_blocks,
            128,
        );
        assert!(
            num_splits > 1,
            "expected splitkv path for shape (b={b}, h={h}, sq={sq}, sk={sk}, d={d}) on a {num_sm}-SM device, but heuristic chose num_splits={num_splits}",
        );
    }

    // Flash-attn input layout is (batch, seq, heads, head_dim).
    let q = (Tensor::arange(0u32, (b * sq * h * d) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((b, sq, h, d))?
        / 1024.)?;
    let k = (Tensor::arange(0u32, (b * sk * h * d) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((b, sk, h, d))?
        / 4096.)?;
    let v = (Tensor::arange(0u32, (b * sk * h * d) as u32, &device)?
        .to_dtype(DType::F16)?
        .reshape((b, sk, h, d))?
        / 8192.)?;

    // Reference attention: collapse (batch, heads) into a single batch axis
    // so `fa_acausal`'s rank-3 matmul matches per-head, then unflatten.
    let ys_ref = {
        let qref = q.transpose(1, 2)?.contiguous()?.reshape((b * h, sq, d))?;
        let kref = k.transpose(1, 2)?.contiguous()?.reshape((b * h, sk, d))?;
        let vref = v.transpose(1, 2)?.contiguous()?.reshape((b * h, sk, d))?;
        fa_acausal(&qref, &kref, &vref, scale)?
            .reshape((b, h, sq, d))?
            .transpose(1, 2)?
            .contiguous()?
    };

    let ys = candle_flash_attn::flash_attn(&q, &k, &v, scale, false)?;

    let ys = ys.to_dtype(DType::F32)?;
    let ys_ref = ys_ref.to_dtype(DType::F32)?;
    let diff = ys.sub(&ys_ref)?.abs()?.flatten_all()?.max(0)?;
    assert_eq!(ys.dims(), &[b, sq, h, d]);
    let diff_v = diff.to_vec0::<f32>()?;
    assert!(
        diff_v < 5e-3,
        "splitkv vs fa_acausal max abs diff = {diff_v} (expected < 5e-3)"
    );
    Ok(())
}

#[test]
fn flash_attn_varlen() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let seqlens_q = Tensor::new(&[0u32, 2u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 2u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        candle_flash_attn::flash_attn_varlen(
            &q, &k, &v, &seqlens_q, &seqlens_k, 32, 32, 0.5, false,
        )?
        .transpose(0, 1)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );
    Ok(())
}
