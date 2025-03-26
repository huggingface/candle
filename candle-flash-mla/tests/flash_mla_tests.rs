use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};
use candle_flash_mla;
use rstest::rstest;

pub trait RepeatInterleaveOp {
    fn repeat_interleave(&self, repeats: usize, dim: usize) -> candle::Result<Tensor>;
}

impl RepeatInterleaveOp for Tensor {
    fn repeat_interleave(&self, repeats: usize, dim: usize) -> candle::Result<Tensor> {
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..self.dim(dim)?)
                .flat_map(|i| vec![i as u32; repeats])
                .collect::<Vec<_>>(),
            self.device(),
        )?;
        self.index_select(&indices, dim)
    }
}

fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    h_q: usize,
    h_kv: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    let k = k.repeat_interleave(h_q / h_kv, 0)?;
    let v = v.repeat_interleave(h_q / h_kv, 0)?;

    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

#[rstest(
    b => [128],
    s_k => [4096, 8192],
    h_q => [16, 32, 64, 128], // TP = 8, 4, 2, 1
    // s_q => [1, 2], // MTP = 1, 2
    s_q => [1],
)]
fn flash_mla_param(b: usize, s_k: usize, h_q: usize, s_q: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;

    let h_kv = 1;
    let d = 576;
    let dv = 512;

    let cache_seqlens_vec = vec![s_k as i32; b];
    let cache_seqlens = Tensor::new(cache_seqlens_vec.clone(), &device)?;
    let max_seqlen = cache_seqlens.max(0)?.to_scalar::<i32>()? as usize;
    let max_seqlen_pad = max_seqlen.div_ceil(256) * 256;

    let q = Tensor::randn(0., 1., (b, s_q, h_q, d), &device)?.to_dtype(DType::BF16)?;
    let block_size = 64;
    let block_table = Tensor::arange(0i32, (b * max_seqlen_pad / block_size) as i32, &device)?
        .reshape((b, max_seqlen_pad / block_size))?;
    let blocked_k = Tensor::randn(
        0.,
        1.,
        (block_table.elem_count(), block_size, h_kv, d),
        &device,
    )?
    .to_dtype(DType::BF16)?;
    let blocked_v = blocked_k.narrow(D::Minus1, 0, dv)?.copy()?;

    let softmax_scale = 1. / (q.dim(D::Minus1)? as f32).sqrt();

    let out_flash = candle_flash_mla::flash_attn_mla(
        &q,
        &blocked_k,
        block_table,
        cache_seqlens,
        softmax_scale,
        dv,
    )?;

    let truth = {
        let mut out = Vec::new();
        for i in 0..b {
            let begin = i * max_seqlen_pad;
            let end = begin + cache_seqlens_vec[i] as usize;

            let q = q.i(i)?.transpose(0, 1)?;
            let k = blocked_k
                .reshape(((), h_kv, d))?
                .i(begin..end)?
                .transpose(0, 1)?;
            let v = blocked_v
                .reshape(((), h_kv, dv))?
                .i(begin..end)?
                .transpose(0, 1)?;

            let res = sdpa(&q, &k, &v, h_q, h_kv, softmax_scale)?;
            out.push(res.transpose(0, 1)?);
        }

        Tensor::stack(&out, 0)?
    };

    assert_eq!(out_flash.dims(), truth.dims());

    {
        let out_flash = out_flash.to_dtype(DType::F64)?;
        let truth = truth.to_dtype(DType::F64)?;

        let cos_diff = 1.
            - 2. * (&out_flash * &truth)?.sum_all()?.to_scalar::<f64>()?
                / (out_flash.sqr()? + truth.sqr()?)?
                    .sum_all()?
                    .to_scalar::<f64>()?
                    .max(1e-12);
        assert!(cos_diff < 1e-5, "{cos_diff} > {}", 1e-5);
    }

    Ok(())
}
