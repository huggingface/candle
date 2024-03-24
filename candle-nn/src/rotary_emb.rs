#![allow(unused)]
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Interleaved variant of rotary embeddings.
/// The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
/// The resulting y0 and y1 are also interleaved with:
///   y0 = x0*cos - x1*sin
///   y1 = x0*sin + x1*cos
#[derive(Debug, Clone)]
struct RotaryEmbI;

impl candle::CustomOp3 for RotaryEmbI {
    fn name(&self) -> &'static str {
        "rotary-emb-int"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .for_each(|(src, dst)| {
                    for i_over_2 in 0..t * d / 2 {
                        let i = 2 * i_over_2;
                        dst[i] = src[i] * cos[i_over_2] - src[i + 1] * sin[i_over_2];
                        dst[i + 1] = src[i] * sin[i_over_2] + src[i + 1] * cos[i_over_2];
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for embs {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }
}

pub fn rope_i(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = cos.dims2()?;
    let (sin_seq_len, sin_n_embd) = cos.dims2()?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbI)
}

pub fn rope_i_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
    let rope = rope.flatten_from(D::Minus2)?;
    Ok(rope)
}
