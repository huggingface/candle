//! CPU Gated DeltaNet scan. Fuse decay, memory read, rank-one update and output
//! read into one mutable state buffer, avoiding per-token tensor allocations.
use candle::{Result, Tensor};
use rayon::prelude::*;

/// Short depthwise causal convolution over an already padded [batch, channel, time] input.
pub fn depthwise(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if input.device().is_cuda() {
        return crate::cuda::depthwise(input, weights);
    }
    if !input.device().is_cpu() {
        candle::bail!("fused depthwise convolution requires CPU or CUDA");
    }
    let (batch, channels, time) = input.dims3()?;
    let (_, _, width) = weights.dims3()?;
    let len = time - width + 1;
    let x = input.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let w = weights.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; batch * channels * len];
    out.par_chunks_mut(len).enumerate().for_each(|(bc, row)| {
        let channel = bc % channels;
        for (t, y) in row.iter_mut().enumerate() {
            for j in 0..width {
                *y += x[bc * time + t + j] * w[channel * width + j];
            }
        }
    });
    Tensor::from_vec(out, (batch, channels, len), input.device())
}

pub fn scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return crate::cuda::scan(q, k, v, decay, beta, state);
    }
    if !q.device().is_cpu() {
        candle::bail!("fused DeltaNet requires CPU or CUDA");
    }
    let (batch, heads, len, kd) = q.dims4()?;
    let vd = v.dim(3)?;
    let qv = q.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let kv = k.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let vv = v.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let gv = decay.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let bv = beta.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut sv = state.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; batch * heads * len * vd];
    sv.par_chunks_mut(kd * vd)
        .zip(out.par_chunks_mut(len * vd))
        .enumerate()
        .for_each(|(head, (s, o))| {
            let mut memory = vec![0f32; vd];
            let mut delta = vec![0f32; vd];
            for t in 0..len {
                let qq = &qv[(head * len + t) * kd..][..kd];
                let kk = &kv[(head * len + t) * kd..][..kd];
                let val = &vv[(head * len + t) * vd..][..vd];
                let gate = gv[head * len + t];
                let rate = bv[head * len + t];
                memory.fill(0.);
                for (i, row) in s.chunks_mut(vd).enumerate() {
                    for j in 0..vd {
                        row[j] *= gate;
                        memory[j] += row[j] * kk[i];
                    }
                }
                for j in 0..vd {
                    delta[j] = (val[j] - memory[j]) * rate;
                }
                let output = &mut o[t * vd..][..vd];
                for (i, row) in s.chunks_mut(vd).enumerate() {
                    for j in 0..vd {
                        row[j] += kk[i] * delta[j];
                        output[j] += row[j] * qq[i];
                    }
                }
            }
        });
    Ok((
        Tensor::from_vec(out, (batch, heads, len, vd), q.device())?.transpose(1, 2)?,
        Tensor::from_vec(sv, (batch, heads, kd, vd), q.device())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn scan_matches_independent_tensor_rule_and_continuation() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, n, kdim, vdim) = (2, 3, 7, 4, 5);
        let data = |len: usize, phase: f32| {
            (0..len)
                .map(|i| ((i as f32 + phase) * 0.37).sin() * 0.2)
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_vec(data(b * h * n * kdim, 0.), (b, h, n, kdim), &dev)?;
        let k = Tensor::from_vec(data(b * h * n * kdim, 1.), (b, h, n, kdim), &dev)?;
        let v = Tensor::from_vec(data(b * h * n * vdim, 2.), (b, h, n, vdim), &dev)?;
        let g = Tensor::full(0.8f32, (b, h, n, 1), &dev)?;
        let beta = Tensor::full(0.4f32, (b, h, n, 1), &dev)?;
        let initial = Tensor::zeros((b, h, kdim, vdim), DType::F32, &dev)?;
        let (actual, state) = scan(&q, &k, &v, &g, &beta, &initial)?;
        let mut expected_state = initial.clone();
        let mut expected_out = vec![];
        for t in 0..n {
            let kt = k.narrow(2, t, 1)?;
            let qt = q.narrow(2, t, 1)?;
            expected_state = (expected_state * 0.8)?;
            let memory = kt.matmul(&expected_state)?;
            let delta = ((v.narrow(2, t, 1)? - memory)? * 0.4)?;
            expected_state = (&expected_state + kt.transpose(2, 3)?.matmul(&delta)?)?;
            expected_out.push(qt.matmul(&expected_state)?);
        }
        let expected = Tensor::cat(&expected_out, 2)?.transpose(1, 2)?;
        let maxerr = |a: &Tensor, b: &Tensor| -> Result<f32> {
            Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
        };
        assert!(maxerr(&actual, &expected)? < 1e-6);
        assert!(maxerr(&state, &expected_state)? < 1e-6);
        let (first, cached) = scan(
            &q.narrow(2, 0, 3)?,
            &k.narrow(2, 0, 3)?,
            &v.narrow(2, 0, 3)?,
            &g.narrow(2, 0, 3)?,
            &beta.narrow(2, 0, 3)?,
            &initial,
        )?;
        let (rest, final_state) = scan(
            &q.narrow(2, 3, 4)?,
            &k.narrow(2, 3, 4)?,
            &v.narrow(2, 3, 4)?,
            &g.narrow(2, 3, 4)?,
            &beta.narrow(2, 3, 4)?,
            &cached,
        )?;
        assert!(maxerr(&actual, &Tensor::cat(&[first, rest], 1)?)? < 1e-6);
        assert!(maxerr(&state, &final_state)? < 1e-6);
        Ok(())
    }

    #[test]
    fn depthwise_matches_grouped_convolution() -> Result<()> {
        let d = Device::Cpu;
        let x = Tensor::arange(0f32, 60f32, &d)?.reshape((2, 3, 10))?;
        let w = Tensor::arange(0f32, 12f32, &d)?.reshape((3, 1, 4))?;
        let actual = depthwise(&x, &w)?;
        let expected = x.conv1d(&w, 0, 1, 1, 3)?;
        let err = (&actual - &expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(err < 1e-5);
        Ok(())
    }
}
