//! CUDA inference kernels. All model tensors and recurrent caches stay on GPU.
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, CudaStorage, CustomOp1, CustomOp2, DType, Layout, Result, Shape, Tensor};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kev.ptx"));

struct Depthwise;

impl CustomOp2 for Depthwise {
    fn name(&self) -> &'static str {
        "kev-cuda-depthwise"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("CUDA depthwise called on CPU")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, channels, time) = xl.shape().dims3()?;
        let (_, _, width) = wl.shape().dims3()?;
        let len = time - width + 1;
        let count = batch * channels * len;
        let dev = x.device().clone();
        let xs = x.as_cuda_slice::<f32>()?;
        let ws = w.as_cuda_slice::<f32>()?;
        let xs = xs.slice(xl.start_offset()..xl.start_offset() + xl.shape().elem_count());
        let ws = ws.slice(wl.start_offset()..wl.start_offset() + wl.shape().elem_count());
        // Every element of dst is written by exactly one kernel thread.
        let dst = unsafe { dev.alloc::<f32>(count) }?;
        let func = dev.get_or_load_custom_func("kev_depthwise", "kev-fp32-v1", PTX)?;
        let mut builder = func.builder();
        builder.arg(&xs).arg(&ws).arg(&dst);
        candle::builder_arg!(
            builder,
            count as u32,
            channels as u32,
            time as u32,
            width as u32,
            len as u32
        );
        unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(dst, dev),
            (batch, channels, len).into(),
        ))
    }
}

pub fn depthwise(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (_, channels, time) = input.dims3()?;
    let (wc, one, width) = weights.dims3()?;
    if wc != channels || one != 1 || width == 0 || time < width {
        candle::bail!("invalid depthwise convolution dimensions");
    }
    if input.dtype() != DType::F32 || weights.dtype() != DType::F32 {
        candle::bail!("Kev CUDA convolution requires fp32");
    }
    input
        .contiguous()?
        .apply_op2_no_bwd(&weights.contiguous()?, &Depthwise)
}

struct Scan {
    bh: usize,
    len: usize,
    kd: usize,
    vd: usize,
}

impl CustomOp1 for Scan {
    fn name(&self) -> &'static str {
        "kev-cuda-delta-scan"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("CUDA DeltaNet called on CPU")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = storage.device().clone();
        let src = storage.as_cuda_slice::<f32>()?;
        let src =
            src.slice(layout.start_offset()..layout.start_offset() + layout.shape().elem_count());
        let count = self.bh * (self.len + self.kd) * self.vd;
        // The kernel initializes its shared state and writes the full output.
        let dst = unsafe { dev.alloc::<f32>(count) }?;
        let func = dev.get_or_load_custom_func("kev_delta_scan", "kev-fp32-v1", PTX)?;
        let config = LaunchConfig {
            grid_dim: (self.bh as u32, self.vd.div_ceil(32) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: (self.kd * 32 * std::mem::size_of::<f32>()) as u32,
        };
        let mut builder = func.builder();
        builder.arg(&src).arg(&dst);
        candle::builder_arg!(
            builder,
            self.bh as u32,
            self.len as u32,
            self.kd as u32,
            self.vd as u32
        );
        unsafe { builder.launch(config) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(dst, dev), (count,).into()))
    }
}

pub fn scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, heads, len, kd) = q.dims4()?;
    let vd = v.dim(3)?;
    if kd == 0 || kd > 384 || len == 0 || vd == 0 || batch == 0 || heads == 0 {
        candle::bail!("unsupported CUDA DeltaNet dimensions");
    }
    if k.dims() != q.dims()
        || v.dims() != [batch, heads, len, vd]
        || decay.dims() != [batch, heads, len, 1]
        || beta.dims() != decay.dims()
        || state.dims() != [batch, heads, kd, vd]
    {
        candle::bail!("CUDA DeltaNet shape mismatch");
    }
    let mut inputs = Vec::with_capacity(6);
    for t in [q, k, v, decay, beta, state] {
        if t.dtype() != DType::F32 || !t.device().same_device(q.device()) {
            candle::bail!("CUDA DeltaNet requires fp32 tensors on the same GPU");
        }
        inputs.push(t.contiguous()?.flatten_all()?);
    }
    let output = Tensor::cat(&inputs, 0)?.apply_op1_no_bwd(&Scan {
        bh: batch * heads,
        len,
        kd,
        vd,
    })?;
    let readouts = batch * heads * len * vd;
    Ok((
        output
            .narrow(0, 0, readouts)?
            .reshape((batch, heads, len, vd))?
            .transpose(1, 2)?,
        output
            .narrow(0, readouts, batch * heads * kd * vd)?
            .reshape((batch, heads, kd, vd))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn error(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_device(&Device::Cpu)? - b)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    #[test]
    fn cuda_scan_matches_cpu_and_preserves_initial_cache() -> Result<()> {
        let gpu = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        // Real head size plus partial column tiles; nonzero caches and strided inputs.
        for (kd, vd) in [(128, 128), (4, 5)] {
            let (b, h, n) = (2, 3, 11);
            let make = |shape: Shape, phase: f32| -> Result<Tensor> {
                let values = (0..shape.elem_count())
                    .map(|i| ((i as f32 + phase) * 0.37).sin() * 0.2)
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, shape, &cpu)
            };
            let q = make((b, h, n, kd).into(), 0.)?;
            let k = make((b, h, n, kd).into(), 1.)?;
            let v = make((b, h, n, vd).into(), 2.)?;
            let g = Tensor::full(0.8f32, (b, h, n, 1), &cpu)?;
            let beta = Tensor::full(0.4f32, (b, h, n, 1), &cpu)?;
            let initial = make((b, h, kd, vd).into(), 3.)?;
            let (expected, expected_state) = crate::delta::scan(&q, &k, &v, &g, &beta, &initial)?;
            let ts = [&q, &k, &v, &g, &beta, &initial]
                .iter()
                .map(|t| t.to_device(&gpu))
                .collect::<Result<Vec<_>>>()?;
            let (out, final_state) = scan(&ts[0], &ts[1], &ts[2], &ts[3], &ts[4], &ts[5])?;
            assert!(error(&out, &expected)? < 1e-6);
            assert!(error(&final_state, &expected_state)? < 1e-6);
            assert_eq!(error(&ts[5], &initial)?, 0.);
            let parts = |start, len| {
                ts[..5]
                    .iter()
                    .map(|t| t.narrow(2, start, len))
                    .collect::<Result<Vec<_>>>()
            };
            let first = parts(0, 3)?;
            let (a, cached) = scan(
                &first[0], &first[1], &first[2], &first[3], &first[4], &ts[5],
            )?;
            let rest = parts(3, 8)?;
            let (z, cached) = scan(&rest[0], &rest[1], &rest[2], &rest[3], &rest[4], &cached)?;
            assert!(error(&Tensor::cat(&[a, z], 1)?, &expected)? < 1e-6);
            assert!(error(&cached, &expected_state)? < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn cuda_depthwise_matches_cpu_with_strided_input() -> Result<()> {
        let cpu = Device::Cpu;
        let gpu = Device::new_cuda(0)?;
        let x = Tensor::arange(0f32, 60f32, &cpu)?
            .reshape((2, 10, 3))?
            .transpose(1, 2)?
            .narrow(2, 1, 8)?;
        let w = Tensor::arange(0f32, 12f32, &cpu)?.reshape((3, 1, 4))?;
        let expected = crate::delta::depthwise(&x, &w)?;
        let actual = depthwise(&x.to_device(&gpu)?, &w.to_device(&gpu)?)?;
        assert!(error(&actual, &expected)? < 1e-5);
        Ok(())
    }
}
