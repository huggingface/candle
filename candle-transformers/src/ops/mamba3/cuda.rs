//! CUDA implementations for Mamba-3 inference operators.

#[rustfmt::skip]
mod cuda_kernels {
    include!(concat!(env!("OUT_DIR"), "/mamba3_kernels.rs"));
}

use super::cpu::{Mamba3FwdOutput, Mamba3StepOutput};
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::cuda_backend::WrapErr;
use candle::{CudaStorage, DType, Device, Result, Tensor, D};

fn launch_siso_step_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    adt: &Tensor,
    dt: &Tensor,
    trap: &Tensor,
    q_bias: &Tensor,
    k_bias: &Tensor,
    angles: &Tensor,
    d: Option<&Tensor>,
    z: Option<&Tensor>,
    angle_state: &Tensor,
    ssm_state: &Tensor,
    k_state: &Tensor,
    v_state: &Tensor,
) -> Result<Mamba3StepOutput> {
    let (batch, nheads, headdim_qk) = q.dims3()?;
    let headdim_v = v.dim(D::Minus1)?;
    let headdim_angles = angles.dim(D::Minus1)?;
    let device = q.device().clone();

    let out = Tensor::zeros((batch, nheads, headdim_v), DType::F32, &device)?;
    let out_angle = Tensor::zeros(angle_state.shape(), DType::F32, &device)?;
    let out_ssm = Tensor::zeros(ssm_state.shape(), DType::F32, &device)?;
    let out_k = Tensor::zeros(k.shape(), DType::F32, &device)?;

    let d_dummy = Tensor::zeros(1, DType::F32, &device)?;
    let z_dummy = Tensor::zeros(1, DType::F32, &device)?;
    let d_t = d.unwrap_or(&d_dummy);
    let z_t = z.unwrap_or(&z_dummy);

    let q = q.contiguous()?.to_dtype(DType::F32)?;
    let k = k.contiguous()?.to_dtype(DType::F32)?;
    let v = v.contiguous()?.to_dtype(DType::F32)?;
    let adt = adt.contiguous()?.to_dtype(DType::F32)?;
    let dt = dt.contiguous()?.to_dtype(DType::F32)?;
    let trap = trap.contiguous()?.to_dtype(DType::F32)?;
    let q_bias = q_bias.contiguous()?.to_dtype(DType::F32)?;
    let k_bias = k_bias.contiguous()?.to_dtype(DType::F32)?;
    let angles = angles.contiguous()?.to_dtype(DType::F32)?;
    let angle_state = angle_state.contiguous()?.to_dtype(DType::F32)?;
    let ssm_state = ssm_state.contiguous()?.to_dtype(DType::F32)?;
    let k_state = k_state.contiguous()?.to_dtype(DType::F32)?;
    let v_state = v_state.contiguous()?.to_dtype(DType::F32)?;

    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("expected cuda device"),
    };

    let func = dev.get_or_load_custom_func(
        "mamba3_siso_step_f32",
        "mamba3",
        cuda_kernels::MAMBA3_KERNELS,
    )?;

    macro_rules! slice {
        ($t:expr) => {{
            let s = $t.storage().as_cuda_storage()?.as_cuda_slice::<f32>()?;
            s.slice($t.layout().start_offset()..)
        }};
    }

    let cfg = LaunchConfig {
        grid_dim: (nheads as u32, batch as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&slice!(out));
    builder.arg(&slice!(out_angle));
    builder.arg(&slice!(out_ssm));
    builder.arg(&slice!(out_k));
    builder.arg(&slice!(q));
    builder.arg(&slice!(k));
    builder.arg(&slice!(v));
    builder.arg(&slice!(adt));
    builder.arg(&slice!(dt));
    builder.arg(&slice!(trap));
    builder.arg(&slice!(q_bias));
    builder.arg(&slice!(k_bias));
    builder.arg(&slice!(angles));
    builder.arg(&slice!(d_t));
    builder.arg(&slice!(z_t));
    builder.arg(&slice!(angle_state));
    builder.arg(&slice!(ssm_state));
    builder.arg(&slice!(k_state));
    builder.arg(&slice!(v_state));
    candle::builder_arg!(
        builder,
        batch as u32,
        nheads as u32,
        headdim_qk as u32,
        headdim_v as u32,
        headdim_angles as u32,
        d.is_some() as u32,
        z.is_some() as u32
    );
    unsafe { builder.launch(cfg) }.w()?;

    Ok(Mamba3StepOutput {
        out,
        angle_state: out_angle,
        ssm_state: out_ssm,
        k_state: out_k,
    })
}

pub fn mamba3_siso_step_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    adt: &Tensor,
    dt: &Tensor,
    trap: &Tensor,
    q_bias: &Tensor,
    k_bias: &Tensor,
    angles: &Tensor,
    d: Option<&Tensor>,
    z: Option<&Tensor>,
    angle_state: &Tensor,
    ssm_state: &Tensor,
    k_state: &Tensor,
    v_state: &Tensor,
) -> Result<Mamba3StepOutput> {
    match q.dtype() {
        DType::F32 => launch_siso_step_f32(
            q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, angle_state, ssm_state,
            k_state, v_state,
        ),
        dtype => {
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let out = launch_siso_step_f32(
                &q,
                &k.to_dtype(DType::F32)?,
                &v.to_dtype(DType::F32)?,
                &adt.to_dtype(DType::F32)?,
                &dt.to_dtype(DType::F32)?,
                &trap.to_dtype(DType::F32)?,
                &q_bias.to_dtype(DType::F32)?,
                &k_bias.to_dtype(DType::F32)?,
                &angles.to_dtype(DType::F32)?,
                d.map(|t| t.to_dtype(DType::F32).unwrap()).as_ref(),
                z.map(|t| t.to_dtype(DType::F32).unwrap()).as_ref(),
                &angle_state.to_dtype(DType::F32)?,
                &ssm_state.to_dtype(DType::F32)?,
                &k_state.to_dtype(DType::F32)?,
                &v_state.to_dtype(DType::F32)?,
            )?;
            Ok(Mamba3StepOutput {
                out: out.out.to_dtype(dtype)?,
                angle_state: out.angle_state.to_dtype(dtype)?,
                ssm_state: out.ssm_state.to_dtype(dtype)?,
                k_state: out.k_state.to_dtype(dtype)?,
            })
        }
    }
}

pub fn mamba3_siso_fwd_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    adt: &Tensor,
    dt: &Tensor,
    trap: &Tensor,
    q_bias: &Tensor,
    k_bias: &Tensor,
    angles: &Tensor,
    d: Option<&Tensor>,
    z: Option<&Tensor>,
    angle_state: Option<&Tensor>,
    ssm_state: Option<&Tensor>,
    k_state: Option<&Tensor>,
    v_state: Option<&Tensor>,
) -> Result<Mamba3FwdOutput> {
    // Prefill uses sequential CUDA steps (same semantics as CPU reference).
    let (batch, seqlen, nheads, _) = q.dims4()?;
    let device = q.device();
    let dtype = q.dtype();
    let headdim_v = v.dim(D::Minus1)?;
    let d_state = q.dim(D::Minus1)?;

    let mut angle_s = match angle_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, angles.dim(D::Minus1)?), dtype, device)?,
    };
    let mut ssm_s = match ssm_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, headdim_v, d_state), dtype, device)?,
    };
    let mut k_s = match k_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, d_state), dtype, device)?,
    };
    let mut v_s = match v_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, headdim_v), dtype, device)?,
    };

    let mut outs = Vec::with_capacity(seqlen);
    for t in 0..seqlen {
        let step = mamba3_siso_step_cuda(
            &q.i((.., t))?,
            &k.i((.., t))?,
            &v.i((.., t))?,
            &adt.i((.., t))?,
            &dt.i((.., t))?,
            &trap.i((.., t))?,
            q_bias,
            k_bias,
            &angles.i((.., t))?,
            d,
            z.map(|z| z.i((.., t)).unwrap()).as_ref(),
            &angle_s,
            &ssm_s,
            &k_s,
            &v_s,
        )?;
        angle_s = step.angle_state;
        ssm_s = step.ssm_state;
        k_s = step.k_state;
        v_s = v.i((.., t))?;
        outs.push(step.out.unsqueeze(1)?);
    }

    Ok(Mamba3FwdOutput {
        out: Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1)?,
        angle_state: angle_s,
        ssm_state: ssm_s,
        k_state: k_s,
        v_state: v_s,
    })
}

pub fn mamba3_mimo_step_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    adt: &Tensor,
    dt: &Tensor,
    trap: &Tensor,
    q_bias: &Tensor,
    k_bias: &Tensor,
    angles: &Tensor,
    mimo_v: &Tensor,
    mimo_z: Option<&Tensor>,
    mimo_o: Option<&Tensor>,
    d: Option<&Tensor>,
    z: Option<&Tensor>,
    angle_state: &Tensor,
    ssm_state: &Tensor,
    k_state: &Tensor,
    v_state: &Tensor,
) -> Result<Mamba3StepOutput> {
    let _ = (mimo_z, mimo_o);
    super::cpu::mamba3_mimo_step(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_v, mimo_z, mimo_o, d, z,
        angle_state, ssm_state, k_state, v_state,
    )
}

pub fn mamba3_mimo_fwd_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    adt: &Tensor,
    dt: &Tensor,
    trap: &Tensor,
    q_bias: &Tensor,
    k_bias: &Tensor,
    angles: &Tensor,
    mimo_v: &Tensor,
    mimo_z: Option<&Tensor>,
    mimo_o: Option<&Tensor>,
    d: Option<&Tensor>,
    z: Option<&Tensor>,
    angle_state: Option<&Tensor>,
    ssm_state: Option<&Tensor>,
    k_state: Option<&Tensor>,
    v_state: Option<&Tensor>,
) -> Result<Mamba3FwdOutput> {
    super::cpu::mamba3_mimo_fwd(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_v, mimo_z, mimo_o, d, z,
        angle_state, ssm_state, k_state, v_state,
    )
}
