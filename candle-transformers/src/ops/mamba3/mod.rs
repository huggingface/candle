//! Mamba-3 GPU inference operators (SISO + MIMO).
//!
//! Reference: [state-spaces/mamba](https://github.com/state-spaces/mamba) Triton kernels
//! and the Mamba-3 paper (exponential-trapezoidal discretization + complex/RoPE state).

mod cpu;
mod params;
mod utils;

#[cfg(feature = "cuda")]
mod cuda;

pub use cpu::{Mamba3FwdOutput, Mamba3StepOutput};
pub use params::Mamba3Params;
pub use utils::{sigmoid, softplus};

use candle::{Device, Result, Tensor};

fn use_cuda(device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    {
        matches!(device, Device::Cuda(_))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        false
    }
}

/// SISO single-token decode step.
pub fn mamba3_siso_step(
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
    if use_cuda(q.device()) {
        #[cfg(feature = "cuda")]
        {
            return cuda::mamba3_siso_step_cuda(
                q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, angle_state, ssm_state,
                k_state, v_state,
            );
        }
    }
    cpu::mamba3_siso_step(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, angle_state, ssm_state, k_state,
        v_state,
    )
}

/// SISO chunked prefill forward pass.
pub fn mamba3_siso_fwd(
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
    if use_cuda(q.device()) {
        #[cfg(feature = "cuda")]
        {
            return cuda::mamba3_siso_fwd_cuda(
                q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, angle_state, ssm_state,
                k_state, v_state,
            );
        }
    }
    cpu::mamba3_siso_fwd(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, angle_state, ssm_state, k_state,
        v_state,
    )
}

/// MIMO single-token decode step.
pub fn mamba3_mimo_step(
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
    if use_cuda(q.device()) {
        #[cfg(feature = "cuda")]
        {
            return cuda::mamba3_mimo_step_cuda(
                q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_v, mimo_z, mimo_o, d, z,
                angle_state, ssm_state, k_state, v_state,
            );
        }
    }
    cpu::mamba3_mimo_step(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_v, mimo_z, mimo_o, d, z,
        angle_state, ssm_state, k_state, v_state,
    )
}

/// MIMO chunked prefill forward pass.
pub fn mamba3_mimo_fwd(
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
    if use_cuda(q.device()) {
        #[cfg(feature = "cuda")]
        {
            return cuda::mamba3_mimo_fwd_cuda(
                q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_v, mimo_z, mimo_o, d, z,
                angle_state, ssm_state, k_state, v_state,
            );
        }
    }
    cpu::mamba3_mimo_fwd(
        q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_v, mimo_z, mimo_o, d, z,
        angle_state, ssm_state, k_state, v_state,
    )
}
