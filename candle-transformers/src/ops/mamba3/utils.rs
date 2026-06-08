//! Shared helpers for Mamba-3 operators.

use candle::{Result, Tensor, D};

const PI: f32 = std::f32::consts::PI;

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    Ok(((x.exp()? + 1.)?).log()?)
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones(x.shape(), x.dtype(), x.device())?;
    Ok((one / (x.neg()? + 1.)?)?)
}

pub fn tanh(x: &Tensor) -> Result<Tensor> {
    let two = Tensor::new(2f32, x.device())?.to_dtype(x.dtype())?;
    let exp2x = x.broadcast_mul(&two)?.exp()?;
    let num = (&exp2x - 1.)?;
    let den = (&exp2x + 1.)?;
    Ok((num / den)?)
}

/// Apply pairwise RoPE to the last dimension (must be even).
pub fn apply_rope_pairwise(x: &Tensor, angles: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    if d % 2 != 0 {
        candle::bail!("apply_rope_pairwise requires even last dim, got {d}");
    }
    let half = d / 2;
    let x0 = x.narrow(D::Minus1, 0, half)?;
    let x1 = x.narrow(D::Minus1, half, half)?;
    let cos_a = angles.cos()?;
    let sin_a = angles.sin()?;
    let y0 = (x0.broadcast_mul(&cos_a)? - x1.broadcast_mul(&sin_a)?)?;
    let y1 = (x0.broadcast_mul(&sin_a)? + x1.broadcast_mul(&cos_a)?)?;
    Tensor::cat(&[&y0, &y1], D::Minus1)
}

/// Apply half-split RoPE (first/second half rotation) used by MIMO.
pub fn apply_rope_half_split(x: &Tensor, angles: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    if d % 2 != 0 {
        candle::bail!("apply_rope_half_split requires even last dim, got {d}");
    }
    let half = d / 2;
    let x_first = x.narrow(D::Minus1, 0, half)?;
    let x_second = x.narrow(D::Minus1, half, half)?;
    let cos_a = angles.cos()?;
    let sin_a = angles.sin()?;
    let y_first = (x_first.broadcast_mul(&cos_a)? - x_second.broadcast_mul(&sin_a)?)?;
    let y_second = (x_first.broadcast_mul(&sin_a)? + x_second.broadcast_mul(&cos_a)?)?;
    Tensor::cat(&[&y_first, &y_second], D::Minus1)
}

/// Compute updated angle state: angle_state + tanh(angle_proj) * dt * pi.
pub fn update_angle_state(angle_state: &Tensor, angle_proj: &Tensor, dt: &Tensor) -> Result<Tensor> {
    let dt = dt.unsqueeze(D::Minus1)?;
    let pi = Tensor::new(PI, angle_proj.device())?.to_dtype(angle_proj.dtype())?;
    let delta = tanh(angle_proj)?.broadcast_mul(&dt)?.broadcast_mul(&pi)?;
    angle_state.broadcast_add(&delta)
}

pub fn trapezoidal_coeffs(adt: &Tensor, dt: &Tensor, trap_raw: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let trap = sigmoid(trap_raw)?;
    let alpha = adt.exp()?;
    let one = Tensor::ones(trap.shape(), trap.dtype(), trap.device())?;
    let one_minus_trap = (&one - &trap)?;
    let beta = (alpha.broadcast_mul(dt)? * &one_minus_trap)?;
    let gamma = (&trap * dt)?;
    Ok((alpha, beta, gamma))
}
