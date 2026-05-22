//! CPU reference implementations for Mamba-3 inference operators.

use super::utils::{apply_rope_half_split, apply_rope_pairwise, trapezoidal_coeffs, update_angle_state};
use candle::{IndexOp, Result, Tensor, D};

#[derive(Debug, Clone)]
pub struct Mamba3StepOutput {
    pub out: Tensor,
    pub angle_state: Tensor,
    pub ssm_state: Tensor,
    pub k_state: Tensor,
}

#[derive(Debug, Clone)]
pub struct Mamba3FwdOutput {
    pub out: Tensor,
    pub angle_state: Tensor,
    pub ssm_state: Tensor,
    pub k_state: Tensor,
    pub v_state: Tensor,
}

fn ssm_matvec(ssm: &Tensor, q: &Tensor) -> Result<Tensor> {
    // ssm: (B, H, V, QK), q: (B, H, QK) -> (B, H, V)
    ssm.matmul(&q.unsqueeze(D::Minus1)?)?.squeeze(D::Minus1)
}

fn outer_update(v: &Tensor, k: &Tensor) -> Result<Tensor> {
    // v: (B, H, V), k: (B, H, QK) -> (B, H, V, QK)
    v.unsqueeze(D::Minus1)?.broadcast_mul(&k.unsqueeze(D::Minus2)?)
}

/// SISO single-token decode step (reference port of `mamba3_siso_step`).
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
    let q = q.broadcast_add(&q_bias.unsqueeze(0)?)?;
    let k = k.broadcast_add(&k_bias.unsqueeze(0)?)?;

    let out_angle = update_angle_state(angle_state, angles, dt)?;
    let q = apply_rope_pairwise(&q, &out_angle)?;
    let k = apply_rope_pairwise(&k, &out_angle)?;

    let (alpha, beta, gamma) = trapezoidal_coeffs(adt, dt, trap)?;

    let alpha = alpha.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
    let beta = beta.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
    let gamma = gamma.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;

    let diff = (beta.broadcast_mul(&outer_update(v_state, k_state)?)? + gamma.broadcast_mul(&outer_update(v, &k)?)?)?;
    let ssm_state = (ssm_state.broadcast_mul(&alpha)? + diff)?;

    let mut out = ssm_matvec(&ssm_state, &q)?;
    if let Some(d) = d {
        out = (out + v.broadcast_mul(&d.unsqueeze(0)?.unsqueeze(D::Minus1)?)?)?;
    }
    if let Some(z) = z {
        out = (out * candle_nn::ops::silu(z)?)?;
    }

    Ok(Mamba3StepOutput {
        out,
        angle_state: out_angle,
        ssm_state,
        k_state: k,
    })
}

/// SISO prefill via sequential steps (correct reference for verification).
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
    let (batch, seqlen, nheads, _) = q.dims4()?;
    let device = q.device();
    let dtype = q.dtype();

    let mut angle_s = match angle_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, angles.dim(D::Minus1)?), dtype, device)?,
    };
    let headdim_v = v.dim(D::Minus1)?;
    let d_state = q.dim(D::Minus1)?;
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
        let step = mamba3_siso_step(
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
    let rank = q.dim(1)?;
    let nheads = q.dim(2)?;
    let d_state = q.dim(3)?;
    let batch = q.dim(0)?;
    let headdim = v.dim(D::Minus1)?;

    let q_bias = q_bias.permute((1, 0, 2))?;
    let k_bias = k_bias.permute((1, 0, 2))?;

    let mut q = q.broadcast_add(&q_bias.unsqueeze(0)?)?;
    let mut k = k.broadcast_add(&k_bias.unsqueeze(0)?)?;

    let out_angle = update_angle_state(angle_state, angles, dt)?;
    q = apply_rope_half_split(&q.reshape((batch * rank, nheads, d_state))?, &out_angle)?
        .reshape((batch, rank, nheads, d_state))?;
    k = apply_rope_half_split(&k.reshape((batch * rank, nheads, d_state))?, &out_angle)?
        .reshape((batch, rank, nheads, d_state))?;

    let mimo_v = mimo_v.permute((1, 0, 2))?;
    let v_rank = v.unsqueeze(1)?.broadcast_mul(&mimo_v.unsqueeze(0)?)?;

    let (alpha, beta, gamma) = trapezoidal_coeffs(adt, dt, trap)?;

    let k_prev = k_state.unsqueeze(3)?;
    let v_prev = v_state.unsqueeze(1)?.unsqueeze(D::Minus1)?;
    let diff_prev = v_prev.broadcast_mul(&k_prev)?;

    let v_rank_outer = v_rank.unsqueeze(D::Minus1)?;
    let k_outer = k.unsqueeze(3)?;
    let diff_cur = v_rank_outer.broadcast_mul(&k_outer)?;

    let diff_sum = (beta.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?.unsqueeze(1)?
        .broadcast_mul(&diff_prev)?
        + gamma.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?.unsqueeze(1)?
            .broadcast_mul(&diff_cur)?)?
    .sum(1)?;

    let ssm_state = (ssm_state.broadcast_mul(&alpha.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)? + diff_sum)?;

    let mut out = Tensor::zeros((batch, nheads, headdim), q.dtype(), q.device())?;
    for r in 0..rank {
        let qr = q.i((.., r))?;
        let yr = ssm_matvec(&ssm_state, &qr)?;
        out = (out + yr)?;
    }

    if let Some(mimo_o) = mimo_o {
        let mimo_o = mimo_o.permute((1, 0, 2))?;
        let mut acc = Tensor::zeros(out.shape(), out.dtype(), out.device())?;
        for r in 0..rank {
            let yr = out.clone();
            acc = (acc + yr.broadcast_mul(&mimo_o.i((r, ..))?.unsqueeze(0)?)?)?;
        }
        out = acc;
    }

    if let Some(d) = d {
        out = (out + v.broadcast_mul(&d.unsqueeze(0)?.unsqueeze(D::Minus1)?)?)?;
    }
    if let (Some(z), Some(mimo_z)) = (z, mimo_z) {
        let mimo_z = mimo_z.permute((1, 0, 2))?;
        let mut z_acc = Tensor::zeros(z.shape(), z.dtype(), z.device())?;
        for r in 0..rank {
            z_acc = (z_acc + z.broadcast_mul(&mimo_z.i((r, ..))?.unsqueeze(0)?)?)?;
        }
        out = (out * candle_nn::ops::silu(&z_acc)?)?;
    } else if let Some(z) = z {
        out = (out * candle_nn::ops::silu(z)?)?;
    }

    Ok(Mamba3StepOutput {
        out,
        angle_state: out_angle,
        ssm_state,
        k_state: k,
    })
}

/// MIMO prefill via sequential MIMO steps.
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
    let (batch, seqlen, rank, nheads, _) = q.dims5()?;
    let device = q.device();
    let dtype = q.dtype();
    let headdim = v.dim(D::Minus1)?;
    let d_state = q.dim(D::Minus1)?;

    let mut angle_s = match angle_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, angles.dim(D::Minus1)?), dtype, device)?,
    };
    let mut ssm_s = match ssm_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, headdim, d_state), dtype, device)?,
    };
    let mut k_s = match k_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, rank, nheads, d_state), dtype, device)?,
    };
    let mut v_s = match v_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((batch, nheads, headdim), dtype, device)?,
    };

    let mut outs = Vec::with_capacity(seqlen);
    for t in 0..seqlen {
        let step = mamba3_mimo_step(
            &q.i((.., t))?,
            &k.i((.., t))?,
            &v.i((.., t))?,
            &adt.i((.., t))?,
            &dt.i((.., t))?,
            &trap.i((.., t))?,
            q_bias,
            k_bias,
            &angles.i((.., t))?,
            mimo_v,
            mimo_z,
            mimo_o,
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, DType};

    #[test]
    fn siso_step_runs() -> Result<()> {
        let device = Device::Cpu;
        let b = 2;
        let h = 4;
        let d_qk = 8;
        let d_v = 8;
        let q = Tensor::randn(0f32, 1f32, (b, h, d_qk), &device)?;
        let k = Tensor::randn(0f32, 1f32, (b, h, d_qk), &device)?;
        let v = Tensor::randn(0f32, 1f32, (b, h, d_v), &device)?;
        let adt = Tensor::randn(0f32, 0.1f32, (b, h), &device)?;
        let dt = (adt.abs()? + 0.01)?;
        let trap = Tensor::randn(0f32, 1f32, (b, h), &device)?;
        let q_bias = Tensor::randn(0f32, 1f32, (h, d_qk), &device)?;
        let k_bias = Tensor::randn(0f32, 1f32, (h, d_qk), &device)?;
        let angles = Tensor::randn(0f32, 0.1f32, (b, h, d_qk / 2), &device)?;
        let d = Tensor::ones(h, DType::F32, &device)?;
        let z = Tensor::randn(0f32, 1f32, (b, h, d_v), &device)?;
        let angle_state = Tensor::zeros((b, h, d_qk / 2), DType::F32, &device)?;
        let ssm_state = Tensor::zeros((b, h, d_v, d_qk), DType::F32, &device)?;
        let k_state = Tensor::zeros((b, h, d_qk), DType::F32, &device)?;
        let v_state = Tensor::zeros((b, h, d_v), DType::F32, &device)?;

        let out = mamba3_siso_step(
            &q,
            &k,
            &v,
            &adt,
            &dt,
            &trap,
            &q_bias,
            &k_bias,
            &angles,
            Some(&d),
            Some(&z),
            &angle_state,
            &ssm_state,
            &k_state,
            &v_state,
        )?;
        assert_eq!(out.out.dims(), &[b, h, d_v]);
        assert_eq!(out.angle_state.dims(), &[b, h, d_qk / 2]);
        assert_eq!(out.ssm_state.dims(), &[b, h, d_v, d_qk]);
        assert_eq!(out.k_state.dims(), &[b, h, d_qk]);
        Ok(())
    }

    #[test]
    fn siso_fwd_runs() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let h = 2;
        let d_qk = 4;
        let d_v = 4;
        let seq = 3;
        let q = Tensor::randn(0f32, 1f32, (b, seq, h, d_qk), &device)?;
        let k = Tensor::randn(0f32, 1f32, (b, seq, h, d_qk), &device)?;
        let v = Tensor::randn(0f32, 1f32, (b, seq, h, d_v), &device)?;
        let adt = Tensor::randn(0f32, 0.1f32, (b, seq, h), &device)?;
        let dt = (adt.abs()? + 0.01)?;
        let trap = Tensor::randn(0f32, 1f32, (b, seq, h), &device)?;
        let q_bias = Tensor::randn(0f32, 1f32, (h, d_qk), &device)?;
        let k_bias = Tensor::randn(0f32, 1f32, (h, d_qk), &device)?;
        let angles = Tensor::randn(0f32, 0.1f32, (b, seq, h, d_qk / 2), &device)?;
        let d = Tensor::ones(h, DType::F32, &device)?;
        let z = Tensor::randn(0f32, 1f32, (b, seq, h, d_v), &device)?;

        let out = mamba3_siso_fwd(
            &q, &k, &v, &adt, &dt, &trap, &q_bias, &k_bias, &angles,
            Some(&d), Some(&z), None, None, None, None,
        )?;
        assert_eq!(out.out.dims(), &[b, seq, h, d_v]);
        Ok(())
    }

    #[test]
    fn mimo_step_runs() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let rank = 2;
        let h = 2;
        let headdim = 4;
        let d_state = 4;
        let q = Tensor::randn(0f32, 1f32, (b, rank, h, d_state), &device)?;
        let k = Tensor::randn(0f32, 1f32, (b, rank, h, d_state), &device)?;
        let v = Tensor::randn(0f32, 1f32, (b, h, headdim), &device)?;
        let adt = Tensor::randn(0f32, 0.1f32, (b, h), &device)?;
        let dt = (adt.abs()? + 0.01)?;
        let trap = Tensor::randn(0f32, 1f32, (b, h), &device)?;
        let q_bias = Tensor::randn(0f32, 1f32, (h, rank, d_state), &device)?;
        let k_bias = Tensor::randn(0f32, 1f32, (h, rank, d_state), &device)?;
        let angles = Tensor::randn(0f32, 0.1f32, (b, h, d_state / 2), &device)?;
        let mimo_v = Tensor::randn(0f32, 1f32, (h, rank, headdim), &device)?;
        let d = Tensor::ones(h, DType::F32, &device)?;
        let z = Tensor::randn(0f32, 1f32, (b, h, headdim), &device)?;
        let angle_state = Tensor::zeros((b, h, d_state / 2), DType::F32, &device)?;
        let ssm_state = Tensor::zeros((b, h, headdim, d_state), DType::F32, &device)?;
        let k_state = Tensor::zeros((b, rank, h, d_state), DType::F32, &device)?;
        let v_state = Tensor::zeros((b, h, headdim), DType::F32, &device)?;

        let out = mamba3_mimo_step(
            &q, &k, &v, &adt, &dt, &trap, &q_bias, &k_bias, &angles,
            &mimo_v, None, None, Some(&d), Some(&z),
            &angle_state, &ssm_state, &k_state, &v_state,
        )?;
        assert_eq!(out.out.dims(), &[b, h, headdim]);
        Ok(())
    }
}
