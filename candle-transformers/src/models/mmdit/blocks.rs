use candle::{Module, Result, Tensor, D};
use candle_nn as nn;

use super::projections::{AttnProjections, Mlp, Qkv, QkvOnlyAttnProjections};

pub struct ModulateIntermediates {
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
}

pub struct DiTBlock {
    norm1: LayerNormNoAffine,
    attn: AttnProjections,
    norm2: LayerNormNoAffine,
    mlp: Mlp,
    ada_ln_modulation: nn::Sequential,
}

pub struct LayerNormNoAffine {
    eps: f64,
}

impl LayerNormNoAffine {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }
}

impl Module for LayerNormNoAffine {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        nn::LayerNorm::new_no_bias(Tensor::ones_like(x)?, self.eps).forward(x)
    }
}

impl DiTBlock {
    pub fn new(hidden_size: usize, num_heads: usize, vb: nn::VarBuilder) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = AttnProjections::new(hidden_size, num_heads, vb.pp("attn"))?;
        let norm2 = LayerNormNoAffine::new(1e-6);
        let mlp_ratio = 4;
        let mlp = Mlp::new(hidden_size, hidden_size * mlp_ratio, vb.pp("mlp"))?;
        let n_mods = 6;
        let ada_ln_modulation = nn::seq().add(nn::Activation::Silu).add(nn::linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
        )?);

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            ada_ln_modulation,
        })
    }

    pub fn pre_attention(&self, x: &Tensor, c: &Tensor) -> Result<(Qkv, ModulateIntermediates)> {
        let modulation = self.ada_ln_modulation.forward(c)?;
        let chunks = modulation.chunk(6, D::Minus1)?;
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            chunks[0].clone(),
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
            chunks[4].clone(),
            chunks[5].clone(),
        );

        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        let qkv = self.attn.pre_attention(&modulated_x)?;

        Ok((
            qkv,
            ModulateIntermediates {
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            },
        ))
    }

    pub fn post_attention(
        &self,
        attn: &Tensor,
        x: &Tensor,
        mod_interm: &ModulateIntermediates,
    ) -> Result<Tensor> {
        let attn_out = self.attn.post_attention(attn)?;
        let x = x.add(&attn_out.broadcast_mul(&mod_interm.gate_msa.unsqueeze(1)?)?)?;

        let norm_x = self.norm2.forward(&x)?;
        let modulated_x = modulate(&norm_x, &mod_interm.shift_mlp, &mod_interm.scale_mlp)?;
        let mlp_out = self.mlp.forward(&modulated_x)?;
        let x = x.add(&mlp_out.broadcast_mul(&mod_interm.gate_mlp.unsqueeze(1)?)?)?;

        Ok(x)
    }
}

pub struct SelfAttnModulateIntermediates {
    gate_msa: Tensor,
    shift_mlp: Tensor,
    scale_mlp: Tensor,
    gate_mlp: Tensor,
    gate_msa2: Tensor,
}

pub struct SelfAttnDiTBlock {
    norm1: LayerNormNoAffine,
    attn: AttnProjections,
    attn2: AttnProjections,
    norm2: LayerNormNoAffine,
    mlp: Mlp,
    ada_ln_modulation: nn::Sequential,
}

impl SelfAttnDiTBlock {
    pub fn new(hidden_size: usize, num_heads: usize, vb: nn::VarBuilder) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = AttnProjections::new(hidden_size, num_heads, vb.pp("attn"))?;
        let attn2 = AttnProjections::new(hidden_size, num_heads, vb.pp("attn2"))?;
        let norm2 = LayerNormNoAffine::new(1e-6);
        let mlp_ratio = 4;
        let mlp = Mlp::new(hidden_size, hidden_size * mlp_ratio, vb.pp("mlp"))?;
        let n_mods = 9;
        let ada_ln_modulation = nn::seq().add(nn::Activation::Silu).add(nn::linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
        )?);

        Ok(Self {
            norm1,
            attn,
            attn2,
            norm2,
            mlp,
            ada_ln_modulation,
        })
    }

    pub fn pre_attention(
        &self,
        x: &Tensor,
        c: &Tensor,
    ) -> Result<(Qkv, Qkv, SelfAttnModulateIntermediates)> {
        let modulation = self.ada_ln_modulation.forward(c)?;
        let chunks = modulation.chunk(9, D::Minus1)?;
        let (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = (
            chunks[0].clone(),
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
            chunks[4].clone(),
            chunks[5].clone(),
            chunks[6].clone(),
            chunks[7].clone(),
            chunks[8].clone(),
        );

        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        let qkv = self.attn.pre_attention(&modulated_x)?;

        let modulated_x2 = modulate(&norm_x, &shift_msa2, &scale_msa2)?;
        let qkv2 = self.attn2.pre_attention(&modulated_x2)?;

        Ok((
            qkv,
            qkv2,
            SelfAttnModulateIntermediates {
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                gate_msa2,
            },
        ))
    }

    pub fn post_attention(
        &self,
        attn: &Tensor,
        attn2: &Tensor,
        x: &Tensor,
        mod_interm: &SelfAttnModulateIntermediates,
    ) -> Result<Tensor> {
        let attn_out = self.attn.post_attention(attn)?;
        let x = x.add(&attn_out.broadcast_mul(&mod_interm.gate_msa.unsqueeze(1)?)?)?;
        let attn_out2 = self.attn2.post_attention(attn2)?;
        let x = x.add(&attn_out2.broadcast_mul(&mod_interm.gate_msa2.unsqueeze(1)?)?)?;

        let norm_x = self.norm2.forward(&x)?;
        let modulated_x = modulate(&norm_x, &mod_interm.shift_mlp, &mod_interm.scale_mlp)?;
        let mlp_out = self.mlp.forward(&modulated_x)?;
        let x = x.add(&mlp_out.broadcast_mul(&mod_interm.gate_mlp.unsqueeze(1)?)?)?;
        Ok(x)
    }
}

pub struct QkvOnlyDiTBlock {
    norm1: LayerNormNoAffine,
    attn: QkvOnlyAttnProjections,
    ada_ln_modulation: nn::Sequential,
}

impl QkvOnlyDiTBlock {
    pub fn new(hidden_size: usize, num_heads: usize, vb: nn::VarBuilder) -> Result<Self> {
        let norm1 = LayerNormNoAffine::new(1e-6);
        let attn = QkvOnlyAttnProjections::new(hidden_size, num_heads, vb.pp("attn"))?;
        let n_mods = 2;
        let ada_ln_modulation = nn::seq().add(nn::Activation::Silu).add(nn::linear(
            hidden_size,
            n_mods * hidden_size,
            vb.pp("adaLN_modulation.1"),
        )?);

        Ok(Self {
            norm1,
            attn,
            ada_ln_modulation,
        })
    }

    pub fn pre_attention(&self, x: &Tensor, c: &Tensor) -> Result<Qkv> {
        let modulation = self.ada_ln_modulation.forward(c)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift_msa, scale_msa) = (chunks[0].clone(), chunks[1].clone());

        let norm_x = self.norm1.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift_msa, &scale_msa)?;
        self.attn.pre_attention(&modulated_x)
    }
}

pub struct FinalLayer {
    norm_final: LayerNormNoAffine,
    linear: nn::Linear,
    ada_ln_modulation: nn::Sequential,
}

impl FinalLayer {
    pub fn new(
        hidden_size: usize,
        patch_size: usize,
        out_channels: usize,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let norm_final = LayerNormNoAffine::new(1e-6);
        let linear = nn::linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            vb.pp("linear"),
        )?;
        let ada_ln_modulation = nn::seq().add(nn::Activation::Silu).add(nn::linear(
            hidden_size,
            2 * hidden_size,
            vb.pp("adaLN_modulation.1"),
        )?);

        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let modulation = self.ada_ln_modulation.forward(c)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift, scale) = (chunks[0].clone(), chunks[1].clone());

        let norm_x = self.norm_final.forward(x)?;
        let modulated_x = modulate(&norm_x, &shift, &scale)?;
        let output = self.linear.forward(&modulated_x)?;

        Ok(output)
    }
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let shift = shift.unsqueeze(1)?;
    let scale = scale.unsqueeze(1)?;
    let scale_plus_one = scale.add(&Tensor::ones_like(&scale)?)?;
    shift.broadcast_add(&x.broadcast_mul(&scale_plus_one)?)
}

pub trait JointBlock {
    fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)>;
}

pub struct MMDiTJointBlock {
    x_block: DiTBlock,
    context_block: DiTBlock,
    num_heads: usize,
    use_flash_attn: bool,
}

impl MMDiTJointBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let x_block = DiTBlock::new(hidden_size, num_heads, vb.pp("x_block"))?;
        let context_block = DiTBlock::new(hidden_size, num_heads, vb.pp("context_block"))?;

        Ok(Self {
            x_block,
            context_block,
            num_heads,
            use_flash_attn,
        })
    }
}

impl JointBlock for MMDiTJointBlock {
    fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let (context_qkv, context_interm) = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_interm) = self.x_block.pre_attention(x, c)?;
        let (context_attn, x_attn) =
            joint_attn(&context_qkv, &x_qkv, self.num_heads, self.use_flash_attn)?;
        let context_out =
            self.context_block
                .post_attention(&context_attn, context, &context_interm)?;
        let x_out = self.x_block.post_attention(&x_attn, x, &x_interm)?;
        Ok((context_out, x_out))
    }
}

pub struct MMDiTXJointBlock {
    x_block: SelfAttnDiTBlock,
    context_block: DiTBlock,
    num_heads: usize,
    use_flash_attn: bool,
}

impl MMDiTXJointBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let x_block = SelfAttnDiTBlock::new(hidden_size, num_heads, vb.pp("x_block"))?;
        let context_block = DiTBlock::new(hidden_size, num_heads, vb.pp("context_block"))?;

        Ok(Self {
            x_block,
            context_block,
            num_heads,
            use_flash_attn,
        })
    }
}

impl JointBlock for MMDiTXJointBlock {
    fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let (context_qkv, context_interm) = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_qkv2, x_interm) = self.x_block.pre_attention(x, c)?;
        let (context_attn, x_attn) =
            joint_attn(&context_qkv, &x_qkv, self.num_heads, self.use_flash_attn)?;
        let x_attn2 = attn(&x_qkv2, self.num_heads, self.use_flash_attn)?;
        let context_out =
            self.context_block
                .post_attention(&context_attn, context, &context_interm)?;
        let x_out = self
            .x_block
            .post_attention(&x_attn, &x_attn2, x, &x_interm)?;
        Ok((context_out, x_out))
    }
}

pub struct ContextQkvOnlyJointBlock {
    x_block: DiTBlock,
    context_block: QkvOnlyDiTBlock,
    num_heads: usize,
    use_flash_attn: bool,
}

impl ContextQkvOnlyJointBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let x_block = DiTBlock::new(hidden_size, num_heads, vb.pp("x_block"))?;
        let context_block = QkvOnlyDiTBlock::new(hidden_size, num_heads, vb.pp("context_block"))?;
        Ok(Self {
            x_block,
            context_block,
            num_heads,
            use_flash_attn,
        })
    }

    pub fn forward(&self, context: &Tensor, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let context_qkv = self.context_block.pre_attention(context, c)?;
        let (x_qkv, x_interm) = self.x_block.pre_attention(x, c)?;

        let (_, x_attn) = joint_attn(&context_qkv, &x_qkv, self.num_heads, self.use_flash_attn)?;

        let x_out = self.x_block.post_attention(&x_attn, x, &x_interm)?;
        Ok(x_out)
    }
}

// A QKV-attention that is compatible with the interface of candle_flash_attn::flash_attn
// Flash attention regards q, k, v dimensions as (batch_size, seqlen, nheads, headdim)
fn flash_compatible_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let q_dims_for_matmul = q.transpose(1, 2)?.dims().to_vec();
    let rank = q_dims_for_matmul.len();
    let q = q.transpose(1, 2)?.flatten_to(rank - 3)?;
    let k = k.transpose(1, 2)?.flatten_to(rank - 3)?;
    let v = v.transpose(1, 2)?.flatten_to(rank - 3)?;
    let attn_weights = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    attn_scores.reshape(q_dims_for_matmul)?.transpose(1, 2)
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

fn joint_attn(
    context_qkv: &Qkv,
    x_qkv: &Qkv,
    num_heads: usize,
    use_flash_attn: bool,
) -> Result<(Tensor, Tensor)> {
    let qkv = Qkv {
        q: Tensor::cat(&[&context_qkv.q, &x_qkv.q], 1)?,
        k: Tensor::cat(&[&context_qkv.k, &x_qkv.k], 1)?,
        v: Tensor::cat(&[&context_qkv.v, &x_qkv.v], 1)?,
    };

    let seqlen = qkv.q.dim(1)?;
    let attn = attn(&qkv, num_heads, use_flash_attn)?;
    let context_qkv_seqlen = context_qkv.q.dim(1)?;
    let context_attn = attn.narrow(1, 0, context_qkv_seqlen)?;
    let x_attn = attn.narrow(1, context_qkv_seqlen, seqlen - context_qkv_seqlen)?;

    Ok((context_attn, x_attn))
}

fn attn(qkv: &Qkv, num_heads: usize, use_flash_attn: bool) -> Result<Tensor> {
    let batch_size = qkv.q.dim(0)?;
    let seqlen = qkv.q.dim(1)?;
    let qkv = Qkv {
        q: qkv.q.reshape((batch_size, seqlen, num_heads, ()))?,
        k: qkv.k.reshape((batch_size, seqlen, num_heads, ()))?,
        v: qkv.v.clone(),
    };

    let headdim = qkv.q.dim(D::Minus1)?;
    let softmax_scale = 1.0 / (headdim as f64).sqrt();
    let attn = if use_flash_attn {
        flash_attn(&qkv.q, &qkv.k, &qkv.v, softmax_scale as f32, false)?
    } else {
        flash_compatible_attention(&qkv.q, &qkv.k, &qkv.v, softmax_scale as f32)?
    };
    attn.reshape((batch_size, seqlen, ()))
}
