// Implement the MMDiT model originally introduced for Stable Diffusion 3 (https://arxiv.org/abs/2403.03206),
// as well as the MMDiT-X variant introduced for Stable Diffusion 3.5-medium (https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
// This follows the implementation of the MMDiT model in the ComfyUI repository.
// https://github.com/comfyanonymous/ComfyUI/blob/78e133d0415784924cd2674e2ee48f3eeca8a2aa/comfy/ldm/modules/diffusionmodules/mmdit.py#L1
// with MMDiT-X support following the Stability-AI/sd3.5 repository.
// https://github.com/Stability-AI/sd3.5/blob/4e484e05308d83fb77ae6f680028e6c313f9da54/mmditx.py#L1
use candle::{Module, Result, Tensor, D};
use candle_nn as nn;

use super::blocks::{
    ContextQkvOnlyJointBlock, FinalLayer, JointBlock, MMDiTJointBlock, MMDiTXJointBlock,
};
use super::embedding::{
    PatchEmbedder, PositionEmbedder, TimestepEmbedder, Unpatchifier, VectorEmbedder,
};

#[derive(Debug, Clone)]
pub struct Config {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub depth: usize,
    pub head_size: usize,
    pub adm_in_channels: usize,
    pub pos_embed_max_size: usize,
    pub context_embed_size: usize,
    pub frequency_embedding_size: usize,
}

impl Config {
    pub fn sd3_medium() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            depth: 24,
            head_size: 64,
            adm_in_channels: 2048,
            pos_embed_max_size: 192,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }

    pub fn sd3_5_medium() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            depth: 24,
            head_size: 64,
            adm_in_channels: 2048,
            pos_embed_max_size: 384,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }

    pub fn sd3_5_large() -> Self {
        Self {
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            depth: 38,
            head_size: 64,
            adm_in_channels: 2048,
            pos_embed_max_size: 192,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }
}

pub struct MMDiT {
    core: MMDiTCore,
    patch_embedder: PatchEmbedder,
    pos_embedder: PositionEmbedder,
    timestep_embedder: TimestepEmbedder,
    vector_embedder: VectorEmbedder,
    context_embedder: nn::Linear,
    unpatchifier: Unpatchifier,
}

impl MMDiT {
    pub fn new(cfg: &Config, use_flash_attn: bool, vb: nn::VarBuilder) -> Result<Self> {
        let hidden_size = cfg.head_size * cfg.depth;
        let core = MMDiTCore::new(
            cfg.depth,
            hidden_size,
            cfg.depth,
            cfg.patch_size,
            cfg.out_channels,
            use_flash_attn,
            vb.clone(),
        )?;
        let patch_embedder = PatchEmbedder::new(
            cfg.patch_size,
            cfg.in_channels,
            hidden_size,
            vb.pp("x_embedder"),
        )?;
        let pos_embedder = PositionEmbedder::new(
            hidden_size,
            cfg.patch_size,
            cfg.pos_embed_max_size,
            vb.clone(),
        )?;
        let timestep_embedder = TimestepEmbedder::new(
            hidden_size,
            cfg.frequency_embedding_size,
            vb.pp("t_embedder"),
        )?;
        let vector_embedder =
            VectorEmbedder::new(cfg.adm_in_channels, hidden_size, vb.pp("y_embedder"))?;
        let context_embedder = nn::linear(
            cfg.context_embed_size,
            hidden_size,
            vb.pp("context_embedder"),
        )?;
        let unpatchifier = Unpatchifier::new(cfg.patch_size, cfg.out_channels)?;

        Ok(Self {
            core,
            patch_embedder,
            pos_embedder,
            timestep_embedder,
            vector_embedder,
            context_embedder,
            unpatchifier,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        // Following the convention of the ComfyUI implementation.
        // https://github.com/comfyanonymous/ComfyUI/blob/78e133d0415784924cd2674e2ee48f3eeca8a2aa/comfy/ldm/modules/diffusionmodules/mmdit.py#L919
        //
        // Forward pass of DiT.
        // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        // t: (N,) tensor of diffusion timesteps
        // y: (N,) tensor of class labels
        let h = x.dim(D::Minus2)?;
        let w = x.dim(D::Minus1)?;
        let cropped_pos_embed = self.pos_embedder.get_cropped_pos_embed(h, w)?;
        let x = self
            .patch_embedder
            .forward(x)?
            .broadcast_add(&cropped_pos_embed)?;
        let c = self.timestep_embedder.forward(t)?;
        let y = self.vector_embedder.forward(y)?;
        let c = (c + y)?;
        let context = self.context_embedder.forward(context)?;

        let x = self.core.forward(&context, &x, &c, skip_layers)?;
        let x = self.unpatchifier.unpatchify(&x, h, w)?;
        x.narrow(2, 0, h)?.narrow(3, 0, w)
    }
}

pub struct MMDiTCore {
    joint_blocks: Vec<Box<dyn JointBlock>>,
    context_qkv_only_joint_block: ContextQkvOnlyJointBlock,
    final_layer: FinalLayer,
}

impl MMDiTCore {
    pub fn new(
        depth: usize,
        hidden_size: usize,
        num_heads: usize,
        patch_size: usize,
        out_channels: usize,
        use_flash_attn: bool,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let mut joint_blocks = Vec::with_capacity(depth - 1);
        for i in 0..depth - 1 {
            let joint_block_vb_pp = format!("joint_blocks.{}", i);
            let joint_block: Box<dyn JointBlock> =
                if vb.contains_tensor(&format!("{}.x_block.attn2.qkv.weight", joint_block_vb_pp)) {
                    Box::new(MMDiTXJointBlock::new(
                        hidden_size,
                        num_heads,
                        use_flash_attn,
                        vb.pp(&joint_block_vb_pp),
                    )?)
                } else {
                    Box::new(MMDiTJointBlock::new(
                        hidden_size,
                        num_heads,
                        use_flash_attn,
                        vb.pp(&joint_block_vb_pp),
                    )?)
                };
            joint_blocks.push(joint_block);
        }

        Ok(Self {
            joint_blocks,
            context_qkv_only_joint_block: ContextQkvOnlyJointBlock::new(
                hidden_size,
                num_heads,
                use_flash_attn,
                vb.pp(format!("joint_blocks.{}", depth - 1)),
            )?,
            final_layer: FinalLayer::new(
                hidden_size,
                patch_size,
                out_channels,
                vb.pp("final_layer"),
            )?,
        })
    }

    pub fn forward(
        &self,
        context: &Tensor,
        x: &Tensor,
        c: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        let (mut context, mut x) = (context.clone(), x.clone());
        for (i, joint_block) in self.joint_blocks.iter().enumerate() {
            if let Some(skip_layers) = &skip_layers {
                if skip_layers.contains(&i) {
                    continue;
                }
            }
            (context, x) = joint_block.forward(&context, &x, c)?;
        }
        let x = self.context_qkv_only_joint_block.forward(&context, &x, c)?;
        self.final_layer.forward(&x, c)
    }
}
