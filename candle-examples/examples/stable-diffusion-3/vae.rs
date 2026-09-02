use anyhow::{Ok, Result};
use candle_transformers::models::stable_diffusion::vae;

pub fn build_sd3_vae_autoencoder(vb: candle_nn::VarBuilder) -> Result<vae::AutoEncoderKL> {
    let config = vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 16,
        norm_num_groups: 32,
        use_quant_conv: false,
        use_post_quant_conv: false,
    };
    Ok(vae::AutoEncoderKL::new(vb, 3, 3, config)?)
}

pub fn sd3_vae_vb_rename(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < parts.len() {
        match parts[i] {
            "down_blocks" => {
                result.push("down");
            }
            "mid_block" => {
                result.push("mid");
            }
            "up_blocks" => {
                result.push("up");
                match parts[i + 1] {
                    // Reverse the order of up_blocks.
                    "0" => result.push("3"),
                    "1" => result.push("2"),
                    "2" => result.push("1"),
                    "3" => result.push("0"),
                    _ => {}
                }
                i += 1; // Skip the number after up_blocks.
            }
            "resnets" => {
                if i > 0 && parts[i - 1] == "mid_block" {
                    match parts[i + 1] {
                        "0" => result.push("block_1"),
                        "1" => result.push("block_2"),
                        _ => {}
                    }
                    i += 1; // Skip the number after resnets.
                } else {
                    result.push("block");
                }
            }
            "downsamplers" => {
                result.push("downsample");
                i += 1; // Skip the 0 after downsamplers.
            }
            "conv_shortcut" => {
                result.push("nin_shortcut");
            }
            "attentions" => {
                if parts[i + 1] == "0" {
                    result.push("attn_1")
                }
                i += 1; // Skip the number after attentions.
            }
            "group_norm" => {
                result.push("norm");
            }
            "query" => {
                result.push("q");
            }
            "key" => {
                result.push("k");
            }
            "value" => {
                result.push("v");
            }
            "proj_attn" => {
                result.push("proj_out");
            }
            "conv_norm_out" => {
                result.push("norm_out");
            }
            "upsamplers" => {
                result.push("upsample");
                i += 1; // Skip the 0 after upsamplers.
            }
            part => result.push(part),
        }
        i += 1;
    }
    result.join(".")
}
