//! F0 Predictor for CosyVoice3 HiFT
//!
//! Predicts fundamental frequency (F0) from mel spectrogram features.
//! Uses causal convolutions for streaming inference support.

use candle::{Device, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

use crate::models::cosyvoice::activations::Elu;

/// Causal type
#[derive(Debug, Clone, Copy)]
pub enum CausalType {
    /// Left padding (sees history)
    Left,
    /// Right padding (previews future)
    Right,
}

/// Causal 1D convolution
#[derive(Debug, Clone)]
pub struct CausalConv1d {
    conv: Conv1d,
    causal_type: CausalType,
    causal_padding: usize,
}

impl CausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        causal_type: CausalType,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Convolution without padding
        let conv_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = candle_nn::conv1d(in_channels, out_channels, kernel_size, conv_config, vb)?;

        // Calculate causal_padding
        // = ((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        let causal_padding = ((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2;

        Ok(Self {
            conv,
            causal_type,
            causal_padding,
        })
    }

    pub fn causal_padding(&self) -> usize {
        self.causal_padding
    }

    /// Forward pass with cache (for streaming inference)
    pub fn forward_with_cache(&self, x: &Tensor, cache: &Tensor) -> Result<Tensor> {
        assert_eq!(cache.dim(2)?, self.causal_padding);

        let x = match self.causal_type {
            CausalType::Left => Tensor::cat(&[cache, x], 2)?,
            CausalType::Right => Tensor::cat(&[x, cache], 2)?,
        };

        self.conv.forward(&x)
    }
}

impl Module for CausalConv1d {
    /// Standard forward pass (auto padding)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let cache = Tensor::zeros(
            (x.dim(0)?, x.dim(1)?, self.causal_padding),
            x.dtype(),
            x.device(),
        )?;
        self.forward_with_cache(x, &cache)
    }
}

/// Causal convolution F0 predictor
///
/// Structure: 5 layers of CausalConv1d + ELU + Linear classifier
///
/// # CPU Fallback
/// Important note from source: F0 predictor accuracy is critical for causal inference.
/// Numerical precision issues may occur on GPU, so the official implementation forces CPU execution.
#[derive(Debug)]
pub struct CausalConvRNNF0Predictor {
    /// 5 conv layers (alternating conv and activation)
    condnet: Vec<(CausalConv1d, Elu)>,
    /// Output classifier
    classifier: Linear,
    /// Input channels
    in_channels: usize,
    /// Condition channels
    cond_channels: usize,
    /// Whether to force CPU execution (recommended: true)
    force_cpu: bool,
}

impl CausalConvRNNF0Predictor {
    /// Create F0 predictor
    ///
    /// # Arguments
    /// * `in_channels` - Input channels (default 80)
    /// * `cond_channels` - Condition channels (default 512)
    /// * `force_cpu` - Whether to force CPU execution (recommended true for precision)
    /// * `vb` - VarBuilder
    pub fn new(
        in_channels: usize,
        cond_channels: usize,
        force_cpu: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut condnet = Vec::new();

        // Layer 1: in_channels -> cond_channels, kernel=4, causal_type='right'
        let conv0 = CausalConv1d::new(
            in_channels,
            cond_channels,
            4,
            1,
            CausalType::Right,
            vb.pp("condnet.0"),
        )?;
        condnet.push((conv0, Elu::new()));

        // Layers 2-5: cond_channels -> cond_channels, kernel=3, causal_type='left'
        for i in 1..5 {
            let conv = CausalConv1d::new(
                cond_channels,
                cond_channels,
                3,
                1,
                CausalType::Left,
                vb.pp(format!("condnet.{}", i * 2)),
            )?;
            condnet.push((conv, Elu::new()));
        }

        // Output layer
        let classifier = candle_nn::linear(cond_channels, 1, vb.pp("classifier"))?;

        Ok(Self {
            condnet,
            classifier,
            in_channels,
            cond_channels,
            force_cpu,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Mel spectrogram [batch, in_channels, time]
    /// * `finalize` - Whether this is the final chunk (affects padding handling)
    pub fn forward(&self, x: &Tensor, finalize: bool) -> Result<Tensor> {
        // If CPU fallback is enabled, move input to CPU first
        let original_device = x.device().clone();
        let x = if self.force_cpu && !original_device.is_cpu() {
            x.to_device(&Device::Cpu)?
        } else {
            x.clone()
        };

        let mut x = x;

        // Special handling for first layer (right causal)
        if finalize {
            x = self.condnet[0].0.forward(&x)?;
        } else {
            // Streaming mode: separate main body and cache portion
            let causal_padding = self.condnet[0].0.causal_padding();
            if x.dim(2)? > causal_padding {
                let main = x.narrow(2, 0, x.dim(2)? - causal_padding)?;
                let cache = x.narrow(2, x.dim(2)? - causal_padding, causal_padding)?;
                x = self.condnet[0].0.forward_with_cache(&main, &cache)?;
            } else {
                x = self.condnet[0].0.forward(&x)?;
            }
        }
        x = self.condnet[0].1.forward(&x)?;

        // Subsequent layers
        for (conv, elu) in self.condnet.iter().skip(1) {
            x = conv.forward(&x)?;
            x = elu.forward(&x)?;
        }

        // Classifier
        let x = x.transpose(1, 2)?; // [batch, time, channels]
        let f0 = self.classifier.forward(&x)?;
        let f0 = f0.squeeze(2)?; // [batch, time]

        // F0 must be positive
        let f0 = f0.abs()?;

        // Move result back to original device
        if self.force_cpu && !original_device.is_cpu() {
            f0.to_device(&original_device)
        } else {
            Ok(f0)
        }
    }

    /// Get configuration info
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn cond_channels(&self) -> usize {
        self.cond_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;

    #[test]
    fn test_causal_conv1d_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create dummy weights
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

        // Initialize weights
        varmap.data().lock().unwrap().insert(
            "weight".to_string(),
            candle::Var::from_tensor(&Tensor::randn(0f32, 0.1, (64, 80, 4), &device)?)?,
        );
        varmap.data().lock().unwrap().insert(
            "bias".to_string(),
            candle::Var::from_tensor(&Tensor::zeros((64,), dtype, &device)?)?,
        );

        let conv = CausalConv1d::new(80, 64, 4, 1, CausalType::Right, vb)?;

        // Test input
        let x = Tensor::randn(0f32, 1.0, (2, 80, 100), &device)?;
        let output = conv.forward(&x)?;

        // Output length should match input (causal)
        assert_eq!(output.dims(), &[2, 64, 100]);
        Ok(())
    }
}

