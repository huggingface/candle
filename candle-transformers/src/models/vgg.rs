//! VGG-16 model implementation.
//!
//! See Very Deep Convolutional Networks for Large-Scale Image Recognition
//! <https://arxiv.org/abs/1409.1556>
use candle::{ModuleT, Result, Tensor};
use candle_nn::{FuncT, VarBuilder};

// Enum representing the different VGG models
pub enum Models {
    Vgg13,
    Vgg16,
    Vgg19,
}

// Struct representing a VGG model
#[derive(Debug)]
pub struct Vgg<'a> {
    blocks: Vec<FuncT<'a>>,
}

// Struct representing the configuration for the pre-logit layer
struct PreLogitConfig {
    in_dim: (usize, usize, usize, usize),
    target_in: usize,
    target_out: usize,
}

// Implementation of the VGG model
impl<'a> Vgg<'a> {
    // Function to create a new VGG model
    pub fn new(vb: VarBuilder<'a>, model: Models) -> Result<Self> {
        let blocks = match model {
            Models::Vgg13 => vgg13_blocks(vb)?,
            Models::Vgg16 => vgg16_blocks(vb)?,
            Models::Vgg19 => vgg19_blocks(vb)?,
        };
        Ok(Self { blocks })
    }
}

// Implementation of the forward pass for the VGG model
impl ModuleT for Vgg<'_> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.unsqueeze(0)?;
        for block in self.blocks.iter() {
            xs = xs.apply_t(block, train)?;
        }
        Ok(xs)
    }
}

// Function to create a conv2d block
// The block is composed of two conv2d layers followed by a max pool layer
fn conv2d_block(convs: &[(usize, usize, &str)], vb: &VarBuilder) -> Result<FuncT<'static>> {
    let layers = convs
        .iter()
        .enumerate()
        .map(|(_, &(in_c, out_c, name))| {
            candle_nn::conv2d(
                in_c,
                out_c,
                3,
                candle_nn::Conv2dConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
                vb.pp(name),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(FuncT::new(move |xs, _train| {
        let mut xs = xs.clone();
        for layer in layers.iter() {
            xs = xs.apply(layer)?.relu()?
        }
        xs = xs.max_pool2d_with_stride(2, 2)?;
        Ok(xs)
    }))
}

// Function to create a fully connected layer
// The layer is composed of two linear layers followed by a dropout layer
fn fully_connected(
    num_classes: usize,
    pre_logit_1: PreLogitConfig,
    pre_logit_2: PreLogitConfig,
    vb: VarBuilder,
) -> Result<FuncT> {
    let lin = get_weights_and_biases(
        &vb.pp("pre_logits.fc1"),
        pre_logit_1.in_dim,
        pre_logit_1.target_in,
        pre_logit_1.target_out,
    )?;
    let lin2 = get_weights_and_biases(
        &vb.pp("pre_logits.fc2"),
        pre_logit_2.in_dim,
        pre_logit_2.target_in,
        pre_logit_2.target_out,
    )?;
    let dropout1 = candle_nn::Dropout::new(0.5);
    let dropout2 = candle_nn::Dropout::new(0.5);
    let dropout3 = candle_nn::Dropout::new(0.5);
    Ok(FuncT::new(move |xs, train| {
        let xs = xs.reshape((1, pre_logit_1.target_out))?;
        let xs = xs.apply_t(&dropout1, train)?.apply(&lin)?.relu()?;
        let xs = xs.apply_t(&dropout2, train)?.apply(&lin2)?.relu()?;
        let lin3 = candle_nn::linear(4096, num_classes, vb.pp("head.fc"))?;
        let xs = xs.apply_t(&dropout3, train)?.apply(&lin3)?.relu()?;
        Ok(xs)
    }))
}

// Function to get the weights and biases for a layer
// This is required because the weights and biases are stored in different format than our linear layer expects
fn get_weights_and_biases(
    vs: &VarBuilder,
    in_dim: (usize, usize, usize, usize),
    target_in: usize,
    target_out: usize,
) -> Result<candle_nn::Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get_with_hints(in_dim, "weight", init_ws)?;
    let ws = ws.reshape((target_in, target_out))?;
    let bound = 1. / (target_out as f64).sqrt();
    let init_bs = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vs.get_with_hints(target_in, "bias", init_bs)?;
    Ok(candle_nn::Linear::new(ws, Some(bs)))
}

fn vgg13_blocks(vb: VarBuilder) -> Result<Vec<FuncT>> {
    let num_classes = 1000;
    let blocks = vec![
        conv2d_block(&[(3, 64, "features.0"), (64, 64, "features.2")], &vb)?,
        conv2d_block(&[(64, 128, "features.5"), (128, 128, "features.7")], &vb)?,
        conv2d_block(&[(128, 256, "features.10"), (256, 256, "features.12")], &vb)?,
        conv2d_block(&[(256, 512, "features.15"), (512, 512, "features.17")], &vb)?,
        conv2d_block(&[(512, 512, "features.20"), (512, 512, "features.22")], &vb)?,
        fully_connected(
            num_classes,
            PreLogitConfig {
                in_dim: (4096, 512, 7, 7),
                target_in: 4096,
                target_out: 512 * 7 * 7,
            },
            PreLogitConfig {
                in_dim: (4096, 4096, 1, 1),
                target_in: 4096,
                target_out: 4096,
            },
            vb.clone(),
        )?,
    ];
    Ok(blocks)
}

fn vgg16_blocks(vb: VarBuilder) -> Result<Vec<FuncT>> {
    let num_classes = 1000;
    let blocks = vec![
        conv2d_block(&[(3, 64, "features.0"), (64, 64, "features.2")], &vb)?,
        conv2d_block(&[(64, 128, "features.5"), (128, 128, "features.7")], &vb)?,
        conv2d_block(
            &[
                (128, 256, "features.10"),
                (256, 256, "features.12"),
                (256, 256, "features.14"),
            ],
            &vb,
        )?,
        conv2d_block(
            &[
                (256, 512, "features.17"),
                (512, 512, "features.19"),
                (512, 512, "features.21"),
            ],
            &vb,
        )?,
        conv2d_block(
            &[
                (512, 512, "features.24"),
                (512, 512, "features.26"),
                (512, 512, "features.28"),
            ],
            &vb,
        )?,
        fully_connected(
            num_classes,
            PreLogitConfig {
                in_dim: (4096, 512, 7, 7),
                target_in: 4096,
                target_out: 512 * 7 * 7,
            },
            PreLogitConfig {
                in_dim: (4096, 4096, 1, 1),
                target_in: 4096,
                target_out: 4096,
            },
            vb.clone(),
        )?,
    ];
    Ok(blocks)
}

fn vgg19_blocks(vb: VarBuilder) -> Result<Vec<FuncT>> {
    let num_classes = 1000;
    let blocks = vec![
        conv2d_block(&[(3, 64, "features.0"), (64, 64, "features.2")], &vb)?,
        conv2d_block(&[(64, 128, "features.5"), (128, 128, "features.7")], &vb)?,
        conv2d_block(
            &[
                (128, 256, "features.10"),
                (256, 256, "features.12"),
                (256, 256, "features.14"),
                (256, 256, "features.16"),
            ],
            &vb,
        )?,
        conv2d_block(
            &[
                (256, 512, "features.19"),
                (512, 512, "features.21"),
                (512, 512, "features.23"),
                (512, 512, "features.25"),
            ],
            &vb,
        )?,
        conv2d_block(
            &[
                (512, 512, "features.28"),
                (512, 512, "features.30"),
                (512, 512, "features.32"),
                (512, 512, "features.34"),
            ],
            &vb,
        )?,
        fully_connected(
            num_classes,
            PreLogitConfig {
                in_dim: (4096, 512, 7, 7),
                target_in: 4096,
                target_out: 512 * 7 * 7,
            },
            PreLogitConfig {
                in_dim: (4096, 4096, 1, 1),
                target_in: 4096,
                target_out: 4096,
            },
            vb.clone(),
        )?,
    ];
    Ok(blocks)
}
