use candle_lora::LoraConv2dConfigBuilder;

#[test]
fn conv2d() -> candle::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle::{DType, Device, Result, Tensor};
    use candle_lora::{Conv2dLayerLike, Lora, NewLayers, SelectedLayers};
    use candle_nn::{init, Conv2d, Conv2dConfig, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Conv,
    }

    #[derive(Debug)]
    struct Model {
        conv: Box<dyn Conv2dLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.conv.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, conv) in new.conv2d {
                match name {
                    ModelLayers::Conv => self.conv = Box::new(conv),
                }
            }
        }
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    let out_channels = 10;
    let in_channels = 10;
    let kernel = 2;

    let cfg = Conv2dConfig::default();

    //Create the model
    let map = VarMap::new();
    let conv_weight = map.get(
        (
            out_channels,
            in_channels / cfg.groups, //cfg.groups in this case are 1
            kernel,
            kernel,
        ),
        "conv.weight",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;
    let conv_bias = map.get(
        out_channels,
        "conv.bias",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;

    let mut model = Model {
        conv: Box::new(Conv2d::new(
            conv_weight.clone(),
            Some(conv_bias.clone()),
            cfg,
        )),
    };

    let shape = [2, in_channels, 20, 20]; //(BS, K, X, Y)
    let dummy_image = Tensor::zeros(&shape, DType::F32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let linear_layers = HashMap::new();
    let conv1d_layers = HashMap::new();
    let mut conv2d_layers = HashMap::new();
    conv2d_layers.insert(ModelLayers::Conv, &*model.conv);
    let embed_layers = HashMap::new();
    let selected = SelectedLayers {
        linear: linear_layers,
        linear_config: None,
        conv1d: conv1d_layers,
        conv1d_config: None,
        conv2d: conv2d_layers,
        conv2d_config: Some(
            LoraConv2dConfigBuilder::default(&device, dtype, in_channels, out_channels).build(),
        ),
        embed: embed_layers,
        embed_config: None,
    };

    //Create new LoRA layers from our layers
    let new_layers = Lora::convert_model(selected);

    //Custom methods to implement
    model.insert_new(new_layers);

    //Test the model
    let lora_output = model.forward(&dummy_image).unwrap();
    println!("LoRA Output: {lora_output:?}");

    assert_eq!(lora_output.shape(), output.shape());

    Ok(())
}
