#[test]
fn conv1d() -> candle::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle::{DType, Device, Result, Tensor};
    use candle_lora::{Conv1dLayerLike, Lora, LoraConv1dConfigBuilder, NewLayers, SelectedLayers};
    use candle_nn::{init, Conv1d, Conv1dConfig, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Conv,
    }

    #[derive(Debug)]
    struct Model {
        conv: Box<dyn Conv1dLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.conv.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, conv) in new.conv1d {
                match name {
                    ModelLayers::Conv => self.conv = Box::new(conv),
                }
            }
        }
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    //Create the model
    let map = VarMap::new();
    let conv_weight = map.get(
        (1, 10, 10),
        "conv.weight",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;
    let conv_bias = map.get(
        10,
        "conv.bias",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;

    let mut model = Model {
        conv: Box::new(Conv1d::new(
            conv_weight.clone(),
            Some(conv_bias.clone()),
            Conv1dConfig::default(),
        )),
    };

    let dummy_image = Tensor::zeros((1, 10, 10), DType::F32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let linear_layers = HashMap::new();
    let mut conv1d_layers = HashMap::new();
    let conv2d_layers = HashMap::new();
    conv1d_layers.insert(ModelLayers::Conv, &*model.conv);
    let embed_layers = HashMap::new();
    let selected = SelectedLayers {
        linear: linear_layers,
        linear_config: None,
        conv1d: conv1d_layers,
        conv1d_config: Some(LoraConv1dConfigBuilder::default(&device, dtype, 1, 10, 10).build()),
        conv2d: conv2d_layers,
        conv2d_config: None,
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
