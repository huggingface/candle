use candle_lora::{NewLayers, SelectedLayers};

#[test]
fn single_linear() -> candle::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle::{DType, Device, Result, Tensor};
    use candle_lora::{LinearLayerLike, Lora, LoraLinearConfigBuilder};
    use candle_nn::{init, Linear, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Layer,
    }

    #[derive(Debug)]
    struct Model {
        layer: Box<dyn LinearLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.layer.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, conv) in new.linear {
                match name {
                    ModelLayers::Layer => self.layer = Box::new(conv),
                }
            }
        }
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    //Create the model
    let map = VarMap::new();
    let layer_weight = map.get(
        (10, 10),
        "layer.weight",
        init::DEFAULT_KAIMING_NORMAL,
        dtype,
        &device,
    )?;

    let mut model = Model {
        layer: Box::new(Linear::new(layer_weight.clone(), None)),
    };

    let dummy_image = Tensor::zeros((10, 10), DType::F32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let mut linear_layers = HashMap::new();
    linear_layers.insert(ModelLayers::Layer, &*model.layer);
    let conv1d_layers = HashMap::new();
    let conv2d_layers = HashMap::new();
    let embed_layers = HashMap::new();
    let selected = SelectedLayers {
        linear: linear_layers,
        linear_config: Some(LoraLinearConfigBuilder::default(&device, dtype, 10, 10).build()),
        conv1d: conv1d_layers,
        conv1d_config: None,
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
