use candle_lora::LoraEmbeddingConfigBuilder;

#[test]
fn embed() -> candle::Result<()> {
    use std::{collections::HashMap, hash::Hash};

    use candle::{DType, Device, Result, Tensor};
    use candle_lora::{EmbeddingLayerLike, Lora, NewLayers, SelectedLayers};
    use candle_nn::{init, Embedding, Module, VarMap};

    #[derive(PartialEq, Eq, Hash)]
    enum ModelLayers {
        Embed,
    }

    #[derive(Debug)]
    struct Model {
        embed: Box<dyn EmbeddingLayerLike>,
    }

    impl Module for Model {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            self.embed.forward(input)
        }
    }

    impl Model {
        fn insert_new(&mut self, new: NewLayers<ModelLayers>) {
            for (name, conv) in new.embed {
                match name {
                    ModelLayers::Embed => self.embed = Box::new(conv),
                }
            }
        }
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 10;
    let hidden_size = 3;

    //Create the model
    let map = VarMap::new();
    let embed_weight = map.get(
        (in_size, hidden_size),
        "embed.weight",
        init::ZERO,
        dtype,
        &device,
    )?;

    let mut model = Model {
        embed: Box::new(Embedding::new(embed_weight, hidden_size)),
    };

    let dummy_image = Tensor::zeros((2, 4), DType::U32, &device)?;

    //Test the model
    let output = model.forward(&dummy_image).unwrap();
    println!("Output: {output:?}");

    //Select layers we want to convert
    let linear_layers = HashMap::new();
    let conv1d_layers = HashMap::new();
    let conv2d_layers = HashMap::new();
    let mut embed_layers = HashMap::new();
    embed_layers.insert(ModelLayers::Embed, &*model.embed);
    let selected = SelectedLayers {
        linear: linear_layers,
        linear_config: None,
        conv1d: conv1d_layers,
        conv1d_config: None,
        conv2d: conv2d_layers,
        conv2d_config: None,
        embed: embed_layers,
        embed_config: Some(
            LoraEmbeddingConfigBuilder::default(&device, dtype, in_size, hidden_size).build(),
        ),
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
