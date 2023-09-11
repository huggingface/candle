use candle::{Shape, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Embedding, Linear, Module};
pub use loraconv1d::{LoraConv1d, LoraConv1dConfig, LoraConv1dConfigBuilder};
pub use loraconv2d::{LoraConv2d, LoraConv2dConfig, LoraConv2dConfigBuilder};
pub use loraembed::{LoraEmbedding, LoraEmbeddingConfig, LoraEmbeddingConfigBuilder};
pub use loralinear::{LoraLinear, LoraLinearConfig, LoraLinearConfigBuilder};
use std::{collections::HashMap, hash::Hash};

mod frozenconv;
mod frozenembed;
mod frozenlinear;
mod loraconv1d;
mod loraconv2d;
mod loraembed;
mod loralinear;

pub struct Lora;

impl Lora {
    /// Convert the selected layers into their LoRA counterparts.
    pub fn convert_model<T: Eq + PartialEq + Hash>(
        selected: SelectedLayers<'_, T>,
    ) -> NewLayers<T> {
        let mut new = NewLayers {
            linear: HashMap::new(),
            conv1d: HashMap::new(),
            conv2d: HashMap::new(),
            embed: HashMap::new(),
        };

        for (name, layer) in selected.linear {
            new.linear.insert(
                name,
                LoraLinear::new(layer, selected.linear_config.as_ref().unwrap()).unwrap(),
            );
        }

        for (name, layer) in selected.conv1d {
            new.conv1d.insert(
                name,
                LoraConv1d::new(layer, selected.conv1d_config.as_ref().unwrap()).unwrap(),
            );
        }

        for (name, layer) in selected.conv2d {
            new.conv2d.insert(
                name,
                LoraConv2d::new(layer, selected.conv2d_config.as_ref().unwrap()).unwrap(),
            );
        }

        for (name, layer) in selected.embed {
            new.embed.insert(
                name,
                LoraEmbedding::new(layer, selected.embed_config.as_ref().unwrap()).unwrap(),
            );
        }

        new
    }
}

/// Each configurations is applied to all layers of its respective type
pub struct SelectedLayers<'a, T: Eq + PartialEq + Hash> {
    pub linear: HashMap<T, &'a dyn LinearLayerLike>,
    pub linear_config: Option<LoraLinearConfig<'a>>,
    pub conv1d: HashMap<T, &'a dyn Conv1dLayerLike>,
    pub conv1d_config: Option<LoraConv1dConfig<'a>>,
    pub conv2d: HashMap<T, &'a dyn Conv2dLayerLike>,
    pub conv2d_config: Option<LoraConv2dConfig<'a>>,
    pub embed: HashMap<T, &'a dyn EmbeddingLayerLike>,
    pub embed_config: Option<LoraEmbeddingConfig<'a>>,
}

/// New layers, after conversion
pub struct NewLayers<T: Eq + PartialEq + Hash> {
    pub linear: HashMap<T, LoraLinear>,
    pub conv1d: HashMap<T, LoraConv1d>,
    pub conv2d: HashMap<T, LoraConv2d>,
    pub embed: HashMap<T, LoraEmbedding>,
}

/// Any layer that is linear-like.
pub trait LinearLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn shape(&self) -> &Shape;
}

impl LinearLayerLike for Linear {
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn shape(&self) -> &Shape {
        self.weight().shape()
    }
}

/// Any layer that is conv1d-like.
pub trait Conv1dLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn config(&self) -> &Conv1dConfig;
}

impl Conv1dLayerLike for Conv1d {
    fn config(&self) -> &Conv1dConfig {
        self.config()
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
}

/// Any layer that is conv2d-like.
pub trait Conv2dLayerLike: Module {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn config(&self) -> &Conv2dConfig;
}

impl Conv2dLayerLike for Conv2d {
    fn config(&self) -> &Conv2dConfig {
        self.config()
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
}

/// Any layer that is embedding-like.
pub trait EmbeddingLayerLike: Module {
    fn embeddings(&self) -> &Tensor;
    fn hidden_size(&self) -> usize;
}

impl EmbeddingLayerLike for Embedding {
    fn embeddings(&self) -> &Tensor {
        self.embeddings()
    }
    fn hidden_size(&self) -> usize {
        self.embeddings().dim(1).unwrap() //Reason: 2nd dim is always the hidden
    }
}
