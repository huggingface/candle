use candle::{DType, Device, Tensor}; 
use candle_nn::{Embedding, Module, VarBuilder, init};
use anyhow::Ok;
use serde::{Serialize, Deserialize};
use anyhow::{bail, Error as E, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MllamaConfig {
    pub vision_config: MllamaVisionConfig,
    pub text_config: MllamaTextConfig,
    pub image_token_index: i32
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MllamaTextConfig {

}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MllamaVisionConfig {
    pub hidden_size: usize,
    pub hidden_act: String,
    pub num_hidden_layers: usize,
    pub num_global_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub intermediate_size: usize,
    pub vision_output_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub norm_eps: f32,
    pub max_num_tiles: usize,
    pub intermediate_layers_indices: Vec<i32>,
    pub supported_aspect_ratios: Vec<Vec<i32>>,
    pub initializer_range: f32,
}

fn hidden_size: int = 1280,
fn hidden_act: str = "gelu",
fn num_hidden_layers: int = 32,
fn num_global_layers: int = 8,
fn num_attention_heads: int = 16,
fn num_channels: int = 3,
fn intermediate_size: int = 5120,
fn vision_output_dim: int = 7680,
fn image_size: int = 448,
fn patch_size: int = 14,
fn norm_eps: float = 1e-5,
fn max_num_tiles: int = 4,
fn intermediate_layers_indices: Optional[List[int]] = None,
fn supported_aspect_ratios: Optional[List[List[int]]] = None,
fn initializer_range: float = 0.02,