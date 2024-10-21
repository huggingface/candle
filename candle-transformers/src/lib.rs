#![cfg_attr(all(target_arch = "wasm32", feature = "wgpu"), allow(deprecated))] //for wasm32 and wgpu, async functions may be used instead of sync functions. 
                                                                               //this will allow the deprecated warnings inside this crate
                                                                               
pub mod generation;
pub mod models;
pub mod object_detection;
pub mod pipelines;
pub mod quantized_nn;
pub mod quantized_var_builder;
pub mod utils;
