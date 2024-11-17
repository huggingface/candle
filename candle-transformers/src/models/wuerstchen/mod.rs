//! Würstchen Efficient Diffusion Model
//!
//! Würstchen is an efficient diffusion model architecture for generating images using
//! a two-stage approach with a small decoder and prior network.
//!
//! - 💻 [GH Link](https://github.com/dome272/Wuerstchen)
//! - 🤗 [HF Link](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen.py)
//! - 📝 [Paper](https://openreview.net/pdf?id=gU58AyJlYz)
//!
//! ## Example
//!
//! <div align=center>
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/wuerstchen/assets/cat.jpg" alt="" width=320>
//!   <p>"Anthropomorphic cat dressed as a fire fighter"</p>
//! </div>

pub mod attention_processor;
pub mod common;
pub mod ddpm;
pub mod diffnext;
pub mod paella_vq;
pub mod prior;
