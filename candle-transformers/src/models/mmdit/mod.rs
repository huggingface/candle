//! Mix of Multi-scale Dilated and Traditional Convolutions
//!
//! Mix of Multi-scale Dilated and Traditional Convolutions (MMDiT) is an architecture
//! introduced for Stable Diffusion 3, with the MMDiT-X variant used in Stable Diffusion 3.5.
//!
//! - 📝 [Research Paper](https://arxiv.org/abs/2403.03206)
//! - 💻 ComfyUI [reference implementation](https://github.com/comfyanonymous/ComfyUI/blob/78e133d0415784924cd2674e2ee48f3eeca8a2aa/comfy/ldm/modules/diffusionmodules/mmdit.py)
//! - 💻 Stability-AI [MMDiT-X implementation](https://github.com/Stability-AI/sd3.5/blob/4e484e05308d83fb77ae6f680028e6c313f9da54/mmditx.py)

//! - ⚡ [Interactive Wasm Example](https://huggingface.co/spaces/radames/Candle-BLIP-Image-Captioning)
//! - 💻 [GH Link](https://github.com/salesforce/BLIP)
//! - 🤗 [HF Link](https://huggingface.co/Salesforce/blip-image-captioning-base)
//! - 📝 [Paper](https://arxiv.org/abs/2201.12086)
//!

pub mod blocks;
pub mod embedding;
pub mod model;
pub mod projections;
