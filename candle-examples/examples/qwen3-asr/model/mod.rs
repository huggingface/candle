//! Candle model implementation (audio tower + thinker LM + generation).

use std::path::PathBuf;

pub mod attention;
pub mod audio_encoder;
pub mod generation;
pub mod kv_cache;
pub mod name_map;
pub mod rope;
pub mod thinker;
pub mod thinker_text;
pub mod weights;

#[derive(Debug)]
pub struct AsrModel {
    pub model_dir: PathBuf,
    pub weights_paths: Vec<PathBuf>,
    pub thinker: thinker::ThinkerForConditionalGeneration,
}
