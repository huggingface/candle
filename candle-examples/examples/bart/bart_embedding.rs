#[derive(Debug, Clone, Copy)]
pub struct EmbeddingConfig {
    pub sparse: bool,
    pub scale_grad_by_freq: bool,
    // pub ws_init: super::Init,
    pub padding_idx: i64,
}
