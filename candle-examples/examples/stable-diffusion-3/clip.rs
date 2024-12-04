use anyhow::{Error as E, Ok, Result};
use candle::{DType, IndexOp, Module, Tensor, D};
use candle_transformers::models::{stable_diffusion, t5};
use std::path::PathBuf;
use tokenizers::tokenizer::Tokenizer;

struct ClipWithTokenizer {
    clip: stable_diffusion::clip::ClipTextTransformer,
    config: stable_diffusion::clip::Config,
    tokenizer: Tokenizer,
    max_position_embeddings: usize,
}

impl ClipWithTokenizer {
    fn new(
        vb: candle_nn::VarBuilder,
        config: stable_diffusion::clip::Config,
        tokenizer_path: &str,
        max_position_embeddings: usize,
    ) -> Result<Self> {
        let clip = stable_diffusion::clip::ClipTextTransformer::new(vb, &config)?;
        let path_buf = hf_hub::api::sync::Api::new()?
            .model(tokenizer_path.to_string())
            .get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(path_buf.to_str().ok_or(E::msg(
            "Failed to serialize huggingface PathBuf of CLIP tokenizer",
        ))?)
        .map_err(E::msg)?;
        Ok(Self {
            clip,
            config,
            tokenizer,
            max_position_embeddings,
        })
    }

    fn encode_text_to_embedding(
        &self,
        prompt: &str,
        device: &candle::Device,
    ) -> Result<(Tensor, Tensor)> {
        let pad_id = match &self.config.pad_with {
            Some(padding) => *self
                .tokenizer
                .get_vocab(true)
                .get(padding.as_str())
                .ok_or(E::msg("Failed to tokenize CLIP padding."))?,
            None => *self
                .tokenizer
                .get_vocab(true)
                .get("<|endoftext|>")
                .ok_or(E::msg("Failed to tokenize CLIP end-of-text."))?,
        };

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let eos_position = tokens.len() - 1;

        while tokens.len() < self.max_position_embeddings {
            tokens.push(pad_id)
        }
        let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let (text_embeddings, text_embeddings_penultimate) = self
            .clip
            .forward_until_encoder_layer(&tokens, usize::MAX, -2)?;
        let text_embeddings_pooled = text_embeddings.i((0, eos_position, ..))?;

        Ok((text_embeddings_penultimate, text_embeddings_pooled))
    }
}

struct T5WithTokenizer {
    t5: t5::T5EncoderModel,
    tokenizer: Tokenizer,
    max_position_embeddings: usize,
}

impl T5WithTokenizer {
    fn new(vb: candle_nn::VarBuilder, max_position_embeddings: usize) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            "google/t5-v1_1-xxl".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/2".to_string(),
        ));
        let config_filename = repo.get("config.json")?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: t5::Config = serde_json::from_str(&config)?;
        let model = t5::T5EncoderModel::load(vb, &config)?;

        let tokenizer_filename = api
            .model("lmz/mt5-tokenizers".to_string())
            .get("t5-v1_1-xxl.tokenizer.json")?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok(Self {
            t5: model,
            tokenizer,
            max_position_embeddings,
        })
    }

    fn encode_text_to_embedding(
        &mut self,
        prompt: &str,
        device: &candle::Device,
    ) -> Result<Tensor> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        tokens.resize(self.max_position_embeddings, 0);
        let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let embeddings = self.t5.forward_dt(&input_token_ids, Some(DType::F32))?;
        Ok(embeddings)
    }
}

pub struct StableDiffusion3TripleClipWithTokenizer {
    clip_l: ClipWithTokenizer,
    clip_g: ClipWithTokenizer,
    clip_g_text_projection: candle_nn::Linear,
    t5: T5WithTokenizer,
}

impl StableDiffusion3TripleClipWithTokenizer {
    pub fn new_split(
        clip_g_file: &PathBuf,
        clip_l_file: &PathBuf,
        t5xxl_file: &PathBuf,
        device: &candle::Device,
    ) -> Result<Self> {
        let vb_clip_g = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[clip_g_file], DType::F16, device)?
        };
        let vb_clip_l = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[clip_l_file], DType::F16, device)?
        };
        let vb_t5 = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[t5xxl_file], DType::F16, device)?
        };
        let max_position_embeddings = 77usize;
        let clip_l = ClipWithTokenizer::new(
            vb_clip_l,
            stable_diffusion::clip::Config::sdxl(),
            "openai/clip-vit-large-patch14",
            max_position_embeddings,
        )?;

        let text_projection =
            candle_nn::linear_no_bias(1280, 1280, vb_clip_g.pp("text_projection"))?;

        let clip_g = ClipWithTokenizer::new(
            vb_clip_g,
            stable_diffusion::clip::Config::sdxl2(),
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            max_position_embeddings,
        )?;

        let t5 = T5WithTokenizer::new(vb_t5, max_position_embeddings)?;
        Ok(Self {
            clip_l,
            clip_g,
            clip_g_text_projection: text_projection,
            t5,
        })
    }

    pub fn new(vb: candle_nn::VarBuilder) -> Result<Self> {
        let max_position_embeddings = 77usize;
        let clip_l = ClipWithTokenizer::new(
            vb.pp("clip_l.transformer"),
            stable_diffusion::clip::Config::sdxl(),
            "openai/clip-vit-large-patch14",
            max_position_embeddings,
        )?;

        let clip_g = ClipWithTokenizer::new(
            vb.pp("clip_g.transformer"),
            stable_diffusion::clip::Config::sdxl2(),
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            max_position_embeddings,
        )?;

        let text_projection =
            candle_nn::linear_no_bias(1280, 1280, vb.pp("clip_g.transformer.text_projection"))?;

        let t5 = T5WithTokenizer::new(vb.pp("t5xxl.transformer"), max_position_embeddings)?;
        Ok(Self {
            clip_l,
            clip_g,
            clip_g_text_projection: text_projection,
            t5,
        })
    }

    pub fn encode_text_to_embedding(
        &mut self,
        prompt: &str,
        device: &candle::Device,
    ) -> Result<(Tensor, Tensor)> {
        let (clip_l_embeddings, clip_l_embeddings_pooled) =
            self.clip_l.encode_text_to_embedding(prompt, device)?;
        let (clip_g_embeddings, clip_g_embeddings_pooled) =
            self.clip_g.encode_text_to_embedding(prompt, device)?;

        let clip_g_embeddings_pooled = self
            .clip_g_text_projection
            .forward(&clip_g_embeddings_pooled.unsqueeze(0)?)?
            .squeeze(0)?;

        let y = Tensor::cat(&[&clip_l_embeddings_pooled, &clip_g_embeddings_pooled], 0)?
            .unsqueeze(0)?;
        let clip_embeddings_concat = Tensor::cat(
            &[&clip_l_embeddings, &clip_g_embeddings],
            D::Minus1,
        )?
        .pad_with_zeros(D::Minus1, 0, 2048)?;

        let t5_embeddings = self
            .t5
            .encode_text_to_embedding(prompt, device)?
            .to_dtype(DType::F16)?;
        let context = Tensor::cat(&[&clip_embeddings_concat, &t5_embeddings], D::Minus2)?;
        Ok((context, y))
    }
}
