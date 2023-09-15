#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use candle_transformers::models::t5;

use anyhow::{anyhow, Error as E, Result};
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Run offline (you must have the files already cached)
    #[arg(long)]
    offline: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// Compute embeddings for this prompt, otherwise compute sentence similarities.
    #[arg(long)]
    prompt: Option<String>,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(t5::T5EncoderModel, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "t5-small".to_string();
        let default_revision = "refs/pr/15".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = if self.offline {
            let cache = Cache::default().repo(repo);
            (
                cache
                    .get("config.json")
                    .ok_or(anyhow!("Missing config file in cache"))?,
                cache
                    .get("tokenizer.json")
                    .ok_or(anyhow!("Missing tokenizer file in cache"))?,
                cache
                    .get("model.safetensors")
                    .ok_or(anyhow!("Missing weights file in cache"))?,
            )
        } else {
            let api = Api::new()?;
            let api = api.repo(repo);
            (
                api.get("config.json")?,
                api.get("tokenizer.json")?,
                api.get("model.safetensors")?,
            )
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: t5::Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let weights = unsafe { candle::safetensors::MmapedFile::new(weights_filename)? };
        let weights = weights.deserialize()?;
        let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
        let model = t5::T5EncoderModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    match args.prompt {
        Some(prompt) => {
            let tokens = tokenizer
                .encode(prompt, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            let token_ids = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
            for idx in 0..args.n {
                let start = std::time::Instant::now();
                let ys = model.forward(&token_ids)?;
                if idx == 0 {
                    println!("{ys}");
                }
                println!("Took {:?}", start.elapsed());
            }
        }
        None => {
            let sentences = [
                "The cat sits outside",
                "A man is playing guitar",
                "I love pasta",
                "The new movie is awesome",
                "The cat plays in the garden",
                "A woman watches TV",
                "The new movie is so great",
                "Do you like pizza?",
            ];
            let n_sentences = sentences.len();
            let mut all_embeddings = Vec::with_capacity(n_sentences);
            for sentence in sentences {
                let tokens = tokenizer
                    .encode(sentence, true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let token_ids = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
                let embeddings = model.forward(&token_ids)?;
                println!("generated embeddings {:?}", embeddings.shape());
                // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
                let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
                let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
                let embeddings = if args.normalize_embeddings {
                    normalize_l2(&embeddings)?
                } else {
                    embeddings
                };
                println!("pooled embeddings {:?}", embeddings.shape());
                all_embeddings.push(embeddings)
            }

            let mut similarities = vec![];
            for (i, e_i) in all_embeddings.iter().enumerate() {
                for (j, e_j) in all_embeddings
                    .iter()
                    .enumerate()
                    .take(n_sentences)
                    .skip(i + 1)
                {
                    let sum_ij = (e_i * e_j)?.sum_all()?.to_scalar::<f32>()?;
                    let sum_i2 = (e_i * e_i)?.sum_all()?.to_scalar::<f32>()?;
                    let sum_j2 = (e_j * e_j)?.sum_all()?.to_scalar::<f32>()?;
                    let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                    similarities.push((cosine_similarity, i, j))
                }
            }
            similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
            for &(score, i, j) in similarities[..5].iter() {
                println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
            }
        }
    }
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
