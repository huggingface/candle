#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, IndexOp, Shape, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::nvembed_v2::model::Model;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    model: Option<String>,

    /// Comma-separated list of model files (e.g., '/path/file1.safetensors,/path/file2.safetensors,/path/file3.safetensors')
    #[arg(long)]
    model_files: Option<String>,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> anyhow::Result<(Model, tokenizers::Tokenizer)> {
        let model_name = match self.model.as_ref() {
            Some(model) => model.to_string(),
            None => "nvidia/NV-Embed-v2".to_string(),
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));

        let model_files = match &self.model_files {
            Some(files) => files
                .split(',')
                .map(std::path::PathBuf::from)
                .collect::<Vec<_>>(),
            None => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        };

        let tokenizer_file = match &self.tokenizer {
            Some(file) => std::path::PathBuf::from(file),
            None => repo.get("tokenizer.json")?,
        };

        let device = candle_examples::device(self.cpu)?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

        let _ = tokenizer
            .with_padding(Some(PaddingParams {
                direction: PaddingDirection::Right,
                pad_id: 2,
                pad_token: "</s>".to_string(),
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: 32768,
                ..Default::default()
            }));

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, &device) }?;

        let nvembed_model = Model::new(vb);
        Ok((nvembed_model?, tokenizer))
    }
}

fn encode(
    model: &mut Model,
    tokenizer: &Tokenizer,
    examples: Vec<String>,
    instruction: &str,
) -> Result<Tensor> {
    let device = &model.device;
    let dtype = model.dtype;

    // Format input text
    let eos_token = if let Some(padding) = tokenizer.get_padding() {
        padding.pad_token.clone()
    } else {
        "".to_string()
    };
    let bos = "<s>".to_string();
    let input_texts = examples
        .iter()
        .map(|input_example| format!("{bos}{instruction}{input_example}{eos_token}"))
        .collect::<Vec<String>>();

    // Tokenize
    let encodings = tokenizer.encode_batch(input_texts, false).map_err(E::msg)?;

    let input_ids_list = encodings
        .iter()
        .map(|encoding| {
            Tensor::from_slice(
                encoding.get_ids(),
                Shape::from(encoding.get_ids().len()),
                device,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_ids = Tensor::stack(&input_ids_list, 0)?;

    // Mask out padding tokens for both embedding model and latent attention model
    let attention_masks: Vec<Tensor> = encodings
        .iter()
        .map(|encoding| {
            Tensor::from_slice(
                encoding.get_attention_mask(),
                Shape::from(encoding.get_attention_mask().len()),
                device,
            )?
            .to_dtype(dtype)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let attention_mask = Tensor::stack(&attention_masks, 0)?;

    // Mask out instruction tokens for latent attention model
    let pool_mask = if !instruction.is_empty() {
        let encoded_instruction = tokenizer.encode(instruction, false).map_err(E::msg)?;
        let instruction_lens = encoded_instruction.get_tokens().len();
        let zeros = Tensor::zeros(
            attention_mask.i((.., ..instruction_lens))?.shape(),
            dtype,
            device,
        )?;
        let b = attention_mask.dims()[0];
        attention_mask.slice_assign(&[..b, ..instruction_lens], &zeros)?
    } else {
        attention_mask.clone()
    };

    let hiddens = model
        .forward(&input_ids, &attention_mask, &pool_mask)?
        .squeeze(1)?;

    // Normalize embedding
    div_l2_norm(&hiddens)
}

fn div_l2_norm(v: &Tensor) -> Result<Tensor> {
    let l2_norm = v.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(v.broadcast_div(&l2_norm)?)
}

fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let (mut model, tokenizer) = args.build_model_and_tokenizer()?;

    if let Some(prompt) = args.prompt {
        let emb = encode(&mut model, &tokenizer, vec![prompt], "")?;
        println!("Embedding: {emb}");
    } else {
        let queries = [
            "are judo throws allowed in wrestling?",
            "how to become a radiology technician in michigan?",
        ];

        let passages = [
            "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
            "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
            ];
        let passage_instruction = "".to_string();
        let query_instruction =
            "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
                .to_string();

        let passages: Vec<String> = passages.iter().map(|s| s.to_string()).collect();
        let queries: Vec<String> = queries.iter().map(|s| s.to_string()).collect();

        let emb_query = encode(&mut model, &tokenizer, queries, &query_instruction)?;
        let emb_passage = encode(&mut model, &tokenizer, passages, &passage_instruction)?;

        let scores = (emb_query.matmul(&emb_passage.t()?)? * 100.0)?;

        println!("scores: {scores}");
    }
    Ok(())
}
