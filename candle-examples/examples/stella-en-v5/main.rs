#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::path::Path;

use anyhow::{anyhow, Error as E, Result};
use clap::Parser;

use candle_transformers::models::stella_en_v5::{
    Config, EmbedDim as StellaEmbedDim, EmbeddingModel,
};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

struct Embedding {
    model: EmbeddingModel,
    device: Device,
    tokenizer: Tokenizer,
}

impl Embedding {
    fn new(model: EmbeddingModel, tokenizer: Tokenizer, device: &Device) -> Self {
        Self {
            model,
            tokenizer,
            device: device.clone(),
        }
    }

    fn encode(&mut self, task: EncodeTask, text: Option<String>) -> Result<()> {
        // Just shocasing embeddings, this has no real value
        if let Some(text) = text {
            let qry = task.query_preproc(&[text]);
            let encoding = self.tokenizer.encode(qry, true).map_err(|e| anyhow!(e))?;

            let shape = (1, encoding.len());
            let input = Tensor::from_slice(encoding.get_ids(), shape, &self.device)?;
            let mask = Tensor::from_slice(encoding.get_attention_mask(), shape, &self.device)?;

            let result = self.model.forward(&input, &mask)?;
            println!("embeddings: {result}");
        } else {
            // Examples copied from [Model Card](https://huggingface.co/dunzhang/stella_en_1.5B_v5#transformers)
            let queries = [
                "What are some ways to reduce stress?".to_string(),
                "What are the benefits of drinking green tea?".to_string(),
            ];

            let docs = [
                "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.".to_string(),
                "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.".to_string(),
            ];

            // We only encode the queries and not the data
            let qry = task.query_preproc(&queries);
            let mut qry_encoded = self
                .tokenizer
                .encode_batch(qry, true)
                .map_err(|e| anyhow!(e))?;

            let mut docs_encoded = self
                .tokenizer
                .encode_batch(docs.to_vec(), true)
                .map_err(|e| anyhow!(e))?;

            let qry_embed = {
                // Now, we generate the tensors for the `input` and `mask`
                let shape = (qry_encoded.len(), qry_encoded[1].len());
                let mut ids = Tensor::zeros(shape, DType::U32, &self.device)?;
                let mut masks = Tensor::zeros(shape, DType::U8, &self.device)?;

                for (i, e) in qry_encoded.drain(..).enumerate() {
                    let input_id =
                        Tensor::from_iter(e.get_ids().to_vec(), &self.device)?.unsqueeze(0)?;
                    let mask = Tensor::from_iter(e.get_attention_mask().to_vec(), &self.device)?
                        .to_dtype(DType::U8)?
                        .unsqueeze(0)?;

                    ids =
                        ids.slice_assign(&[i..i + 1, 0..input_id.dims2().unwrap().1], &input_id)?;
                    masks = masks.slice_assign(&[i..i + 1, 0..mask.dims2().unwrap().1], &mask)?;
                }

                // Let's generate the embeddings for the query, we are going to be normalizing the result.
                // For larger datasets, you can call `.forward()` on batches and run a `l2 norm` pass on the entire data
                self.model.forward_norm(&ids, &masks)?
            };

            let doc_embed = {
                let shape = (docs_encoded.len(), docs_encoded[1].len());
                let mut ids = Tensor::zeros(shape, DType::U32, &self.device)?;
                let mut masks = Tensor::zeros(shape, DType::U8, &self.device)?;

                for (i, e) in docs_encoded.drain(..).enumerate() {
                    let input_id =
                        Tensor::from_iter(e.get_ids().to_vec(), &self.device)?.unsqueeze(0)?;
                    let mask = Tensor::from_iter(e.get_attention_mask().to_vec(), &self.device)?
                        .to_dtype(DType::U8)?
                        .unsqueeze(0)?;

                    ids =
                        ids.slice_assign(&[i..i + 1, 0..input_id.dims2().unwrap().1], &input_id)?;
                    masks = masks.slice_assign(&[i..i + 1, 0..mask.dims2().unwrap().1], &mask)?;
                }

                // Let's generate the embeddings for the query, we are going to be normalizing the result.
                // For larger datasets, you can call `.forward()` on batches and run a `l2 norm` pass on the entire data
                self.model.forward_norm(&ids, &masks)?
            };

            println!(
                "Embed shapes:\nQuery: {:?}\nDocs: {:?}",
                qry_embed.shape(),
                doc_embed.shape()
            ); // [2, 1024] for head dim `1024`

            // a matmul to generate the `similarity` score
            let res = qry_embed.matmul(&doc_embed.t()?)?;
            for (k, v) in queries.iter().enumerate() {
                let tnsr = res.get(k)?;
                let max = tnsr.argmax(0)?.to_scalar::<u32>()?;
                println!(
                    "\nScore: {}\nQuery: {}\nAnswer: {}\n\n",
                    tnsr.get(max as usize)?.to_scalar::<f32>()?,
                    v,
                    docs[k]
                );
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum EmbedDim {
    #[value(name = "256")]
    Dim256,
    #[value(name = "768")]
    Dim768,
    #[value(name = "1024")]
    Dim1024,
    #[value(name = "2048")]
    Dim2048,
    #[value(name = "4096")]
    Dim4096,
    #[value(name = "6144")]
    Dim6144,
    #[value(name = "8192")]
    Dim8192,
}

impl EmbedDim {
    /// Returns dir path to the embed head weights int he repo
    pub fn embed_dim_default_dir(&self) -> &'static str {
        match self {
            Self::Dim256 => "2_Dense_256",
            Self::Dim768 => "2_Dense_768",
            Self::Dim1024 => "2_Dense_1024",
            Self::Dim2048 => "2_Dense_2048",
            Self::Dim4096 => "2_Dense_4096",
            Self::Dim6144 => "2_Dense_6144",
            Self::Dim8192 => "2_Dense_8192",
        }
    }

    /// Resolves the `EmbedDim` for given variant
    pub fn embed_dim(&self) -> StellaEmbedDim {
        match self {
            Self::Dim256 => StellaEmbedDim::Dim256,
            Self::Dim768 => StellaEmbedDim::Dim768,
            Self::Dim1024 => StellaEmbedDim::Dim1024,
            Self::Dim2048 => StellaEmbedDim::Dim2048,
            Self::Dim4096 => StellaEmbedDim::Dim4096,
            Self::Dim6144 => StellaEmbedDim::Dim6144,
            Self::Dim8192 => StellaEmbedDim::Dim8192,
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum EncodeTask {
    /// `s2p` is the `retrieval` task
    /// Default in this example
    #[value(name = "s2p")]
    S2P,
    /// `s2s` is the semantic similarity task
    #[value(name = "s2s")]
    S2S,
}

impl EncodeTask {
    /// Preprocess a set of inputs basef on a template suggested by the model authors
    /// See: https://huggingface.co/dunzhang/stella_en_1.5B_v5#introduction
    pub fn query_preproc(&self, txt: &[String]) -> Vec<String> {
        let instruct = match self {
            Self::S2P => {
                "Given a web search query, retrieve relevant passages that answer the query."
            }
            Self::S2S => "Retrieve semantically similar text.",
        };

        txt.iter()
            .map(|s| format!("Instruct: {instruct}\nQuery: {s}"))
            .collect::<Vec<_>>()
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "1.5b")]
    Large,
    #[value(name = "400m")]
    Small,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    which: Which,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    query: Option<String>,

    #[arg(long, default_value = "1024")]
    embed_dim: Option<EmbedDim>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    base_weight_files: Option<String>,

    #[arg(long)]
    embed_head_weight_files: Option<String>,

    /// `Stella` is trained on 2 tasks: See [`Model Card`](https://huggingface.co/dunzhang/stella_en_1.5B_v5)
    /// `s2s`: Semantic textual similarity
    /// `s2p`: Retrieval task - `Default` in this example
    #[arg(long, default_value = "s2p")]
    task: Option<EncodeTask>,
}

// Tokenizer creation is super critical in our case.
// We are going to be `padding: Left` for each batch
fn create_tokenizer(tokenizer_file: &Path, which: Which) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

    if which == Which::Large {
        let pad_id = if let Some(pad_id) = tokenizer.token_to_id("<|endoftext|>") {
            pad_id
        } else {
            return Err(anyhow!(
                "Tokenizer doesn't contain expected `<|endoftext|>` token"
            ));
        };

        // This part is super important, we are padding the tokens to the *`left`* and not the usual *`right`* padding
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_id,
            pad_token: "<|endoftext|>".to_string(),
            ..Default::default()
        }));
    } else {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            ..Default::default()
        }));
    }

    Ok(tokenizer)
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let embed_dim = match args.embed_dim {
        Some(d) => d,
        None => EmbedDim::Dim1024,
    };

    let (repo, cfg) = match args.which {
        Which::Large => (
            "dunzhang/stella_en_1.5B_v5",
            Config::new_1_5_b_v5(embed_dim.embed_dim()),
        ),
        Which::Small => (
            "dunzhang/stella_en_400M_v5",
            Config::new_400_m_v5(embed_dim.embed_dim()),
        ),
    };

    let repo = api.repo(Repo::model(repo.to_string()));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    // Note, if you are providing `weight_files`, ensure that the `--embed_dim` dimensions provided matches the weights
    // E.g. if you are using `--embed_dim 1024`, the weight files should include the `.safetensors` file from `2_Dense_1024` dir of the repo
    let base_weight_files = match args.base_weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            vec![repo.get("model.safetensors")?]
        }
    };

    let embed_weight_files = match args.embed_head_weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            let head_w_path = format!("{}/model.safetensors", embed_dim.embed_dim_default_dir());
            vec![repo.get(&head_w_path)?]
        }
    };

    println!("retrieved the files in {:?}", start.elapsed());

    // Initializing the tokenizer which would require us to add padding to the `left` for batch encoding
    let tokenizer = create_tokenizer(tokenizer_filename.as_path(), args.which)?;

    let start = std::time::Instant::now();

    let device = candle_examples::device(args.cpu)?;
    let dtype = DType::F32;

    let base_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&base_weight_files, dtype, &device)? };
    // Embedding layer is always built on F32 for accuracy
    let embed_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&embed_weight_files, DType::F32, &device)? };

    let model = EmbeddingModel::new(&cfg, base_vb, embed_vb)?;

    println!("loaded the model in {:?}", start.elapsed());

    let mut embedding = Embedding::new(model, tokenizer, &device);

    let task = args.task.map_or(EncodeTask::S2P, |t| t);

    embedding.encode(task, args.query)
}
