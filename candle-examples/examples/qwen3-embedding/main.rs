//! Qwen3-Embedding example: embed JSONL text fields into vectors using a GGUF model.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{BufRead, Write as IoWrite};

use candle::quantized::gguf_file;
use candle::Tensor;
use candle_transformers::models::quantized_qwen3_embed::EmbeddingModel;
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-Embedding-8B";
const DEFAULT_GGUF_REPO: &str = "Qwen/Qwen3-Embedding-8B-GGUF";
const DEFAULT_GGUF_FILE: &str = "Qwen3-Embedding-8B-Q4_K_M.gguf";

#[derive(Parser, Debug)]
#[command(author, version, about = "Embed text using Qwen3-Embedding (GGUF)")]
struct Args {
    #[arg(long)]
    input: String,

    #[arg(long, default_value = "enriched_embeddings.npy")]
    output: String,

    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    #[arg(long, default_value = DEFAULT_GGUF_REPO)]
    gguf_repo: String,

    #[arg(long, default_value = DEFAULT_GGUF_FILE)]
    gguf_file: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    tracing: bool,

    #[arg(long, default_value_t = 8192)]
    max_seq_len: usize,

    #[arg(long, default_value = "english_description")]
    field: String,

    #[arg(long)]
    instruct: Option<String>,

    #[arg(long, default_value_t = 100)]
    log_every: usize,
}

// String-based field extraction tolerates non-standard JSON values (e.g. NaN)
// that may appear in unrelated numeric fields on the same line.
fn extract_field(line: &str, field: &str) -> Option<String> {
    let key = format!("\"{}\":", field);
    let start = line.find(&key)? + key.len();
    let rest = &line[start..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let rest = &rest[1..];
    let mut chars = rest.chars();
    let mut value = String::new();
    loop {
        match chars.next()? {
            '\\' => {
                let escaped = chars.next()?;
                match escaped {
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    'u' => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(cp) {
                                value.push(c);
                            }
                        }
                    }
                    other => {
                        value.push('\\');
                        value.push(other);
                    }
                }
            }
            '"' => return Some(value),
            c => value.push(c),
        }
    }
}

fn get_tokenizer(args: &Args) -> Result<Tokenizer> {
    let path = match &args.tokenizer {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(args.model_id.clone());
            repo.get("tokenizer.json")
                .context("Failed to download tokenizer.json")?
        }
    };
    Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))
}

fn get_model_path(args: &Args) -> Result<std::path::PathBuf> {
    match &args.model {
        Some(p) => Ok(std::path::PathBuf::from(p)),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.repo(hf_hub::Repo::with_revision(
                args.gguf_repo.clone(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ));
            repo.get(&args.gguf_file)
                .context("Failed to download GGUF model file")
        }
    }
}

fn write_npy(path: &str, data: &[f32], rows: usize, cols: usize) -> Result<()> {
    let mut f = std::fs::File::create(path)?;

    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows, cols
    );

    let prefix_len = 10;
    let total = prefix_len + header.len() + 1;
    let padding = (64 - (total % 64)) % 64;
    let header_len = (header.len() + padding + 1) as u16;

    f.write_all(&[0x93])?;
    f.write_all(b"NUMPY")?;
    f.write_all(&[1, 0])?;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for _ in 0..padding {
        f.write_all(b" ")?;
    }
    f.write_all(b"\n")?;

    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes)?;

    Ok(())
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
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

    let model_path = get_model_path(&args)?;
    eprintln!("Loading model from {}", model_path.display());
    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;

    let mut file = std::fs::File::open(&model_path)?;
    let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;

    let mut total_size_in_bytes = 0;
    for (_, tensor) in ct.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    eprintln!(
        "Loaded {} tensors ({}) in {:.1}s",
        ct.tensor_infos.len(),
        format_size(total_size_in_bytes),
        start.elapsed().as_secs_f64(),
    );

    let model = EmbeddingModel::from_gguf(ct, &mut file, &device)?;
    eprintln!(
        "Model ready (hidden_size={}) in {:.1}s",
        model.hidden_size(),
        start.elapsed().as_secs_f64(),
    );

    let tokenizer = get_tokenizer(&args)?;

    let input_file =
        std::fs::File::open(&args.input).with_context(|| format!("Cannot open {}", args.input))?;
    let reader = std::io::BufReader::new(input_file);

    let mut descriptions: Vec<String> = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Error reading line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let text = extract_field(&line, &args.field).with_context(|| {
            format!(
                "Missing or non-string field '{}' at line {}",
                args.field,
                i + 1
            )
        })?;
        descriptions.push(text);
    }
    eprintln!(
        "Read {} descriptions from {}",
        descriptions.len(),
        args.input
    );

    if descriptions.is_empty() {
        anyhow::bail!("No descriptions found in input file");
    }

    let hidden_size = model.hidden_size();
    let n = descriptions.len();
    let mut all_embeddings: Vec<f32> = Vec::with_capacity(n * hidden_size);

    let embed_start = std::time::Instant::now();

    for (i, desc) in descriptions.iter().enumerate() {
        let text = match &args.instruct {
            Some(inst) => format!("Instruct: {inst}\nQuery: {desc}"),
            None => desc.clone(),
        };

        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

        let mut token_ids = encoding.get_ids().to_vec();
        if token_ids.len() > args.max_seq_len {
            token_ids.truncate(args.max_seq_len);
        }

        let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
        let embedding = model.forward(&input)?;

        let emb_vec = embedding
            .squeeze(0)?
            .to_dtype(candle::DType::F32)?
            .to_vec1::<f32>()?;
        all_embeddings.extend_from_slice(&emb_vec);

        if (i + 1) % args.log_every == 0 || i + 1 == n {
            let elapsed = embed_start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let eta = (n - i - 1) as f64 / rate;
            eprintln!("[{}/{}] {:.1} desc/s, ETA {:.0}s", i + 1, n, rate, eta,);
        }
    }

    write_npy(&args.output, &all_embeddings, n, hidden_size)?;

    let total_elapsed = embed_start.elapsed().as_secs_f64();
    eprintln!(
        "Done: {} embeddings ({}x{}) written to {} in {:.1}s ({:.1} desc/s)",
        n,
        n,
        hidden_size,
        args.output,
        total_elapsed,
        n as f64 / total_elapsed,
    );

    Ok(())
}
