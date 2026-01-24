use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use candle::quantized::gguf_file;
use candle::quantized::tokenizer::TokenizerFromGguf;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the GGUF file that stores the tokenizer metadata.
    #[arg(long)]
    model: String,
    /// Optional revision (branch/tag/commit) when pulling from the Hugging Face Hub.
    #[arg(long)]
    revision: Option<String>,
    /// Text prompt to tokenize with the GGUF tokenizer.
    #[arg(long, default_value = "Hello Candle!")]
    prompt: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let gguf_path = resolve_model_path(&args.model, args.revision.clone())
        .with_context(|| format!("failed to locate GGUF file {}", args.model))?;

    let file = File::open(&gguf_path)
        .with_context(|| format!("failed to open GGUF file {}", gguf_path.display()))?;
    let mut reader = BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader).context("failed to load GGUF metadata")?;

    // Build the tokenizer directly from the GGUF metadata (tokens, merges, and post-processing).
    let tokenizer =
        Tokenizer::from_gguf(&content).context("failed to initialize tokenizer from GGUF")?;

    let encoding = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(anyhow::Error::msg)
        .context("failed to tokenize prompt")?;

    println!("Prompt: {}", args.prompt);
    println!("Source: {}", gguf_path.display());
    println!("Token ids: {:?}", encoding.get_ids());
    println!("Tokens: {:?}", encoding.get_tokens());
    println!(
        "Special tokens mask: {:?}",
        encoding.get_special_tokens_mask()
    );

    let decoded = tokenizer
        .decode(encoding.get_ids(), true)
        .map_err(anyhow::Error::msg)
        .context("failed to decode tokens")?;

    println!("Decoded (special tokens stripped): {decoded}");

    Ok(())
}

fn resolve_model_path(model: &str, revision: Option<String>) -> Result<PathBuf> {
    // Local path: use as-is if it exists.
    let candidate = Path::new(model);
    if candidate.exists() {
        return Ok(candidate.to_path_buf());
    }

    // Hugging Face Hub: accept strings like `author/repo/weights.gguf` or
    // `author/repo/subdir/weights.gguf`. An optional `revision` can be provided.
    let trimmed = model
        .trim_start_matches("hf://")
        .trim_start_matches("https://huggingface.co/")
        .trim_start_matches("huggingface.co/");
    let parts: Vec<_> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() < 3 {
        anyhow::bail!(
            "model must be a local file or an HF path like `author/repo/file.gguf`, got `{model}`"
        );
    }

    let repo_id = format!("{}/{}", parts[0], parts[1]);
    let filename = parts[2..].join("/");

    let api = Api::new()?;
    let repo = Repo::with_revision(
        repo_id,
        RepoType::Model,
        revision.unwrap_or_else(|| "main".to_string()),
    );
    let path = api.repo(repo).get(&filename)?;
    Ok(path)
}
