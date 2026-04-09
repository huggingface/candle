//! Batch inference with batched beam search for BART models.
//!
//! Efficiently process multiple inputs with parallel beam search decoding.
//! All batch_size × beam_size hypotheses are processed in parallel with
//! efficient KV cache reuse and reordering.
//!
//! ```
//! cargo run --example bart_batch --release -- \
//!     --model-id facebook/bart-large-cnn \
//!     --input-file prompts.txt \
//!     --batch-size 4 \
//!     --beam-size 4 \
//!     --length-penalty 1.0
//! ```

use anyhow::{Error as E, Result};
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bart;
use clap::Parser;
use tokenizers::{Encoding, PaddingParams, PaddingStrategy, Tokenizer};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "facebook/bart-large-cnn")]
    model_id: String,

    /// File with one prompt per line
    #[arg(long)]
    input_file: String,

    #[arg(long, default_value_t = 4)]
    batch_size: usize,

    #[arg(long, default_value_t = 50)]
    max_length: usize,

    /// Beam size for beam search (1 = greedy decoding)
    #[arg(long, default_value_t = 4)]
    beam_size: usize,

    /// Length penalty for beam search (higher = longer outputs)
    #[arg(long, default_value_t = 1.0)]
    length_penalty: f64,

    /// Minimum generation length before EOS is allowed
    #[arg(long, default_value_t = 0)]
    min_length: usize,

    /// Block n-gram repetition (0 = disabled)
    #[arg(long, default_value_t = 3)]
    no_repeat_ngram_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    // Load prompts
    let prompts: Vec<String> = std::fs::read_to_string(&args.input_file)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(String::from)
        .collect();

    if prompts.is_empty() {
        anyhow::bail!("No prompts found in {}", args.input_file);
    }

    println!("Loaded {} prompts", prompts.len());

    // Load model
    println!("Loading model: {}", args.model_id);
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));

    let config_file = repo.get("config.json")?;
    let config: bart::BartConfig = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;

    let tokenizer_file = repo.get("tokenizer.json")?;
    let mut tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;

    // Enable padding to longest in batch
    let pad_id = config.pad_token_id;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id,
        pad_token: "[PAD]".to_string(),
        ..Default::default()
    }));

    let model_file = repo.get("model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)? };

    let mut model = bart::BartForConditionalGeneration::new(&config, vb)?;
    println!("Model loaded!");

    // Create batched beam search config
    let beam_config = bart::BatchedBeamSearchConfig::new(
        args.beam_size,
        args.length_penalty,
        args.min_length,
        args.max_length,
        args.no_repeat_ngram_size,
    );

    println!(
        "Batched beam search config: beam_size={}, length_penalty={}, min_length={}, max_length={}, no_repeat_ngram={}",
        args.beam_size, args.length_penalty, args.min_length, args.max_length, args.no_repeat_ngram_size
    );

    // Process in batches
    let mut all_outputs = Vec::new();

    for (batch_idx, batch_prompts) in prompts.chunks(args.batch_size).enumerate() {
        println!(
            "\nProcessing batch {}/{}",
            batch_idx + 1,
            prompts.len().div_ceil(args.batch_size)
        );

        // Tokenize batch with padding
        let encodings: Vec<Encoding> = tokenizer
            .encode_batch(batch_prompts.to_vec(), true)
            .map_err(E::msg)?;

        // Extract input_ids
        let input_ids: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_ids())
            .copied()
            .collect();

        let seq_len = encodings[0].len();
        let batch_size = batch_prompts.len();

        let input_tensor =
            Tensor::new(input_ids.as_slice(), &device)?.reshape((batch_size, seq_len))?;

        println!(
            "Batch shape: {:?}, Max length: {}",
            input_tensor.dims(),
            seq_len
        );

        // Encode all inputs in batch
        let encoder_output = model.encode(&input_tensor)?;

        // Run batched beam search (all batch_size × beam_size hypotheses in parallel)
        let results =
            bart::batched_beam_search(&model, &encoder_output, &config, &beam_config, &device)?;

        // Decode outputs
        let num_start = config.initial_decoder_tokens().len();
        for (prompt, hypotheses) in batch_prompts.iter().zip(results.iter()) {
            let (best_tokens, score) = &hypotheses[0];

            // Skip start tokens when decoding
            let decode_tokens = if best_tokens.len() > num_start {
                &best_tokens[num_start..]
            } else {
                best_tokens.as_slice()
            };

            let decoded = tokenizer.decode(decode_tokens, true).map_err(E::msg)?;

            println!("\nPrompt: {}", prompt);
            println!("Output (score={:.3}): {}", score, decoded);

            all_outputs.push((prompt.clone(), decoded, *score));
        }
    }

    println!("\n=== Summary ===");
    println!(
        "Processed {} prompts in {} batches",
        prompts.len(),
        prompts.len().div_ceil(args.batch_size)
    );

    Ok(())
}
