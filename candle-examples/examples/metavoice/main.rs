#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;

use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::encodec;
use candle_transformers::models::metavoice::{adapters, gpt, tokenizers, transformer};

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use rand::{distributions::Distribution, SeedableRng};

pub const ENCODEC_NTOKENS: u32 = 1024;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// The guidance scale.
    #[arg(long, default_value_t = 3.0)]
    guidance_scale: f64,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The output file using the wav format.
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    #[arg(long)]
    first_stage_meta: Option<String>,

    #[arg(long)]
    first_stage_weights: Option<String>,

    #[arg(long)]
    second_stage_weights: Option<String>,

    #[arg(long)]
    encodec_weights: Option<String>,

    #[arg(long)]
    spk_emb: Option<String>,
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
    let device = candle_examples::device(args.cpu)?;
    let api = Api::new()?;
    let repo = api.model("lmz/candle-metavoice".to_string());
    let first_stage_meta = match &args.first_stage_meta {
        Some(w) => std::path::PathBuf::from(w),
        None => repo.get("first_stage.meta.json")?,
    };
    let first_stage_meta: serde_json::Value =
        serde_json::from_reader(&std::fs::File::open(first_stage_meta)?)?;
    let first_stage_tokenizer = match first_stage_meta.as_object() {
        None => anyhow::bail!("not a json object"),
        Some(j) => match j.get("tokenizer") {
            None => anyhow::bail!("no tokenizer key"),
            Some(j) => j,
        },
    };
    let fs_tokenizer = tokenizers::BPE::from_json(first_stage_tokenizer, 512)?;

    let first_stage_weights = match &args.first_stage_weights {
        Some(w) => std::path::PathBuf::from(w),
        None => repo.get("first_stage.safetensors")?,
    };
    let second_stage_weights = match &args.first_stage_weights {
        Some(w) => std::path::PathBuf::from(w),
        None => repo.get("second_stage.safetensors")?,
    };
    let encodec_weights = match args.encodec_weights {
        Some(w) => std::path::PathBuf::from(w),
        None => Api::new()?
            .model("facebook/encodec_24khz".to_string())
            .get("model.safetensors")?,
    };
    let first_stage_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[first_stage_weights], DType::F32, &device)?
    };
    let first_stage_config = transformer::Config::cfg1b_v0_1();
    let mut first_stage_model = transformer::Model::new(&first_stage_config, first_stage_vb)?;

    let second_stage_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[second_stage_weights], DType::F32, &device)?
    };
    let second_stage_config = gpt::Config::cfg1b_v0_1();
    let second_stage_model = gpt::Model::new(second_stage_config, second_stage_vb)?;

    let encodec_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[encodec_weights], DType::F32, &device)? };
    let encodec_config = encodec::Config::default();
    let encodec_model = encodec::Model::new(&encodec_config, encodec_vb)?;

    println!("prompt: '{}'", args.prompt);
    let prompt_tokens = fs_tokenizer.encode(&args.prompt)?;
    let mut tokens = prompt_tokens.clone();
    println!("{tokens:?}");
    let spk_emb_file = match &args.spk_emb {
        Some(w) => std::path::PathBuf::from(w),
        None => repo.get("spk_emb.safetensors")?,
    };
    let spk_emb = candle::safetensors::load(&spk_emb_file, &device)?;
    let spk_emb = match spk_emb.get("spk_emb") {
        None => anyhow::bail!("missing spk_emb tensor in {spk_emb_file:?}"),
        Some(spk_emb) => spk_emb.to_dtype(DType::F32)?,
    };
    let mut logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), None);

    // First stage generation.
    for index in 0.. {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &device)?;
        let input = Tensor::stack(&[&input, &input], 0)?;
        let logits = first_stage_model.forward(&input, &spk_emb, index)?;
        let logits0 = logits.i((0, 0))?;
        let logits1 = logits.i((1, 0))?;
        let logits = ((logits0 * args.guidance_scale)? + logits1 * (1. - args.guidance_scale))?;
        let logits = logits.to_dtype(DType::F32)?;
        let next_token = logits_processor.sample(&logits)?;
        println!("{} {next_token}", tokens.len());
        tokens.push(next_token);
        if next_token == 2048 {
            break;
        }
    }
    let fie2c = adapters::FlattenedInterleavedEncodec2Codebook::new(ENCODEC_NTOKENS);
    let (text_ids, ids1, ids2) = fie2c.decode(&tokens);
    println!("text ids len: {}", text_ids.len());
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed + 1337);
    // TODO: Use the config rather than hardcoding the offset here.
    let encoded_text: Vec<_> = prompt_tokens.iter().map(|v| v - 1024).collect();
    let hierarchies_in1 = [encoded_text.as_slice(), ids1.as_slice(), &[ENCODEC_NTOKENS]].concat();
    let hierarchies_in2 = [
        vec![ENCODEC_NTOKENS; encoded_text.len()].as_slice(),
        ids2.as_slice(),
        &[ENCODEC_NTOKENS],
    ]
    .concat();
    let in_x1 = Tensor::new(hierarchies_in1, &device)?;
    let in_x2 = Tensor::new(hierarchies_in2, &device)?;
    let in_x = Tensor::stack(&[in_x1, in_x2], 0)?.unsqueeze(0)?;
    let logits = second_stage_model.forward(&in_x)?;
    let mut codes = vec![];
    for (idx, logits) in logits.iter().enumerate() {
        println!("{idx} {logits}");
        let logits = logits.squeeze(0)?;
        let (seq_len, _) = logits.dims2()?;
        let mut codes_ = Vec::with_capacity(seq_len);
        for step in 0..seq_len {
            let logits = logits.i(step)?.to_dtype(DType::F32)?;
            let logits = &(&logits / 1.0)?;
            let prs = candle_nn::ops::softmax_last_dim(logits)?.to_vec1::<f32>()?;
            let distr = rand::distributions::WeightedIndex::new(prs.as_slice())?;
            let sample = distr.sample(&mut rng) as u32;
            codes_.push(sample)
        }
        codes.push(codes_)
    }

    let codes = Tensor::new(codes, &device)?.unsqueeze(0)?;
    let codes = Tensor::cat(&[in_x, codes], 1)?;
    println!("codes: {codes}");
    let tilted_encodec = adapters::TiltedEncodec::new(ENCODEC_NTOKENS);
    let codes = codes.i(0)?.to_vec2::<u32>()?;
    let (text_ids, audio_ids) = tilted_encodec.decode(&codes);
    println!("text_ids len: {:?}", text_ids.len());
    let audio_ids = Tensor::new(audio_ids, &device)?.unsqueeze(0)?;
    println!("audio_ids shape: {:?}", audio_ids.shape());
    let pcm = encodec_model.decode(&audio_ids)?;
    println!("output pcm shape: {:?}", pcm.shape());
    let pcm = pcm.i(0)?.i(0)?.to_vec1::<f32>()?;
    let mut output = std::fs::File::create(&args.out_file)?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
    Ok(())
}
