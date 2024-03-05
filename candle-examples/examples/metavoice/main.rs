#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;
use std::io::Write;

use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::encodec;
use candle_transformers::models::metavoice::{
    adapters, gpt, speaker_encoder, tokenizers, transformer,
};

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use rand::{distributions::Distribution, SeedableRng};

pub const ENCODEC_NTOKENS: u32 = 1024;

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    use symphonia::core::audio::Signal;
    use symphonia::core::conv::FromSample;
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

fn pcm_decode<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};

    let src = std::fs::File::open(path)?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .expect("no supported audio tracks");
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .expect("unsupported codec");
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ArgDType {
    F32,
    F16,
    Bf16,
}

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

    /// The maximum number of tokens to generate for the first stage.
    #[arg(long, default_value_t = 2000)]
    max_tokens: u64,

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
    speaker_encoder_weights: Option<String>,

    #[arg(long)]
    encodec_weights: Option<String>,

    /// The speaker embeddings, either an audio files in which case they are extracted, or a
    /// safetensors file with the embeddings already extracted.
    #[arg(long)]
    spk_emb: Option<String>,

    #[arg(long, default_value = "f32")]
    dtype: ArgDType,
}

fn mel_filters() -> Result<Vec<f32>> {
    let mel_bytes = include_bytes!("melfilters40.bytes").as_slice();
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    Ok(mel_filters)
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
    let dtype = match args.dtype {
        ArgDType::F32 => DType::F32,
        ArgDType::F16 => DType::F16,
        ArgDType::Bf16 => DType::BF16,
    };
    let first_stage_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[first_stage_weights], dtype, &device)? };
    let first_stage_config = transformer::Config::cfg1b_v0_1();
    let mut first_stage_model = transformer::Model::new(&first_stage_config, first_stage_vb)?;

    let second_stage_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[second_stage_weights], dtype, &device)? };
    let second_stage_config = gpt::Config::cfg1b_v0_1();
    let second_stage_model = gpt::Model::new(second_stage_config.clone(), second_stage_vb)?;

    let encodec_device = if device.is_metal() {
        &candle::Device::Cpu
    } else {
        &device
    };
    let encodec_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[encodec_weights], dtype, encodec_device)? };
    let encodec_config = encodec::Config::default();
    let encodec_model = encodec::Model::new(&encodec_config, encodec_vb)?;

    println!("prompt: '{}'", args.prompt);
    let prompt_tokens = fs_tokenizer.encode(&args.prompt)?;
    let mut tokens = prompt_tokens.clone();
    println!("{tokens:?}");
    let safetensors_embeddings = args
        .spk_emb
        .as_ref()
        .map_or(true, |v| v.ends_with("safetensors"));
    let spk_emb = if safetensors_embeddings {
        let spk_emb_file = match &args.spk_emb {
            Some(w) => std::path::PathBuf::from(w),
            None => repo.get("spk_emb.safetensors")?,
        };
        let spk_emb = candle::safetensors::load(&spk_emb_file, &candle::Device::Cpu)?;
        match spk_emb.get("spk_emb") {
            None => anyhow::bail!("missing spk_emb tensor in {spk_emb_file:?}"),
            Some(spk_emb) => spk_emb.to_dtype(dtype)?.to_device(&device)?,
        }
    } else {
        let weights = match &args.speaker_encoder_weights {
            Some(w) => std::path::PathBuf::from(w),
            None => repo.get("speaker_encoder.safetensors")?,
        };
        let mel_filters = mel_filters()?;
        let config = speaker_encoder::Config::cfg();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
        let model = speaker_encoder::Model::new(config, vb)?;
        let (pcm, sample_rate) = pcm_decode(&args.spk_emb.unwrap())?;
        if sample_rate != 16_000 {
            eprintln!("WARNING: speaker embedding input should use a 16kHz sample rate!")
        }
        model.embed_utterance(
            &pcm,
            &mel_filters,
            /* rate */ 1.3,
            /* min_c */ 0.75,
            &device,
        )?
    };
    let mut logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), Some(0.95));

    // First stage generation.
    for index in 0..args.max_tokens {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &device)?;
        let input = Tensor::stack(&[&input, &input], 0)?;
        let logits = first_stage_model.forward(&input, &spk_emb, tokens.len() - context_size)?;
        let logits0 = logits.i((0, 0))?;
        let logits1 = logits.i((1, 0))?;
        let logits = ((logits0 * args.guidance_scale)? + logits1 * (1. - args.guidance_scale))?;
        let logits = logits.to_dtype(DType::F32)?;
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        print!(".");
        std::io::stdout().flush()?;
        if next_token == 2048 {
            break;
        }
    }
    println!();
    let fie2c = adapters::FlattenedInterleavedEncodec2Codebook::new(ENCODEC_NTOKENS);
    let (text_ids, ids1, ids2) = fie2c.decode(&tokens);
    println!("text ids len: {}", text_ids.len());
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed + 1337);
    // TODO: Use the config rather than hardcoding the offset here.
    let encoded_text: Vec<_> = prompt_tokens.iter().map(|v| v - 1024).collect();
    let mut hierarchies_in1 =
        [encoded_text.as_slice(), ids1.as_slice(), &[ENCODEC_NTOKENS]].concat();
    let mut hierarchies_in2 = [
        vec![ENCODEC_NTOKENS; encoded_text.len()].as_slice(),
        ids2.as_slice(),
        &[ENCODEC_NTOKENS],
    ]
    .concat();
    hierarchies_in1.resize(second_stage_config.block_size, ENCODEC_NTOKENS);
    hierarchies_in2.resize(second_stage_config.block_size, ENCODEC_NTOKENS);
    let in_x1 = Tensor::new(hierarchies_in1, &device)?;
    let in_x2 = Tensor::new(hierarchies_in2, &device)?;
    let in_x = Tensor::stack(&[in_x1, in_x2], 0)?.unsqueeze(0)?;
    let logits = second_stage_model.forward(&in_x)?;
    println!("sampling from logits...");
    let mut codes = vec![];
    for logits in logits.iter() {
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
    let audio_ids = Tensor::new(audio_ids, encodec_device)?.unsqueeze(0)?;
    println!("audio_ids shape: {:?}", audio_ids.shape());
    let pcm = encodec_model.decode(&audio_ids)?;
    println!("output pcm shape: {:?}", pcm.shape());
    let pcm = pcm.i(0)?.i(0)?.to_dtype(DType::F32)?;
    let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
    let pcm = pcm.to_vec1::<f32>()?;
    let mut output = std::fs::File::create(&args.out_file)?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
    Ok(())
}
