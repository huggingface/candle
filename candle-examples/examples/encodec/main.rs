#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::encodec::{Config, Model};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;

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

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Action {
    AudioToAudio,
    AudioToCode,
    CodeToAudio,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The action to be performed, specifies the format for the input and output data.
    action: Action,

    /// The input file, either an audio file or some encodec tokens stored as safetensors.
    in_file: String,

    /// The output file, either a wave audio file or some encodec tokens stored as safetensors.
    out_file: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("facebook/encodec_24khz".to_string())
            .get("model.safetensors")?,
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let config = Config::default();
    let model = Model::new(&config, vb)?;

    let codes = match args.action {
        Action::CodeToAudio => {
            let codes = candle::safetensors::load(args.in_file, &device)?;
            codes.get("codes").expect("no codes in input file").clone()
        }
        Action::AudioToCode | Action::AudioToAudio => {
            let (pcm, sample_rate) = pcm_decode(args.in_file)?;
            if sample_rate != 24_000 {
                println!("WARNING: encodec uses a 24khz sample rate, input uses {sample_rate}")
            }
            let pcm_len = pcm.len();
            let pcm = Tensor::from_vec(pcm, (1, 1, pcm_len), &device)?;
            println!("input pcm shape: {:?}", pcm.shape());
            model.encode(&pcm)?
        }
    };
    println!("codes shape: {:?}", codes.shape());

    match args.action {
        Action::AudioToCode => {
            codes.save_safetensors("codes", &args.out_file)?;
        }
        Action::AudioToAudio | Action::CodeToAudio => {
            let pcm = model.decode(&codes)?;
            println!("output pcm shape: {:?}", pcm.shape());
            let pcm = pcm.i(0)?.i(0)?;
            let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
            let pcm = pcm.to_vec1::<f32>()?;
            let mut output = std::fs::File::create(&args.out_file)?;
            candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
        }
    }
    Ok(())
}
