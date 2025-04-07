#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::snac::{Config, Model};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;

mod audio_io;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Action {
    AudioToAudio,
    AudioToCode,
    CodeToAudio,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "24khz")]
    S24khz,
    #[value(name = "32khz")]
    S32khz,
    #[value(name = "44khz")]
    S44khz,
}

impl Which {
    fn sample_rate(&self) -> u32 {
        match self {
            Which::S24khz => 24000,
            Which::S32khz => 32000,
            Which::S44khz => 44000,
        }
    }

    fn config_repo(&self) -> &'static str {
        match self {
            Which::S24khz => "hubertsiuzdak/snac_24khz",
            Which::S32khz => "hubertsiuzdak/snac_32khz",
            Which::S44khz => "hubertsiuzdak/snac_44khz",
        }
    }

    fn model_file(&self) -> &'static str {
        match self {
            Which::S24khz => "snac_24khz.safetensors",
            Which::S32khz => "snac_32khz.safetensors",
            Which::S44khz => "snac_44khz.safetensors",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The action to be performed, specifies the format for the input and output data.
    action: Action,

    /// The input file, either an audio file or some snac tokens stored as safetensors.
    in_file: String,

    /// The output file, either a wave audio file or some snac tokens stored as safetensors.
    out_file: String,

    /// The model size to use.
    #[arg(long, default_value = "24khz")]
    which: Which,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// The config file, in safetensor format.
    #[arg(long)]
    config: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let model_sample_rate = args.which.sample_rate();
    let config = match args.config {
        Some(c) => std::path::PathBuf::from(c),
        None => Api::new()?
            .model(args.which.config_repo().to_string())
            .get("config.json")?,
    };
    let config: Config = serde_json::from_slice(&std::fs::read(config)?)?;
    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("lmz/candle-snac".to_string())
            .get(args.which.model_file())?,
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let model = Model::new(&config, vb)?;

    let codes = match args.action {
        Action::CodeToAudio => {
            let codes = candle::safetensors::load(args.in_file, &device)?;
            let num_codebooks = model.num_codebooks();
            (0..num_codebooks)
                .map(|i| {
                    codes
                        .get(&format!("codes-{i}"))
                        .expect("no codes in input file")
                        .clone()
                })
                .collect::<Vec<_>>()
        }
        Action::AudioToCode | Action::AudioToAudio => {
            let pcm = if args.in_file == "-" {
                println!(">>>> RECORDING AUDIO, PRESS ENTER ONCE DONE <<<<");
                let (stream, input_audio) = audio_io::setup_input_stream()?;
                let mut pcms = vec![];
                let stdin = std::thread::spawn(|| {
                    let mut s = String::new();
                    std::io::stdin().read_line(&mut s)
                });
                while !stdin.is_finished() {
                    let input = input_audio.lock().unwrap().take_all();
                    if input.is_empty() {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        continue;
                    }
                    pcms.push(input)
                }
                drop(stream);
                pcms.concat()
            } else {
                let (pcm, sample_rate) = audio_io::pcm_decode(args.in_file)?;
                if sample_rate != model_sample_rate {
                    println!("WARNING: snac uses a {model_sample_rate} sample rate, input uses {sample_rate}, resampling...");
                    audio_io::resample(&pcm, sample_rate, model_sample_rate)?
                } else {
                    pcm
                }
            };
            let pcm_len = pcm.len();
            let pcm = Tensor::from_vec(pcm, (1, 1, pcm_len), &device)?;
            println!("input pcm shape: {:?}", pcm.shape());
            model.encode(&pcm)?
        }
    };
    for codes in codes.iter() {
        println!("codes shape: {:?}", codes.shape());
    }

    match args.action {
        Action::AudioToCode => {
            let mut tensors = std::collections::HashMap::new();
            for (i, codes) in codes.iter().enumerate() {
                tensors.insert(format!("codes-{i}"), codes.clone());
            }
            candle::safetensors::save(&tensors, "codes.safetensors")?;
        }
        Action::AudioToAudio | Action::CodeToAudio => {
            let codes = codes.iter().collect::<Vec<_>>();
            let pcm = model.decode(&codes)?;
            println!("output pcm shape: {:?}", pcm.shape());
            let pcm = pcm.i(0)?.i(0)?;
            let pcm = candle_examples::audio::normalize_loudness(&pcm, model_sample_rate, true)?;
            let pcm = pcm.to_vec1::<f32>()?;
            if args.out_file == "-" {
                let (stream, ad) = audio_io::setup_output_stream()?;
                {
                    let mut ad = ad.lock().unwrap();
                    ad.push_samples(&pcm)?;
                }
                loop {
                    let ad = ad.lock().unwrap();
                    if ad.is_empty() {
                        break;
                    }
                    // That's very weird, calling thread::sleep here triggers the stream to stop
                    // playing (the callback doesn't seem to be called anymore).
                    // std::thread::sleep(std::time::Duration::from_millis(100));
                }
                drop(stream)
            } else {
                let mut output = std::fs::File::create(&args.out_file)?;
                candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, model_sample_rate)?;
            }
        }
    }
    Ok(())
}
