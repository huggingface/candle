#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mimi::{Config, Model};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;

mod audio_io;

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

    /// The input file, either an audio file or some mimi tokens stored as safetensors.
    in_file: String,

    /// The output file, either a wave audio file or some mimi tokens stored as safetensors.
    out_file: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// Whether to use streaming or not, when streaming slices of data of the given size are passed
    /// to the encoder/decoder one at a time.
    #[arg(long)]
    streaming: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("kyutai/mimi".to_string())
            .get("model.safetensors")?,
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let config = Config::v0_1(None);
    let mut model = Model::new(config, vb)?;

    let codes = match args.action {
        Action::CodeToAudio => {
            let codes = candle::safetensors::load(args.in_file, &device)?;
            codes.get("codes").expect("no codes in input file").clone()
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
                if sample_rate != 24_000 {
                    println!("WARNING: mimi uses a 24khz sample rate, input uses {sample_rate}, resampling...");
                    audio_io::resample(&pcm, sample_rate as usize, 24_000)?
                } else {
                    pcm
                }
            };
            match args.streaming {
                Some(chunk_size) => {
                    let mut code_chunks = vec![];
                    for pcm in pcm.chunks(chunk_size) {
                        let pcm = Tensor::new(pcm, &device)?.reshape((1, 1, ()))?;
                        let code_chunk = model.encode(&pcm)?;
                        code_chunks.push(code_chunk)
                    }
                    Tensor::cat(&code_chunks, candle::D::Minus1)?
                }
                None => {
                    let pcm_len = pcm.len();
                    let pcm = Tensor::from_vec(pcm, (1, 1, pcm_len), &device)?;
                    println!("input pcm shape: {:?}", pcm.shape());
                    model.encode(&pcm)?
                }
            }
        }
    };
    println!("codes shape: {:?}", codes.shape());
    model.reset_state();

    match args.action {
        Action::AudioToCode => {
            codes.save_safetensors("codes", &args.out_file)?;
        }
        Action::AudioToAudio | Action::CodeToAudio => {
            let pcm = match args.streaming {
                Some(chunk_size) => {
                    let seq_len = codes.dim(candle::D::Minus1)?;
                    let mut pcm_chunks = vec![];
                    for chunk_start in (0..seq_len).step_by(chunk_size) {
                        let chunk_len = usize::min(chunk_size, seq_len - chunk_start);
                        let codes = codes.narrow(candle::D::Minus1, chunk_start, chunk_len)?;
                        let pcm = model.decode_step(&codes.into())?;
                        if let Some(pcm) = pcm.as_option() {
                            pcm_chunks.push(pcm.clone())
                        }
                    }
                    Tensor::cat(&pcm_chunks, candle::D::Minus1)?
                }
                None => model.decode(&codes)?,
            };
            println!("output pcm shape: {:?}", pcm.shape());
            let pcm = pcm.i(0)?.i(0)?;
            let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
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
                candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
            }
        }
    }
    Ok(())
}
