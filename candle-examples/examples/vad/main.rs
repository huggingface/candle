#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::io::Read;

use anyhow::{Error as E, Result};
use candle_examples::wav::{self, write_pcm_as_wav};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_onnx;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "silero")]
    Silero,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SampleRate {
    #[value(name = "8000")]
    Sr8k,
    #[value(name = "16000")]
    Sr16k,
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
    input: Option<String>,

    #[arg(long)]
    sample_rate: SampleRate,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    /// The model to use.
    #[arg(long, default_value = "silero")]
    which: Which,
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
    let model_id = match &args.model_id {
        Some(model_id) => std::path::PathBuf::from(model_id),
        None => match args.which {
            Which::Silero => hf_hub::api::sync::Api::new()?
                .model("onnx-community/silero-vad".into())
                .get("onnx/model.onnx")?,
            // TODO: candle-onnx doesn't support Int8 dtype
            // Which::SileroQuantized => hf_hub::api::sync::Api::new()?
            //     .model("onnx-community/silero-vad".into())
            //     .get("onnx/model_quantized.onnx")?,
        },
    };
    let (sample_rate, frame_size, context_size): (i64, usize, usize) = match args.sample_rate {
        SampleRate::Sr8k => (8000, 256, 32),
        SampleRate::Sr16k => (16000, 512, 64),
    };
    println!("retrieved the files in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let model = candle_onnx::read_file(model_id)?;

    println!("loaded the model in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    struct State {
        frame_size: usize,
        sample_rate: Tensor,
        state: Tensor,
        context: Tensor,
    }

    let mut pcm = vec![];
    std::io::stdin().read_to_end(&mut pcm)?;
    assert_eq!(pcm.len() % 2, 0);
    let pcm: Vec<_> = pcm
        .chunks(2)
        .map(|bs| match bs {
            &[a, b] => i16::from_le_bytes([a, b]),
            _ => unreachable!(),
        })
        .collect();
    let frame: &[i16] = &pcm;

    let mut state = State {
        frame_size,
        sample_rate: Tensor::new(sample_rate, &device)?,
        state: Tensor::zeros((2, 1, 128), DType::F32, &device)?,
        context: Tensor::zeros((1, context_size), DType::F32, &device)?,
    };
    let mut res = vec![];
    for chunk in frame.chunks(state.frame_size) {
        if chunk.len() < state.frame_size {
            continue;
        }
        let chunk: Vec<_> = chunk
            .into_iter()
            .map(|i| *i as f32 / i16::MAX as f32)
            .collect();
        let next_context = Tensor::from_slice(
            &chunk[state.frame_size - context_size..],
            (1, context_size),
            &device,
        )?;
        let chunk = Tensor::from_iter(chunk.into_iter(), &device)?.unsqueeze(0)?;
        let chunk = Tensor::cat(&[&state.context, &chunk], 1)?;
        let inputs = std::collections::HashMap::from_iter([
            ("input".to_string(), chunk),
            ("sr".to_string(), state.sample_rate.clone()),
            ("state".to_string(), state.state.clone()),
        ]);
        let out = candle_onnx::simple_eval(&model, inputs).unwrap();
        let out_names = &model.graph.as_ref().unwrap().output;
        let output = out.get(&out_names[0].name).unwrap().clone();
        state.state = out.get(&out_names[1].name).unwrap().clone();
        assert_eq!(state.state.dims(), &[2, 1, 128]);
        state.context = next_context;

        let output = output.flatten_all()?.to_vec1::<f32>()?;
        res.extend(output);
    }
    println!("calculated prediction in {:?}", start.elapsed());

    let res_len = res.len() as f32;
    let prediction = res.iter().sum::<f32>() / res_len;
    println!("vad prediction: {prediction}");
    Ok(())
}
