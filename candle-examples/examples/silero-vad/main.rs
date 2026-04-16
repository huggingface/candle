#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;

use candle::{DType, Tensor};

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

/// an iterator which reads consecutive frames of le i16 values from a reader
struct I16Frames<R> {
    rdr: R,
    buf: Box<[u8]>,
    len: usize,
    eof: bool,
}
impl<R> I16Frames<R> {
    fn new(rdr: R, frame_size: usize) -> Self {
        I16Frames {
            rdr,
            buf: vec![0; frame_size * std::mem::size_of::<i16>()].into_boxed_slice(),
            len: 0,
            eof: false,
        }
    }
}
impl<R: std::io::Read> Iterator for I16Frames<R> {
    type Item = std::io::Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof {
            return None;
        }
        self.len += match self.rdr.read(&mut self.buf[self.len..]) {
            Ok(0) => {
                self.eof = true;
                0
            }
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };
        if self.eof || self.len == self.buf.len() {
            let buf = self.buf[..self.len]
                .chunks(2)
                .map(|bs| match bs {
                    [a, b] => i16::from_le_bytes([*a, *b]),
                    _ => unreachable!(),
                })
                .map(|i| i as f32 / i16::MAX as f32)
                .collect();
            self.len = 0;
            Some(Ok(buf))
        } else {
            self.next()
        }
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

    let mut state = State {
        frame_size,
        sample_rate: Tensor::new(sample_rate, &device)?,
        state: Tensor::zeros((2, 1, 128), DType::F32, &device)?,
        context: Tensor::zeros((1, context_size), DType::F32, &device)?,
    };
    let mut res = vec![];
    for chunk in I16Frames::new(std::io::stdin().lock(), state.frame_size) {
        let chunk = chunk.unwrap();
        if chunk.len() < state.frame_size {
            continue;
        }
        let next_context = Tensor::from_slice(
            &chunk[state.frame_size - context_size..],
            (1, context_size),
            &device,
        )?;
        let chunk = Tensor::from_vec(chunk, (1, state.frame_size), &device)?;
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
        assert_eq!(output.len(), 1);
        let output = output[0];
        println!("vad chunk prediction: {output}");
        res.push(output);
    }
    println!("calculated prediction in {:?}", start.elapsed());

    let res_len = res.len() as f32;
    let prediction = res.iter().sum::<f32>() / res_len;
    println!("vad average prediction: {prediction}");
    Ok(())
}
