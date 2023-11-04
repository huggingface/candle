use anyhow::Result;
use candle::{Device, Tensor};

use clap::{Parser, Subcommand};

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Print {
        #[arg(long)]
        file: String,
    },
    SimpleEval {
        #[arg(long)]
        file: String,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Print { file } => {
            let model = candle_onnx::read_file(file)?;
            println!("{model:?}");
            let graph = model.graph.unwrap();
            for node in graph.node.iter() {
                println!("{node:?}");
            }
        }
        Command::SimpleEval { file } => {
            let model = candle_onnx::read_file(file)?;
            let inputs = model
                .graph
                .as_ref()
                .unwrap()
                .input
                .iter()
                .map(|name| {
                    let value = Tensor::new(&[-3.2, 2.7], &Device::Cpu)?;
                    Ok((name.name.clone(), value))
                })
                .collect::<Result<_>>()?;
            let outputs = candle_onnx::simple_eval(&model, inputs)?;
            for (name, value) in outputs.iter() {
                println!("{name}: {value:?}")
            }
        }
    }
    Ok(())
}
