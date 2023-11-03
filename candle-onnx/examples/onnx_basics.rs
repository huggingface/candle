use anyhow::Result;

use clap::{Parser, Subcommand};

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Print {
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
    }
    Ok(())
}
