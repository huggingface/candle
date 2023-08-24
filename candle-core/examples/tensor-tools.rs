use candle_core::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(ValueEnum, Debug, Clone)]
enum Format {
    Safetensors,
    Npz,
    Ggml,
    Gguf,
    Pth,
    Pickle,
}

impl Format {
    fn infer<P: AsRef<std::path::Path>>(p: P) -> Option<Self> {
        p.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|e| match e {
                // We don't infer any format for .bin as it can be used for ggml/gguf or pytorch.
                "safetensors" | "safetensor" => Some(Self::Safetensors),
                "npz" => Some(Self::Npz),
                "pth" | "pt" => Some(Self::Pth),
                "ggml" => Some(Self::Ggml),
                "gguf" => Some(Self::Gguf),
                _ => None,
            })
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Ls {
        files: Vec<std::path::PathBuf>,

        /// The file format to use, if unspecified infer from the file extension.
        #[arg(long, value_enum)]
        format: Option<Format>,

        /// Enable verbose mode.
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Parser, Debug, Clone)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

fn run_ls(file: &std::path::PathBuf, format: Option<Format>, verbose: bool) -> Result<()> {
    let format = match format {
        Some(format) => format,
        None => match Format::infer(file) {
            Some(format) => format,
            None => {
                println!(
                    "{file:?}: cannot infer format from file extension, use the --format flag"
                );
                return Ok(());
            }
        },
    };
    match format {
        Format::Npz => {
            let tensors = candle_core::npy::NpzTensors::new(file)?;
            let mut names = tensors.names();
            names.sort();
            for name in names {
                let shape_dtype = match tensors.get_shape_and_dtype(name) {
                    Ok((shape, dtype)) => format!("[{shape:?}; {dtype:?}]"),
                    Err(err) => err.to_string(),
                };
                println!("{name}: {shape_dtype}")
            }
        }
        Format::Safetensors => {
            let tensors = unsafe { candle_core::safetensors::MmapedFile::new(file)? };
            let tensors = tensors.deserialize()?;
            let mut tensors = tensors.tensors();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, view) in tensors.iter() {
                let dtype = view.dtype();
                let dtype = match candle_core::DType::try_from(dtype) {
                    Ok(dtype) => format!("{dtype:?}"),
                    Err(_) => format!("{dtype:?}"),
                };
                let shape = view.shape();
                println!("{name}: [{shape:?}; {dtype}]")
            }
        }
        Format::Pth => {
            let mut tensors = candle_core::pickle::read_pth_tensor_info(file, verbose)?;
            tensors.sort_by(|a, b| a.name.cmp(&b.name));
            for tensor_info in tensors.iter() {
                println!(
                    "{}: [{:?}; {:?}]",
                    tensor_info.name,
                    tensor_info.layout.shape(),
                    tensor_info.dtype,
                );
                if verbose {
                    println!("    {:?}", tensor_info);
                }
            }
        }
        Format::Pickle => {
            let file = std::fs::File::open(file)?;
            let mut reader = std::io::BufReader::new(file);
            let mut stack = candle_core::pickle::Stack::empty();
            stack.read_loop(&mut reader)?;
            for (i, obj) in stack.stack().iter().enumerate() {
                println!("{i} {obj:?}");
            }
        }
        Format::Ggml => {
            let mut file = std::fs::File::open(file)?;
            let content = candle_core::quantized::ggml_file::Content::read(&mut file)?;
            let mut tensors = content.tensors.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, qtensor) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", qtensor.shape(), qtensor.dtype());
            }
        }
        Format::Gguf => {
            let mut file = std::fs::File::open(file)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
            if verbose {
                let mut metadata = content.metadata.into_iter().collect::<Vec<_>>();
                metadata.sort_by(|a, b| a.0.cmp(&b.0));
                println!("metadata entries ({})", metadata.len());
                for (key, value) in metadata.iter() {
                    println!("  {key}: {value:?}");
                }
            }
            let mut tensors = content.tensor_infos.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, info) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", info.shape, info.ggml_dtype);
            }
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Ls {
            files,
            format,
            verbose,
        } => {
            let multiple_files = files.len() > 1;
            for file in files.iter() {
                if multiple_files {
                    println!("--- {file:?} ---");
                }
                run_ls(file, format.clone(), verbose)?
            }
        }
    }
    Ok(())
}
