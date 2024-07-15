use anyhow::Result;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() -> Result<()> {
    let kernel_files: Vec<PathBuf> = kernel_source_files()?;
    let mut metallib_files: Vec<PathBuf> = Vec::with_capacity(kernel_files.len());

    for kernel_file in kernel_files {
        let ir_path = compile_kernel(kernel_file)?;
        let metallib_path = link_kernel(ir_path)?;
        metallib_files.push(metallib_path);
    }

    gen_metallibs_rs(metallib_files)?;

    Ok(())
}

fn kernel_source_files() -> Result<Vec<PathBuf>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let src_dir = Path::new(&manifest_dir).join("src");

    let mut paths = Vec::new();
    for entry in fs::read_dir(src_dir)? {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map(|ext| ext.to_str().unwrap()) == Some("metal") {
            paths.push(path);
        }
    }

    Ok(paths)
}

fn compile_kernel(kernel_path: impl AsRef<Path>) -> Result<PathBuf> {
    let out_dir = std::env::var("OUT_DIR")?;

    let file_stem = kernel_path.as_ref().file_stem().unwrap().to_str().unwrap();
    let ir_file_name = format!("{}.ir", file_stem,);

    println!("cargo:rerun-if-changed=src/{}.metal", file_stem);

    let output_file = Path::new(&out_dir).join(ir_file_name);

    let mut command = std::process::Command::new("xcrun");
    command.arg("metal");
    command.arg("-c");
    command.arg(format!("{}", kernel_path.as_ref().display()));
    command.arg("-o");
    command.arg(format!("{}", output_file.display()));

    let status = command.status()?;

    if !status.success() {
        return Err(anyhow::anyhow!(
            "Failed to compile kernel file: {:?}",
            kernel_path.as_ref()
        ));
    }

    Ok(output_file)
}

fn link_kernel(ir_path: impl AsRef<Path>) -> Result<PathBuf> {
    let out_dir = std::env::var("OUT_DIR")?;

    let metallib_file_name = format!(
        "{}.metallib",
        ir_path.as_ref().file_stem().unwrap().to_str().unwrap()
    );

    let output_file = Path::new(&out_dir).join(metallib_file_name);

    let mut command = std::process::Command::new("xcrun");
    command.arg("metallib");
    command.arg(format!("{}", ir_path.as_ref().display()));
    command.arg("-o");
    command.arg(format!("{}", output_file.display()));

    let status = command.status()?;

    if !status.success() {
        return Err(anyhow::anyhow!(
            "Failed to link kernel file: {:?}",
            ir_path.as_ref()
        ));
    }

    Ok(output_file)
}

fn gen_metallibs_rs(metallibs: Vec<PathBuf>) -> Result<()> {
    use convert_case::{Case, Casing};

    // generate a rust source file that contains an include_bytes constant
    // for every metallib file
    let out_dir = std::env::var("OUT_DIR")?;
    let out_file = Path::new(&out_dir).join("candle_metallibs.rs");

    let mut file = fs::File::create(&out_file)?;

    for metallib in metallibs {
        let name = metallib.file_stem().unwrap().to_str().unwrap();
        writeln!(
            file,
            "pub const {}: &'static [u8] = include_bytes!(\"{}\");",
            name.to_case(Case::ScreamingSnake),
            metallib.display()
        )?;
    }

    Ok(())
}
