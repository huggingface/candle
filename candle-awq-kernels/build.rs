// Build script compiling the AWQ fused dequant+GEMM CUDA kernel into a static lib.
//
// The CUDA compile only runs when the `cuda` feature is enabled. With `--features metal` and no
// CUDA the crate has no native build step (the Metal kernel is compiled at runtime).
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return Ok(());
    }

    use cudaforge::KernelBuilder;

    println!("cargo::rerun-if-changed=kernels/awq_gemm.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    let builder = KernelBuilder::new()
        .source_files(vec!["kernels/awq_gemm.cu"])
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Xcompiler")
        .arg("-fPIC");

    let out_file = out_dir.join("libawqkernels.a");
    builder.build_lib(out_file)?;

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=awqkernels");
    println!("cargo::rustc-link-lib=dylib=cudart");
    Ok(())
}
