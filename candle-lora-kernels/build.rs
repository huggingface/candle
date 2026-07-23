// Build script that compiles the LoRA BGMV CUDA kernels into a static library
// and links it. Requires the CUDA toolchain (nvcc); this crate is CUDA-only and
// is pulled in by candle-nn's `lora-cuda` feature.
use cudaforge::{KernelBuilder, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/lora_sgmv.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    let mut builder = KernelBuilder::default()
        .source_files(vec!["src/lora_sgmv.cu"])
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--expt-relaxed-constexpr")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__");

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }
    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    builder.build_lib(out_dir.join("liblorakernels.a"))?;

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=lorakernels");
    println!("cargo::rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}
