// Build script compiling the single-token decode attention CUDA kernel into a static lib.
use cudaforge::KernelBuilder;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels/decode_attention.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    let mut builder = KernelBuilder::new()
        .source_files(vec!["kernels/decode_attention.cu"])
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr");

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

    let out_file = out_dir.join("libflashinferkernels.a");
    builder.build_lib(out_file)?;

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=flashinferkernels");
    println!("cargo::rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}
