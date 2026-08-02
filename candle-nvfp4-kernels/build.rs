use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return Ok(());
    }

    println!("cargo::rerun-if-changed=kernels/nvfp4_gemm.cu");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let mut builder = cudaforge::KernelBuilder::new()
        .source_files(vec!["kernels/nvfp4_gemm.cu"])
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("-O3");
    if !std::env::var("TARGET").is_ok_and(|target| target.contains("msvc")) {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }
    builder.build_lib(out_dir.join("libnvfp4kernels.a"))?;
    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=nvfp4kernels");
    println!("cargo::rustc-link-lib=dylib=cudart");
    if !std::env::var("TARGET").is_ok_and(|target| target.contains("msvc")) {
        println!("cargo::rustc-link-lib=stdc++");
    }
    Ok(())
}
