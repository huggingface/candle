use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=cuda/kev.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kev.ptx");
    let status = Command::new(env::var_os("NVCC").unwrap_or_else(|| "nvcc".into()))
        .args([
            "-ptx",
            "-arch=compute_75",
            "--fmad=false",
            "-O3",
            "cuda/kev.cu",
            "-o",
        ])
        .arg(output)
        .status()
        .expect("CUDA build requires nvcc on PATH (or set NVCC)");
    assert!(status.success(), "Kev CUDA kernel compilation failed");
}
