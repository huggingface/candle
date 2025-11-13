// TEAM-507: CUDA parity for ROCm build system
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    
    // Determine which backend to build
    let has_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let has_rocm = env::var("CARGO_FEATURE_ROCM").is_ok();
    
    #[cfg(feature = "cuda")]
    if has_cuda {
        build_cuda_kernels();
    }
    
    #[cfg(feature = "rocm")]
    if has_rocm {
        build_rocm_kernels();
    }
    
    if !has_cuda && !has_rocm {
        panic!("Either 'cuda' or 'rocm' feature must be enabled");
    }
}

#[cfg(feature = "cuda")]
fn build_cuda_kernels() {
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo::warning={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(ptx_path).unwrap();
    println!("cargo::warning=CUDA kernels built successfully");
}

// TEAM-507: ROCm build system using rocm_rs::bindgen_rocm
// NOTE: bindgen_rocm is now part of rocm-rs crate (proper location)
// This achieves parity with bindgen_cuda's API
#[cfg(feature = "rocm")]
fn build_rocm_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let hsaco_path = out_dir.join("hsaco.rs");
    
    // TEAM-507: Using rocm_rs::bindgen_rocm::Builder - CUDA parity achieved!
    let builder = rocm_rs::bindgen_rocm::Builder::default();
    let bindings = builder.build_hsaco().unwrap();
    bindings.write(hsaco_path).unwrap();
    
    println!("cargo::warning=ROCm kernels built successfully");
}
