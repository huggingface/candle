use cudaforge::{CudaToolkit, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

const CUTILE_FEATURE: &str = "CARGO_FEATURE_CUTILE";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu", "sort_cub.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    let mut moe_sources = vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
        "src/mmvq_gguf.cu",
        "src/mmq_gguf/mmq_quantize.cu",
        "src/mmq_gguf/mmq_instance_q4_0.cu",
        "src/mmq_gguf/mmq_instance_q4_1.cu",
        "src/mmq_gguf/mmq_instance_q5_0.cu",
        "src/mmq_gguf/mmq_instance_q5_1.cu",
        "src/mmq_gguf/mmq_instance_q8_0.cu",
        "src/mmq_gguf/mmq_instance_q2_k.cu",
        "src/mmq_gguf/mmq_instance_q3_k.cu",
        "src/mmq_gguf/mmq_instance_q4_k.cu",
        "src/mmq_gguf/mmq_instance_q5_k.cu",
        "src/mmq_gguf/mmq_instance_q6_k.cu",
    ];
    if env::var_os(CUTILE_FEATURE).is_some() {
        moe_sources.push("src/moe/moe_align.cu");
    }

    let mut moe_builder = KernelBuilder::default()
        .source_files(moe_sources)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    let ptx_cap = CudaToolkit::detect()?
        .supported_architectures()
        .into_iter()
        .min()
        .unwrap_or(75);
    let ptx_gencode = format!("-gencode=arch=compute_{ptx_cap},code=compute_{ptx_cap}");
    let mut sort_builder = KernelBuilder::default()
        .source_files(["src/sort_cub.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg(&ptx_gencode);
    if !is_target_msvc {
        sort_builder = sort_builder.arg("-Xcompiler").arg("-fPIC");
    }
    sort_builder.build_lib(out_dir.join("libsort_cub.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=sort_cub");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
