use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

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
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    // Detect compute capability up front so we can conditionally select
    // kernel sources. The pre-Volta swap below depends on it; the bf16
    // gate further down also reuses this value.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);

    // moe_wmma.cu and moe_wmma_gguf.cu both rely on `nvcuda::wmma`, which
    // requires Volta+ tensor cores (sm >= 70). On Pascal (sm 60) and older
    // they fail to compile with "namespace wmma not found". Substitute an
    // aborting stub so the static link still resolves; dense models don't
    // exercise these paths and MoE models can't run on Pascal regardless.
    let wmma_sources: &[&str] = if compute_cap < 70 {
        &["src/moe/moe_wmma_stub.cu"]
    } else {
        &["src/moe/moe_wmma.cu", "src/moe/moe_wmma_gguf.cu"]
    };

    let mut moe_builder = KernelBuilder::default()
        .source_files({
            let mut files = vec!["src/moe/moe_gguf.cu"];
            files.extend_from_slice(wmma_sources);
            files.extend_from_slice(&[
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
            ]);
            files
        })
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
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
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
