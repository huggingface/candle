use cudaforge::{detect_compute_cap, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo::rerun-if-env-changed=CARGO_FEATURE_CUDA_LEGACY_BF16");

    let compute_cap = detect_compute_cap().map(|arch| arch.base()).unwrap_or(80);
    let legacy_bf16 = compute_cap < 80 && env::var_os("CARGO_FEATURE_CUDA_LEGACY_BF16").is_some();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut ptx_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if legacy_bf16 {
        ptx_builder = ptx_builder.arg("-DCANDLE_CUDA_BF16_FALLBACK=1");
    }

    let bindings = ptx_builder.build_ptx()?;
    bindings.write(&ptx_path)?;

    let mut moe_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_files(vec![
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
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // WMMA is available starting with Volta (sm_70).
    //
    // Keep the requested compute capability for the general CUDA/MoE
    // kernels, but compile the WMMA-only translation units at their
    // actual architectural floor when targeting pre-Volta devices.
    if legacy_bf16 {
        moe_builder = moe_builder.arg("-DCANDLE_CUDA_BF16_FALLBACK=1");
    }

    if compute_cap < 70 {
        moe_builder = moe_builder
            .with_compute_override("moe_wmma.cu", 70)
            .with_compute_override("moe_wmma_gguf.cu", 70);
    }

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
