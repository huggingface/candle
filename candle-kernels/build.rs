use cudaforge::{detect_compute_cap, get_gpu_arch_string, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Detect CUDA compute capability using cudaforge's nvidia-smi detection
    let arch = detect_compute_cap().unwrap_or_else(|_| cudaforge::GpuArch::new(0));
    let compute_cap = arch.base();

    // WMMA (Tensor Cores) require SM 7.0+ (Volta)
    if compute_cap < 70 {
        println!(
            "cargo::warning={} < 70: Excluding WMMA kernels (Tensor Cores require Volta or newer)",
            get_gpu_arch_string(compute_cap)
        );
    }

    // Build for PTX
    let mut builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .exclude(&["moe_*.cu"]) // Exclude MOE kernels from PTX build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if compute_cap >= 53 && compute_cap < 70 {
        builder = builder.arg("-DALLOW_LEGACY_BF16");
        builder = builder.arg("-DALLOW_LEGACY_FP8");
    }

    let ptx_output = builder.build_ptx()?;
    ptx_output.write(out_dir.join("ptx.rs"))?;

    // Build for MOE Static Library
    let mut moe_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/moe/moe_hfma2.cu",
        ]);

    let target = env::var("TARGET").unwrap_or_default();
    let is_target_msvc = target.contains("msvc");
    if is_target_msvc {
        moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
    } else {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}