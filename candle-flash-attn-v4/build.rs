use anyhow::anyhow;
use cudaforge::{KernelBuilder, Result};
use std::path::PathBuf;

const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

const KERNEL_FILES: &[&str] = &[
    // Main API and stub implementations.
    // Real FlashAttention-4 kernels are implemented in CuTe-DSL (Python)
    // and will be integrated when available as precompiled PTX/SASS or C++.
    "flash_api.cu",
];

const CUTLASS_COMMIT: &str = "4c42f73fdab5787e3bb57717f35a8cb1b3c0dc6d";

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    let is_target_msvc = target.contains("msvc");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");

    for file in KERNEL_FILES {
        println!("cargo:rerun-if-changed=bkernel/{file}");
    }
    println!("cargo:rerun-if-changed=bkernel/**.h");
    println!("cargo:rerun-if-changed=bkernel/**.hpp");
    println!("cargo:rerun-if-changed=bkernel/**.cpp");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize()
                .map_err(|_| {
                    anyhow!(
                        "Directory doesn't exist: {} (the current directory is {})",
                        path.display(),
                        std::env::current_dir().unwrap().display()
                    )
                })
                .expect("Unable to obtain build dir!")
        }
    };

    let kernels: Vec<PathBuf> = KERNEL_FILES
        .iter()
        .map(|f| PathBuf::from("bkernel").join(f))
        .collect();

    let mut builder = KernelBuilder::new()
        .source_files(kernels)
        .out_dir(&build_dir)
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_BFLOAT16_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("-U__CUDA_NO_BFLOAT162_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT162_CONVERSIONS__")
        .arg("-D_USE_MATH_DEFINES")
        .args(["--default-stream", "per-thread"])
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--ptxas-options=-v")
        .arg("--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")
        .arg("--verbose")
        .thread_percentage(0.5);

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    assert!(
        compute_cap >= 100,
        "FlashAttention-4 requires compute capability >=100 (Blackwell), got {}",
        compute_cap
    );

    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }

    let out_file = build_dir.join("libflashattentionv4.a");
    builder.build_lib(out_file)?;

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashattentionv4");

    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    Ok(())
}
