// build.rs

// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright (c) 2024 Michael Feil
// adapted from https://github.com/huggingface/candle-flash-attn-v1 , Oliver Dehaene
// adapted further in 2025 by Eric Buehler for candle repo.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use anyhow::anyhow;
use cudaforge::{KernelBuilder, Result};
use std::path::PathBuf;

const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

const KERNEL_FILES: &[&str] = &[
    "flash_api.cu",
    "flash_fwd_hdim64_fp16_sm90.cu",
    "flash_fwd_hdim64_bf16_sm90.cu",
    "flash_fwd_hdim128_fp16_sm90.cu",
    "flash_fwd_hdim128_bf16_sm90.cu",
    "flash_fwd_hdim256_fp16_sm90.cu",
    "flash_fwd_hdim256_bf16_sm90.cu",
    // "flash_bwd_hdim64_fp16_sm90.cu",
    // "flash_bwd_hdim96_fp16_sm90.cu",
    // "flash_bwd_hdim128_fp16_sm90.cu",
    // commented out in main repo: // "flash_bwd_hdim256_fp16_sm90.cu",
    // "flash_bwd_hdim64_bf16_sm90.cu",
    // "flash_bwd_hdim96_bf16_sm90.cu",
    // "flash_bwd_hdim128_bf16_sm90.cu",
    // "flash_fwd_hdim64_e4m3_sm90.cu",
    // "flash_fwd_hdim128_e4m3_sm90.cu",
    // "flash_fwd_hdim256_e4m3_sm90.cu",
    "flash_fwd_hdim64_fp16_gqa2_sm90.cu",
    "flash_fwd_hdim64_fp16_gqa4_sm90.cu",
    "flash_fwd_hdim64_fp16_gqa8_sm90.cu",
    "flash_fwd_hdim64_fp16_gqa16_sm90.cu",
    "flash_fwd_hdim64_fp16_gqa32_sm90.cu",
    "flash_fwd_hdim128_fp16_gqa2_sm90.cu",
    "flash_fwd_hdim128_fp16_gqa4_sm90.cu",
    "flash_fwd_hdim128_fp16_gqa8_sm90.cu",
    "flash_fwd_hdim128_fp16_gqa16_sm90.cu",
    "flash_fwd_hdim128_fp16_gqa32_sm90.cu",
    "flash_fwd_hdim256_fp16_gqa2_sm90.cu",
    "flash_fwd_hdim256_fp16_gqa4_sm90.cu",
    "flash_fwd_hdim256_fp16_gqa8_sm90.cu",
    "flash_fwd_hdim256_fp16_gqa16_sm90.cu",
    "flash_fwd_hdim256_fp16_gqa32_sm90.cu",
    "flash_fwd_hdim64_bf16_gqa2_sm90.cu",
    "flash_fwd_hdim64_bf16_gqa4_sm90.cu",
    "flash_fwd_hdim64_bf16_gqa8_sm90.cu",
    "flash_fwd_hdim64_bf16_gqa16_sm90.cu",
    "flash_fwd_hdim64_bf16_gqa32_sm90.cu",
    "flash_fwd_hdim128_bf16_gqa2_sm90.cu",
    "flash_fwd_hdim128_bf16_gqa4_sm90.cu",
    "flash_fwd_hdim128_bf16_gqa8_sm90.cu",
    "flash_fwd_hdim128_bf16_gqa16_sm90.cu",
    "flash_fwd_hdim128_bf16_gqa32_sm90.cu",
    "flash_fwd_hdim256_bf16_gqa2_sm90.cu",
    "flash_fwd_hdim256_bf16_gqa4_sm90.cu",
    "flash_fwd_hdim256_bf16_gqa8_sm90.cu",
    "flash_fwd_hdim256_bf16_gqa16_sm90.cu",
    "flash_fwd_hdim256_bf16_gqa32_sm90.cu",
    // "flash_fwd_hdim64_e4m3_gqa2_sm90.cu",
    // "flash_fwd_hdim64_e4m3_gqa4_sm90.cu",
    // "flash_fwd_hdim64_e4m3_gqa8_sm90.cu",
    // "flash_fwd_hdim64_e4m3_gqa16_sm90.cu",
    // "flash_fwd_hdim64_e4m3_gqa32_sm90.cu",
    // "flash_fwd_hdim128_e4m3_gqa2_sm90.cu",
    // "flash_fwd_hdim128_e4m3_gqa4_sm90.cu",
    // "flash_fwd_hdim128_e4m3_gqa8_sm90.cu",
    // "flash_fwd_hdim128_e4m3_gqa16_sm90.cu",
    // "flash_fwd_hdim128_e4m3_gqa32_sm90.cu",
    // "flash_fwd_hdim256_e4m3_gqa2_sm90.cu",
    // "flash_fwd_hdim256_e4m3_gqa4_sm90.cu",
    // "flash_fwd_hdim256_e4m3_gqa8_sm90.cu",
    // "flash_fwd_hdim256_e4m3_gqa16_sm90.cu",
    // "flash_fwd_hdim256_e4m3_gqa32_sm90.cu",
];

const CUTLASS_COMMIT: &str = "4c42f73fdab5787e3bb57717f35a8cb1b3c0dc6d";

fn main() -> Result<()> {
    // Telling Cargo that if any of these files changes, rebuild.
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    let is_target_msvc = target.contains("msvc");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");

    for file in KERNEL_FILES {
        println!("cargo:rerun-if-changed=hkernel/{file}");
    }
    println!("cargo:rerun-if-changed=kernels/**.h");
    println!("cargo:rerun-if-changed=kernels/**.hpp");
    println!("cargo:rerun-if-changed=kernels/**.cpp");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    // You can optionally allow an environment variable to cache the compiled artifacts.
    // If not found, we compile into the standard OUT_DIR.
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
        .map(|f| PathBuf::from("hkernel").join(f))
        .collect();

    let mut builder = KernelBuilder::new()
        .source_files(kernels)
        .out_dir(&build_dir)
        .with_cutlass(Some(CUTLASS_COMMIT)) // ✅ Auto-fetch and include CUTLASS from GitHub
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
        .thread_percentage(0.5); // Use up to 50% of available threads

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    assert!(compute_cap >= 90, "Compute capability must be >=90 (90a)");

    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }
    // Our final library name
    let out_file = build_dir.join("libflashattentionv3.a");
    builder.build_lib(out_file)?;

    // Finally, instruct cargo to link your library
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashattentionv3");

    // Link required system libs
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    Ok(())
}
