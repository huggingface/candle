use cudaforge::{detect_compute_cap, get_gpu_arch_string, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp8)");
    println!("cargo::rustc-check-cfg=cfg(cuda_arch, values(\"53\", \"61\", \"70\", \"75\", \"80\", \"86\", \"89\", \"90\", \"100\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_700, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_61, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_53, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_f16, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_half2_native, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_dp4a, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_pageable_memory_access, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_memory_pools, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_async_copy, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_l2_cache_persistence, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_cuda_graphs, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_wmma, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_wmma_f16, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_independent_thread_scheduling, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_bf16_conversions, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_sparse_tensor_cores, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_wmma_bf16, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_bf16, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_tf32, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_fp8, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_transformer_engine, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_fp4, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_tma, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_clusters, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(has_distributed_shared_memory, values(\"true\", \"false\"))");
    

    // Detect CUDA compute capability using cudaforge's nvidia-smi detection
    let arch = detect_compute_cap().unwrap_or_else(|_| cudaforge::GpuArch::new(0));
    let compute_cap = arch.base();
    println!("cargo:rustc-cfg=cuda_arch=\"{}\"", compute_cap);
    println!("cargo:rustc-cfg=inf_cc_700=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=inf_cc_61=\"{}\"", compute_cap >= 61);
    println!("cargo:rustc-cfg=inf_cc_53=\"{}\"", compute_cap >= 53);
    println!("cargo:rustc-cfg=has_f16=\"{}\"", compute_cap >= 53);
    println!("cargo:rustc-cfg=has_half2_native=\"{}\"", compute_cap == 60 || compute_cap == 62 || compute_cap >= 70);
    println!("cargo:rustc-cfg=has_dp4a=\"{}\"", compute_cap >= 61);
    println!("cargo:rustc-cfg=has_pageable_memory_access=\"{}\"", compute_cap >= 60);
    println!("cargo:rustc-cfg=has_memory_pools=\"{}\"", compute_cap >= 60);
    println!("cargo:rustc-cfg=has_async_copy=\"{}\"", compute_cap >= 80);
    println!("cargo:rustc-cfg=has_l2_cache_persistence=\"{}\"", compute_cap >= 80);
    println!("cargo:rustc-cfg=has_cuda_graphs=\"{}\"", compute_cap >= 30);
    println!("cargo:rustc-cfg=has_wmma=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=has_wmma_f16=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=has_independent_thread_scheduling=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=has_bf16_conversions=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=has_sparse_tensor_cores=\"{}\"", compute_cap >= 80);
    println!("cargo:rustc-cfg=has_wmma_bf16=\"{}\"", compute_cap >= 80);
    println!("cargo:rustc-cfg=has_bf16=\"{}\"", compute_cap >= 80);
    println!("cargo:rustc-cfg=has_tf32=\"{}\"", compute_cap >= 80);
    println!("cargo:rustc-cfg=has_fp8=\"{}\"", compute_cap >= 89);
    println!("cargo:rustc-cfg=has_transformer_engine=\"{}\"", compute_cap >= 89);
    println!("cargo:rustc-cfg=has_fp4=\"{}\"", compute_cap >= 100);
    println!("cargo:rustc-cfg=has_tma=\"{}\"", compute_cap >= 90);
    println!("cargo:rustc-cfg=has_clusters=\"{}\"", compute_cap >= 90);
    println!("cargo:rustc-cfg=has_distributed_shared_memory=\"{}\"", compute_cap >= 90);

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