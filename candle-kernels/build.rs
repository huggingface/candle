use cudaforge::{detect_compute_cap, get_gpu_arch_string, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

```rust
// Macro to register capabilities in Rust and in the CUDA builder
```
macro_rules! set_cfg {
    ($builder:expr, $name:expr, $value:expr) => {
```rust
        // Register for Rust (check-cfg is required for recent Rust versions)
```
        println!("cargo::rustc-check-cfg=cfg({}, values(\"true\", \"false\"))", $name);
        
```rust
        // Apply the configuration if the value is true
```
        if $value {
            println!("cargo:rustc-cfg={}=\"true\"", $name);
            $builder = $builder.arg(format!("-D{}=1", $name.to_uppercase()).as_str());
        } else {
            println!("cargo:rustc-cfg={}=\"false\"", $name);
            // Optionnel : $builder = $builder.arg(format!("-D{}=0", $name.to_uppercase()).as_str());
        }
    };
}

// Macro helper to apply to both builders
macro_rules! dual_set {
    ($b1:expr, $b2:expr, $name:expr, $value:expr) => {
        set_cfg!($b1, $name, $value);
        set_cfg!($b2, $name, $value);
    };
}

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = detect_compute_cap().unwrap_or_else(|_| cudaforge::GpuArch::new(0));
    let compute_cap = arch.base();

    // Initialize builders early
    let mut builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .exclude(&["moe_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

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

    // Register check-cfg for non-macro flags
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp8)");
    println!("cargo::rustc-check-cfg=cfg(cuda_arch, values(\"53\", \"61\", \"70\", \"75\", \"80\", \"86\", \"89\", \"90\", \"100\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_700, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_61, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_53, values(\"true\", \"false\"))");

    // Apply hardware capabilities using the macro (affects Rust and both builders)
    println!("cargo:rustc-cfg=cuda_arch=\"{}\"", compute_cap);
    println!("cargo:rustc-cfg=inf_cc_700=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=inf_cc_61=\"{}\"", compute_cap >= 61);
    println!("cargo:rustc-cfg=inf_cc_53=\"{}\"", compute_cap >= 53);



    dual_set!(builder, moe_builder, "has_f16", compute_cap >= 53);
    dual_set!(builder, moe_builder, "has_half2_native", compute_cap == 60 || compute_cap == 62 || compute_cap >= 70);
    dual_set!(builder, moe_builder, "has_dp4a", compute_cap >= 61);
    dual_set!(builder, moe_builder, "has_pageable_memory_access", compute_cap >= 60);
    dual_set!(builder, moe_builder, "has_memory_pools", compute_cap >= 60);
    dual_set!(builder, moe_builder, "has_async_copy", compute_cap >= 80);
    dual_set!(builder, moe_builder, "has_l2_cache_persistence", compute_cap >= 80);
    dual_set!(builder, moe_builder, "has_cuda_graphs", compute_cap >= 30);
    dual_set!(builder, moe_builder, "has_wmma", compute_cap >= 70);
    dual_set!(builder, moe_builder, "has_wmma_f16", compute_cap >= 70);
    dual_set!(builder, moe_builder, "has_independent_thread_scheduling", compute_cap >= 70);
    dual_set!(builder, moe_builder, "has_bf16_conversions", compute_cap >= 70);
    dual_set!(builder, moe_builder, "has_sparse_tensor_cores", compute_cap >= 80);
    dual_set!(builder, moe_builder, "has_wmma_bf16", compute_cap >= 80);
    dual_set!(builder, moe_builder, "has_bf16", compute_cap >= 80);
    dual_set!(builder, moe_builder, "has_tf32", compute_cap >= 80);
    dual_set!(builder, moe_builder, "has_fp8", compute_cap >= 89);
    dual_set!(builder, moe_builder, "has_transformer_engine", compute_cap >= 89);
    dual_set!(builder, moe_builder, "has_fp4", compute_cap >= 100);
    dual_set!(builder, moe_builder, "has_tma", compute_cap >= 90);
    dual_set!(builder, moe_builder, "has_clusters", compute_cap >= 90);
    dual_set!(builder, moe_builder, "has_distributed_shared_memory", compute_cap >= 90);

    // WMMA (Tensor Cores) require SM 7.0+ (Volta)
    if compute_cap < 70 {
        println!(
            "cargo::warning={} < 70: Excluding WMMA kernels (Tensor Cores require Volta or newer)",
            get_gpu_arch_string(compute_cap)
        );
    }

    if compute_cap >= 53 && compute_cap < 70 {
        builder = builder.arg("-DALLOW_LEGACY_BF16");
        builder = builder.arg("-DALLOW_LEGACY_FP8");
    }

    // Build for PTX
    let ptx_output = builder.build_ptx()?;
    ptx_output.write(out_dir.join("ptx.rs"))?;

    // Build for MOE Static Library
    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}