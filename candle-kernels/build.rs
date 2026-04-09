use cudaforge::{detect_compute_cap, KernelBuilder, Result};
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

/// Parse the ALLOW_LEGACY env var into a set of permitted legacy features.
///
/// - `ALLOW_LEGACY=all`       → enables all legacy fallbacks
/// - `ALLOW_LEGACY=bf16,fp8`  → enables only bf16 and fp8
/// - unset / empty            → nothing enabled
fn parse_allow_legacy() -> HashSet<String> {
    let raw = env::var("ALLOW_LEGACY").unwrap_or_default();
    raw.to_lowercase()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Determine whether a legacy fallback feature should be enabled.
///
/// Each legacy feature has a CC range where it makes sense to offer
/// emulation (e.g. bf16 via fp32 on GPUs without native bf16).
/// The feature is enabled only when:
///   1. The GPU CC is within the valid legacy range AND
///   2. The user opted in via `ALLOW_LEGACY=all` or `ALLOW_LEGACY=<feature>`
struct LegacyCapability {
    name: &'static str,        // e.g. "allow_legacy_bf16"
    feature_key: &'static str, // e.g. "bf16"  (what appears in ALLOW_LEGACY)
    min_cc: usize,             // inclusive lower bound for CC
    max_cc: usize,             // exclusive upper bound for CC
}

const LEGACY_CAPABILITIES: &[LegacyCapability] = &[
    LegacyCapability {
        name: "allow_legacy_bf16",
        feature_key: "bf16",
        min_cc: 53,
        max_cc: 80,
    },
    LegacyCapability {
        name: "allow_legacy_fp8",
        feature_key: "fp8",
        min_cc: 53,
        max_cc: 89,
    },
];

/// A hardware capability determined purely from compute_cap.
struct HwCapability {
    name: &'static str,
    min_cc: usize,
}

/// The set of hardware capabilities that candle-kernels actually uses.
/// Each one is enabled when `compute_cap >= min_cc`.
const HW_CAPABILITIES: &[HwCapability] = &[
    HwCapability {
        name: "has_half2_native",
        min_cc: 60,
    },
    HwCapability {
        name: "has_f16_arithmetic",
        min_cc: 53,
    },
    HwCapability {
        name: "has_bf16",
        min_cc: 80,
    },
    HwCapability {
        name: "has_fp8",
        min_cc: 89,
    },
    HwCapability {
        name: "has_wmma",
        min_cc: 70,
    },
    HwCapability {
        name: "has_wmma_f16",
        min_cc: 70,
    },
    HwCapability {
        name: "has_wmma_bf16",
        min_cc: 80,
    },
    HwCapability {
        name: "has_tensor_cores",
        min_cc: 70,
    },
];

/// Emit `cargo::rustc-check-cfg` declarations so that Cargo doesn't warn
/// about unknown cfgs.
fn emit_check_cfgs() {
    for cap in HW_CAPABILITIES {
        println!("cargo::rustc-check-cfg=cfg({})", cap.name);
    }
    for cap in LEGACY_CAPABILITIES {
        println!("cargo::rustc-check-cfg=cfg({})", cap.name);
    }
}

/// Apply a capability flag to both NVCC (-DNAME=1/0) and Rust (cargo:rustc-cfg).
fn apply_flag(builder: &mut KernelBuilder, name: &str, enabled: bool) -> KernelBuilder {
    let define = format!("-D{}={}", name.to_uppercase(), if enabled { 1 } else { 0 });
    let b = std::mem::replace(builder, KernelBuilder::new());
    let b = b.arg(&define);
    if enabled {
        println!("cargo:rustc-cfg={}", name);
    }
    b
}

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=ALLOW_LEGACY");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = detect_compute_cap().unwrap_or_else(|_| cudaforge::GpuArch::new(0));
    let compute_cap = env::var("CUDA_COMPUTE_CAP")
        .map(|s| s.parse::<usize>().unwrap_or(80))
        .unwrap_or_else(|_| arch.base());

    let permitted = parse_allow_legacy();

    // ── Declare check-cfgs ──────────────────────────────────────────────────
    emit_check_cfgs();

    // ── Main PTX builder ────────────────────────────────────────────────────
    let mut builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .watch([
            "src/compatibility.cuh",
            "src/cuda_utils.cuh",
            "src/binary_op_macros.cuh",
        ])
        .exclude(&["moe_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Apply hardware capabilities
    for cap in HW_CAPABILITIES {
        let enabled = compute_cap >= cap.min_cc;
        builder = apply_flag(&mut builder, cap.name, enabled);
    }

    // Apply legacy fallback capabilities
    for cap in LEGACY_CAPABILITIES {
        let enabled = compute_cap >= cap.min_cc
            && compute_cap < cap.max_cc
            && (permitted.contains("all") || permitted.contains(cap.feature_key));
        builder = apply_flag(&mut builder, cap.name, enabled);
    }

    // ── MOE static library builder ──────────────────────────────────────────
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

    // Apply the same capability flags to the MOE builder
    for cap in HW_CAPABILITIES {
        let enabled = compute_cap >= cap.min_cc;
        let define = format!(
            "-D{}={}",
            cap.name.to_uppercase(),
            if enabled { 1 } else { 0 }
        );
        moe_builder = moe_builder.arg(&define);
    }
    for cap in LEGACY_CAPABILITIES {
        let enabled = compute_cap >= cap.min_cc
            && compute_cap < cap.max_cc
            && (permitted.contains("all") || permitted.contains(cap.feature_key));
        let define = format!(
            "-D{}={}",
            cap.name.to_uppercase(),
            if enabled { 1 } else { 0 }
        );
        moe_builder = moe_builder.arg(&define);
    }

    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    // ── Build PTX ───────────────────────────────────────────────────────────
    let ptx_output = builder.build_ptx()?;
    ptx_output.write(out_dir.join("ptx.rs"))?;

    // ── Build MOE Static Library ────────────────────────────────────────────
    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
