use cudarc::driver;
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Detect CUDA compute capability for FP8 support and flashinfer features
    let compute_cap = get_compute_cap();

    // WMMA (Tensor Cores) require SM 7.0+ (Volta)
    // Note: compute_cap is in format XY (e.g., 61 for SM 6.1, 70 for SM 7.0)
    let has_tensor_cores = compute_cap < 70;
    if !has_tensor_cores {
        println!("cargo::warning=SM {} < 70: Excluding WMMA kernels (Tensor Cores require Volta or newer)", compute_cap);
    }

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut builder = bindgen_cuda::Builder::default()
        .compute_cap(compute_cap)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if compute_cap >= 53 {
        if compute_cap < 70 {
            builder = builder.arg("-DALLOW_LEGACY_BF16");
            builder = builder.arg("-DALLOW_LEGACY_FP8");
        }
    }

    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs (these are built as static lib, not PTX)
    remove_lines(
        &ptx_path,
        &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF", "MOE_HFMA2"],
    );

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
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

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
        "src/moe/moe_hfma2.cu",
    ]);

    // Note: MoE kernels now use __CUDA_ARCH__ directly for WMMA fallback,

    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}

/// Get CUDA compute capability using cudarc driver detection.
/// Falls back to CUDA_COMPUTE_CAP env var if driver detection fails.
/// Returns the MINIMUM compute cap to ensure compatibility with all GPUs.
fn get_compute_cap() -> usize {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // First try to detect from actual GPU hardware via cudarc
    if let Ok(caps) = list_compute_caps() {
        if !caps.is_empty() {
            // Use minimum compute cap to ensure all kernels work on all GPUs
            // (for multi-GPU setups with different architectures)
            // TODO: support multiple compute caps (related to https://github.com/Narsil/bindgen_cuda/pull/16)
            let min_cap = *caps.iter().min().unwrap();
            println!(
                "cargo:warning=Using detected compute cap: {} (from {:?})",
                min_cap, caps
            );
            return min_cap as usize;
        }
    }

    // Fallback to environment variable
    if let Ok(compute_cap_str) = env::var("CUDA_COMPUTE_CAP") {
        if let Ok(cap) = compute_cap_str.parse::<usize>() {
            println!("cargo:warning=Using CUDA_COMPUTE_CAP from env: {}", cap);
            return cap;
        }
    }

    // Default to 0 if nothing worked - bindgen_cuda will try to detect it
    println!("cargo:warning=Could not detect compute cap, defaulting to 0");
    0
}

fn list_compute_caps() -> Result<Vec<usize>, ()> {
    // Try to initialize the CUDA driver and query devices
    if driver::result::init().is_err() {
        println!("cargo:warning=CUDA driver init failed; falling back to nvidia-smi or env var");
        return Err(());
    }

    let n = driver::result::device::get_count()
        .map(|x| x as usize)
        .unwrap_or(0);

    let mut seen = std::collections::HashSet::new();
    let mut devices_cc = Vec::with_capacity(n);

    for i in 0..n {
        let ctx = match driver::CudaContext::new(i) {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "cargo:warning=Failed to create CUDA context for device {}: {:?}",
                    i, e
                );
                continue;
            }
        };

        let cc = match ctx.compute_capability() {
            // Format: XY (e.g., 61 for SM 6.1, 70 for SM 7.0) - matches nvcc's sm_XY format
            Ok((major, minor)) => (major as usize) * 10 + (minor as usize),
            Err(e) => {
                println!(
                    "cargo:warning=Failed to get compute cap for device {}: {:?}",
                    i, e
                );
                continue;
            }
        };

        println!(
            "cargo:warning=CUDA device id {} has compute capability {}",
            i, cc
        );

        if seen.insert(cc) {
            devices_cc.push(cc);
        }
    }

    if devices_cc.len() > 1 {
        println!(
            "cargo:warning=Multiple compute capabilities detected: {:?}",
            devices_cc
        );
    }

    devices_cc.sort_unstable();
    Ok(devices_cc)
}