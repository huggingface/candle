use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Detect compute capability
    let compute_cap = compute_cap();

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable BF16 WMMA for pre-Ampere GPUs (sm < 80)
    if compute_cap < 80 {
        builder = builder.arg("-DNO_BF16_WMMA");
        println!(
            "cargo:warning=Disabling BF16 WMMA kernels (compute cap {} < 80)",
            compute_cap
        );
    }

    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable BF16 WMMA for pre-Ampere GPUs (sm < 80)
    // BF16 Tensor Core support requires Ampere or newer
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_WMMA");
        println!(
            "cargo:warning=Disabling BF16 WMMA kernels (compute cap {} < 80)",
            compute_cap
        );
    }

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
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn compute_cap() -> usize {
    // Check for override via environment variable
    if let Ok(val) = env::var("CUDA_COMPUTE_CAP") {
        if let Ok(cap) = val.parse::<usize>() {
            println!(
                "cargo:warning=Using CUDA_COMPUTE_CAP from environment: {}",
                cap
            );
            return cap;
        }
    }

    // Try to detect via nvidia-smi
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse "7.5" -> 75, "8.6" -> 86, etc.
            if let Some(cap) = stdout.lines().next() {
                let cap = cap.trim().replace('.', "");
                if let Ok(cap) = cap.parse::<usize>() {
                    println!("cargo:warning=Detected compute cap: {}", cap);
                    return cap;
                }
            }
        }
    }

    // Default to 80 if detection fails (safe: won't disable BF16 unnecessarily)
    println!("cargo:warning=Could not detect compute cap, defaulting to 80");
    80
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
