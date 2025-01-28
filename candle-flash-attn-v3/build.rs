// build.rs
use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;

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

fn main() -> Result<()> {
    // Use RAYON_NUM_THREADS or else default to the number of physical CPUs
    let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
        |_| num_cpus::get_physical(),
        |s| usize::from_str(&s).unwrap_or_else(|_| num_cpus::get_physical()),
    );
    // limit to 16 cpus to not use to much ram on large servers
    let num_cpus = num_cpus.min(16);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .unwrap();

    // Telling Cargo that if any of these files changes, rebuild.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");

    for file in KERNEL_FILES {
        println!("cargo:rerun-if-changed=hkernel/{file}");
    }
    println!("cargo:rerun-if-changed=kernels/**.h");
    println!("cargo:rerun-if-changed=kernels/**.hpp");
    println!("cargo:rerun-if-changed=kernels/**.cpp");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    // You can optionally allow an environment variable to cache the compiled artifacts.
    // If not found, we compile into the standard OUT_DIR.
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().map_err(|_| {
                anyhow!(
                    "Directory doesn't exist: {} (the current directory is {})",
                    path.display(),
                    std::env::current_dir().unwrap().display()
                )
            })?
        }
    };

    // Ensure we set CUDA_INCLUDE_DIR for our crates that might rely on it.
    set_cuda_include_dir()?;

    // If set, pass along the custom compiler for NVCC
    let ccbin_env = std::env::var("CANDLE_NVCC_CCBIN").ok();

    // Determine the GPU architecture we’re targeting, e.g. 90 for `sm_90`.
    let compute_cap = compute_cap()?;
    // assert compute cap is sm90
    assert!(compute_cap == 90, "Compute capability must be 90 (90a)");

    // Our final library name
    let out_file = build_dir.join("libflashattentionv3.a");

    // Construct the list of (input_file -> output_object_file)
    let kernel_dir = PathBuf::from("hkernel");
    let cu_files: Vec<(PathBuf, PathBuf)> = KERNEL_FILES
        .iter()
        .map(|f| {
            let mut obj_file = out_dir.join(f);
            obj_file.set_extension("o");
            (kernel_dir.join(f), obj_file)
        })
        .collect();

    // Decide whether to skip recompile if outputs are up to date.
    // This is a simplistic approach,
    // so feel free to refine if you need more robust up-to-date checks.
    let out_modified = out_file
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .unwrap_or_else(|| std::time::SystemTime::UNIX_EPOCH);
    let should_compile = !out_file.exists()
        || cu_files.iter().any(|(input, _)| {
            let input_modified = input
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            input_modified.duration_since(out_modified).is_ok() // True if input_modified >= out_modified
        });

    if should_compile {
        // 1) Compile each .cu/.cpp -> .o
        cu_files
            .par_iter()
            .try_for_each(|(input, obj)| -> Result<()> {
                let mut command = std::process::Command::new("nvcc");

                // Optimization and standard
                command.arg("-O3");
                command.arg("-std=c++17");

                // GPU architecture, hard code sm_90a instead of sm90
                command.arg(format!("--gpu-architecture={}", "sm_90a"));

                // Compile to object file
                command.arg("-c");
                command.args(["-o", obj.to_str().unwrap()]);

                // Default stream per-thread
                command.args(["--default-stream", "per-thread"]);

                // Include path
                command.arg("-Icutlass/include");

                // Undefine CUDA “no half/bfloat” macros
                command.arg("-U__CUDA_NO_HALF_OPERATORS__");
                command.arg("-U__CUDA_NO_HALF_CONVERSIONS__");
                command.arg("-U__CUDA_NO_BFLOAT16_OPERATORS__");
                command.arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__");
                command.arg("-U__CUDA_NO_BFLOAT162_OPERATORS__");
                command.arg("-U__CUDA_NO_BFLOAT162_CONVERSIONS__");

                // Enable relaxed/extended lambda and fast math
                command.arg("--expt-relaxed-constexpr");
                command.arg("--expt-extended-lambda");
                command.arg("--use_fast_math");

                // PTXAS options: verbose output, register usage info, etc.
                command.arg("--ptxas-options=-v");
                command.arg("--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage");

                // Additional debug/performance flags
                command.arg("-lineinfo");
                command.arg("-DCUTLASS_DEBUG_TRACE_LEVEL=0");
                command.arg("-DNDEBUG");

                if let Some(ccbin_path) = &ccbin_env {
                    command.arg("-allow-unsupported-compiler");
                    command.args(["-ccbin", ccbin_path]);
                }

                // Add the source file
                command.arg(input);

                let output = command
                    .spawn()
                    .with_context(|| format!("Failed to spawn nvcc for {input:?}"))?
                    .wait_with_output()
                    .with_context(|| format!("Failed during nvcc invocation for {input:?}"))?;

                if !output.status.success() {
                    return Err(anyhow!(
                        "nvcc error:\nCommand: {:?}\nstdout:\n{}\nstderr:\n{}",
                        command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }

                Ok(())
            })?;

        // 2) Create static library from the .o files
        let obj_files = cu_files
            .iter()
            .map(|(_, obj)| obj.clone())
            .collect::<Vec<_>>();

        let mut command = std::process::Command::new("nvcc");
        command.arg("--lib");
        command.args(["-o", out_file.to_str().unwrap()]);
        command.args(obj_files);

        let output = command
            .spawn()
            .context("Failed spawning nvcc to archive .o files")?
            .wait_with_output()
            .context("Failed during nvcc archive step")?;

        if !output.status.success() {
            return Err(anyhow!(
                "nvcc error (archiving):\nCommand: {:?}\nstdout:\n{}\nstderr:\n{}",
                command,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }

    // Finally, instruct cargo to link your library
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashattentionv3");

    // Link required system libs
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

/// This function attempts to find a CUDA toolkit root that contains `include/cuda.h`,
/// and prints that path as `CUDA_INCLUDE_DIR`.
fn set_cuda_include_dir() -> Result<()> {
    // Adapted from cudarc build.rs
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .filter_map(|v| std::env::var(v).ok())
        .map(Into::<PathBuf>::into);

    let common_roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let candidates = env_vars.chain(common_roots.into_iter().map(Into::into));

    let root = candidates
        .filter(|path| path.join("include").join("cuda.h").is_file())
        .next()
        .ok_or_else(|| anyhow!("Cannot find a valid CUDA root with include/cuda.h"))?;

    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        root.join("include").display()
    );
    Ok(())
}

/// Determine the compute capability we should target.
/// If the user sets `CUDA_COMPUTE_CAP` we trust that.
/// Otherwise, we attempt to parse it from `nvidia-smi`.
fn compute_cap() -> Result<usize> {
    if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        let cc = compute_cap_str
            .parse::<usize>()
            .context("Failed to parse CUDA_COMPUTE_CAP")?;
        Ok(cc)
    } else {
        // parse from nvidia-smi
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv"])
            .output()
            .context("Failed to run nvidia-smi. Make sure it's in PATH.")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        if lines.next().unwrap_or("") != "compute_cap" {
            return Err(anyhow!("Unexpected output from nvidia-smi: {stdout}"));
        }
        if let Some(cap_line) = lines.next() {
            // e.g. "9.0" -> "90"
            let cc_str = cap_line.trim().replace('.', "");
            let cc = cc_str.parse::<usize>()?;
            Ok(cc)
        } else {
            Err(anyhow!("nvidia-smi did not return a compute_cap line"))
        }
    }
}
