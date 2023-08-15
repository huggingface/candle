// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;

const KERNEL_FILES: [&str; 9] = [
    "flash_api.cu",
    "flash_fwd_hdim128_fp16_sm80.cu",
    "flash_fwd_hdim160_fp16_sm80.cu",
    "flash_fwd_hdim192_fp16_sm80.cu",
    "flash_fwd_hdim224_fp16_sm80.cu",
    "flash_fwd_hdim256_fp16_sm80.cu",
    "flash_fwd_hdim32_fp16_sm80.cu",
    "flash_fwd_hdim64_fp16_sm80.cu",
    "flash_fwd_hdim96_fp16_sm80.cu",
    // "flash_fwd_hdim128_bf16_sm80.cu",
    // "flash_fwd_hdim160_bf16_sm80.cu",
    // "flash_fwd_hdim192_bf16_sm80.cu",
    // "flash_fwd_hdim224_bf16_sm80.cu",
    // "flash_fwd_hdim256_bf16_sm80.cu",
    // "flash_fwd_hdim32_bf16_sm80.cu",
    // "flash_fwd_hdim64_bf16_sm80.cu",
    // "flash_fwd_hdim96_bf16_sm80.cu",
];

fn main() -> Result<()> {
    let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
        |_| num_cpus::get_physical(),
        |s| usize::from_str(&s).unwrap(),
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .unwrap();

    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed=kernels/{kernel_file}");
    }
    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash.h");
    println!("cargo:rerun-if-changed=kernels/philox.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo:rerun-if-changed=kernels/block_info.h");
    println!("cargo:rerun-if-changed=kernels/static_switch.h");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => PathBuf::from(build_dir),
    };
    set_cuda_include_dir()?;
    let compute_cap = compute_cap()?;

    let out_file = build_dir.join("libflashattention.a");

    let kernel_dir = PathBuf::from("kernels");
    let cu_files: Vec<_> = KERNEL_FILES
        .iter()
        .map(|f| {
            let mut obj_file = out_dir.join(f);
            obj_file.set_extension("o");
            (kernel_dir.join(f), obj_file)
        })
        .collect();
    let should_compile = if out_file.exists() {
        cu_files.iter().any(|(cu_file, _)| {
            let out_modified = out_file.metadata().unwrap().modified().unwrap();
            let in_modified = cu_file.metadata().unwrap().modified().unwrap();
            in_modified.duration_since(out_modified).is_ok()
        })
    } else {
        true
    };
    if should_compile {
        cu_files
            .par_iter()
            .map(|(cu_file, obj_file)| {
                let mut command = std::process::Command::new("nvcc");
                command
                    .arg("-std=c++17")
                    .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                    .arg("-c")
                    .args(["-o", obj_file.to_str().unwrap()])
                    .args(["--default-stream", "per-thread"])
                    .arg("-Icutlass/include")
                    .arg("--expt-relaxed-constexpr")
                    .arg(cu_file);
                let output = command
                    .spawn()
                    .context("failed spawning nvcc")?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                        "nvcc error while compiling:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
                Ok(())
            })
            .collect::<Result<()>>()?;
        let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
        let mut command = std::process::Command::new("nvcc");
        command
            .arg("--lib")
            .args(["-o", out_file.to_str().unwrap()])
            .args(obj_files);
        let output = command
            .spawn()
            .context("failed spawning nvcc")?
            .wait_with_output()?;
        if !output.status.success() {
            anyhow::bail!(
                "nvcc error while linking:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        }
    }
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    /* laurent: I tried using the cc cuda integration as below but this lead to ptaxs never
       finishing to run for some reason. Calling nvcc manually worked fine.
    cc::Build::new()
        .cuda(true)
        .include("cutlass/include")
        .flag("--expt-relaxed-constexpr")
        .flag("--default-stream")
        .flag("per-thread")
        .flag(&format!("--gpu-architecture=sm_{compute_cap}"))
        .file("kernels/flash_fwd_hdim32_fp16_sm80.cu")
        .compile("flashattn");
    */
    Ok(())
}

fn set_cuda_include_dir() -> Result<()> {
    // NOTE: copied from cudarc build.rs.
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::<PathBuf>::into);
    let root = env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
        .context("cannot find include/cuda.h")?;
    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        root.join("include").display()
    );
    Ok(())
}

#[allow(unused)]
fn compute_cap() -> Result<usize> {
    // Grab compute code from nvidia-smi
    let mut compute_cap = {
        let out = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=compute_cap")
                    .arg("--format=csv")
                    .output()
                    .context("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.")?;
        let out = std::str::from_utf8(&out.stdout).context("stdout is not a utf8 string")?;
        let mut lines = out.lines();
        assert_eq!(
            lines.next().context("missing line in stdout")?,
            "compute_cap"
        );
        let cap = lines
            .next()
            .context("missing line in stdout")?
            .replace('.', "");
        cap.parse::<usize>()
            .with_context(|| format!("cannot parse as int {cap}"))?
    };

    // Grab available GPU codes from nvcc and select the highest one
    let max_nvcc_code = {
        let out = std::process::Command::new("nvcc")
                    .arg("--list-gpu-code")
                    .output()
                    .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).unwrap();

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        if !codes.contains(&compute_cap) {
            anyhow::bail!(
                "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {codes:?}."
            );
        }
        *codes.last().unwrap()
    };

    // If nvidia-smi compute_cap is higher than the highest gpu code from nvcc,
    // then choose the highest gpu code in nvcc
    if compute_cap > max_nvcc_code {
        println!(
            "cargo:warning=Lowering gpu arch {compute_cap} to max nvcc target {max_nvcc_code}."
        );
        compute_cap = max_nvcc_code;
    }

    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        compute_cap = compute_cap_str
            .parse::<usize>()
            .with_context(|| format!("cannot parse as usize '{compute_cap_str}'"))?;
        println!("cargo:warning=Using gpu arch {compute_cap} from $CUDA_COMPUTE_CAP");
    }
    println!("cargo:rustc-env=CUDA_COMPUTE_CAP=sm_{compute_cap}");
    Ok(compute_cap)
}
