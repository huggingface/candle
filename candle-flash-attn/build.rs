#![allow(unused)]
use anyhow::{Context, Result};
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_hdim32_fp16_sm80.cu");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo:rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=kernels/flash.h");
    println!("cargo:rerun-if-changed=kernels/philox.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo:rerun-if-changed=kernels/block_info.h");
    println!("cargo:rerun-if-changed=kernels/static_switch.h");

    let out_dir = std::env::var("OUT_DIR").context("OUT_DIR not set")?;
    let mut out_dir = PathBuf::from(out_dir);
    // TODO: Getting up two levels avoid having to recompile this too often, however it's likely
    // not a safe assumption.
    out_dir.pop();
    out_dir.pop();
    set_cuda_include_dir()?;
    let compute_cap = compute_cap()?;

    let mut command = std::process::Command::new("nvcc");
    let out_file = out_dir.join("libflashattention.a");

    let cu_file = PathBuf::from("kernels/flash_fwd_hdim32_fp16_sm80.cu");
    let should_compile = if out_file.exists() {
        let out_modified = out_file.metadata()?.modified()?;
        let in_modified = cu_file.metadata()?.modified()?;
        in_modified.duration_since(out_modified).is_ok()
    } else {
        true
    };
    if should_compile {
        command
            .arg(format!("--gpu-architecture=sm_{compute_cap}"))
            .arg("--lib")
            .args(["-o", out_file.to_str().unwrap()])
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
    }
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=flashattention");

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
