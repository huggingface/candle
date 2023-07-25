#![allow(unused)]
use anyhow::{Context, Result};
use std::io::Write;
use std::path::PathBuf;

struct KernelDirectories {
    kernel_dir: &'static str,
    rust_target: &'static str,
    include_dirs: &'static [&'static str],
}

const DIRS: [KernelDirectories; 1] = [KernelDirectories {
    kernel_dir: "examples/custom-ops/kernels/",
    rust_target: "examples/custom-ops/cuda_kernels.rs",
    include_dirs: &[],
}];

impl KernelDirectories {
    fn maybe_build_ptx(
        &self,
        cu_file: &std::path::Path,
        ptx_file: &std::path::Path,
        compute_cap: usize,
    ) -> Result<()> {
        let should_compile = if ptx_file.exists() {
            let ptx_modified = ptx_file.metadata()?.modified()?;
            let cu_modified = cu_file.metadata()?.modified()?;
            cu_modified.duration_since(ptx_modified).is_ok()
        } else {
            true
        };
        if should_compile {
            #[cfg(feature = "cuda")]
            {
                let mut command = std::process::Command::new("nvcc");
                let out_dir = ptx_file.parent().context("no parent for ptx file")?;
                let include_dirs: Vec<String> =
                    self.include_dirs.iter().map(|c| format!("-I{c}")).collect();
                command
                    .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                    .arg("--ptx")
                    .args(["--default-stream", "per-thread"])
                    .args(["--output-directory", out_dir.to_str().unwrap()])
                    .arg(format!("-I/{}", self.kernel_dir))
                    .args(include_dirs)
                    .arg(cu_file);
                let output = command
                    .spawn()
                    .context("failed spawning nvcc")?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                    "nvcc error while compiling {cu_file:?}:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
                }
            }
            #[cfg(not(feature = "cuda"))]
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .open(ptx_file)?;
        }
        Ok(())
    }
    fn process(&self, out_dir: &std::path::Path, compute_cap: usize) -> Result<()> {
        println!("cargo:rerun-if-changed={}", self.kernel_dir);
        let kernel_dir = PathBuf::from(self.kernel_dir);
        let out_dir = out_dir.join(self.kernel_dir);
        if !out_dir.exists() {
            std::fs::create_dir_all(&out_dir)?;
        }
        let mut cu_files = vec![];
        let mut cuh_files = vec![];
        for file in std::fs::read_dir(kernel_dir)?.flatten() {
            let file = file.path();
            match file.extension().and_then(|v| v.to_str()) {
                Some("cu") => cu_files.push(file),
                Some("cuh") => cuh_files.push(file),
                _ => {}
            }
        }

        let mut ptx_paths = vec![];
        for cu_file in cu_files.iter() {
            let file_stem = cu_file
                .file_stem()
                .with_context(|| format!("no stem {cu_file:?}"))?;
            let file_stem = file_stem.to_string_lossy().into_owned();
            let ptx_file = out_dir.join(&format!("{file_stem}.ptx"));
            self.maybe_build_ptx(cu_file, &ptx_file, compute_cap)?;
            ptx_paths.push(ptx_file);
        }

        let regenerate_rs_file = true;
        if regenerate_rs_file {
            let mut file = std::fs::File::create(self.rust_target)?;
            for ptx_path in ptx_paths {
                let name = ptx_path
                    .file_stem()
                    .context("empty stem")?
                    .to_string_lossy();
                file.write_all(b"#[rustfmt::skip]\n")?;
                let const_definition = format!(
                    r#"pub const {}: &str = include_str!(concat!(env!("OUT_DIR"), "/{}/{name}.ptx"));"#,
                    name.to_uppercase().replace('.', "_"),
                    self.kernel_dir,
                );
                file.write_all(const_definition.as_bytes())?;
                file.write_all(b"\n")?;
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").context("OUT_DIR not set")?;
    let out_dir = PathBuf::from(out_dir);
    #[cfg(feature = "cuda")]
    set_cuda_include_dir()?;
    #[cfg(feature = "cuda")]
    let compute_cap = compute_cap()?;
    #[cfg(not(feature = "cuda"))]
    let compute_cap = 0;
    for d in DIRS {
        d.process(&out_dir, compute_cap)?
    }
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
