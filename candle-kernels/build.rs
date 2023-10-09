use std::io::Write;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    cuda::set_include_dir();
    let kernel_paths = cuda::build_ptx();
    let mut file = std::fs::File::create("src/lib.rs").unwrap();
    file.write_all(
        br#"use phf::phf_map;"#
    ).unwrap();
    file.write_all(&[b'\n']).unwrap();
    file.write_all(
        br#"pub static KERNELS: phf::Map<&'static str, &'static str> = phf_map! {"#
    ).unwrap();


    for kernel_path in kernel_paths {
        let name = kernel_path.file_stem().unwrap().to_str().unwrap();
        file.write_all(
            format!(
                r#""{}" => include_str!(concat!(env!("OUT_DIR"), "/{}.ptx")),"#,
                name.to_uppercase().replace('.', "_"),
                name
            )
                .as_bytes(),
        )
            .unwrap();
        file.write_all(&[b'\n']).unwrap();
    }
    file.write_all(
        br#"};"#
    ).unwrap();
    file.write_all(&[b'\n']).unwrap();
}

mod cuda {
    use anyhow::{Context, Result};

    pub fn set_include_dir() {
        use std::path::PathBuf;
        // NOTE: copied from cudarc build.rs.
        // We can't actually set a env!() value from another crate,
        // so we have to do that here.

        // use PathBuf;

        let env_vars = [
            "CUDA_PATH",
            "CUDA_ROOT",
            "CUDA_TOOLKIT_ROOT_DIR",
            "CUDNN_LIB",
        ];
        #[allow(unused)]
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
        #[allow(unused)]
            let roots = roots.into_iter().map(Into::<PathBuf>::into);

        #[cfg(feature = "ci-check")]
            let root: PathBuf = "ci".into();

        #[cfg(not(feature = "ci-check"))]
            let root = env_vars
            .chain(roots)
            .find(|path| path.join("include").join("cuda.h").is_file())
            .unwrap();

        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            root.join("include").display()
        );
    }

    pub fn build_ptx() -> Vec<std::path::PathBuf> {
        use rayon::prelude::*;
        use std::path::PathBuf;
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let kernel_paths: Vec<PathBuf> = glob::glob("src/*.cu")
            .unwrap()
            .map(|p| p.unwrap())
            .collect();
        let mut include_directories: Vec<PathBuf> = glob::glob("src/**/*.cuh")
            .unwrap()
            .map(|p| p.unwrap())
            .collect();

        println!("cargo:rerun-if-changed=src/");
        // for path in &kernel_paths {
        //     println!("cargo:rerun-if-changed={}", path.display());
        // }

        for path in &mut include_directories {
            // println!("cargo:rerun-if-changed={}", path.display());
            let destination =
                std::format!("{out_dir}/{}", path.file_name().unwrap().to_str().unwrap());
            std::fs::copy(path.clone(), destination).unwrap();
            // remove the filename from the path so it's just the directory
            path.pop();
        }

        include_directories.sort();
        include_directories.dedup();

        #[allow(unused)]
            let include_options: Vec<String> = include_directories
            .into_iter()
            .map(|s| "-I".to_string() + &s.into_os_string().into_string().unwrap())
            .collect::<Vec<_>>();

        let ccbin_env = std::env::var("CANDLE_NVCC_CCBIN");
        println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");

        let compute_caps = compute_caps().unwrap();

        let ptx_paths = compute_caps
            .iter()
            .map(|compute_cap| {
                let ptx_paths = kernel_paths.par_iter().map(
                    |p| {
                        let output_filename = std::path::Path::new(&out_dir).to_path_buf().join("out").with_file_name(format!("{}_{compute_cap}.ptx", p.file_stem().unwrap().to_str().unwrap()));

                        let ignore = if output_filename.exists() {
                            let out_modified = output_filename.metadata().unwrap().modified().unwrap();
                            let in_modified = p.metadata().unwrap().modified().unwrap();
                            out_modified.duration_since(in_modified).is_ok()
                        } else {
                            false
                        };
                        if ignore {
                            Ok(output_filename)
                        } else {
                            let mut command = std::process::Command::new("nvcc");
                            command.arg(format!("--gpu-architecture=sm_{compute_cap}"))
                                .arg("--ptx")
                                .args(["--default-stream", "per-thread"])
                                .args(["--output-file", &output_filename.to_str().unwrap()])
                                // Flash attention only
                                // .arg("--expt-relaxed-constexpr")
                                .args(&include_options);
                            if let Ok(ccbin_path) = &ccbin_env {
                                command
                                    .arg("-allow-unsupported-compiler")
                                    .args(["-ccbin", ccbin_path]);
                            }
                            command.arg(p);
                            let output = command.spawn().context("nvcc failed to start. Ensure that you have CUDA installed and that `nvcc` is in your PATH.")?.wait_with_output()?;

                            if !output.status.success() {
                                anyhow::bail!(
                        "nvcc error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        &command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                            }

                            Ok(output_filename)
                        }
                    }).collect::<Result<Vec<PathBuf>>>().unwrap();
                ptx_paths
            })
            .flatten().collect::<Vec<PathBuf>>();

        ptx_paths
    }

    #[allow(unused)]
    fn compute_caps() -> Result<Vec<usize>> {
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAPS");

        // Try to parse compute caps from env
        let mut compute_caps = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAPS") {
            println!("cargo:rustc-env=CUDA_COMPUTE_CAPS={compute_cap_str}");
            compute_cap_str
                .split(',')
                .map(|code_str| code_str.parse::<usize>().context("Could not parse code"))
                .collect::<Result<Vec<usize>>>()?
        } else {
            // Use nvidia-smi to get the current compute cap
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
            let cap = cap
                .parse::<usize>()
                .with_context(|| format!("cannot parse as int {cap}"))?;
            println!("cargo:rustc-env=CUDA_COMPUTE_CAPS={cap}");
            vec![cap]
        };

        // Grab available GPU codes from nvcc and select the highest one
        let (supported_nvcc_codes, max_nvcc_code) = {
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
            let max_nvcc_code = *codes.last().unwrap();
            (codes, max_nvcc_code)
        };

        // Check that nvcc supports the asked compute caps
        for cap in compute_caps.iter() {
            if !supported_nvcc_codes.contains(cap) {
                anyhow::bail!(
                "nvcc cannot target gpu arch {cap}. Available nvcc targets are {supported_nvcc_codes:?}."
            );
            }
            if cap > &max_nvcc_code {
                anyhow::bail!(
                "CUDA compute cap {cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
            );
            }
        }

        Ok(compute_caps)
    }
}

