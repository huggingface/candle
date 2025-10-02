#![allow(unused)]
use anyhow::{Context, Result};
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};

// For build_bert_single_file_binary()
use std::fs::{self, File};
use std::io::copy;

struct KernelDirectories {
    kernel_glob: &'static str,
    rust_target: &'static str,
    include_dirs: &'static [&'static str],
}

const KERNEL_DIRS: [KernelDirectories; 1] = [KernelDirectories {
    kernel_glob: "examples/custom-ops/kernels/*.cu",
    rust_target: "examples/custom-ops/cuda_kernels.rs",
    include_dirs: &[],
}];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        // Added: Get the safe output directory from the environment.
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        for kdir in KERNEL_DIRS.iter() {
            let builder = bindgen_cuda::Builder::default().kernel_paths_glob(kdir.kernel_glob);
            println!("cargo:info={builder:?}");
            let bindings = builder.build_ptx().unwrap();

            // Changed: This now writes to a safe path inside $OUT_DIR.
            let safe_target = out_dir.join(
                Path::new(kdir.rust_target)
                    .file_name()
                    .context("Failed to get filename from rust_target")?,
            );
            bindings.write(safe_target).unwrap()
        }
    }

    build_bert_single_file_binary()?;

    Ok(())
}

fn build_bert_single_file_binary() -> Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").context("CARGO_MANIFEST_DIR not set")?;
    let example_name = "bert_single_file_binary";

    let dest_path = Path::new(&manifest_dir)
        .join("examples")
        .join(example_name)
        .join("files");

    let files = ["config.json", "tokenizer.json", "model.safetensors"];

    let all_files_exist = files
        .iter()
        .all(|filename| dest_path.join(filename).exists());

    if all_files_exist {
        eprintln!(
            "All {} files already exist, skipping download",
            example_name
        );
        return Ok(());
    }

    eprintln!("Downloading {} files...", example_name);

    fs::create_dir_all(&dest_path).context("Failed to create destination directory")?;

    // Use specific commit vs main to reduce chance of URL breaking later from directory layout changes, etc.
    let base_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf";

    for filename in &files {
        let dest_file = dest_path.join(filename);

        if dest_file.exists() {
            eprintln!("File already exists, skipping: {}", filename);
            continue;
        }

        let url = format!("{}/{}", base_url, filename);
        eprintln!("Downloading {} from {}...", filename, url);

        let response = ureq::get(&url)
            .call()
            .context(format!("Failed to download {}", url))?;

        if response.status() != 200 {
            anyhow::bail!(
                "Download failed for {} with status: {}",
                filename,
                response.status()
            );
        }

        let mut reader = response.into_reader();
        let mut file =
            File::create(&dest_file).context(format!("Failed to create file {:?}", dest_file))?;

        let bytes_written =
            copy(&mut reader, &mut file).context(format!("Failed to write {}", filename))?;

        eprintln!("Downloaded {} ({} bytes)", filename, bytes_written);
    }

    eprintln!("All {} files downloaded successfully", example_name);
    Ok(())
}
