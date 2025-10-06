#![allow(unused)]
use anyhow::{Context, Result};
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};

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
    Ok(())
}
