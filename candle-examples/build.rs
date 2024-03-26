#![allow(unused)]
use anyhow::{Context, Result};
use std::io::Write;
use std::path::PathBuf;

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
        for kdir in KERNEL_DIRS.iter() {
            let builder = bindgen_cuda::Builder::default().kernel_paths_glob(kdir.kernel_glob);
            println!("cargo:info={builder:?}");
            let bindings = builder.build_ptx().unwrap();
            bindings.write(kdir.rust_target).unwrap()
        }
    }
    Ok(())
}
