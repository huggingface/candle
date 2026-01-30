use std::env;
use std::path::{Path, PathBuf};
mod buildtime_downloader;
#[allow(unused_imports)]
use buildtime_downloader::download_model;
use cudaforge::{KernelBuilder, Result};

struct KernelDirectories {
    kernel_glob: &'static str,
    rust_target: &'static str,
}

const KERNEL_DIRS: [KernelDirectories; 1] = [KernelDirectories {
    kernel_glob: "examples/custom-ops/kernels/*.cu",
    rust_target: "examples/custom-ops/cuda_kernels.rs",
}];

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        // Added: Get the safe output directory from the environment.
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        for kdir in KERNEL_DIRS.iter() {
            // Changed: This now writes to a safe path inside $OUT_DIR.
            let safe_target = out_dir.join(
                Path::new(kdir.rust_target)
                    .file_name()
                    .expect("Failed to get filename from rust_target"),
            );

            let bindings = KernelBuilder::new()
                .source_glob(kdir.kernel_glob)
                .build_ptx()?; // ✅ Propagate errors
            bindings.write(safe_target)?;
        }
    }

    // Download config, tokenizer, and model files from hf at build time.
    // option_env! automatically detects changes in the env var and trigger rebuilds correctly.
    // Example value:
    // CANDLE_BUILDTIME_MODEL_REVISION="sentence-transformers/all-MiniLM-L6-v2:c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    if let Some(model_rev) = core::option_env!("CANDLE_BUILDTIME_MODEL_REVISION") {
        buildtime_downloader::download_model(model_rev).expect("Model download failed!");
    }
    Ok(())
}
