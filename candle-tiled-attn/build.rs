fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        use std::{env, path::PathBuf};

        println!("cargo::rerun-if-changed=src/cuda/tiled_attn.cu");

        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
        let bindings = cudaforge::KernelBuilder::new()
            .source_glob("src/cuda/*.cu")
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O3")
            .build_ptx()
            .expect("failed to build tiled attention PTX");

        bindings
            .write(out_dir.join("cuda_kernels.rs"))
            .expect("failed to write tiled attention PTX bindings");
    }
}
