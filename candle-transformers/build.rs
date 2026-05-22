fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/mamba3");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target = out_dir.join("mamba3_kernels.rs");

    let bindings = cudaforge::KernelBuilder::new()
        .source_glob("kernels/mamba3/*.cu")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()
        .expect("failed to build mamba3 cuda kernels");

    bindings.write(target).expect("failed to write mamba3 ptx bindings");
}
