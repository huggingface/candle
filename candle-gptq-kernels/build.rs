// Build script compiling the GPTQ fused dequant+GEMM CUDA kernel into a static lib.
//
// The CUDA compile only runs when the `cuda` feature is enabled (i.e. `CARGO_FEATURE_CUDA` is
// set). With `--features metal` and no CUDA the crate has no native build step: the Metal kernel
// source is compiled at runtime from the embedded `.metal` string, so this script is a no-op.
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        // No CUDA requested (e.g. a Metal-only build): nothing to compile or link.
        return Ok(());
    }

    use cudaforge::KernelBuilder;

    println!("cargo::rerun-if-changed=kernels/gptq_gemm.cu");
    println!("cargo::rerun-if-changed=kernels/gptq_gemm_tc.cu");
    println!("cargo::rerun-if-changed=kernels/marlin/marlin_cuda_kernel.cu");
    println!("cargo::rerun-if-changed=kernels/marlin/marlin_shim.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));

    let builder = KernelBuilder::new()
        .source_files(vec![
            "kernels/gptq_gemm.cu",
            "kernels/gptq_gemm_tc.cu",
            "kernels/marlin/marlin_cuda_kernel.cu",
            "kernels/marlin/marlin_shim.cu",
        ])
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Xcompiler")
        .arg("-fPIC")
        // The vendored Marlin kernel (kernels/marlin/marlin_cuda_kernel.cu) calls the constexpr
        // __host__ helper `ceildiv` from __global__/__device__ code; recent nvcc rejects this
        // without relaxing the constexpr rules. Passed as a build flag rather than editing the
        // vendored kernel, which must stay an unmodified copy of upstream IST-DASLab/marlin.
        .arg("--expt-relaxed-constexpr");

    let out_file = out_dir.join("libgptqkernels.a");
    builder.build_lib(out_file)?;

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=gptqkernels");
    println!("cargo::rustc-link-lib=dylib=cudart");
    Ok(())
}
