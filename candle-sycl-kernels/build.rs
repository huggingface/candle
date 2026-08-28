// Compiles the SYCL C++ sources in `csrc/` into a shared library with `icpx`
// and links it into the crate. Runs ONLY when `candle-core` is built with
// `--features sycl` (this crate is excluded from the workspace), so a plain
// `cargo build` never reaches here and never needs oneAPI installed.
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCES: &[&str] = &[
    "runtime.cpp",
    "elementwise.cpp",
    "reduce.cpp",
    "ternary.cpp",
    "indexing.cpp",
    "pool.cpp",
    "im2col.cpp",
    "conv_transpose.cpp",
    "quant.cpp",
    "mmvq.cpp",
    "norm.cpp",
    "gemm.cpp",
];
const HEADERS: &[&str] = &["candle_sycl.h", "common.hpp"];

fn find_icpx() -> String {
    if let Ok(p) = std::env::var("CANDLE_SYCL_ICPX") {
        return p;
    }
    if Command::new("icpx").arg("--version").output().is_ok() {
        return "icpx".to_string();
    }
    for root in [
        std::env::var("ONEAPI_ROOT").unwrap_or_default(),
        "/opt/intel/oneapi".to_string(),
    ] {
        if root.is_empty() {
            continue;
        }
        let cand = format!("{root}/compiler/latest/bin/icpx");
        if Path::new(&cand).exists() {
            return cand;
        }
    }
    panic!(
        "candle-sycl-kernels: `icpx` not found. Install Intel oneAPI (or run in the \
         `intel/oneapi-basekit` container) and ensure `icpx` is on PATH, or set \
         CANDLE_SYCL_ICPX to its full path. This crate is only built for \
         `--features sycl`."
    );
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc = manifest.join("csrc");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let icpx = find_icpx();

    for f in SOURCES.iter().chain(HEADERS) {
        println!("cargo:rerun-if-changed=csrc/{f}");
    }
    println!("cargo:rerun-if-env-changed=CANDLE_SYCL_ICPX");

    let common = [
        "-fsycl",
        "-O2",
        "-fPIC",
        "-std=c++20",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-Wno-unknown-pragmas",
        "-Wno-deprecated-declarations",
    ];

    let mut objs = Vec::new();
    for src in SOURCES {
        let obj = out.join(format!("{src}.o"));
        let status = Command::new(&icpx)
            .args(common)
            .arg("-qmkl=sequential")
            .arg("-c")
            .arg(csrc.join(src))
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("failed to spawn icpx");
        assert!(status.success(), "icpx failed to compile {src}");
        objs.push(obj);
    }

    let lib = out.join("libcandle_sycl.so");
    let status = Command::new(&icpx)
        .args(common)
        .arg("-shared")
        .args(&objs)
        .arg("-o")
        .arg(&lib)
        .arg("-qmkl=sequential")
        .arg("-lmkl_sycl")
        .arg("-lsycl")
        .status()
        .expect("failed to spawn icpx for link");
    assert!(status.success(), "icpx failed to link libcandle_sycl.so");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=candle_sycl");
    // So the .so is found at runtime without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
    // oneAPI runtime libs (libsycl, libmkl_*) live here.
    if let Ok(root) = std::env::var("ONEAPI_ROOT").or_else(|_| {
        if Path::new("/opt/intel/oneapi").exists() {
            Ok("/opt/intel/oneapi".to_string())
        } else {
            Err(std::env::VarError::NotPresent)
        }
    }) {
        for sub in [
            "compiler/latest/lib",
            "mkl/latest/lib",
            "mkl/latest/lib/intel64",
        ] {
            let p = format!("{root}/{sub}");
            if Path::new(&p).exists() {
                println!("cargo:rustc-link-search=native={p}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
            }
        }
    }
}
