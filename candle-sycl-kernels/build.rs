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
// Every header the sources include, so that editing one rebuilds the library.
const HEADERS: &[&str] = &["candle_sycl.h", "common.hpp", "quant_blocks.hpp"];

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
    println!("cargo:rerun-if-env-changed=CANDLE_SYCL_TARGETS");
    println!("cargo:rerun-if-env-changed=MKLROOT");

    let oneapi_root = std::env::var("ONEAPI_ROOT").ok().or_else(|| {
        Path::new("/opt/intel/oneapi")
            .exists()
            .then(|| "/opt/intel/oneapi".to_string())
    });

    let mut common = vec![
        "-fsycl".to_string(),
        "-O2".into(),
        "-fPIC".into(),
        "-std=c++20".into(),
        "-fno-fast-math".into(),
        "-ffp-contract=off".into(),
        "-Wno-unknown-pragmas".into(),
        "-Wno-deprecated-declarations".into(),
    ];
    // Ahead-of-time device code, e.g. `intel_gpu_bmg_g31` (`sycl-ls --verbose`
    // prints a device's architecture) or `intel_gpu_bmg_g31,spir64` to keep a
    // JIT fallback for other devices. Unset, kernels ship as SPIR-V only and
    // every process JIT-compiles each one on its first launch (seconds).
    if let Ok(targets) = std::env::var("CANDLE_SYCL_TARGETS") {
        common.push(format!("-fsycl-targets={targets}"));
    }
    // `-qmkl` finds oneMKL's headers through `MKLROOT`, which only `setvars.sh`
    // sets; without it, point at the install like `find_icpx` does.
    if std::env::var_os("MKLROOT").is_none() {
        if let Some(root) = oneapi_root.as_deref() {
            common.push(format!("-I{root}/mkl/latest/include"));
        }
    }

    let mut objs = Vec::new();
    for src in SOURCES {
        let obj = out.join(format!("{src}.o"));
        let status = Command::new(&icpx)
            .args(&common)
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

    // oneAPI runtime directories, baked into the .so as an rpath so that a
    // binary which finds this library resolves libsycl/libmkl without the
    // caller having sourced `setvars.sh`.
    let mut runtime_dirs: Vec<String> = Vec::new();
    if let Some(root) = oneapi_root.as_deref() {
        for sub in [
            "compiler/latest/lib",
            "mkl/latest/lib",
            "mkl/latest/lib/intel64",
            "tcm/latest/lib",
            "umf/latest/lib",
        ] {
            let p = format!("{root}/{sub}");
            if Path::new(&p).exists() {
                runtime_dirs.push(p);
            }
        }
    }

    let lib = out.join("libcandle_sycl.so");
    let status = Command::new(&icpx)
        .args(&common)
        .arg("-shared")
        .args(&objs)
        .arg("-o")
        .arg(&lib)
        .args(runtime_dirs.iter().map(|d| format!("-L{d}")))
        .args(runtime_dirs.iter().map(|d| format!("-Wl,-rpath,{d}")))
        .arg("-qmkl=sequential")
        .arg("-lmkl_sycl")
        .arg("-lsycl")
        .args(preload_libs(&runtime_dirs))
        .status()
        .expect("failed to spawn icpx for link");
    assert!(status.success(), "icpx failed to link libcandle_sycl.so");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=candle_sycl");
    // `links` metadata: direct dependents' build scripts get it as
    // `DEP_CANDLE_SYCL_LIB_DIR`, to rpath the .so into their binaries (the
    // `rustc-link-arg` below only applies to this crate's own targets).
    println!("cargo:lib_dir={}", out.display());
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

/// The Level Zero adapter that libsycl `dlopen`s needs libumf, which needs
/// libhwloc, and neither is on the adapter's own RUNPATH: they are found only
/// via `LD_LIBRARY_PATH` from `setvars.sh`. Making them direct dependencies of
/// libcandle_sycl.so loads them through its rpath at startup, and the dynamic
/// loader then reuses the loaded copies for the adapter.
fn preload_libs(runtime_dirs: &[String]) -> Vec<String> {
    let mut args = vec!["-Wl,--push-state,--no-as-needed".to_string()];
    for (lib, file) in [("hwloc", "libhwloc.so.15"), ("umf", "libumf.so.1")] {
        if runtime_dirs.iter().any(|d| Path::new(d).join(file).exists()) {
            args.push(format!("-l{lib}"));
        }
    }
    args.push("-Wl,--pop-state".into());
    args
}
