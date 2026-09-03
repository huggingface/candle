fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(candle_nightly)");
    // The optimized aarch64 fp16 kernels rely on `float16x8_t`, which is
    // still gated behind the unstable `stdarch_neon_f16` library feature.
    // Detect a nightly compiler so those kernels can be enabled there while
    // stable builds fall back to the widening implementation.
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default();
    if version.contains("nightly") || version.contains("dev") {
        println!("cargo::rustc-cfg=candle_nightly");
    }
}
