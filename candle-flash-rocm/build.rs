//! Build script for candle-flash-rocm
//! 
//! Compiles AMD's Composable Kernel Flash Attention implementation
//! Created by: TEAM-509

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-changed=ck/");
    
    // Check if ROCm is available
    let rocm_path = std::env::var("ROCM_PATH")
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    
    let rocm_include = format!("{}/include", rocm_path);
    let rocm_lib = format!("{}/lib", rocm_path);
    
    // Check if ROCm exists
    if !PathBuf::from(&rocm_include).exists() {
        println!("cargo:warning=ROCm not found at {}. Flash Attention will not be available.", rocm_path);
        println!("cargo:warning=Set ROCM_PATH environment variable if ROCm is installed elsewhere.");
        return;
    }
    
    println!("cargo:rustc-link-search=native={}", rocm_lib);
    println!("cargo:rustc-link-lib=amdhip64");
    println!("cargo:rustc-link-lib=hipblas");
    
    // Check if CK is cloned
    let ck_dir = PathBuf::from("ck");
    if !ck_dir.exists() {
        println!("cargo:warning=Composable Kernel not found at ./ck");
        println!("cargo:warning=Run: git clone https://github.com/ROCm/composable_kernel.git ck");
        println!("cargo:warning=Flash Attention will not be available.");
        return;
    }
    
    // Build Composable Kernel Flash Attention
    build_composable_kernel(&ck_dir, &rocm_path);
    
    // Compile our C wrapper
    compile_wrapper(&rocm_path);
}

fn build_composable_kernel(ck_dir: &PathBuf, rocm_path: &str) {
    let build_dir = ck_dir.join("build");
    
    // Create build directory
    if !build_dir.exists() {
        std::fs::create_dir_all(&build_dir).expect("Failed to create CK build directory");
    }
    
    // Detect GPU architecture
    let gpu_target = std::env::var("CK_GPU_TARGETS")
        .unwrap_or_else(|_| "gfx942".to_string()); // Default to MI300
    
    println!("cargo:warning=Building Composable Kernel for GPU target: {}", gpu_target);
    
    // Configure with CMake
    let cmake_status = Command::new("cmake")
        .current_dir(&build_dir)
        .args(&[
            "-D", &format!("CMAKE_PREFIX_PATH={}", rocm_path),
            "-D", &format!("CMAKE_CXX_COMPILER={}/bin/hipcc", rocm_path),
            "-D", "CMAKE_BUILD_TYPE=Release",
            "-D", &format!("GPU_TARGETS={}", gpu_target),
            "-D", "BUILD_DEV=OFF",
            ".."
        ])
        .status();
    
    match cmake_status {
        Ok(status) if status.success() => {
            println!("cargo:warning=CK CMake configuration successful");
        }
        _ => {
            println!("cargo:warning=CK CMake configuration failed. Flash Attention may not work.");
            return;
        }
    }
    
    // Build only Flash Attention examples (faster than full CK)
    let make_status = Command::new("make")
        .current_dir(&build_dir)
        .args(&["-j8", "tile_example_fmha_fwd"])
        .status();
    
    match make_status {
        Ok(status) if status.success() => {
            println!("cargo:warning=CK Flash Attention built successfully");
        }
        _ => {
            println!("cargo:warning=CK build failed. Flash Attention may not work.");
            return;
        }
    }
    
    // Link CK library
    let ck_lib_dir = build_dir.join("lib");
    if ck_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", ck_lib_dir.display());
        println!("cargo:rustc-link-lib=composable_kernel");
    }
}

fn compile_wrapper(rocm_path: &str) {
    cc::Build::new()
        .cpp(true)
        .file("csrc/fmha_wrapper.cpp")
        .include(format!("{}/include", rocm_path))
        .include("ck/include")
        .flag("-std=c++17")
        .flag("-D__HIP_PLATFORM_AMD__")
        .compile("fmha_wrapper");
    
    println!("cargo:warning=C wrapper compiled successfully");
}
