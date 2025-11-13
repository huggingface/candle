// TEAM-506: CUDA parity for ROCm build system
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    
    // Determine which backend to build
    let has_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let has_rocm = env::var("CARGO_FEATURE_ROCM").is_ok();
    
    if has_cuda {
        build_cuda_kernels();
    } else if has_rocm {
        build_rocm_kernels();
    } else {
        panic!("Either 'cuda' or 'rocm' feature must be enabled");
    }
}

fn build_cuda_kernels() {
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo::warning={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(ptx_path).unwrap();
    println!("cargo::warning=CUDA kernels built successfully");
}

fn build_rocm_kernels() {
    use std::process::Command;
    use std::fs;
    
    println!("cargo::warning=Building ROCm/HIP kernels...");
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let hsaco_rs_path = out_dir.join("hsaco.rs");
    
    // Get ROCm path
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let hipcc = format!("{}/bin/hipcc", rocm_path);
    
    // Check if hipcc exists
    if !std::path::Path::new(&hipcc).exists() {
        panic!("hipcc not found at {}. Set ROCM_PATH environment variable.", hipcc);
    }
    
    // All kernel modules (must match CUDA exactly)
    let kernels = [
        "affine", "binary", "cast", "conv", "fill",
        "indexing", "quantized", "reduce", "sort", "ternary", "unary"
    ];
    
    let mut hsaco_code = String::from("// TEAM-506: Auto-generated ROCm HSACO bindings\n\n");
    
    for kernel in &kernels {
        let cu_file = format!("src/{}.cu", kernel);
        let hip_file = format!("src/{}.hip", kernel);
        let hsaco_file = out_dir.join(format!("{}.hsaco", kernel));
        
        println!("cargo::rerun-if-changed={}", cu_file);
        
        // Step 1: Convert .cu to .hip using hipify-perl
        let hipify_status = Command::new("hipify-perl")
            .arg(&cu_file)
            .stdout(std::fs::File::create(&hip_file).unwrap())
            .status();
        
        if hipify_status.is_err() || !hipify_status.unwrap().success() {
            println!("cargo::warning=hipify-perl failed for {}, trying direct compilation", kernel);
            // If hipify fails, try compiling .cu directly (hipcc can handle some CUDA code)
        }
        
        // Step 2: Compile HIP → HSACO
        let source_file = if std::path::Path::new(&hip_file).exists() {
            &hip_file
        } else {
            &cu_file
        };
        
        println!("cargo::warning=Compiling {} → {}.hsaco", source_file, kernel);
        
        let status = Command::new(&hipcc)
            .args(&[
                "-c", source_file,
                "-o", hsaco_file.to_str().unwrap(),
                "--offload-arch=gfx1030",  // RDNA2: RX 6000 series
                "--offload-arch=gfx1100",  // RDNA3: RX 7000 series
                "--offload-arch=gfx90a",   // CDNA2: MI200 series
                "-O3",
                "-ffast-math",
                "-fgpu-rdc",  // Relocatable device code
            ])
            .status()
            .expect(&format!("Failed to execute hipcc for {}", kernel));
        
        if !status.success() {
            panic!("Failed to compile {} to HSACO", kernel);
        }
        
        // Step 3: Read HSACO binary
        let hsaco_bytes = fs::read(&hsaco_file)
            .expect(&format!("Failed to read HSACO for {}", kernel));
        
        println!("cargo::warning=Generated {}.hsaco ({} bytes)", kernel, hsaco_bytes.len());
        
        // Step 4: Generate Rust constant (exactly like CUDA's ptx.rs)
        hsaco_code.push_str(&format!(
            "pub const {}: &[u8] = &[\n",
            kernel.to_uppercase()
        ));
        
        // Format as byte array for readability
        for (i, byte) in hsaco_bytes.iter().enumerate() {
            if i % 16 == 0 {
                hsaco_code.push_str("    ");
            }
            hsaco_code.push_str(&format!("0x{:02x}, ", byte));
            if i % 16 == 15 {
                hsaco_code.push('\n');
            }
        }
        
        hsaco_code.push_str("\n];\n\n");
    }
    
    // Write hsaco.rs (exactly like ptx.rs)
    fs::write(&hsaco_rs_path, hsaco_code)
        .expect("Failed to write hsaco.rs");
    
    println!("cargo::warning=ROCm kernels built successfully: {} modules", kernels.len());
}
