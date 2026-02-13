use cudaforge::{detect_compute_cap, get_gpu_arch_string, KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

// Macro to register capabilities in Rust and in the CUDA builder
macro_rules! set_cfg {
    ($builder:expr, $name:expr, $value:expr) => {
        // Register for Rust (check-cfg is required for recent Rust versions)
        println!("cargo::rustc-check-cfg=cfg({}, values(\"true\", \"false\"))", $name);
        
        // Apply the configuration if the value is true
        if $value {
            println!("cargo:rustc-cfg={}=\"true\"", $name);
            $builder = $builder.arg(format!("-D{}=1", $name.to_uppercase()).as_str());
        } else {
            println!("cargo:rustc-cfg={}=\"false\"", $name);
            // Optionnel : $builder = $builder.arg(format!("-D{}=0", $name.to_uppercase()).as_str());
        }
    };
}

// Macro helper to apply to both builders
macro_rules! dual_set {
    ($b1:expr, $b2:expr, $name:expr, $value:expr) => {
        set_cfg!($b1, $name, $value);
        set_cfg!($b2, $name, $value);
    };
}

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = detect_compute_cap().unwrap_or_else(|_| cudaforge::GpuArch::new(0));
    let compute_cap = arch.base();

    // Initialize builders early
    let mut builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .exclude(&["moe_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    let mut moe_builder = KernelBuilder::new()
        .compute_cap(compute_cap)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/moe/moe_hfma2.cu",
        ]);

    let target = env::var("TARGET").unwrap_or_default();
    let is_target_msvc = target.contains("msvc");
    if is_target_msvc {
        moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
    } else {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    // Register check-cfg for non-macro flags
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp8)");
    println!("cargo::rustc-check-cfg=cfg(cuda_arch, values(\"53\", \"61\", \"70\", \"75\", \"80\", \"86\", \"89\", \"90\", \"100\", \"120\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_70, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_61, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_53, values(\"true\", \"false\"))");

    // Apply hardware capabilities using the macro (affects Rust and both builders)
    println!("cargo:rustc-cfg=cuda_arch=\"{}\"", compute_cap);
    println!("cargo:rustc-cfg=inf_cc_70=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=inf_cc_61=\"{}\"", compute_cap >= 61);
    println!("cargo:rustc-cfg=inf_cc_53=\"{}\"", compute_cap >= 53);


    // ── Maxwell/Pascal (CC 5.3 – 6.x) ── Arithmetic foundations ──────────────
    // FP16 (half precision): 16-bit float, half the size of f32. Enables faster
    // inference on memory-bound workloads. Available since Maxwell (CC 5.3).
    dual_set!(builder, moe_builder, "has_f16", compute_cap >= 53);
    // half2 native: Packed FP16 arithmetic — two FP16 ops in a single instruction,
    // doubling throughput. Only on GP100 (6.0), GP10B (6.2), and Volta+ (7.0+).
    dual_set!(builder, moe_builder, "has_half2_native", compute_cap == 60 || compute_cap == 62 || compute_cap >= 70);
    // DP4A: Dot Product of 4x8-bit integers accumulated into 32-bit.
    // First HW-accelerated INT8 inference instruction. (CC 6.1+, e.g. GTX 1080)
    dual_set!(builder, moe_builder, "has_dp4a", compute_cap >= 61);
    // Pageable memory access: GPU can directly access system (CPU) memory
    // without explicit pinning via cudaMallocHost. (Unified Memory, CC 6.0+)
    dual_set!(builder, moe_builder, "has_pageable_memory_access", compute_cap >= 60);
    // CUDA Memory Pools: Efficient stream-ordered memory allocation/deallocation
    // via cudaMallocAsync/cudaFreeAsync, reducing allocation overhead. (CC 6.0+)
    dual_set!(builder, moe_builder, "has_memory_pools", compute_cap >= 60);
    // Peer-to-Peer copy: Direct GPU-to-GPU memory transfers over NVLink/PCIe
    // without CPU staging. Essential for multi-GPU inference. (CC 6.0+)
    dual_set!(builder, moe_builder, "has_p2p_copy", compute_cap >= 60);
    // CUDA Graphs: Capture a sequence of GPU operations and replay them with
    // minimal CPU overhead. Reduces kernel launch latency. (CC 3.0+)
    dual_set!(builder, moe_builder, "has_cuda_graphs", compute_cap >= 30);

    // ── Volta/Turing (CC 7.0 – 7.5) ── Tensor Cores era ─────────────────────
    // WMMA (Warp Matrix Multiply-Accumulate): First-gen Tensor Core API.
    // 16x16x16 matrix ops in hardware. The foundation of fast GEMM. (CC 7.0+)
    dual_set!(builder, moe_builder, "has_wmma", compute_cap >= 70);
    // WMMA FP16: Tensor Core matrix multiply with FP16 inputs/outputs. (CC 7.0+)
    dual_set!(builder, moe_builder, "has_wmma_f16", compute_cap >= 70);
    // Independent thread scheduling: Each thread has its own program counter.
    // Enables fine-grained synchronization & divergent warp execution. (CC 7.0+)
    dual_set!(builder, moe_builder, "has_independent_thread_scheduling", compute_cap >= 70);
    // BF16 conversions: Hardware-accelerated FP32↔BF16 conversion instructions.
    // Fast type casting for mixed-precision training/inference. (CC 7.0+)
    dual_set!(builder, moe_builder, "has_bf16_conversions", compute_cap >= 70);

    // ── Ampere (CC 8.0 – 8.9) ── Precision & efficiency ─────────────────────
    // Sparse Tensor Cores: 2:4 structured sparsity — prune 50% of weights
    // and get ~2x speedup on matrix ops with no accuracy loss. (CC 8.0+)
    dual_set!(builder, moe_builder, "has_sparse_tensor_cores", compute_cap >= 80);
    // WMMA BF16: Tensor Core matrix multiply with BF16 inputs. BF16 has the
    // same exponent range as FP32 (better stability than FP16). (CC 8.0+)
    dual_set!(builder, moe_builder, "has_wmma_bf16", compute_cap >= 80);
    // BF16 native: Full BF16 arithmetic support (add, mul, fma). (CC 8.0+)
    dual_set!(builder, moe_builder, "has_bf16", compute_cap >= 80);
    // TF32 (TensorFloat-32): 19-bit format (8-bit exp + 10-bit mantissa + sign).
    // Transparent FP32→TF32 on Tensor Cores, ~8x faster than FP32. (CC 8.0+)
    dual_set!(builder, moe_builder, "has_tf32", compute_cap >= 80);
    // Async copy: memcpy_async from global to shared memory without blocking
    // the compute pipeline. Key enabler for software pipelining. (CC 8.0+)
    dual_set!(builder, moe_builder, "has_async_copy", compute_cap >= 80);
    // L2 cache persistence: Pin frequently accessed data (e.g. KV cache) in L2.
    // Reduces DRAM bandwidth pressure during inference. (CC 8.0+)
    dual_set!(builder, moe_builder, "has_l2_cache_persistence", compute_cap >= 80);
    // mbarrier: Hardware-assisted asynchronous barriers. Threads can continue
    // working while waiting for async copies to land. Required for Warp
    // Specialization (producer/consumer kernel patterns). (CC 8.0+)
    dual_set!(builder, moe_builder, "has_mbarrier", compute_cap >= 80);

    // ── Ada Lovelace (CC 8.9) ── FP8 & Transformer Engine ───────────────────
    // FP8 (E4M3/E5M2): 8-bit floats for inference. ~2x throughput vs FP16
    // with minimal accuracy loss when combined with per-tensor scaling. (CC 8.9+)
    dual_set!(builder, moe_builder, "has_fp8", compute_cap >= 89);
    // Transformer Engine: Automatic FP8 mixed-precision for attention & FFN.
    // Handles dynamic scaling and format selection (E4M3 vs E5M2). (CC 8.9+)
    dual_set!(builder, moe_builder, "has_transformer_engine", compute_cap >= 89);

    // ── Hopper (CC 9.0) ── Data logistics revolution ─────────────────────────
    // TMA (Tensor Memory Accelerator): DMA engine that moves N-dimensional
    // tiles between global↔shared memory. Zero CUDA cores used. (CC 9.0+)
    dual_set!(builder, moe_builder, "has_tma", compute_cap >= 90);
    // Thread Block Clusters: Group blocks that can cooperate via distributed
    // shared memory. Enables cross-block synchronization. (CC 9.0+)
    dual_set!(builder, moe_builder, "has_clusters", compute_cap >= 90);
    // Distributed Shared Memory: Blocks within a cluster can directly load/store
    // from each other's shared memory. Ideal for wide attention heads. (CC 9.0+)
    dual_set!(builder, moe_builder, "has_distributed_shared_memory", compute_cap >= 90);
    // L2 multicast: Broadcast data from L2 cache to multiple SMs simultaneously.
    // Eliminates redundant DRAM reads when broadcasting model weights. (CC 9.0+)
    dual_set!(builder, moe_builder, "has_l2_multicast", compute_cap >= 90);
    // Confidential Computing: Hardware-enforced memory encryption (AES) on VRAM
    // and PCIe bus. Data remains encrypted even during GPU processing. Adds latency
    // to memory copies and may disable P2P direct access. Relevant for cloud
    // inference on secure enclaves (Azure/AWS). (CC 9.0+)
    dual_set!(builder, moe_builder, "has_confidential_computing", compute_cap >= 90);

    // ── Blackwell (CC 10.0 / 12.0) ── Density & micro-scaling ────────────────
    //
    // IMPORTANT: Blackwell has TWO distinct SM variants:
    //   - SM 100 (100a/100f) = B200/B100 data center: Full TMA v2, HW all-reduce,
    //     TMEM, cluster multicast, Transformer Engine 2nd gen.
    //   - SM 120 = RTX 5090 consumer: FP4/MXFP4 compute, AMP scheduler, but
    //     limited inter-GPU communication (no HW all-reduce, no cluster multicast).
    //   Kernels written for SM 100a (e.g. cutlass 3.x) may NOT compile or run
    //   efficiently on SM 120 if they rely on data center communication primitives.
    //
    // ─ Shared capabilities (both SM 100 and SM 120) ─
    //
    // NVFP4: NVIDIA's native 4-bit float format. Requires strict per-tensor
    // or per-group scaling. Doubles throughput vs FP8. (CC 10.0+)
    dual_set!(builder, moe_builder, "has_fp4", compute_cap >= 100);
    // MXFP4 (OCP Microscaling): Standardized 4-bit format from the Open Compute
    // Project. Uses micro-scaling with per-group (16/32 elements) scale factors.
    // ~4x compression vs FP16 with near-FP16 accuracy. (CC 10.0+)
    dual_set!(builder, moe_builder, "has_mxfp4", compute_cap >= 100);
    // Microscaling: General support for the MX format family (MXFP4, MXFP6,
    // MXFP8, MXINT8). Enables per-group scale factors shared across 16/32
    // elements, which is the key to Blackwell's precision at low bitwidths. (CC 10.0+)
    dual_set!(builder, moe_builder, "has_microscaling", compute_cap >= 100);
    // TMEM (Tensor Memory): Dedicated on-chip memory directly coupled to Tensor
    // Cores. Replaces register file (RF) usage for TC operands, drastically
    // reducing register pressure and improving warp occupancy. Critical for
    // high-throughput GEMM in CUTLASS 3.x+. (CC 10.0+)
    dual_set!(builder, moe_builder, "has_tmem", compute_cap >= 100);
    // Dynamic Sparsity: Extends the 2:4 structured sparsity (Ampere) with
    // data-dependent sparsity rates that vary per format (FP4/FP8). Allows
    // further bandwidth savings on "information-poor" layers. (CC 10.0+)
    dual_set!(builder, moe_builder, "has_dynamic_sparsity", compute_cap >= 100);
    //
    // ─ Data center only (SM 100 / B200 / B100) ─
    //
    // TMA v2: Enhanced Tensor Memory Accelerator with queuing & async scheduling
    // primitives not available on SM 120. (CC == 100)
    dual_set!(builder, moe_builder, "has_tma_v2", compute_cap == 100);
    // Cluster Multicast: HW-accelerated broadcast from global memory to shared
    // memory across all blocks in a cluster. Enables zero-copy weight distribution
    // to compute units. Available on B200, but REMOVED from RTX 5090. (CC == 100)
    dual_set!(builder, moe_builder, "has_cluster_multicast", compute_cap == 100);
    // HW All-Reduce: Hardware-accelerated multi-GPU reduction without using
    // compute cores. Crucial for Tensor Parallelism across 8+ GPUs. (CC == 100)
    dual_set!(builder, moe_builder, "has_hw_allreduce", compute_cap == 100);
    // TMA Indirect: Scatter/gather via TMA — the accelerator can follow pointers
    // to load non-contiguous data tiles. Game-changer for MoE: loads only the
    // activated experts asynchronously without touching CUDA cores. (CC == 100)
    dual_set!(builder, moe_builder, "has_tma_indirect", compute_cap == 100);
    // Data center GPU: Distinguishes SM 100 (B200/B100) and SM 90 (H100) from
    // consumer parts. Guards kernels relying on advanced inter-GPU communication,
    // warp-specialized TMA v2, cluster multicast, or HW-synchronized collectives.
    dual_set!(builder, moe_builder, "is_data_center_gpu", compute_cap == 100 || compute_cap == 90);
    //
    // ─ Consumer only (SM 120 / RTX 5090) ─
    //
    // AMP (AI Management Processor): Dedicated HW scheduler for AI workloads on
    // consumer Blackwell. Manages AI task queuing independently of the CPU,
    // reducing inference latency and "stuttering" on desktop systems. (CC 12.0+)
    dual_set!(builder, moe_builder, "has_amp", compute_cap >= 120);

    // WMMA (Tensor Cores) require SM 7.0+ (Volta)
    if compute_cap < 70 {
        println!(
            "cargo::warning={} < 70: Excluding WMMA kernels (Tensor Cores require Volta or newer)",
            get_gpu_arch_string(compute_cap)
        );
    }

    if compute_cap >= 53 && compute_cap < 70 {
        builder = builder.arg("-DALLOW_LEGACY_BF16");
        builder = builder.arg("-DALLOW_LEGACY_FP8");
    }

    // Build for PTX
    let ptx_output = builder.build_ptx()?;
    ptx_output.write(out_dir.join("ptx.rs"))?;

    // Build for MOE Static Library
    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}