#[macro_export]
macro_rules! set_cfg {
    ($builder:expr, $name:expr, $value:expr) => {
        // Register for Rust (check-cfg is required for recent Rust versions)
        println!(
            "cargo::rustc-check-cfg=cfg({}, values(\"true\", \"false\"))",
            $name
        );

        // Apply the configuration if the value is true
        if $value {
            println!("cargo:rustc-cfg={}=\"true\"", $name);
            $builder = $builder.arg(format!("-D{}=1", $name.to_uppercase()).as_str());
        } else {
            println!("cargo:rustc-cfg={}=\"false\"", $name);
        }
    };
}

#[macro_export]
macro_rules! dual_set {
    ($b1:expr, $b2:expr, $name:expr, $value:expr) => {
        set_cfg!($b1, $name, $value);
        set_cfg!($b2, $name, $value);
    };
}

macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:warning=\r\x1b[32;1m   {}", format!($($tokens)*))
    }
}

pub struct Capability {
    pub name: &'static str,
    pub description: &'static str,
    pub check: fn(usize) -> bool,
}

pub const CAPABILITIES: &[Capability] = &[
    // ── Maxwell/Pascal (CC 5.3 – 6.x) ── Arithmetic foundations ──────────────
    Capability {
        name: "has_f16",
        description: "FP16 (half precision): 16-bit float, half the size of f32. Enables faster inference on memory-bound workloads. Available since Maxwell (CC 5.3).",
        check: |cc| cc >= 53,
    },
    Capability {
        name: "has_half2_native",
        description: "half2 native: Packed FP16 arithmetic — two FP16 ops in a single instruction, doubling throughput. Only on GP100 (6.0), GP10B (6.2), and Volta+ (7.0+).",
        check: |cc| cc == 60 || cc == 62 || cc >= 70,
    },
    Capability {
        name: "has_dp4a",
        description: "DP4A: Dot Product of 4x8-bit integers accumulated into 32-bit. First HW-accelerated INT8 inference instruction. (CC 6.1+, e.g. GTX 1080)",
        check: |cc| cc >= 61,
    },
    Capability {
        name: "has_pageable_memory_access",
        description: "Pageable memory access: GPU can directly access system (CPU) memory without explicit pinning via cudaMallocHost. (Unified Memory, CC 6.0+)",
        check: |cc| cc >= 60,
    },
    Capability {
        name: "has_memory_pools",
        description: "CUDA Memory Pools: Efficient stream-ordered memory allocation/deallocation via cudaMallocAsync/cudaFreeAsync, reducing allocation overhead. (CC 6.0+)",
        check: |cc| cc >= 60,
    },
    Capability {
        name: "has_p2p_copy",
        description: "Peer-to-Peer copy: Direct GPU-to-GPU memory transfers over NVLink/PCIe without CPU staging. Essential for multi-GPU inference. (CC 6.0+)",
        check: |cc| cc >= 60,
    },
    Capability {
        name: "has_cuda_graphs",
        description: "CUDA Graphs: Capture a sequence of GPU operations and replay them with minimal CPU overhead. Reduces kernel launch latency. (CC 3.0+)",
        check: |cc| cc >= 30,
    },
    // ── Volta/Turing (CC 7.0 – 7.5) ── Tensor Cores era ─────────────────────
    Capability {
        name: "has_wmma",
        description: "WMMA (Warp Matrix Multiply-Accumulate): First-gen Tensor Core API. 16x16x16 matrix ops in hardware. The foundation of fast GEMM. (CC 7.0+)",
        check: |cc| cc >= 70,
    },
    Capability {
        name: "has_wmma_f16",
        description: "WMMA FP16: Tensor Core matrix multiply with FP16 inputs/outputs. (CC 7.0+)",
        check: |cc| cc >= 70,
    },
    Capability {
        name: "has_independent_thread_scheduling",
        description: "Independent thread scheduling: Each thread has its own program counter. Enables fine-grained synchronization & divergent warp execution. (CC 7.0+)",
        check: |cc| cc >= 70,
    },
    Capability {
        name: "has_bf16_conversions",
        description: "BF16 conversions: Hardware-accelerated FP32↔BF16 conversion instructions. Fast type casting for mixed-precision training/inference. (CC 7.0+)",
        check: |cc| cc >= 70,
    },
    // ── Ampere (CC 8.0 – 8.9) ── Precision & efficiency ─────────────────────
    Capability {
        name: "has_sparse_tensor_cores",
        description: "Sparse Tensor Cores: 2:4 structured sparsity — prune 50% of weights and get ~2x speedup on matrix ops with no accuracy loss. (CC 8.0+)",
        check: |cc| cc >= 80,
    },
    Capability {
        name: "has_wmma_bf16",
        description: "WMMA BF16: Tensor Core matrix multiply with BF16 inputs. BF16 has the same exponent range as FP32 (better stability than FP16). (CC 8.0+)",
        check: |cc| cc >= 80,
    },
    Capability {
        name: "has_bf16",
        description: "BF16 native: Full BF16 arithmetic support (add, mul, fma). (CC 8.0+)",
        check: |cc| cc >= 80,
    },
    Capability {
        name: "has_tf32",
        description: "TF32 (TensorFloat-32): 19-bit format (8-bit exp + 10-bit mantissa + sign). Transparent FP32→TF32 on Tensor Cores, ~8x faster than FP32. (CC 8.0+)",
        check: |cc| cc >= 80,
    },
    Capability {
        name: "has_async_copy",
        description: "Async copy: memcpy_async from global to shared memory without blocking the compute pipeline. Key enabler for software pipelining. (CC 8.0+)",
        check: |cc| cc >= 80,
    },
    Capability {
        name: "has_l2_cache_persistence",
        description: "L2 cache persistence: Pin frequently accessed data (e.g. KV cache) in L2. Reduces DRAM bandwidth pressure during inference. (CC 8.0+)",
        check: |cc| cc >= 80,
    },
    Capability {
        name: "has_mbarrier",
        description: "mbarrier: Hardware-assisted asynchronous barriers. Threads can continue working while waiting for async copies to land. Required for Warp Specialization (producer/consumer kernel patterns). (CC 8.0+)",
        check: |cc| cc >= 80,
    },
    // ── Ada Lovelace (CC 8.9) ── FP8 & Transformer Engine ───────────────────
    Capability {
        name: "has_fp8",
        description: "FP8 (E4M3/E5M2): 8-bit floats for inference. ~2x throughput vs FP16 with minimal accuracy loss when combined with per-tensor scaling. (CC 8.9+)",
        check: |cc| cc >= 89,
    },
    Capability {
        name: "has_transformer_engine",
        description: "Transformer Engine: Automatic FP8 mixed-precision for attention & FFN. Handles dynamic scaling and format selection (E4M3 vs E5M2). (CC 8.9+)",
        check: |cc| cc >= 89,
    },
    // ── Hopper (CC 9.0) ── Data logistics revolution ─────────────────────────
    Capability {
        name: "has_tma",
        description: "TMA (Tensor Memory Accelerator): DMA engine that moves N-dimensional tiles between global↔shared memory. Zero CUDA cores used. (CC 9.0+)",
        check: |cc| cc >= 90,
    },
    Capability {
        name: "has_clusters",
        description: "Thread Block Clusters: Group blocks that can cooperate via distributed shared memory. Enables cross-block synchronization. (CC 9.0+)",
        check: |cc| cc >= 90,
    },
    Capability {
        name: "has_distributed_shared_memory",
        description: "Distributed Shared Memory: Blocks within a cluster can directly load/store from each other's shared memory. Ideal for wide attention heads. (CC 9.0+)",
        check: |cc| cc >= 90,
    },
    Capability {
        name: "has_l2_multicast",
        description: "L2 multicast: Broadcast data from L2 cache to multiple SMs simultaneously. Eliminates redundant DRAM reads when broadcasting model weights. (CC 9.0+)",
        check: |cc| cc >= 90,
    },
    Capability {
        name: "has_confidential_computing",
        description: "Confidential Computing: Hardware-enforced memory encryption (AES) on VRAM and PCIe bus. Data remains encrypted even during GPU processing. (CC 9.0+)",
        check: |cc| cc >= 90,
    },
    // ── Blackwell (CC 10.0 / 12.0) ── Density & micro-scaling ────────────────
    Capability {
        name: "has_fp4",
        description: "NVFP4: NVIDIA's native 4-bit float format. Requires strict per-tensor or per-group scaling. Doubles throughput vs FP8. (CC 10.0+)",
        check: |cc| cc >= 100,
    },
    Capability {
        name: "has_mxfp4",
        description: "MXFP4 (OCP Microscaling): Standardized 4-bit format from the Open Compute Project. Uses micro-scaling with per-group (16/32 elements) scale factors. (CC 10.0+)",
        check: |cc| cc >= 100,
    },
    Capability {
        name: "has_microscaling",
        description: "Microscaling: General support for the MX format family (MXFP4, MXFP6, MXFP8, MXINT8). Enables per-group scale factors shared across 16/32 elements. (CC 10.0+)",
        check: |cc| cc >= 100,
    },
    Capability {
        name: "has_tmem",
        description: "TMEM (Tensor Memory): Dedicated on-chip memory directly coupled to Tensor Cores. Replaces register file (RF) usage for TC operands. (CC 10.0+)",
        check: |cc| cc >= 100,
    },
    Capability {
        name: "has_dynamic_sparsity",
        description: "Dynamic Sparsity: Extends the 2:4 structured sparsity (Ampere) with data-dependent sparsity rates that vary per format (FP4/FP8). (CC 10.0+)",
        check: |cc| cc >= 100,
    },
    Capability {
        name: "has_tma_v2",
        description: "TMA v2: Enhanced Tensor Memory Accelerator with queuing & async scheduling primitives not available on SM 120. (CC == 100)",
        check: |cc| cc == 100,
    },
    Capability {
        name: "has_cluster_multicast",
        description: "Cluster Multicast: HW-accelerated broadcast from global memory to shared memory across all blocks in a cluster. Available on B200, but REMOVED from RTX 5090. (CC == 100)",
        check: |cc| cc == 100,
    },
    Capability {
        name: "has_hw_allreduce",
        description: "HW All-Reduce: Hardware-accelerated multi-GPU reduction without using compute cores. Crucial for Tensor Parallelism across 8+ GPUs. (CC == 100)",
        check: |cc| cc == 100,
    },
    Capability {
        name: "has_tma_indirect",
        description: "TMA Indirect: Scatter/gather via TMA — the accelerator can follow pointers to load non-contiguous data tiles. (CC == 100)",
        check: |cc| cc == 100,
    },
    Capability {
        name: "is_data_center_gpu",
        description: "Data center GPU: Distinguishes SM 100 (B200/B100) and SM 90 (H100) from consumer parts. Guards kernels relying on advanced inter-GPU communication. (CC 100 or 90)",
        check: |cc| cc == 100 || cc == 90,
    },
    Capability {
        name: "has_amp",
        description: "AMP (AI Management Processor): Dedicated HW scheduler for AI workloads on consumer Blackwell. Manages AI task queuing independently of the CPU. (CC 12.0+)",
        check: |cc| cc >= 120,
    },
    // ── Emulation & Software Fallbacks ───────────────────────────────────────
    Capability {
        name: "allow_legacy_bf16",
        description: "Enables BF16 emulation/fallback via FP32 for architectures older than Volta (CC 5.3 - 6.x).",
        check: |cc| cc >= 53 && cc < 70,
    },
    Capability {
        name: "allow_legacy_fp8",
        description: "Enables FP8 emulation/fallback via FP16/FP32 for architectures older than Ada Lovelace (CC 5.3 - 6.x).",
        check: |cc| cc >= 53 && cc < 70,
    },
];

pub fn emit_check_cfgs() {
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp8)");
    println!("cargo::rustc-check-cfg=cfg(cuda_arch, values(\"53\", \"61\", \"70\", \"75\", \"80\", \"86\", \"89\", \"90\", \"100\", \"120\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_70, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_61, values(\"true\", \"false\"))");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_53, values(\"true\", \"false\"))");
}

pub fn emit_rustc_cfgs(compute_cap: usize) {
    println!("cargo:rustc-cfg=cuda_arch=\"{}\"", compute_cap);
    println!("cargo:rustc-cfg=inf_cc_70=\"{}\"", compute_cap >= 70);
    println!("cargo:rustc-cfg=inf_cc_61=\"{}\"", compute_cap >= 61);
    println!("cargo:rustc-cfg=inf_cc_53=\"{}\"", compute_cap >= 53);
}

/// Prints a detailed summary of enabled hardware features and their descriptions.
pub fn emit_detailed_feature_summary(compute_cap: usize, results: &[(&'static str, bool)]) {
    p!(
        "─── CUDA Hardware Capability Summary (SM {}) ───",
        compute_cap
    );
    for (name, enabled) in results {
        let cap = CAPABILITIES.iter().find(|c| c.name == *name).unwrap();
        if *enabled {
            p!("[X] {:<30} | {}", cap.name, cap.description);
        } else {
            p!("[ ] {:<30} | {}", cap.name, cap.description);
        }
    }
    p!("──────────────────────────────────────────────────");
}

/// Emits all registrations and returns a list of results for each capability.
pub fn get_capabilities_results(compute_cap: usize) -> Vec<(&'static str, bool)> {
    println!("cargo::rerun-if-env-changed=ALLOW_LEGACY");
    emit_check_cfgs();
    emit_rustc_cfgs(compute_cap);

    // Parse ALLOW_LEGACY environment variable
    let allow_legacy = std::env::var("ALLOW_LEGACY").unwrap_or_else(|_| "all".to_string());
    let allow_legacy = allow_legacy.to_lowercase();
    let permitted: Vec<&str> = allow_legacy.split(',').map(|s| s.trim()).collect();
    let allow_all = permitted.contains(&"all");

    let results: Vec<(&'static str, bool)> = CAPABILITIES
        .iter()
        .map(|cap| {
            let mut enabled = (cap.check)(compute_cap);

            // Filter legacy features if they are enabled by CC
            if enabled && cap.name.starts_with("allow_legacy_") {
                let feature_type = &cap.name["allow_legacy_".len()..];
                if !allow_all && !permitted.contains(&feature_type) {
                    enabled = false;
                }
            }
            (cap.name, enabled)
        })
        .collect();

    emit_detailed_feature_summary(compute_cap, &results);
    results
}
