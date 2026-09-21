//! Matmul through the wax compiler.
//!
//! [`call_wax_gemm`] mirrors [`crate::call_mlx_gemm`]'s signature.
//! It uses the standard [`wax_kernels::gemm`] specification, and lowers via
//! [`wax_metal::gemm::lower_gemm`].
//! If lowering fails we fall back to [`crate::call_mlx_gemm`].

use crate::metal::ComputePipeline;
use crate::utils::EncoderProvider;
use crate::{Buffer, ComputeCommandEncoder, Device, MetalKernelError};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};
use wax_ir::CompileKey;

/// Unique key for a GEMM kernel at runtime.
///
/// The module dsl is not unique on it's own. We also have to include the runtime parameters,
/// like `m`, `n`, etc.
///
/// The key does not describe what the leading dimension of the dimension is - it is always `n`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct PlanKey {
    dtype: crate::GemmDType,
    m: usize,
    n: usize,
    k: usize,
    lda: i32,
    ldb: i32,
    trans_b: bool,
}

/// Dispatch specification ready to launch.
struct Dispatch {
    pipeline: ComputePipeline,
    launch: wax_metal::Launch,
    /// Promoted extents, bound with `setBytes` at `3 + i`.
    scalars: Vec<i32>,
    role: BufferRole,
}

/// The role a buffer has in a GEMM.
/// Primary { false } -> standard one-pass GEMM.
/// Primary { true }  -> first stage of two-pass GEMM. Writes to scratch.
/// Reduce            -> second stage of two-pass GEMM. Reduces to dst.
enum BufferRole {
    Primary { dest_is_scratch: bool },
    Reduce,
}

impl BufferRole {
    fn apply(
        &self,
        enc: &crate::metal::encoder::ComputeCommandEncoder,
        scratch: Option<&crate::metal::Buffer>,
        lhs: &crate::metal::Buffer,
        rhs: &crate::metal::Buffer,
        out: &crate::metal::Buffer,
    ) {
        match self {
            BufferRole::Primary { dest_is_scratch } => {
                if *dest_is_scratch {
                    if let Some(s) = scratch {
                        enc.set_output_buffer(0, Some(s), 0);
                    }
                } else {
                    enc.set_output_buffer(0, Some(out), 0);
                }
                enc.set_input_buffer(1, Some(lhs), 0);
                enc.set_input_buffer(2, Some(rhs), 0);
            }
            BufferRole::Reduce => {
                enc.set_output_buffer(0, Some(out), 0);
                if let Some(s) = scratch {
                    enc.set_input_buffer(1, Some(s), 0);
                }
            }
        }
    }
}

/// A GEMM plan ready to launch.
struct Plan {
    dispatches: Vec<Dispatch>,
    two_pass: bool,
}

impl Plan {
    fn is_two_pass(&self) -> bool {
        self.two_pass
    }
}

fn cache() -> &'static Mutex<HashMap<PlanKey, Arc<Plan>>> {
    static C: OnceLock<Mutex<HashMap<PlanKey, Arc<Plan>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Cached wax IR gemm module.
fn gemm_module(key: &CompileKey) -> Result<Rc<wax_ir::Module>, MetalKernelError> {
    wax_ir::cached_module(key, || {
        let strides: Vec<(&str, &[i32])> = key
            .strides
            .iter()
            .map(|(name, st)| (name.as_str(), st.as_slice()))
            .collect();
        wax_frontend::KernelCompiler::new(
            wax_kernels::gemm::module_ast,
            wax_kernels::gemm::MODULE_NAME,
            key.entry,
        )
        .generics(key.generics.clone())
        .strides(&strides)
        .compile()
        .map_err(|e| MetalKernelError::LoadLibraryError(format!("gemm compile: {e:?}")))?
        .to_wax_module(key.entry)
        .map_err(|e| MetalKernelError::LoadLibraryError(format!("gemm module: {e}")))
    })
}

fn dtype_tag(dtype: crate::GemmDType) -> &'static str {
    match dtype {
        crate::GemmDType::F32 => "f32",
        crate::GemmDType::F16 => "f16",
        crate::GemmDType::BF16 => "bf16",
    }
}

/// Compile kernel and build pipeline for the provided plan.
fn ready(device: &Device, key: &PlanKey) -> Result<Arc<Plan>, MetalKernelError> {
    if let Some(s) = cache().lock().unwrap().get(key) {
        return Ok(s.clone());
    }
    let spec = wax_metal::gemm::GemmSpec {
        trans_b: key.trans_b,
        m: key.m,
        n: key.n,
        k: key.k,
        elem_bytes: if matches!(key.dtype, crate::GemmDType::F32) {
            4
        } else {
            2
        },
        ldc: key.n,
    };
    let Some((bm, bn, bk)) = wax_metal::gemm::module_tile(&spec) else {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "wax gemm: K={} is not a multiple of any K block the tile loop accepts",
            key.k,
        )));
    };
    // The plain store is only correct when the tiling divides the shape. Padded uses masked
    // stores.
    // CompileKey is made up of the entire input to the frontend. M and N etc are provided at
    // runtime, and thus are not part of the key. This means one wax gemm module serves every
    // combination of M and N.
    let padded = !key.m.is_multiple_of(bm) || !key.n.is_multiple_of(bn);
    let ckey = CompileKey {
        source_hash: wax_kernels::gemm::SOURCE_HASH,
        module: wax_kernels::gemm::MODULE_NAME,
        entry: if padded {
            wax_kernels::gemm::ENTRY_PADDED
        } else {
            wax_kernels::gemm::ENTRY
        },
        // <E, BM, BN, BK, K> for both entries.
        generics: vec![
            dtype_tag(key.dtype).to_string(),
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
            key.k.to_string(),
        ],
        // `z` is dst, packed. Leading dim is `n`.
        // `[1, ld]` is read as `Layout::Transposed` by `gemm_pattern::view_facts`. Recogniser
        // derives `trans_b` from the module.
        strides: vec![
            ("z".into(), vec![key.n as i32, 1]),
            ("x".into(), vec![key.lda, 1]),
            (
                "y".into(),
                if key.trans_b {
                    vec![1, key.ldb]
                } else {
                    vec![key.ldb, 1]
                },
            ),
        ],
    };
    let profile = wax_metal::TargetProfile::host();
    // Cache key for `wax_cache`.
    let cache_key = format!(
        "{}|{ckey:?}|{spec:?}|{profile:?}",
        crate::wax_cache::EMITTER
    )
    .into_bytes();

    let cached = crate::wax_cache::load(&cache_key);
    let hit = cached.is_some();
    let kernel = match cached {
        Some(k) => k,
        None => {
            let module = gemm_module(&ckey)?;
            let lowering = wax_metal::gemm::lower_gemm(&module, &spec, &profile).map_err(|e| {
                MetalKernelError::LoadLibraryError(format!("wax gemm lowering: {e:?}"))
            })?;
            if std::env::var_os("CANDLE_WAX_TRACE").is_some() {
                eprintln!(
                    "[wax-gemm] m={} n={} k={} {} transB={} padded={} -> {:?} ({:?}){}",
                    key.m,
                    key.n,
                    key.k,
                    dtype_tag(key.dtype),
                    key.trans_b,
                    padded,
                    lowering.wanted,
                    lowering.reason,
                    lowering
                        .fell_back
                        .map(|w| format!(" - fell back: {w}"))
                        .unwrap_or_default()
                );
            }
            crate::wax_cache::store(&cache_key, &lowering.kernel);
            lowering.kernel
        }
    };
    if hit && std::env::var_os("CANDLE_WAX_TRACE").is_some() {
        eprintln!(
            "[wax-gemm] m={} n={} k={} {} transB={} padded={} -> {:?} (cached)",
            key.m,
            key.n,
            key.k,
            dtype_tag(key.dtype),
            key.trans_b,
            padded,
            wax_metal::gemm::route(&spec).plan
        );
    }
    let k = &kernel;
    let two_pass = k.second.is_some();
    let mut dispatches = vec![Dispatch {
        pipeline: crate::kernel::load_pipeline_from_metallib(device, &k.metallib, &k.entry)?,
        launch: k.launch,
        scalars: k.scalar_args.clone(),
        role: BufferRole::Primary {
            dest_is_scratch: two_pass,
        },
    }];
    if let Some(second) = &k.second {
        dispatches.push(Dispatch {
            pipeline: crate::kernel::load_pipeline_from_metallib(
                device,
                &k.metallib,
                &second.entry,
            )?,
            launch: second.launch,
            scalars: second.scalar_args.clone(),
            role: BufferRole::Reduce,
        });
    }
    let plan = Arc::new(Plan {
        dispatches,
        two_pass,
    });
    cache().lock().unwrap().insert(key.clone(), plan.clone());
    Ok(plan)
}

/// Computes `C = A @ B`. The specific implementation is chosen by the compiler from the shape.
///
/// Cases not yet supported by [`wax_metal::gemm::lower_gemm`] fall back to
/// [`crate::call_mlx_gemm`]:
///
/// - `b > 1`: `GemmSpec` does not model a batch dimension.
/// - non-zero offset: no expression in the kernel's views.
/// - an operand that is not row-major and not a plain transpose, or a transposed left
///   operand: rejected rather than guess a leading dimension.
/// - K not divisble by the kernel's K-block.
#[allow(clippy::too_many_arguments)]
pub fn call_wax_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    dtype: crate::GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let unsupported = |msg: &str| {
        // Use `CANDLE_WAX_TRACE=1` to enable wax tracing
        if std::env::var_os("CANDLE_WAX_TRACE").is_some() {
            eprintln!("[wax-gemm] decline {b}x{m}x{n}x{k} {dtype:?}: {msg}");
        }
        Err(MetalKernelError::LoadLibraryError(format!(
            "wax gemm: {msg}"
        )))
    };
    // `CANDLE_WAX_GEMM=0` to disable wax and always use MLX gemm.
    if std::env::var("CANDLE_WAX_GEMM").ok().as_deref() == Some("0") {
        return unsupported("disabled by CANDLE_WAX_GEMM=0");
    }
    if b != 1 {
        return unsupported("batched matmul not supported");
    }
    if lhs_offset != 0 || rhs_offset != 0 {
        return unsupported("a non-zero buffer offset has no expression in the kernel's views");
    }

    // Get leading dimension
    // `[.., ld, 1]` == row major.
    // `[.., 1, ld]` == transposed.
    let ld = |st: &[usize]| -> Option<(i32, bool)> {
        match st {
            [.., a, 1] => Some((*a as i32, false)),
            [.., 1, a] => Some((*a as i32, true)),
            _ => None,
        }
    };
    let (Some((lda, a_trans)), Some((ldb, trans_b))) = (ld(lhs_stride), ld(rhs_stride)) else {
        return unsupported("operands are not row-major nor a plain transpose");
    };
    if a_trans {
        return unsupported("transposed left operand has no expression in the kernel's views");
    }
    let key = PlanKey {
        dtype,
        m,
        n,
        k,
        lda,
        ldb,
        trans_b,
    };
    let plan = ready(device, &key)?;
    // Two-pass (split-K) writes f32 partials which the the reduce pass then sums.
    // For this we need to allocate a scratch buffer.
    let scratch = if plan.is_two_pass() {
        let elems = split_k_scratch_elems(m, n, k, dtype, trans_b).ok_or(
            MetalKernelError::LoadLibraryError(
                "wax gemm: the two-pass plan has no scratch size".to_string(),
            ),
        )?;
        Some(device.new_buffer(elems * 4, crate::RESOURCE_OPTIONS)?)
    } else {
        None
    };

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    // The plan's dispatches launched in order.
    for dispatch in &plan.dispatches {
        encoder.set_compute_pipeline_state(&dispatch.pipeline);
        dispatch
            .role
            .apply(encoder, scratch.as_ref(), lhs_buffer, rhs_buffer, output);
        for (i, v) in dispatch.scalars.iter().enumerate() {
            encoder.set_bytes(3 + i, v);
        }
        let g = dispatch.launch.grid;
        let t = dispatch.launch.threadgroup;
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: g.0,
                height: g.1,
                depth: g.2,
            },
            objc2_metal::MTLSize {
                width: t.0,
                height: t.1,
                depth: t.2,
            },
        );
    }
    Ok(())
}

/// Size of the scratch buffer required for split-k GEMM.
pub fn split_k_scratch_elems(
    m: usize,
    n: usize,
    k: usize,
    dtype: crate::GemmDType,
    trans_b: bool,
) -> Option<usize> {
    use wax_metal::gemm::{route, GemmSpec, Plan as GemmPlan};
    let spec = GemmSpec {
        trans_b,
        m,
        n,
        k,
        elem_bytes: if matches!(dtype, crate::GemmDType::F32) {
            4
        } else {
            2
        },
        ldc: n,
    };
    // `splits > 1` implies split-k approach.
    match route(&spec).plan {
        GemmPlan::Lower { splits, .. } if splits > 1 => Some(splits as usize * n),
        // MPP split-K uses the whole `m * n` matrix in f32 as scratch buffer.
        // Technically an over-estimation, but it's decided by MPP - not us.
        GemmPlan::Mpp { splits, .. } if splits > 1 => Some(splits as usize * m * n),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile_key(padded: bool, trans_b: bool) -> CompileKey {
        let y = if trans_b { vec![1, 32] } else { vec![32, 1] };
        CompileKey {
            source_hash: wax_kernels::gemm::SOURCE_HASH,
            module: wax_kernels::gemm::MODULE_NAME,
            entry: if padded {
                wax_kernels::gemm::ENTRY_PADDED
            } else {
                wax_kernels::gemm::ENTRY
            },
            generics: vec![
                "f16".into(),
                "16".into(),
                "32".into(),
                "8".into(),
                "64".into(),
            ],
            strides: vec![
                ("z".into(), vec![32, 1]),
                ("x".into(), vec![64, 1]),
                ("y".into(), y),
            ],
        }
    }

    #[test]
    fn all_wax_gemms_compile() {
        for &padded in &[false, true] {
            for &trans_b in &[false, true] {
                gemm_module(&compile_key(padded, trans_b)).expect("compiles");
            }
        }
    }

    #[test]
    fn gemm_module_caches_correctly() {
        let k = compile_key(false, false);
        let a = gemm_module(&k).expect("compiles");
        let again = gemm_module(&k).expect("hit");
        assert_eq!(
            Rc::as_ptr(&a),
            Rc::as_ptr(&again),
            "module should not recompile"
        );
        let b = gemm_module(&compile_key(true, false)).expect("compiles");
        assert_ne!(Rc::as_ptr(&a), Rc::as_ptr(&b), "module keys must differ");
    }
}
