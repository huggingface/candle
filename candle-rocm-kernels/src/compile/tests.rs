use super::{cache, KernelCache, ModuleName};
use std::sync::Arc;

/// A source this crate does not own, written in the same dialect as
/// `candle-kernels/src/*.cu`. The `#include` is the point of the test as
/// much as the kernel is: a downstream module has to reach candle's staged
/// headers, not just the shim.
const CUSTOM: &str = r#"
#include "cuda_utils.cuh"

extern "C" __global__ void custom_scale_f32(const float *x, float *y, const size_t n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] * 3.0f;
    }
}
"#;

/// A revision of [`CUSTOM`] under a different symbol, so a stale module is
/// caught by the resolve rather than by a wrong-kernel launch.
const CUSTOM_V2: &str = r#"
#include "cuda_utils.cuh"

extern "C" __global__ void custom_scale_v2_f32(const float *x, float *y, const size_t n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] * 5.0f;
    }
}
"#;

fn custom_key(name: &str, source: &str) -> ModuleName {
    ModuleName::Custom {
        name: name.to_string(),
        source: cache::source_digest(source),
    }
}

/// The in-memory key has to separate two sources sharing a name — the same
/// statement `a_revised_source_is_not_shadowed_by_the_name_it_reuses` makes
/// end to end, but without a GPU, so it runs everywhere.
#[test]
fn a_custom_key_separates_two_sources_under_one_name() {
    assert_eq!(custom_key("gdn", CUSTOM), custom_key("gdn", CUSTOM));
    assert_ne!(custom_key("gdn", CUSTOM), custom_key("gdn", CUSTOM_V2));
    assert_ne!(custom_key("gdn", CUSTOM), custom_key("other", CUSTOM));
}

/// A caller-supplied source has to compile and resolve through the same
/// cache the built-ins use — the whole point of the custom entry point.
#[test]
fn a_custom_source_compiles_and_resolves() {
    let cache = match KernelCache::new(0) {
        Ok(cache) => cache,
        // No ROCm device on this machine.
        Err(_) => return,
    };
    cache
        .custom_function("candle_rocm_kernels_test", CUSTOM, "custom_scale_f32")
        .expect("custom module failed to compile");
    // Second call is the in-memory hit.
    cache
        .custom_function("candle_rocm_kernels_test", CUSTOM, "custom_scale_f32")
        .expect("custom module failed to resolve from cache");
}

/// The same module name, a revised source. Anything keyed on the name alone
/// hands the second call the *first* module: the kernel it asks for is
/// missing, or worse, still there and stale.
#[test]
fn a_revised_source_is_not_shadowed_by_the_name_it_reuses() {
    let cache = match KernelCache::new(0) {
        Ok(cache) => cache,
        Err(_) => return,
    };
    cache
        .custom_function("candle_rocm_kernels_revision", CUSTOM, "custom_scale_f32")
        .expect("custom module failed to compile");
    cache
        .custom_function(
            "candle_rocm_kernels_revision",
            CUSTOM_V2,
            "custom_scale_v2_f32",
        )
        .expect("the revised source was shadowed by the module it replaced");
    // The first source keeps resolving: both revisions stay resident, so a
    // handle taken before the revision is still launchable.
    cache
        .custom_function("candle_rocm_kernels_revision", CUSTOM, "custom_scale_f32")
        .expect("the first revision was evicted");
}

/// Custom modules live in their own namespace. Keying both kinds by a bare
/// string would hand a downstream crate that named its module `unary`
/// candle's `unary` instead — a wrong-kernel launch, not an error.
#[test]
fn a_custom_module_does_not_shadow_a_built_in_of_the_same_name() {
    let cache = match KernelCache::new(0) {
        Ok(cache) => cache,
        Err(_) => return,
    };
    // Load the custom one first, so a shared namespace would poison the
    // built-in lookup rather than the other way round.
    cache
        .custom_function("unary", CUSTOM, "custom_scale_f32")
        .expect("custom module failed to compile");
    cache
        .function(&crate::UNARY, "ucopy_f32")
        .expect("built-in `unary` was shadowed by the custom module");
    // And the reverse: the built-in must not satisfy the custom lookup.
    assert!(cache.custom_function("unary", CUSTOM, "ucopy_f32").is_err());
}

/// Dropping the outer `Mutex<KernelCache>` the ROCm backend used to hold
/// exposed two races on the compile-and-load path, both of which faulted the
/// GPU rather than failing cleanly: two threads writing the same
/// process-named hipcc intermediates, and the loser of an insert race
/// `hipModuleUnload`ing a module whose functions the caller had already
/// resolved. Ten threads racing on one module is what reproduced them.
#[test]
fn concurrent_first_use_of_a_module_is_safe() {
    let cache = match KernelCache::new(0) {
        Ok(cache) => Arc::new(cache),
        // No ROCm device on this machine.
        Err(_) => return,
    };
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let cache = cache.clone();
            std::thread::spawn(move || {
                cache
                    .function(&crate::UNARY, "ucopy_f32")
                    .map(|f| f.as_raw() as usize)
            })
        })
        .collect();

    let resolved: Vec<usize> = handles
        .into_iter()
        .map(|h| h.join().expect("thread panicked").expect("resolve failed"))
        .collect();
    // One module, one symbol: every thread must have landed on the same
    // handle, which is only true if exactly one load won.
    assert!(resolved.windows(2).all(|w| w[0] == w[1]), "{resolved:?}");
}

/// A unit's compile gate is scaffolding for the compile, not for the launches
/// after it: once the module is resident the gate is dead weight, and an
/// autotuner generating a module per shape would otherwise retain one lock per
/// shape for the life of the process.
#[test]
fn a_compile_gate_is_released_once_its_module_is_resident() {
    let cache = match KernelCache::new(0) {
        Ok(cache) => cache,
        Err(_) => return,
    };
    cache
        .function(&crate::UNARY, "ucopy_f32")
        .expect("built-in module failed to compile");
    assert!(cache
        .compiling
        .lock()
        .expect("gate map is poisoned")
        .is_empty());
}
