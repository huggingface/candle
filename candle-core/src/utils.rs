//! Useful functions for checking features.
use std::str::FromStr;
use std::sync::OnceLock;

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        _ => default_num_threads(),
    }
}

fn default_num_threads() -> usize {
    let physical = {
        #[cfg(target_os = "macos")]
        {
            perf_core_count().unwrap_or_else(num_cpus::get_physical)
        }
        #[cfg(not(target_os = "macos"))]
        {
            num_cpus::get_physical()
        }
    };
    let physical = physical.max(1); // safeguard against bad number
    if physical <= 4 {
        physical
    } else {
        // NOTE: When the CPU backend is optimized further so as not to trash shared
        // caches etc the amount of threads can be increased.
        physical.min(physical / 2)
    }
}

/// On Apple Silicon: P-core count via `hw.perflevel0.logicalcpu`.
/// Returns None on Intel Macs (key absent) or on error.
#[cfg(target_os = "macos")]
fn perf_core_count() -> Option<usize> {
    use std::os::raw::c_void;
    let mut count: u32 = 0;
    let mut size = std::mem::size_of::<u32>();
    let ret = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr().cast(),
            &mut count as *mut u32 as *mut c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (ret == 0 && count > 0).then_some(count as usize)
}

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

pub(crate) fn candle_pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(get_num_threads())
            .start_handler(|_| set_thread_affinity())
            .build()
            .expect("failed to build candle rayon threadpool")
    })
}

#[cfg(target_os = "macos")]
/// Elevate the thread QoS so macOS prefers running on Performance (P) cores.
fn set_thread_affinity() {
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
#[inline(always)]
fn set_thread_affinity() {
    // On non‑macOS platforms we currently leave thread affinity untouched.
}

pub(crate) fn with_threadpool<F: FnOnce() -> R + Send, R: Send>(f: F) -> R {
    candle_pool().install(f)
}

pub fn has_accelerate() -> bool {
    cfg!(feature = "accelerate")
}

pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

pub fn with_avx() -> bool {
    cfg!(target_feature = "avx2")
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}
