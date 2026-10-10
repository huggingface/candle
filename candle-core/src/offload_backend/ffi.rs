//! libomptarget bindings and transfer/launch accounting.
//!
//! `std::offload` exposes exactly four public items and none of them allocate
//! device memory, so a backend cannot be built on the safe API alone (G1).
//! Everything here is the runtime capability that is already present underneath
//! `offload!` but has no Rust surface: allocate, free, copy, enumerate.
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

// The OpenMP device API is split across two libraries, which is easy to get
// wrong: `omp_target_*` are in libomptarget, but `omp_get_*_device` and
// `omp_get_num_devices` are in libomp. A single `#[link(name = "omptarget")]`
// block appears to work only because `-Zoffload` implicitly adds `-lomp` --
// take the offload flag away, as this crate does, and the omp_get_* symbols go
// undefined at link time.
#[link(name = "omptarget")]
extern "C" {
    pub fn omp_target_alloc(size: usize, device_num: i32) -> *mut c_void;
    pub fn omp_target_free(ptr: *mut c_void, device_num: i32);
    pub fn omp_target_memcpy(
        dst: *mut c_void,
        src: *mut c_void,
        length: usize,
        dst_offset: usize,
        src_offset: usize,
        dst_device: i32,
        src_device: i32,
    ) -> i32;
}

#[link(name = "omp")]
extern "C" {
    pub fn omp_get_default_device() -> i32;
    pub fn omp_get_initial_device() -> i32;
    pub fn omp_get_num_devices() -> i32;
}

/// Counters for the two things this project is measuring: how much data crosses
/// the PCIe boundary, and how many kernels are launched to do it.
pub struct Stats {
    pub h2d_bytes: AtomicU64,
    pub d2h_bytes: AtomicU64,
    pub h2d_calls: AtomicU64,
    pub d2h_calls: AtomicU64,
    pub launches: AtomicU64,
    pub allocs: AtomicU64,
    pub alloc_bytes: AtomicU64,
    pub frees: AtomicU64,
    /// Ops that ran on the GPU.
    pub gpu_ops: AtomicU64,
    /// Ops that had to round-trip through the CPU backend.
    pub fallback_ops: AtomicU64,
}

pub static STATS: Stats = Stats {
    h2d_bytes: AtomicU64::new(0),
    d2h_bytes: AtomicU64::new(0),
    h2d_calls: AtomicU64::new(0),
    d2h_calls: AtomicU64::new(0),
    launches: AtomicU64::new(0),
    allocs: AtomicU64::new(0),
    alloc_bytes: AtomicU64::new(0),
    frees: AtomicU64::new(0),
    gpu_ops: AtomicU64::new(0),
    fallback_ops: AtomicU64::new(0),
};

/// Which ops fell back, and how often. Kept separately from the atomic counters
/// because it needs a map and is only read at report time.
pub static FALLBACKS: std::sync::Mutex<
    Option<std::collections::BTreeMap<&'static str, (u64, u64)>>,
> = std::sync::Mutex::new(None);

/// Total bytes moved across the bus so far, in both directions. Sampling this
/// before and after a fallback attributes the traffic to the op that caused it,
/// which is the number that actually matters -- a fallback on a 256 MiB
/// embedding table costs far more than one on a 4 KiB bias.
pub fn bus_bytes() -> u64 {
    STATS.h2d_bytes.load(Ordering::Relaxed) + STATS.d2h_bytes.load(Ordering::Relaxed)
}

pub fn note_fallback(op: &'static str, bytes: u64) {
    STATS.fallback_ops.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut g) = FALLBACKS.lock() {
        let e = g.get_or_insert_with(Default::default).entry(op).or_insert((0, 0));
        e.0 += 1;
        e.1 += bytes;
    }
}

#[inline]
pub fn note_launch() {
    STATS.launches.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn note_gpu_op() {
    STATS.gpu_ops.fetch_add(1, Ordering::Relaxed);
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

/// Human-readable dump of everything the counters saw. Called by the examples.
pub fn report() -> String {
    use std::fmt::Write;
    let s = &STATS;
    let ld = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let mut o = String::new();
    let _ = writeln!(o, "offload backend: data transfer and kernel launches");
    let _ = writeln!(
        o,
        "  host -> device   {:>10} calls  {:>12.2} MiB",
        ld(&s.h2d_calls),
        mib(ld(&s.h2d_bytes))
    );
    let _ = writeln!(
        o,
        "  device -> host   {:>10} calls  {:>12.2} MiB",
        ld(&s.d2h_calls),
        mib(ld(&s.d2h_bytes))
    );
    let _ = writeln!(
        o,
        "  device allocs    {:>10} calls  {:>12.2} MiB   ({} frees)",
        ld(&s.allocs),
        mib(ld(&s.alloc_bytes)),
        ld(&s.frees)
    );
    let _ = writeln!(o, "  kernel launches  {:>10}", ld(&s.launches));
    let total_ops = ld(&s.gpu_ops) + ld(&s.fallback_ops);
    let pct = if total_ops == 0 {
        0.0
    } else {
        100.0 * ld(&s.gpu_ops) as f64 / total_ops as f64
    };
    let _ = writeln!(
        o,
        "  ops on GPU       {:>10} / {} ({:.1}%)",
        ld(&s.gpu_ops),
        total_ops,
        pct
    );
    if let Ok(g) = FALLBACKS.lock() {
        if let Some(map) = g.as_ref() {
            let mut v: Vec<_> = map.iter().collect();
            v.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
            let _ = writeln!(
                o,
                "  CPU fallbacks by op (each is a D2H + H2D round trip):{:>21}{:>14}",
                "calls", "MiB moved"
            );
            for (name, (count, bytes)) in v.iter().take(20) {
                let _ = writeln!(o, "      {:<38} {:>10} {:>13.2}", name, count, mib(*bytes));
            }
        }
    }
    o
}
