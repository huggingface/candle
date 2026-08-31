//! Useful functions for checking features.
use std::any::Any;
use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::Thread;

/// Per-worker control on its own 64-byte cache line to prevent false sharing.
#[repr(C, align(64))]
struct Slot {
    /// `Release`-stored by parent to signal work. `Acquire-loaded by worker.
    go: AtomicUsize,
    /// `Release`-stored by worker (and its subtree) when done. `Acquire`-loaded by parent.
    done: AtomicUsize,
}

/// Closure pointer written by main before the `Release` on each root slot's `go`.
struct WorkDesc {
    trampoline: unsafe fn(*const (), usize),
    data: *const (),
}

struct BarrierPoolInner {
    slots: Box<[Slot]>,
    /// Thread handles for unpark.
    handles: OnceLock<Box<[Thread]>>,
    work: UnsafeCell<WorkDesc>,
    stop: AtomicBool,
    panic_payload: Mutex<Option<Box<dyn Any + Send>>>,
}

unsafe impl Sync for BarrierPoolInner {}
unsafe impl Send for BarrierPoolInner {}

/// Two-level tree barrier pool.
///
/// Main signals one leader per CPU cluster (O(n_clusters) stores, not O(n_workers)).
/// Leaders fan work out to intra-cluster followers while main executes its own
/// slice. On-completion reduction mirrors the fan-out: followers -> leader -> main.
/// Main drains only n_cluster slots instead of n_worker slots on completion.
pub struct BarrierPool {
    inner: Box<BarrierPoolInner>,
    threads: Vec<std::thread::JoinHandle<()>>,
    /// Workers that main signals directly
    root_workers: Vec<usize>,
    /// Serializes concurrent callers. Stores the current generation.
    call_lock: Mutex<usize>,
}

unsafe impl Sync for BarrierPool {}
unsafe impl Send for BarrierPool {}

unsafe fn call_trampoline<F: Fn(usize)>(data: *const (), tid: usize) {
    (*(data as *const F))(tid);
}

/// Build the two-level fan-out tree. First worker in each cluster is leader.
fn compute_tree(n_workers: usize, cluster_size: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut children = vec![vec![]; n_workers];
    if n_workers == 0 || cluster_size <= 1 || cluster_size == usize::MAX {
        // Flat: every worker is a root.
        return (children, (0..n_workers).collect());
    }

    let mut roots = Vec::new();

    // First cluster: main + (cluster_size - 1) workers.
    let c0 = (cluster_size - 1).min(n_workers);
    roots.push(0);
    for f in 1..c0 {
        children[0].push(f);
    }

    // Remaining clusters: cluster_size workers each.
    let mut next = c0;
    while next < n_workers {
        let leader = next;
        roots.push(leader);
        let end = n_workers.min(next + cluster_size);
        for f in (leader + 1)..end {
            children[leader].push(f);
        }
        next = end;
    }

    (children, roots)
}

fn get_cluster_size() -> usize {
    // Later we should use `sysctlbyname("hw.perflevel0.cpusperl2")` and similar for other systems to get precise cluster size.
    // For now we use these hardcoded numbers.
    // Using `usize::MAX` simply means that the tree is flat. Every core is it's own leader.
    #[cfg(target_os = "macos")]
    return 6;
    #[cfg(not(target_os = "macos"))]
    usize::MAX
}

impl BarrierPool {
    fn new(n_workers: usize) -> Self {
        let inner = Box::new(BarrierPoolInner {
            slots: (0..n_workers)
                .map(|_| Slot {
                    go: AtomicUsize::new(0),
                    done: AtomicUsize::new(0),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            handles: OnceLock::new(),
            work: UnsafeCell::new(WorkDesc {
                trampoline: |_, _| {},
                data: std::ptr::null(),
            }),
            stop: AtomicBool::new(false),
            panic_payload: Mutex::new(None),
        });

        let inner_ptr = &*inner as *const BarrierPoolInner as usize;
        let cluster_size = get_cluster_size();
        let (children, root_workers) = compute_tree(n_workers, cluster_size);

        let threads: Vec<_> = (0..n_workers)
            .map(|tid| {
                let children = children[tid].clone();
                std::thread::Builder::new()
                    .name(format!("candle-bp-{tid}"))
                    .spawn(move || {
                        set_thread_affinity();
                        let inner = unsafe { &*(inner_ptr as *const BarrierPoolInner) };
                        let slot = &inner.slots[tid];
                        let mut gen = 0usize;

                        const SPIN_LIMIT: u32 = 10_000;
                        loop {
                            // Spin briefly then park, waiting for next work item.
                            let mut spins = 0u32;
                            loop {
                                let g = slot.go.load(Ordering::Acquire);
                                if g != gen {
                                    gen = g;
                                    break;
                                }
                                if inner.stop.load(Ordering::Relaxed) {
                                    return;
                                }
                                spins += 1;
                                if spins < SPIN_LIMIT {
                                    spin_loop();
                                } else {
                                    // Park deposits a token; unpark() before park() returns immediately — no race.
                                    std::thread::park();
                                    spins = 0;
                                }
                            }
                            if inner.stop.load(Ordering::Relaxed) {
                                break;
                            }

                            // Fan work out to children.
                            let handles = inner.handles.get().unwrap();
                            for &child in &children {
                                inner.slots[child].go.store(gen, Ordering::Release);
                                handles[child].unpark();
                            }

                            // Execute own work, catching panics so done is always signalled.
                            if let Err(payload) =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let work = unsafe { &*inner.work.get() };
                                    unsafe { (work.trampoline)(work.data, tid) };
                                }))
                            {
                                let mut p = inner.panic_payload.lock().unwrap();
                                if p.is_none() {
                                    *p = Some(payload);
                                }
                            }

                            // Reduce: wait for all children before signalling parent.
                            for &child in &children {
                                while inner.slots[child].done.load(Ordering::Acquire) != gen {
                                    spin_loop();
                                }
                            }

                            // Signal done unconditionally so main/parent never deadlocks.
                            slot.done.store(gen, Ordering::Release);
                        }
                    })
                    .expect("failed to spawn candle barrier pool worker")
            })
            .collect();

        inner
            .handles
            .set(
                threads
                    .iter()
                    .map(|jh| jh.thread().clone())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
            .ok();
        BarrierPool {
            inner,
            threads,
            root_workers,
            call_lock: Mutex::new(0),
        }
    }

    #[inline]
    pub fn n_workers(&self) -> usize {
        self.threads.len()
    }
}

/// RAII guard that blocks until all root workers have finished.
struct BarrierGuard<'a> {
    inner: &'a BarrierPoolInner,
    roots: &'a [usize],
    gen: usize,
}

impl Drop for BarrierGuard<'_> {
    fn drop(&mut self) {
        for &root in self.roots {
            while self.inner.slots[root].done.load(Ordering::Acquire) != self.gen {
                spin_loop();
            }
        }
        // Always clear any stored panic. Propagate unless main is not already unwinding.
        let payload = self.inner.panic_payload.lock().unwrap().take();
        if !std::thread::panicking() {
            if let Some(p) = payload {
                std::panic::resume_unwind(p);
            }
        }
    }
}

impl BarrierPool {
    pub fn execute<F: Fn(usize) + Sync>(&self, f: F) {
        let n = self.threads.len();
        if n == 0 {
            f(0);
            return;
        }

        let mut guard = self.call_lock.lock().unwrap();
        let new_gen = guard.wrapping_add(1);
        *guard = new_gen;

        unsafe {
            let work = &mut *self.inner.work.get();
            work.trampoline = call_trampoline::<F>;
            work.data = &f as *const F as *const ();
        }

        let handles = self.inner.handles.get().unwrap();
        for &root in &self.root_workers {
            self.inner.slots[root].go.store(new_gen, Ordering::Release);
            handles[root].unpark();
        }

        // Guard waits for workers and propagates panics on drop.
        let _guard = BarrierGuard {
            inner: &self.inner,
            roots: &self.root_workers,
            gen: new_gen,
        };

        // Main thread participates as worker n (workers are 0..n-1).
        f(n);
        // _guard drops here, waiting for root workers.
    }
}

impl Drop for BarrierPool {
    fn drop(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        self.inner.stop.store(true, Ordering::Relaxed);
        let new_gen = self.call_lock.lock().unwrap().wrapping_add(1);
        let handles = self.inner.handles.get();
        // Signal ALL slots and unpark them so parked workers wake immediately.
        for (i, slot) in self.inner.slots.iter().enumerate() {
            slot.go.store(new_gen, Ordering::Release);
            if let Some(hs) = handles {
                hs[i].unpark();
            }
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

static BARRIER_POOL: OnceLock<BarrierPool> = OnceLock::new();

/// Persistent barrier pool
pub fn barrier_pool() -> &'static BarrierPool {
    BARRIER_POOL.get_or_init(|| BarrierPool::new(candle_num_threads().saturating_sub(1)))
}

pub fn with_threadpool<F: FnOnce() -> R + Send, R: Send>(f: F) -> R {
    candle_pool().install(f)
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
    physical.max(1) // safeguard against bad number
}

fn rayon_num_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
        .filter(|nt| nt > &0)
        .unwrap_or_else(default_num_threads)
}

fn candle_num_threads() -> usize {
    std::env::var("CANDLE_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
        .filter(|nt| nt > &0)
        .unwrap_or_else(default_num_threads)
}

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    rayon_num_threads()
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
