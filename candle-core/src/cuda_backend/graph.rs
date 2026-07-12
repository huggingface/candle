//! CUDA Graph capture and replay.
//!
//! During autoregressive decoding the same fixed-shape sequence of kernels is
//! launched once per generated token, and per-kernel launch overhead can
//! dominate once the model is small relative to the GPU. [`CudaGraph`] lets
//! that sequence be captured once and replayed with a single `cuGraphLaunch`
//! call instead of relaunching every individual kernel.
use std::sync::Arc;

use cudarc::driver::sys;

use super::device::EventTrackingGuard;
use super::{CudaDevice, WrapErr};
use crate::{Context, Result};

/// A captured, replayable sequence of CUDA operations.
///
/// Build one with [`CudaGraph::capture`], then call [`CudaGraph::replay`] as
/// many times as needed. The shapes and buffer addresses touched by the
/// captured closure must stay the same across replays: a CUDA graph records
/// the exact kernel launches and memory addresses used during capture, it
/// does not re-derive them.
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<cudarc::driver::CudaStream>,
    // Kept paused for the graph's entire lifetime (capture, every replay, and
    // its eventual destruction), not just the recording call: see the doc
    // comment on `capture` for why re-enabling tracking in between lets
    // cudarc re-record a tensor's dependency event outside of the graph that
    // already owns a captured node for that same event.
    _event_tracking_guard: EventTrackingGuard,
}

impl CudaGraph {
    /// Captures the CUDA operations issued by `f` on `device`'s stream.
    ///
    /// `f` is run exactly once, with the stream in capture mode so that the
    /// kernels it launches are recorded into a graph rather than executed
    /// immediately. While capturing, host-to-device uploads of small
    /// constants are served from a cache (see
    /// [`CudaDevice::enable_cuda_graph_htod_cache`]) since a fresh upload
    /// cannot be recorded as a replayable graph node.
    ///
    /// Callers must run the same operations `f` will perform at least once,
    /// outside of capture, before calling this function, with that warm-up
    /// run itself wrapped in [`CudaDevice::enable_cuda_graph_htod_cache`]
    /// (this function only holds that guard for its own call, not for
    /// whatever warm-up the caller ran beforehand). The warm-up run JIT-loads
    /// any CUDA modules the operations need and, under the guard, populates
    /// the host-to-device upload cache; both module loading and uncached
    /// uploads are disallowed once capture starts and invalidate the capture
    /// (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`) if attempted while it is
    /// active. A warm-up run outside of the guard JIT-loads modules but
    /// leaves the cache empty, so capture still fails on any operation that
    /// needs an uncached host upload (e.g. a kernel launch's non-contiguous
    /// layout parameters, or `Tensor::index_select` against a small index
    /// tensor).
    ///
    /// Any buffer whose contents you need to read between replays must be
    /// allocated *before* capture and written in place by `f` (e.g. via
    /// `Tensor::slice_set`). A tensor allocated *inside* `f` becomes a
    /// graph-owned allocation node whose device memory is only valid while the
    /// graph is executing, so reading it outside of a replay fails with
    /// `CUDA_ERROR_INVALID_VALUE`.
    ///
    /// Event tracking stays paused for the returned [`CudaGraph`]'s entire
    /// lifetime, not just for this call: cudarc unconditionally re-records a
    /// tensor's read/write dependency event on every access to it, regardless
    /// of whether tracking is enabled. A tensor allocated before capture
    /// (e.g. a model-level KV cache written into by every capture) already
    /// owns such events, and recording one from inside a capture turns that
    /// recording into a node of *this* graph. If tracking were re-enabled as
    /// soon as `capture` returns, any later access to that same tensor --
    /// between replays, or by a second, independent `capture` on the same
    /// device -- would record a plain (non-captured) update to an event this
    /// graph still owns a captured node for. Holding the guard until the
    /// graph (and its captured nodes) are destroyed keeps every access to
    /// those tensors ordered by the graph itself for as long as it exists.
    pub fn capture<T>(device: &CudaDevice, f: impl FnOnce() -> Result<T>) -> Result<(Self, T)> {
        let stream = device.cuda_stream();
        let _cache_guard = device.enable_cuda_graph_htod_cache();
        // candle drives cudarc in multi-stream mode, so each kernel launch would
        // otherwise emit cuStreamWaitEvent/cuEventRecord against the tensors'
        // dependency-tracking events. Those event waits reference events recorded
        // outside the capture region and make capture fail with
        // CUDA_ERROR_INVALID_VALUE, so pause event tracking for the capture (and,
        // per the doc comment above, for the whole lifetime of the resulting graph).
        let event_tracking_guard = device.pause_event_tracking();

        unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .result()
        .w()
        .context("cuStreamBeginCapture_v2 failed")?;

        let value = match f() {
            Ok(value) => value,
            Err(err) => {
                let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
                let _ = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut cu_graph) };
                if !cu_graph.is_null() {
                    let _ = unsafe { sys::cuGraphDestroy(cu_graph) };
                }
                return Err(err).context("operation inside cuda graph capture closure failed");
            }
        };

        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut cu_graph) }
            .result()
            .w()
            .context("cuStreamEndCapture failed")?;
        if cu_graph.is_null() {
            crate::bail!("cuda graph capture recorded no operations");
        }

        let mut cu_graph_exec: sys::CUgraphExec = std::ptr::null_mut();
        let instantiate =
            unsafe { sys::cuGraphInstantiateWithFlags(&mut cu_graph_exec, cu_graph, 0) }
                .result()
                .w()
                .context("cuGraphInstantiateWithFlags failed");
        if let Err(err) = instantiate {
            let _ = unsafe { sys::cuGraphDestroy(cu_graph) };
            return Err(err);
        }

        Ok((
            Self {
                cu_graph,
                cu_graph_exec,
                stream,
                _event_tracking_guard: event_tracking_guard,
            },
            value,
        ))
    }

    /// Replays the captured operations on the stream they were captured on.
    pub fn replay(&self) -> Result<()> {
        unsafe { sys::cuGraphLaunch(self.cu_graph_exec, self.stream.cu_stream()) }
            .result()
            .w()
            .context("cuGraphLaunch failed")
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuGraphExecDestroy(self.cu_graph_exec);
            let _ = sys::cuGraphDestroy(self.cu_graph);
        }
        // `_event_tracking_guard` is dropped after this body runs, restoring
        // event tracking only once the captured graph is fully destroyed.
    }
}

// SAFETY: the underlying CUgraph/CUgraphExec handles are only ever
// dereferenced through the CUDA driver API while holding `&self`/`&mut self`,
// matching the thread-safety story of the other cudarc driver handles
// (CudaStream, CudaModule, ...) that candle already shares across threads.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}
