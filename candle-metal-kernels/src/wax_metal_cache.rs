//! Hook into the wax caching system.

use std::sync::OnceLock;

/// Identifies the wax compiler that produced a kernel, from `build.rs`. Changes whenever the
/// emitter does, so entries from an older one are recognised and swept.
pub(crate) const EMITTER: &str = env!("WAX_EMITTER");

/// Persistent cache of wax kernels compiled to metallibs.
pub(crate) fn persistent_cache() -> Option<&'static wax_cache::Cache> {
    static C: OnceLock<Option<wax_cache::Cache>> = OnceLock::new();
    C.get_or_init(|| wax_cache::Cache::open("candle-wax", EMITTER.as_bytes()))
        .as_ref()
}
