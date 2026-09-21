//! Backend neutral persistent cache for compiled wax kernels.
//!
//! [`Cache`] is the persistent cache. The entries are simply `(generation, key, payload)` of
//! byte strings. The cache does not know what a kernel is, what compiled it, how it is encoded,
//! etc. Cached kernels are stored as `<key>.waxk` files in the platform's per-user cache
//! directory (can override with `WAX_CACHE` env var). The cache also ensures it does not grow
//! indefinitely. It adheres to a max byte budget automatically ([`Cache::sweep`]).
//! Any failure in accessing the cache is a miss. Missing file, bad data, key mismatch, etc.
//!
//! [`Memo`] is an in-process helper for a similar task. It's purpose is to track already built
//! artifacts that should not be rebuilt. Similar to `OnceCell` but for multiple artifacts.

mod cache;
mod dir;
mod memo;

pub use cache::{Cache, DEFAULT_MAX_BYTES, Swept};
pub use memo::Memo;
