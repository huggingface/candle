//! Fingerprints the wax compiler stack into the build, for caching.
//!
//! `CompileKey` covers the DSL kernel's source. The `WAX_EMITTER` key covers what turns that
//! source into bytes. The frontend, the IR, optimization passes, lowering, etc. Any change to
//! the pipeline can make a cached entry outdated.
//!
//! We let cargo handle the change detection. We use `rerun-if-changed` on all relevant
//! directories. If there is a change we update `WAX_EMITTER` with system time as nanos.
//!
//! A released crate does not need to do this as its version is already a unqiue key, so it simply
//! uses `CARGO_PKG_VERSION`.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const EMITTER_TREES: [&str; 3] = [
    "wax-metal/src",
    "../candle-wax/wax-ir/src",
    "../candle-wax/wax-frontend/src",
];

fn main() {
    for dir in EMITTER_TREES {
        println!("cargo:rerun-if-changed={dir}");
    }
    let emitter = if EMITTER_TREES.iter().all(|d| Path::new(d).exists()) {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("a clock after 1970")
            .as_nanos()
            .to_string()
    } else {
        env!("CARGO_PKG_VERSION").to_string()
    };
    println!("cargo:rustc-env=WAX_EMITTER={emitter}");
}
