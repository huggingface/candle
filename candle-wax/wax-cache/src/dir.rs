//! Where the cache lives, per platform.

use std::path::PathBuf;

/// Platform's per-user cache directory.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) fn cache_root() -> Option<PathBuf> {
    Some(std::env::home_dir()?.join("Library/Caches"))
}

#[cfg(target_os = "windows")]
pub(crate) fn cache_root() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
}

/// Android and wasm have no clear environment answer.
///
/// Android can get it via `Context.getCacheDir()`.
/// `wasm32-unknown-unknown` has no filesystem. WASI can have one granted by host.
#[cfg(any(target_os = "android", target_family = "wasm"))]
pub(crate) fn cache_root() -> Option<PathBuf> {
    None
}

/// XDG: `$XDG_CACHE_HOME` when it is set and absolute, else `$HOME/.cache`.
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "windows",
    target_os = "android",
    target_family = "wasm"
)))]
pub(crate) fn cache_root() -> Option<PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::home_dir().map(|h| h.join(".cache")))
}

#[cfg(test)]
mod location {
    use super::*;

    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn apple_stores_in_library_cache() {
        let root = cache_root().expect("a home directory");
        assert!(root.ends_with("Library/Caches"), "{}", root.display());
        assert!(root.is_absolute());
    }
}
