//! Cache keys, cache locations and the writes that populate them.

use crate::error::KernelError;
use fslock::LockFile;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// The hipcc flags that affect code generation. One definition, used both to
/// build the command line and to key the cache, so the two cannot drift.
///
/// `--offload-arch` and the include paths are not here: the architecture is
/// hashed separately and the include paths are derived from the cache
/// directory itself.
pub(crate) const COMPILE_FLAGS: &[&str] = &["--genco", "-O3", "-std=c++17", "-D__CUDA_ARCH__=800"];

/// Hex characters kept from a SHA-256; enough to name a file uniquely.
const KEY_LEN: usize = 16;

/// Cache key for one translation unit.
///
/// Everything that can change the emitted code object goes in: the source, the
/// staged headers (names *and* contents), the hipcc flags, the target
/// architecture, the full toolchain version and this crate's version.
///
/// Hashing the source alone was silently wrong. `stage_headers` rewrites the
/// staged headers on every run, but the cache *filename* did not move with
/// them, so editing a shim header produced a hit on the previous object and the
/// compile was skipped entirely.
pub(crate) fn module_key(
    source: &str,
    headers: &[(&str, &str)],
    arch: &str,
    toolchain: &str,
) -> String {
    let mut hasher = Sha256::new();
    field(&mut hasher, env!("CARGO_PKG_VERSION").as_bytes());
    field(&mut hasher, arch.as_bytes());
    field(&mut hasher, toolchain.as_bytes());
    for flag in COMPILE_FLAGS {
        field(&mut hasher, flag.as_bytes());
    }
    field(&mut hasher, source.as_bytes());
    hasher.update(headers_digest(headers));
    truncated_hex(hasher)
}

/// Names the directory the headers are staged into, so two builds whose header
/// sets differ never write over each other's staging area.
pub(crate) fn headers_key(headers: &[(&str, &str)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(headers_digest(headers));
    truncated_hex(hasher)
}

fn headers_digest(headers: &[(&str, &str)]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for (name, contents) in headers {
        field(&mut hasher, name.as_bytes());
        field(&mut hasher, contents.as_bytes());
    }
    hasher.finalize().into()
}

/// Length-prefix each field so no two different splits can hash alike.
fn field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn truncated_hex(hasher: Sha256) -> String {
    format!("{:x}", hasher.finalize())[..KEY_LEN].to_string()
}

/// Directory component for an architecture. `gcnArchName` can carry target
/// features (`gfx942:sramecc+:xnack-`); the raw string still goes to the
/// compiler and into the key, only the path is tamed.
pub(crate) fn arch_dir_name(arch: &str) -> String {
    arch.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Where compiled code objects live.
///
/// `CANDLE_ROCM_CACHE_DIR` wins outright. Otherwise the user cache directory,
/// falling back to a per-uid directory under the system temp dir when that is
/// absent or read-only: containers, CI runners and service accounts frequently
/// have no writable `$HOME`, and the backend has to run there too.
pub(crate) fn base_cache_dir() -> Result<PathBuf, KernelError> {
    if let Some(dir) = std::env::var_os("CANDLE_ROCM_CACHE_DIR") {
        let dir = PathBuf::from(dir);
        if is_writable(&dir) {
            return Ok(dir);
        }
        return Err(KernelError::Io(format!(
            "CANDLE_ROCM_CACHE_DIR is set to {}, which is not writable",
            dir.display()
        )));
    }

    if let Some(dir) = dirs::cache_dir().map(|dir| dir.join("candle-rocm")) {
        if is_writable(&dir) {
            return Ok(dir);
        }
        eprintln!(
            "candle-rocm: {} is not writable, falling back to the system temp directory; \
             set CANDLE_ROCM_CACHE_DIR to choose another location",
            dir.display()
        );
    }

    let fallback = std::env::temp_dir().join(format!("candle-rocm-{}", user_tag()));
    if is_writable(&fallback) {
        return Ok(fallback);
    }
    Err(KernelError::Io(format!(
        "no writable kernel cache directory; tried {}. Set CANDLE_ROCM_CACHE_DIR",
        fallback.display()
    )))
}

/// `CANDLE_ROCM_FORCE_RECOMPILE=1` skips the disk read (the write still
/// happens), which is the documented way out of a cache entry that some earlier
/// crash or race left corrupt.
pub(crate) fn force_recompile() -> bool {
    matches!(
        std::env::var("CANDLE_ROCM_FORCE_RECOMPILE").as_deref(),
        Ok("1" | "true" | "TRUE")
    )
}

fn is_writable(dir: &Path) -> bool {
    if fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(format!(".probe.{}", std::process::id()));
    match fs::write(&probe, b"") {
        Ok(()) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Keeps two users off the same temp directory, which would otherwise be
/// unusable for whoever did not create it.
fn user_tag() -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = fs::metadata("/proc/self") {
            return meta.uid().to_string();
        }
    }
    std::env::var("USER").unwrap_or_else(|_| "shared".to_string())
}

/// Advisory lock over one cache entry, held across the whole
/// miss -> compile -> write sequence.
///
/// Without it, concurrent compilers of the same module read each other's
/// half-written intermediates and can persist a corrupt `.hsaco` that only an
/// `rm -rf` clears. It also stops N processes each spending ~70 s on the same
/// translation unit.
pub(crate) fn lock_entry(cache_dir: &Path, name: &str, key: &str) -> Result<LockFile, KernelError> {
    fs::create_dir_all(cache_dir)
        .map_err(|e| KernelError::Io(format!("could not create {}: {e}", cache_dir.display())))?;
    let path = cache_dir.join(format!("{name}_{key}.lock"));
    let mut lock = LockFile::open(&path)
        .map_err(|e| KernelError::Io(format!("could not open {}: {e}", path.display())))?;
    lock.lock()
        .map_err(|e| KernelError::Io(format!("could not lock {}: {e}", path.display())))?;
    Ok(lock)
}

/// Write via a unique temporary in the destination directory, then rename.
/// Concurrent compilers (parallel test threads, several processes) would
/// otherwise observe a half-written file.
pub(crate) fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), KernelError> {
    let parent = path
        .parent()
        .ok_or_else(|| KernelError::Io(format!("{} has no parent directory", path.display())))?;
    fs::create_dir_all(parent)
        .map_err(|e| KernelError::Io(format!("could not create {}: {e}", parent.display())))?;

    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("staged"),
        std::process::id()
    ));
    fs::write(&tmp, contents)
        .map_err(|e| KernelError::Io(format!("could not write {}: {e}", tmp.display())))?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        KernelError::Io(format!("could not place {}: {e}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCH: &str = "gfx1101";
    const TOOLCHAIN: &str = "HIP version: 7.2.4-53211";
    const HEADERS: &[(&str, &str)] = &[("hip_shim/hip_compat.h", "// shim"), ("a.cuh", "// a")];

    fn key(headers: &[(&str, &str)]) -> String {
        module_key("__global__ void k() {}", headers, ARCH, TOOLCHAIN)
    }

    #[test]
    fn key_is_stable_for_identical_inputs() {
        assert_eq!(key(HEADERS), key(HEADERS));
    }

    #[test]
    fn key_changes_with_header_contents() {
        let edited = [
            ("hip_shim/hip_compat.h", "// shim edited"),
            ("a.cuh", "// a"),
        ];
        assert_ne!(key(HEADERS), key(&edited));
    }

    #[test]
    fn key_changes_with_header_names() {
        let renamed = [("hip_shim/other.h", "// shim"), ("a.cuh", "// a")];
        assert_ne!(key(HEADERS), key(&renamed));
    }

    /// The real header set is what the cache is keyed on in production; flip one
    /// byte of the force-included shim and the key has to move.
    #[test]
    fn key_changes_with_a_real_shim_header() {
        const NAME: &str = "hip_shim/hip_compat.h";
        let original = crate::HEADERS
            .iter()
            .find(|(name, _)| *name == NAME)
            .expect("the force-included shim header is staged")
            .1;
        let patched = format!("{original}\n// edit\n");
        let edited: Vec<(&str, &str)> = crate::HEADERS
            .iter()
            .map(|&(name, contents)| {
                if name == NAME {
                    (name, patched.as_str())
                } else {
                    (name, contents)
                }
            })
            .collect();
        assert_ne!(key(crate::HEADERS), key(&edited));
    }

    #[test]
    fn key_changes_with_source_arch_and_toolchain() {
        let base = module_key("a", HEADERS, ARCH, TOOLCHAIN);
        assert_ne!(base, module_key("b", HEADERS, ARCH, TOOLCHAIN));
        assert_ne!(base, module_key("a", HEADERS, "gfx1100", TOOLCHAIN));
        assert_ne!(base, module_key("a", HEADERS, ARCH, "HIP version: 7.2.5-1"));
    }

    /// Length-prefixing: shifting a byte across a field boundary must not
    /// produce the same digest.
    #[test]
    fn key_fields_do_not_bleed_into_each_other() {
        let a = [("ab", "c"), ("d", "e")];
        let b = [("a", "bc"), ("d", "e")];
        assert_ne!(key(&a), key(&b));
    }

    #[test]
    fn headers_key_tracks_the_header_set() {
        let edited = [
            ("hip_shim/hip_compat.h", "// shim edited"),
            ("a.cuh", "// a"),
        ];
        assert_eq!(headers_key(HEADERS), headers_key(HEADERS));
        assert_ne!(headers_key(HEADERS), headers_key(&edited));
    }

    #[test]
    fn arch_dir_name_strips_target_features() {
        assert_eq!(arch_dir_name("gfx1101"), "gfx1101");
        assert_eq!(
            arch_dir_name("gfx942:sramecc+:xnack-"),
            "gfx942_sramecc__xnack-"
        );
    }
}
