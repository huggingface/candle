//! The persistent cache. `(generation, key, payload)` byte strings in files on disk.

use crate::dir::cache_root;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const MAGIC: &[u8; 4] = b"WAXK";
const FORMAT: u32 = 2;
const EXT: &str = "waxk";

/// Max bytes of the cache before sweeping evicts the oldest.
pub const DEFAULT_MAX_BYTES: u64 = 64 << 20;

/// Fraction of `max_bytes` to not evict.
const SWEEP_TARGET: f64 = 0.8;

/// How old a tmp can wait for a rename before a sweep treats it as abandoned.
const TMP_GRACE: std::time::Duration = std::time::Duration::from_secs(3600);

/// A generation longer than this in a file header means the header is not ours.
const MAX_GENERATION: usize = 1024;

pub struct Cache {
    dir: PathBuf,
    generation: Vec<u8>,
    max_bytes: u64,
    swept: AtomicBool,
    written: AtomicU64,
}

/// What a sweep removed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Swept {
    pub stale: usize,
    pub evicted: usize,
    pub bytes: u64,
}

impl Cache {
    /// `WAX_CACHE=0` disables.
    /// `WAX_CACHE_DIR` relocates.
    /// `WAX_CACHE_MAX_BYTES` resizes.
    ///
    /// `None` where no environment answer exists for the platform's cache directory.
    /// A caller that knows exactly the cache is located should use [`Cache::at`] directly.
    pub fn open(name: &str, generation: &[u8]) -> Option<Self> {
        if std::env::var("WAX_CACHE").ok().as_deref() == Some("0") {
            return None;
        }
        let dir = match std::env::var_os("WAX_CACHE_DIR") {
            Some(d) => PathBuf::from(d),
            None => cache_root()?.join(name),
        };
        let max_bytes = std::env::var("WAX_CACHE_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_BYTES);
        Some(Self::at(dir, generation, max_bytes))
    }

    pub fn at(dir: PathBuf, generation: &[u8], max_bytes: u64) -> Self {
        Self {
            dir,
            generation: generation.to_vec(),
            max_bytes,
            swept: AtomicBool::new(false),
            written: AtomicU64::new(0),
        }
    }

    pub fn load(&self, key: &[u8]) -> Option<Vec<u8>> {
        let raw = std::fs::read(self.path_for(key)).ok()?;
        let (generation, rest) = blob(header(&raw)?)?;
        let (stored, rest) = blob(rest)?;
        if generation != self.generation || stored != key {
            return None;
        }
        Some(blob(rest)?.0.to_vec())
    }

    pub fn store(&self, key: &[u8], payload: &[u8]) {
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let mut v = Vec::with_capacity(payload.len() + key.len() + 64);
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&FORMAT.to_le_bytes());
        for b in [&self.generation[..], key, payload] {
            v.extend_from_slice(&(b.len() as u64).to_le_bytes());
            v.extend_from_slice(b);
        }
        let n = v.len() as u64;
        self.write_atomically(&self.path_for(key), &v);
        // Sweep once per process and/or when a quarter of the budget is reached.
        let first = !self.swept.swap(true, Ordering::Relaxed);
        if self.written.fetch_add(n, Ordering::Relaxed) + n > self.max_bytes / 4 || first {
            self.written.store(0, Ordering::Relaxed);
            self.sweep();
        }
    }

    /// Drop old generation. Evict oldest kernels until within `max_bytes` budget.
    pub fn sweep(&self) -> Swept {
        let mut out = Swept::default();
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return out;
        };
        let mut live: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
        let mut total = 0u64;
        for e in entries.flatten() {
            let p = e.path();
            let Ok(md) = e.metadata() else { continue };
            if p.extension().and_then(|s| s.to_str()) != Some(EXT) {
                // Check if tmp file is old enough to remove.
                let orphaned = md
                    .modified()
                    .ok()
                    .and_then(|t| t.elapsed().ok())
                    .is_some_and(|age| age > TMP_GRACE);
                if p.extension().and_then(|s| s.to_str()) == Some("tmp") && orphaned {
                    let _ = std::fs::remove_file(&p);
                }
                continue;
            }
            let size = md.len();
            if self.generation_of(&p).as_deref() == Some(&self.generation) {
                total += size;
                live.push((p, size, md.modified().unwrap_or(std::time::UNIX_EPOCH)));
            } else if std::fs::remove_file(&p).is_ok() {
                out.stale += 1;
                out.bytes += size;
            }
        }
        if total <= self.max_bytes {
            return out;
        }
        let target = (self.max_bytes as f64 * SWEEP_TARGET) as u64;
        live.sort_by_key(|(_, _, t)| *t);
        for (p, size, _) in live {
            if total <= target {
                break;
            }
            if std::fs::remove_file(&p).is_ok() {
                total -= size;
                out.evicted += 1;
                out.bytes += size;
            }
        }
        out
    }

    /// Get generation header value. Reads only the header.
    fn generation_of(&self, p: &Path) -> Option<Vec<u8>> {
        let mut f = std::fs::File::open(p).ok()?;
        let mut head = [0u8; 16];
        f.read_exact(&mut head).ok()?;
        let (len, _) = header(&head)?.split_first_chunk::<8>()?;
        let n = u64::from_le_bytes(*len) as usize;
        if n > MAX_GENERATION {
            return None;
        }
        let mut g = vec![0u8; n];
        f.read_exact(&mut g).ok()?;
        Some(g)
    }

    fn path_for(&self, key: &[u8]) -> PathBuf {
        self.dir.join(format!("{:016x}.{EXT}", fnv_1a(key)))
    }

    /// Write temp file with unique name based on process id and atomic counter.
    /// Once contents are on disk, rename to correct file name. This ensures concurrent readers
    /// either see the previous contents or the new ones, never something half finished.
    fn write_atomically(&self, path: &Path, bytes: &[u8]) {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let tmp = path.with_extension(format!(
            "{}-{}.tmp",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let wrote = std::fs::File::create(&tmp)
            .and_then(|mut f| f.write_all(bytes).and_then(|_| f.sync_all()));
        if wrote.is_ok() && std::fs::rename(&tmp, path).is_ok() {
            return;
        }
        let _ = std::fs::remove_file(&tmp);
    }
}

/// FNV-1a hash. For filenames.
fn fnv_1a(key: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// `MAGIC` and `FORMAT`, then the tail.
fn header(raw: &[u8]) -> Option<&[u8]> {
    let (magic, rest) = raw.split_first_chunk::<4>()?;
    let (format, rest) = rest.split_first_chunk::<4>()?;
    (magic == MAGIC && u32::from_le_bytes(*format) == FORMAT).then_some(rest)
}

/// Split raw bytes into a 1-byte length-prefixed blob.
fn blob(raw: &[u8]) -> Option<(&[u8], &[u8])> {
    let (len, rest) = raw.split_first_chunk::<8>()?;
    rest.split_at_checked(u64::from_le_bytes(*len) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(name: &str, generation: &[u8], max_bytes: u64) -> Cache {
        let dir = std::env::temp_dir().join(format!("wax-cache-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Cache::at(dir, generation, max_bytes)
    }

    #[test]
    fn cache_roundtrips_correctly() {
        let c = cache("roundtrip", b"g1", DEFAULT_MAX_BYTES);
        c.store(b"key", b"payload");
        assert_eq!(c.load(b"key").as_deref(), Some(&b"payload"[..]));
    }

    #[test]
    fn different_key_same_file_misses() {
        let c = cache("collision", b"g1", DEFAULT_MAX_BYTES);
        c.store(b"key", b"payload");
        let path = c.path_for(b"key");
        let raw = std::fs::read(&path).unwrap();
        c.write_atomically(&c.path_for(b"other"), &raw);
        assert_eq!(c.load(b"other"), None);
    }

    #[test]
    fn different_generation_misses() {
        let c = cache("generation", b"g1", DEFAULT_MAX_BYTES);
        c.store(b"key", b"payload");
        let newer = Cache::at(c.dir.clone(), b"g2", DEFAULT_MAX_BYTES);
        assert_eq!(newer.load(b"key"), None);
    }

    #[test]
    fn truncated_file_is_miss_not_panic() {
        let c = cache("truncated", b"g1", DEFAULT_MAX_BYTES);
        c.store(b"key", b"payload");
        let path = c.path_for(b"key");
        let raw = std::fs::read(&path).unwrap();
        for n in 0..raw.len() {
            std::fs::write(&path, &raw[..n]).unwrap();
            assert_eq!(c.load(b"key"), None, "truncated to {n}");
        }
    }

    #[test]
    fn corrupt_length_prefix_is_miss_not_huge_allocation() {
        let c = cache("corrupt", b"g1", DEFAULT_MAX_BYTES);
        c.store(b"key", b"payload");
        let path = c.path_for(b"key");
        let mut raw = std::fs::read(&path).unwrap();
        raw[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &raw).unwrap();
        assert_eq!(c.load(b"key"), None);
    }

    fn total_bytes(c: &Cache) -> u64 {
        std::fs::read_dir(&c.dir)
            .map(|d| d.flatten().map(|e| e.metadata().unwrap().len()).sum())
            .unwrap_or(0)
    }

    #[test]
    fn sweep_drops_old_generation() {
        let old = cache("stale", b"g1", DEFAULT_MAX_BYTES);
        for i in 0..5u8 {
            old.store(&[i], b"payload");
        }
        let newer = Cache::at(old.dir.clone(), b"g2", DEFAULT_MAX_BYTES);
        let swept = newer.sweep();
        assert_eq!(swept.stale, 5);
        assert_eq!(swept.evicted, 0);
    }

    #[test]
    fn sweep_brings_cache_within_budget_oldest_first() {
        let c = cache("evict", b"g1", 4096);
        for i in 0..20u8 {
            c.store(&[i], &[i; 512]);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        c.sweep();
        assert!(total_bytes(&c) <= 4096, "{} bytes left", total_bytes(&c));
        assert_eq!(c.load(&[19]).as_deref(), Some(&[19u8; 512][..]));
        assert_eq!(c.load(&[0]), None);
    }

    #[test]
    fn cache_store_sweeps_automatically() {
        let c = cache("bounded", b"g1", 4096);
        for i in 0..200u16 {
            c.store(&i.to_le_bytes(), &[0u8; 512]);
            assert!(
                total_bytes(&c) <= 4096 + 4096 / 4 + 1024,
                "unbounded at {i}: {} bytes",
                total_bytes(&c)
            );
        }
    }

    #[test]
    #[cfg(not(any(target_os = "android", target_family = "wasm")))]
    fn the_name_becomes_the_last_component() {
        let c = Cache::open("wax-cache-test-name", b"g").expect("a cache directory");
        assert!(
            c.dir.ends_with("wax-cache-test-name"),
            "{}",
            c.dir.display()
        );
    }
}

#[cfg(test)]
mod concurrency {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn multithreaded_stores_do_not_corrupt_entry() {
        const BIG: usize = 2 << 20;
        let dir = std::env::temp_dir().join(format!("wax-cache-{}-race", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = Arc::new(Cache::at(dir, b"g1", DEFAULT_MAX_BYTES));
        for round in 0..3u8 {
            let hands: Vec<_> = (0..8u8)
                .map(|t| {
                    let c = c.clone();
                    std::thread::spawn(move || c.store(b"same-key", &vec![round * 8 + t + 1; BIG]))
                })
                .collect();
            for h in hands {
                h.join().unwrap();
            }
            let got = c.load(b"same-key").expect("entry did not survive race");
            assert_eq!(got.len(), BIG, "round {round}: wrong length");
            let first = got[0];
            assert!(
                first != 0 && got.iter().all(|&b| b == first),
                "round {round}: entry is a mix of two writers"
            );
        }
    }

    fn filetime_set(p: &Path, t: std::time::SystemTime) {
        let f = std::fs::File::options().write(true).open(p).unwrap();
        f.set_times(std::fs::FileTimes::new().set_modified(t))
            .unwrap();
    }

    #[test]
    fn sweep_collects_abandoned_temp() {
        let dir = std::env::temp_dir().join(format!("wax-cache-{}-orphan", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let c = Cache::at(dir.clone(), b"g1", DEFAULT_MAX_BYTES);
        let fresh = dir.join("aaaa.1-0.tmp");
        std::fs::write(&fresh, b"in flight").unwrap();
        let old = dir.join("bbbb.1-1.tmp");
        std::fs::write(&old, b"abandoned").unwrap();
        let long_ago = std::time::SystemTime::now() - TMP_GRACE * 2;
        filetime_set(&old, long_ago);
        c.sweep();
        assert!(fresh.exists());
        assert!(!old.exists());
    }
}
