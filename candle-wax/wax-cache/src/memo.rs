use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Track artifacts to ensure they are only built once.
pub struct Memo<K, V> {
    built: Mutex<HashMap<K, V>>,
    building: Mutex<HashMap<K, Arc<Mutex<()>>>>,
}

impl<K, V> Default for Memo<K, V> {
    fn default() -> Self {
        Self {
            built: Mutex::new(HashMap::new()),
            building: Mutex::new(HashMap::new()),
        }
    }
}

impl<K: Clone + Eq + Hash, V: Clone> Memo<K, V> {
    pub fn get(&self, key: &K) -> Option<V> {
        self.built.lock().unwrap().get(key).cloned()
    }

    pub fn get_or_build<E>(&self, key: &K, build: impl FnOnce() -> Result<V, E>) -> Result<V, E> {
        if let Some(v) = self.get(key) {
            return Ok(v);
        }
        let entry = self
            .building
            .lock()
            .unwrap()
            .entry(key.clone())
            .or_default()
            .clone();
        let _guard = entry.lock().unwrap();
        // Automatically drops entry from `self.building` by `key`.
        let _done = MemoRelease {
            map: &self.building,
            key,
        };
        // May have finished already
        if let Some(v) = self.get(key) {
            return Ok(v);
        }
        let v = build()?;
        self.built.lock().unwrap().insert(key.clone(), v.clone());
        Ok(v)
    }
}

struct MemoRelease<'a, K: Eq + Hash> {
    map: &'a Mutex<HashMap<K, Arc<Mutex<()>>>>,
    key: &'a K,
}

impl<K: Eq + Hash> Drop for MemoRelease<'_, K> {
    fn drop(&mut self) {
        self.map.lock().unwrap().remove(self.key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn threads_missing_the_same_key_build_it_once() {
        let m: Arc<Memo<u32, String>> = Arc::new(Memo::default());
        let builds = Arc::new(AtomicUsize::new(0));
        let hands: Vec<_> = (0..16)
            .map(|_| {
                let (m, b) = (m.clone(), builds.clone());
                std::thread::spawn(move || {
                    m.get_or_build::<()>(&7, || {
                        b.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(std::time::Duration::from_millis(30));
                        Ok("built".into())
                    })
                    .unwrap()
                })
            })
            .collect();
        for h in hands {
            assert_eq!(h.join().unwrap(), "built");
        }
        assert_eq!(builds.load(Ordering::Relaxed), 1, "built more than once");
    }

    #[test]
    fn distinct_keys_build_in_parallel() {
        let m: Arc<Memo<u32, u32>> = Arc::new(Memo::default());
        let hands: Vec<_> = (0..8u32)
            .map(|i| {
                let m = m.clone();
                std::thread::spawn(move || m.get_or_build::<()>(&i, || Ok(i * 10)).unwrap())
            })
            .collect();
        for (i, h) in hands.into_iter().enumerate() {
            assert_eq!(h.join().unwrap(), i as u32 * 10);
        }
    }

    #[test]
    fn failed_build_is_not_memoed() {
        let m: Memo<u32, u32> = Memo::default();
        assert!(m.get_or_build::<&str>(&1, || Err("boom")).is_err());
        assert_eq!(m.get(&1), None);
        assert!(m.building.lock().unwrap().is_empty(), "gate leaked");
        assert_eq!(m.get_or_build::<&str>(&1, || Ok(5)), Ok(5));
    }
}
