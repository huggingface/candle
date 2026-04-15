//! Flat-ragged tensor cache backed by safetensors.
//!
//! [`RaggedCache`] stores a list of variable-length 2D tensors (shape
//! `(T_i, D)`) as a single concatenated tensor plus an offsets array. This
//! is the standard layout for caching per-sample token embeddings, audio
//! frames, graph-node features, or any other ragged-sequence data that is
//! expensive to compute but cheap to store.
//!
//! Layout on disk (safetensors, two tensors):
//!
//! | Key       | Shape          | DType              | Meaning                |
//! |-----------|----------------|--------------------|------------------------|
//! | `flat`    | `(total_T, D)` | user-chosen        | concatenated features  |
//! | `offsets` | `(n_items+1,)` | `I64`              | cumulative row counts  |
//!
//! Item `i` is recovered by slicing `flat[offsets[i]..offsets[i+1], :]`.
//!
//! All items in a cache must share the same trailing dimension `D`, dtype,
//! and device. Zero-length items (`T_i == 0`) are supported — their slice is
//! an empty tensor.
//!
//! # Example
//!
//! ```no_run
//! use candle::{Device, Tensor};
//! use candle_datasets::ragged::RaggedCache;
//!
//! let dev = Device::Cpu;
//! let items = vec![
//!     Tensor::zeros((7, 64), candle::DType::F32, &dev).unwrap(),
//!     Tensor::zeros((12, 64), candle::DType::F32, &dev).unwrap(),
//!     Tensor::zeros((3, 64), candle::DType::F32, &dev).unwrap(),
//! ];
//! let cache = RaggedCache::from_items(&items).unwrap();
//! cache.save("embeddings.safetensors").unwrap();
//!
//! // Later, in another process:
//! let cache = RaggedCache::load("embeddings.safetensors", &dev).unwrap();
//! assert_eq!(cache.len(), 3);
//! let item1 = cache.get(1).unwrap();
//! assert_eq!(item1.dims(), &[12, 64]);
//! ```

use std::collections::HashMap;
use std::path::Path;

use candle::{safetensors, Device, Result, Tensor};

const KEY_FLAT: &str = "flat";
const KEY_OFFSETS: &str = "offsets";

/// A concatenated store of variable-length 2D tensors indexable by item
/// position.
///
/// Build from a slice of tensors via [`RaggedCache::from_items`], persist via
/// [`RaggedCache::save`], and later rehydrate with [`RaggedCache::load`].
pub struct RaggedCache {
    flat: Tensor,
    offsets: Vec<i64>,
}

impl std::fmt::Debug for RaggedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaggedCache")
            .field("items", &self.len())
            .field("feature_dim", &self.feature_dim())
            .field("total_rows", &self.flat.dims().first().copied().unwrap_or(0))
            .field("dtype", &self.flat.dtype())
            .finish()
    }
}

impl RaggedCache {
    /// Build a cache by concatenating `items` along axis 0.
    ///
    /// All items must:
    /// - be 2D (shape `(T_i, D)`)
    /// - share the same trailing dim `D`
    /// - share the same dtype and device
    pub fn from_items(items: &[Tensor]) -> Result<Self> {
        if items.is_empty() {
            let device = Device::Cpu;
            let flat = Tensor::zeros((0, 0), candle::DType::F32, &device)?;
            return Ok(Self {
                flat,
                offsets: vec![0],
            });
        }

        let first_dims = items[0].dims();
        if first_dims.len() != 2 {
            candle::bail!(
                "RaggedCache::from_items: item 0 has rank {}, expected 2",
                first_dims.len()
            );
        }
        let feature_dim = first_dims[1];
        let dtype = items[0].dtype();
        let device = items[0].device().clone();

        let mut offsets: Vec<i64> = Vec::with_capacity(items.len() + 1);
        offsets.push(0);
        let mut cumulative: i64 = 0;

        for (i, t) in items.iter().enumerate() {
            let dims = t.dims();
            if dims.len() != 2 {
                candle::bail!(
                    "RaggedCache::from_items: item {i} has rank {}, expected 2",
                    dims.len()
                );
            }
            if dims[1] != feature_dim {
                candle::bail!(
                    "RaggedCache::from_items: item {i} has feature dim {}, expected {feature_dim}",
                    dims[1]
                );
            }
            if t.dtype() != dtype {
                candle::bail!(
                    "RaggedCache::from_items: item {i} has dtype {:?}, expected {:?}",
                    t.dtype(),
                    dtype
                );
            }
            if !t.device().same_device(&device) {
                candle::bail!("RaggedCache::from_items: item {i} is on a different device");
            }
            cumulative += dims[0] as i64;
            offsets.push(cumulative);
        }

        // Concatenating zero-length tensors with Tensor::cat is a no-op the
        // candle backend doesn't always love; skip empties.
        let non_empty: Vec<&Tensor> = items.iter().filter(|t| t.dims()[0] > 0).collect();
        let flat = if non_empty.is_empty() {
            Tensor::zeros((0, feature_dim), dtype, &device)?
        } else {
            Tensor::cat(&non_empty, 0)?
        };

        Ok(Self { flat, offsets })
    }

    /// Number of items stored.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// `true` when the cache holds no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Feature dimension `D`, or 0 if the cache is empty.
    pub fn feature_dim(&self) -> usize {
        let dims = self.flat.dims();
        if dims.len() < 2 {
            0
        } else {
            dims[1]
        }
    }

    /// Retrieve item `idx` as a view into the flat storage.
    ///
    /// The returned tensor shares storage with the underlying `flat` tensor
    /// when possible (it is produced via [`Tensor::narrow`]).
    pub fn get(&self, idx: usize) -> Result<Tensor> {
        if idx >= self.len() {
            candle::bail!(
                "RaggedCache::get: index {idx} out of bounds (len = {})",
                self.len()
            );
        }
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        self.flat.narrow(0, start, end - start)
    }

    /// Retrieve a batch of items by index. Empty `indices` yields an empty
    /// `Vec`; out-of-range indices surface as an error.
    pub fn gather(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        indices.iter().map(|&i| self.get(i)).collect()
    }

    /// Save the cache to a safetensors file.
    ///
    /// Writes two named tensors: `flat` (the concatenated storage) and
    /// `offsets` (an `(n+1,)` `I64` tensor).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let offsets_tensor =
            Tensor::from_slice(&self.offsets[..], (self.offsets.len(),), self.flat.device())?;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(KEY_FLAT.to_string(), self.flat.clone());
        map.insert(KEY_OFFSETS.to_string(), offsets_tensor);
        safetensors::save(&map, path.as_ref())
    }

    /// Load a cache from a safetensors file onto the given device.
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let map = safetensors::load(path.as_ref(), device)?;
        let flat = map
            .get(KEY_FLAT)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "RaggedCache::load: missing key `{KEY_FLAT}` in {:?}",
                    path.as_ref()
                ))
            })?
            .clone();
        let offsets_tensor = map.get(KEY_OFFSETS).ok_or_else(|| {
            candle::Error::Msg(format!(
                "RaggedCache::load: missing key `{KEY_OFFSETS}` in {:?}",
                path.as_ref()
            ))
        })?;
        let offsets: Vec<i64> = offsets_tensor.to_vec1::<i64>()?;
        if offsets.is_empty() {
            candle::bail!("RaggedCache::load: offsets array is empty");
        }
        if offsets[0] != 0 {
            candle::bail!(
                "RaggedCache::load: offsets[0] = {} (expected 0)",
                offsets[0]
            );
        }
        for i in 1..offsets.len() {
            if offsets[i] < offsets[i - 1] {
                candle::bail!(
                    "RaggedCache::load: offsets not monotonically non-decreasing at index {i}: {} < {}",
                    offsets[i],
                    offsets[i - 1]
                );
            }
        }
        let total = *offsets.last().unwrap() as usize;
        let flat_dims = flat.dims();
        if flat_dims.len() != 2 {
            candle::bail!(
                "RaggedCache::load: `flat` has rank {}, expected 2",
                flat_dims.len()
            );
        }
        if flat_dims[0] != total {
            candle::bail!(
                "RaggedCache::load: flat row count {} does not match offsets total {}",
                flat_dims[0],
                total
            );
        }
        Ok(Self { flat, offsets })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempPath(std::path::PathBuf);
    impl TempPath {
        fn new(stem: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir().join(format!(
                "candle_datasets_ragged_{}_{nanos}_{n}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir.join(format!("{stem}.safetensors")))
        }
    }
    impl Drop for TempPath {
        fn drop(&mut self) {
            if let Some(p) = self.0.parent() {
                let _ = std::fs::remove_dir_all(p);
            }
        }
    }

    fn make_items(dev: &Device) -> Vec<Tensor> {
        vec![
            Tensor::arange(0.0f32, 8.0, dev)
                .unwrap()
                .reshape((4, 2))
                .unwrap(),
            Tensor::arange(100.0f32, 106.0, dev)
                .unwrap()
                .reshape((3, 2))
                .unwrap(),
            Tensor::arange(200.0f32, 204.0, dev)
                .unwrap()
                .reshape((2, 2))
                .unwrap(),
        ]
    }

    #[test]
    fn from_items_roundtrip_in_memory() {
        let dev = Device::Cpu;
        let items = make_items(&dev);
        let cache = RaggedCache::from_items(&items).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.feature_dim(), 2);

        let got0: Vec<Vec<f32>> = cache.get(0).unwrap().to_vec2().unwrap();
        let got1: Vec<Vec<f32>> = cache.get(1).unwrap().to_vec2().unwrap();
        let got2: Vec<Vec<f32>> = cache.get(2).unwrap().to_vec2().unwrap();
        assert_eq!(got0.len(), 4);
        assert_eq!(got1.len(), 3);
        assert_eq!(got2.len(), 2);
        assert_eq!(got0[0], vec![0.0, 1.0]);
        assert_eq!(got1[0], vec![100.0, 101.0]);
        assert_eq!(got2[1], vec![202.0, 203.0]);
    }

    #[test]
    fn save_load_roundtrip() {
        let dev = Device::Cpu;
        let items = make_items(&dev);
        let cache = RaggedCache::from_items(&items).unwrap();
        let tmp = TempPath::new("save_load");
        cache.save(&tmp.0).unwrap();

        let reloaded = RaggedCache::load(&tmp.0, &dev).unwrap();
        assert_eq!(reloaded.len(), 3);
        for i in 0..3 {
            let a: Vec<Vec<f32>> = cache.get(i).unwrap().to_vec2().unwrap();
            let b: Vec<Vec<f32>> = reloaded.get(i).unwrap().to_vec2().unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn gather_batch() {
        let dev = Device::Cpu;
        let items = make_items(&dev);
        let cache = RaggedCache::from_items(&items).unwrap();
        let batch = cache.gather(&[2, 0]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].dims(), &[2, 2]);
        assert_eq!(batch[1].dims(), &[4, 2]);
    }

    #[test]
    fn out_of_bounds_errors() {
        let dev = Device::Cpu;
        let items = make_items(&dev);
        let cache = RaggedCache::from_items(&items).unwrap();
        let err = cache.get(99).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("out of bounds"), "unexpected: {msg}");
    }

    #[test]
    fn mismatched_feature_dim_errors() {
        let dev = Device::Cpu;
        let items = vec![
            Tensor::zeros((3, 4), DType::F32, &dev).unwrap(),
            Tensor::zeros((2, 5), DType::F32, &dev).unwrap(),
        ];
        let err = RaggedCache::from_items(&items).unwrap_err();
        assert!(format!("{err}").contains("feature dim"));
    }

    #[test]
    fn wrong_rank_errors() {
        let dev = Device::Cpu;
        let items = vec![Tensor::zeros(5, DType::F32, &dev).unwrap()];
        let err = RaggedCache::from_items(&items).unwrap_err();
        assert!(format!("{err}").contains("rank"));
    }

    #[test]
    fn empty_items_len_zero() {
        let cache = RaggedCache::from_items(&[]).unwrap();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn zero_length_item_supported() {
        let dev = Device::Cpu;
        let items = vec![
            Tensor::zeros((3, 2), DType::F32, &dev).unwrap(),
            Tensor::zeros((0, 2), DType::F32, &dev).unwrap(),
            Tensor::zeros((2, 2), DType::F32, &dev).unwrap(),
        ];
        let cache = RaggedCache::from_items(&items).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(0).unwrap().dims(), &[3, 2]);
        assert_eq!(cache.get(1).unwrap().dims(), &[0, 2]);
        assert_eq!(cache.get(2).unwrap().dims(), &[2, 2]);
    }
}
