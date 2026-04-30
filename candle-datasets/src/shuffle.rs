//! Shuffling helpers for training loops.
//!
//! Two entry points:
//!
//! - [`shuffled_indices`] returns a freshly-shuffled `Vec<usize>` for a given
//!   length and seed. Useful for building per-epoch shuffled orderings of a
//!   fixed dataset without copying the underlying rows.
//! - [`Shuffled`] is an iterator adaptor that collects the inner iterator,
//!   shuffles it once, and yields items in shuffled order. Use this when
//!   you want a one-shot shuffled pass over an iterator.
//!
//! Both are seeded deterministically so reruns of the same training script
//! produce identical orderings.
//!
//! # Example
//!
//! ```
//! use candle_datasets::shuffle::{shuffled_indices, Shuffled};
//!
//! // Per-epoch index shuffle: same seed → same order.
//! let a = shuffled_indices(10, 42);
//! let b = shuffled_indices(10, 42);
//! assert_eq!(a, b);
//! assert_eq!(a.len(), 10);
//!
//! // Iterator adaptor: one-shot shuffled pass.
//! let items = vec![1u32, 2, 3, 4, 5];
//! let shuffled: Vec<u32> = Shuffled::new(items.into_iter(), 7).collect();
//! assert_eq!(shuffled.len(), 5);
//! ```

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Return a shuffled `Vec<usize>` containing every index in `0..len`.
///
/// Uses [`rand::rngs::StdRng`] seeded from `seed`, so the same `(len, seed)`
/// pair always produces the same ordering. This is the standard building
/// block for per-epoch shuffling of a fixed training set: pass `seed =
/// base_seed + epoch` to get a fresh ordering each epoch while keeping runs
/// reproducible.
pub fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices
}

/// Iterator adaptor that drains the inner iterator into a `Vec`, shuffles it
/// once using a seeded PRNG, and yields items in shuffled order.
///
/// The inner iterator is consumed entirely on construction, so `Shuffled` is
/// not suitable for unbounded or very large iterators — use
/// [`shuffled_indices`] plus indexed access in that case.
///
/// # Example
///
/// ```
/// use candle_datasets::shuffle::Shuffled;
///
/// let v: Vec<i32> = (0..100).collect();
/// let s1: Vec<i32> = Shuffled::new(v.iter().copied(), 123).collect();
/// let s2: Vec<i32> = Shuffled::new((0..100), 123).collect();
/// assert_eq!(s1, s2); // same seed, same order
/// ```
pub struct Shuffled<T> {
    items: std::vec::IntoIter<T>,
}

impl<T> Shuffled<T> {
    /// Collect `inner` into a `Vec`, shuffle it with a seeded PRNG, and
    /// return an iterator over the shuffled items.
    pub fn new<I: IntoIterator<Item = T>>(inner: I, seed: u64) -> Self {
        let mut items: Vec<T> = inner.into_iter().collect();
        let mut rng = StdRng::seed_from_u64(seed);
        items.shuffle(&mut rng);
        Self {
            items: items.into_iter(),
        }
    }

    /// Number of items remaining in the shuffled iterator.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// `true` when the iterator has been fully consumed.
    pub fn is_empty(&self) -> bool {
        self.items.len() == 0
    }
}

impl<T> Iterator for Shuffled<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.items.size_hint()
    }
}

impl<T> ExactSizeIterator for Shuffled<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffled_indices_is_permutation() {
        let idx = shuffled_indices(50, 1);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn shuffled_indices_is_deterministic() {
        assert_eq!(shuffled_indices(100, 42), shuffled_indices(100, 42));
    }

    #[test]
    fn shuffled_indices_differs_by_seed() {
        // Probabilistically this could fail with equal orderings, but with
        // 20! possible permutations it effectively cannot.
        assert_ne!(shuffled_indices(20, 1), shuffled_indices(20, 2));
    }

    #[test]
    fn shuffled_indices_empty() {
        assert_eq!(shuffled_indices(0, 1), Vec::<usize>::new());
    }

    #[test]
    fn shuffled_iter_preserves_len() {
        let v: Vec<u32> = (0..100).collect();
        let s: Vec<u32> = Shuffled::new(v.clone(), 7).collect();
        assert_eq!(s.len(), 100);
        let mut sorted = s;
        sorted.sort_unstable();
        assert_eq!(sorted, v);
    }

    #[test]
    fn shuffled_iter_is_seeded() {
        let a: Vec<u32> = Shuffled::new(0..100, 99).collect();
        let b: Vec<u32> = Shuffled::new(0..100, 99).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn shuffled_iter_actually_shuffles() {
        let v: Vec<u32> = (0..50).collect();
        let s: Vec<u32> = Shuffled::new(v.clone(), 0).collect();
        assert_ne!(s, v, "shuffling should (almost surely) change order");
    }

    #[test]
    fn shuffled_iter_exact_size() {
        let mut s = Shuffled::new(0..10u32, 0);
        assert_eq!(s.len(), 10);
        let _ = s.next();
        assert_eq!(s.len(), 9);
    }
}
