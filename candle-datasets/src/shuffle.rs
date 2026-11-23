use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    indices
}

pub struct Shuffled<T> {
    items: std::vec::IntoIter<T>,
}

impl<T> Shuffled<T> {
    pub fn new<I: IntoIterator<Item = T>>(inner: I, seed: u64) -> Self {
        let mut items: Vec<T> = inner.into_iter().collect();
        let mut rng = StdRng::seed_from_u64(seed);
        items.shuffle(&mut rng);
        Self {
            items: items.into_iter(),
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

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
