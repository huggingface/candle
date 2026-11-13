use std::fmt;

/// <div class="vulkano-alert-caution">
///
/// > Caution
/// >
/// > This map **permits duplicate keys** as an optimization. It is assumed that this doesn't
/// > happen in the codebase and this is validated with debug assertions on.
///
/// </div>
#[derive(Clone)]
pub struct LinearMap<K, V> {
    inner: Vec<(K, V)>,
}

impl<K, V> Default for LinearMap<K, V> {
    #[inline]
    fn default() -> Self {
        LinearMap::new()
    }
}

impl<K, V> LinearMap<K, V> {
    #[inline]
    pub const fn new() -> Self {
        LinearMap { inner: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        LinearMap {
            inner: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.iter().map(|(k, v)| (k, v))
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.inner.iter_mut().map(|(k, v)| (&*k, v))
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.inner.iter().map(|(k, _)| k)
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.inner.iter().map(|(_, v)| v)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<K: Eq, V> LinearMap<K, V> {
    #[inline]
    pub fn insert(&mut self, key: K, value: V) {
        debug_assert!(!self.keys().any(|k| k == &key));

        self.inner.push((key, value));
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.iter().find_map(|(k, v)| (k == key).then_some(v))
    }

    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.iter_mut().find_map(|(k, v)| (k == key).then_some(v))
    }

    #[inline]
    pub fn get_or_insert(&mut self, key: K, value: V) -> &mut V {
        self.get_or_insert_with(key, || value)
    }

    #[inline]
    pub fn get_or_insert_with(&mut self, key: K, f: impl FnOnce() -> V) -> &mut V {
        let index;

        if let Some(i) = self.inner.iter().position(|(k, _)| k == &key) {
            index = i;
        } else {
            index = self.inner.len();
            self.inner.push((key, f()));
        }

        &mut unsafe { self.inner.get_unchecked_mut(index) }.1
    }

    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.keys().any(|k| k == key)
    }

    #[inline]
    pub fn index_of(&self, key: &K) -> Option<usize> {
        self.keys().position(|k| k == key)
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for LinearMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Eq, V> Extend<(K, V)> for LinearMap<K, V> {
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        if cfg!(debug_assertions) {
            iter.into_iter().for_each(|(k, v)| self.insert(k, v));
        } else {
            self.inner.extend(iter);
        }
    }
}

impl<K: Eq, V> FromIterator<(K, V)> for LinearMap<K, V> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = LinearMap::with_capacity(lower);
        map.extend(iter);

        map
    }
}
