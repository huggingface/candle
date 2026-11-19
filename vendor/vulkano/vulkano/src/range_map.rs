// Most of the code in this module comes from the rangemap crate, which is licensed under either of
// - Apache License, Version 2.0 (https://github.com/jeffparsons/rangemap/blob/master/LICENSE-APACHE
//   or http://www.apache.org/licenses/LICENSE-2.0)
// - MIT (https://github.com/jeffparsons/rangemap/blob/master/LICENSE-MIT or http://opensource.org/licenses/MIT)
// at your option.
//
// The following changes were made:
// - The `RangeStartWrapper` used as key was changed into just the start that's used as the key,
//   and the end is stored in the value (in `Entry`) instead.
// - A `RangeMap::split_at` method was added.
// - Some parts we don't need were removed.

#![allow(dead_code)]

use std::{
    cmp,
    collections::{btree_map, BTreeMap},
    fmt::{Debug, Error as FmtError, Formatter},
    iter::{FromIterator, FusedIterator},
    ops::{Bound, Range},
};

/// A map whose keys are stored as (half-open) ranges bounded
/// inclusively below and exclusively above `(start..end)`.
///
/// Contiguous and overlapping ranges that map to the same value
/// are coalesced into a single range.
#[derive(Clone)]
pub struct RangeMap<K, V> {
    // Stores the range start in the key and the range end in the corresponding value.
    btm: BTreeMap<K, Entry<K, V>>,
}

#[derive(Clone)]
struct Entry<K, V> {
    end: K,
    value: V,
}

impl<K, V> Default for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    /// Makes a new empty `RangeMap`.
    #[inline]
    pub const fn new() -> Self {
        RangeMap {
            btm: BTreeMap::new(),
        }
    }

    /// Returns a reference to the value corresponding to the given key,
    /// if the key is covered by any range in the map.
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.get_key_value(key).map(|(_range, value)| value)
    }

    /// Returns the range-value pair (as a pair of references) corresponding
    /// to the given key, if the key is covered by any range in the map.
    #[inline]
    pub fn get_key_value(&self, key: &K) -> Option<(Range<K>, &V)> {
        self.btm
            // The only stored range that could contain the given key is the
            // last stored range whose start is less than or equal to this key.
            .range((Bound::Unbounded, Bound::Included(key)))
            .next_back()
            .filter(|(_start, Entry { end, value: _ })| {
                // Does the only candidate range contain
                // the requested key?
                end > key
            })
            .map(|(start, Entry { end, value })| (start.clone()..end.clone(), value))
    }

    /// Returns `true` if any range in the map covers the specified key.
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns `true` if any part of the provided range overlaps with a range in the map.
    #[inline]
    pub fn contains_any(&self, range: &Range<K>) -> bool {
        self.range(range).next().is_some()
    }

    /// Gets an iterator over all pairs of key range and value,
    /// ordered by key range.
    ///
    /// The iterator element type is `(&'a Range<K>, &'a V)`.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.btm.iter(),
        }
    }

    /// Gets a mutable iterator over all pairs of key range and value,
    /// ordered by key range.
    ///
    /// The iterator element type is `(&'a Range<K>, &'a mut V)`.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.btm.iter_mut(),
        }
    }

    /// Insert a pair of key range and value into the map.
    ///
    /// If the inserted range partially or completely overlaps any
    /// existing range in the map, then the existing range (or ranges) will be
    /// partially or completely replaced by the inserted range.
    ///
    /// If the inserted range either overlaps or is immediately adjacent
    /// any existing range _mapping to the same value_, then the ranges
    /// will be coalesced into a single contiguous range.
    ///
    /// # Panics
    ///
    /// Panics if range `start >= end`.
    pub fn insert(&mut self, mut range: Range<K>, value: V) {
        // We don't want to have to make empty ranges make sense;
        // they don't represent anything meaningful in this structure.
        assert!(range.start < range.end);

        // Wrap up the given range so that we can "borrow"
        // it as a wrapper reference to either its start or end.
        // See `range_wrapper.rs` for explanation of these hacks.
        let new_value = value;

        // Is there a stored range either overlapping the start of
        // the range to insert or immediately preceding it?
        //
        // If there is any such stored range, it will be the last
        // whose start is less than or equal to the start of the range to insert,
        // or the one before that if both of the above cases exist.
        let mut candidates = self
            .btm
            .range((Bound::Unbounded, Bound::Included(&range.start)))
            .rev()
            .take(2)
            .filter(|(start, Entry { end, value: _ })| {
                // Does the candidate range either overlap
                // or immediately precede the range to insert?
                // (Remember that it might actually cover the _whole_
                // range to insert and then some.)
                (*start..end).touches(&(&range.start..&range.end))
            });

        if let Some(mut candidate) = candidates.next() {
            // Or the one before it if both cases described above exist.
            if let Some(another_candidate) = candidates.next() {
                candidate = another_candidate;
            }

            let (stored_start, stored_entry) = (candidate.0.clone(), candidate.1.clone());

            self.adjust_touching_ranges_for_insert(
                stored_start,
                stored_entry,
                &mut range,
                &new_value,
            );
        }

        // Are there any stored ranges whose heads overlap or immediately
        // follow the range to insert?
        //
        // If there are any such stored ranges (that weren't already caught above),
        // their starts will fall somewhere after the start of the range to insert,
        // and on or before its end.
        //
        // This time around, if the latter holds, it also implies
        // the former so we don't need to check here if they touch.
        while let Some((start, entry)) = self
            .btm
            .range((Bound::Included(&range.start), Bound::Included(&range.end)))
            .next()
        {
            // One extra exception: if we have different values,
            // and the stored range starts at the end of the range to insert,
            // then we don't want to keep looping forever trying to find more!
            #[allow(clippy::suspicious_operation_groupings)]
            if start == &range.end && entry.value != new_value {
                // We're beyond the last stored range that could be relevant.
                // Avoid wasting time on irrelevant ranges, or even worse, looping forever.
                // (`adjust_touching_ranges_for_insert` below assumes that the given range
                // is relevant, and behaves very poorly if it is handed a range that it
                // shouldn't be touching.)
                break;
            }

            let stored_start = start.clone();
            let stored_entry = entry.clone();

            self.adjust_touching_ranges_for_insert(
                stored_start,
                stored_entry,
                &mut range,
                &new_value,
            );
        }

        // Insert the (possibly expanded) new range, and we're done!
        self.btm.insert(
            range.start,
            Entry {
                end: range.end,
                value: new_value,
            },
        );
    }

    /// Removes a range from the map, if all or any of it was present.
    ///
    /// If the range to be removed _partially_ overlaps any ranges
    /// in the map, then those ranges will be contracted to no
    /// longer cover the removed range.
    ///
    ///
    /// # Panics
    ///
    /// Panics if range `start >= end`.
    pub fn remove(&mut self, range: Range<K>) {
        // We don't want to have to make empty ranges make sense;
        // they don't represent anything meaningful in this structure.
        assert!(range.start < range.end);

        // Is there a stored range overlapping the start of
        // the range to remove?
        //
        // If there is any such stored range, it will be the last
        // whose start is less than or equal to the start of the range to remove.
        if let Some((stored_start, stored_entry)) = self
            .btm
            .range((Bound::Unbounded, Bound::Included(&range.start)))
            .next_back()
            .filter(|(start, Entry { end, value: _ })| {
                // Does the only candidate range overlap
                // the range to remove?
                (*start..end).overlaps(&(&range.start..&range.end))
            })
            .map(|(stored_start, stored_entry)| (stored_start.clone(), stored_entry.clone()))
        {
            self.adjust_overlapping_ranges_for_remove(stored_start, stored_entry, &range);
        }

        // Are there any stored ranges whose heads overlap the range to remove?
        //
        // If there are any such stored ranges (that weren't already caught above),
        // their starts will fall somewhere after the start of the range to remove,
        // and before its end.
        while let Some((stored_start, stored_entry)) = self
            .btm
            .range((Bound::Excluded(&range.start), Bound::Excluded(&range.end)))
            .next()
            .map(|(stored_start, stored_entry)| (stored_start.clone(), stored_entry.clone()))
        {
            self.adjust_overlapping_ranges_for_remove(stored_start, stored_entry, &range);
        }
    }

    fn adjust_touching_ranges_for_insert(
        &mut self,
        stored_start: K,
        stored_entry: Entry<K, V>,
        new_range: &mut Range<K>,
        new_value: &V,
    ) {
        if stored_entry.value == *new_value {
            // The ranges have the same value, so we can "adopt"
            // the stored range.
            //
            // This means that no matter how big or where the stored range is,
            // we will expand the new range's bounds to subsume it,
            // and then delete the stored range.
            new_range.start = cmp::min(&new_range.start, &stored_start).clone();
            new_range.end = cmp::max(&new_range.end, &stored_entry.end).clone();
            self.btm.remove(&stored_start);
        } else {
            // The ranges have different values.
            if new_range.overlaps(&(stored_start.clone()..stored_entry.end.clone())) {
                // The ranges overlap. This is a little bit more complicated.
                // Delete the stored range, and then add back between
                // 0 and 2 subranges at the ends of the range to insert.
                self.btm.remove(&stored_start);
                if stored_start < new_range.start {
                    // Insert the piece left of the range to insert.
                    self.btm.insert(
                        stored_start,
                        Entry {
                            end: new_range.start.clone(),
                            value: stored_entry.value.clone(),
                        },
                    );
                }
                if stored_entry.end > new_range.end {
                    // Insert the piece right of the range to insert.
                    self.btm.insert(new_range.end.clone(), stored_entry);
                }
            } else {
                // No-op; they're not overlapping,
                // so we can just keep both ranges as they are.
            }
        }
    }

    fn adjust_overlapping_ranges_for_remove(
        &mut self,
        stored_start: K,
        stored_entry: Entry<K, V>,
        range_to_remove: &Range<K>,
    ) {
        // Delete the stored range, and then add back between
        // 0 and 2 subranges at the ends of the range to remove.
        self.btm.remove(&stored_start);

        if stored_start < range_to_remove.start {
            // Insert the piece left of the range to remove.
            self.btm.insert(
                stored_start,
                Entry {
                    end: range_to_remove.start.clone(),
                    value: stored_entry.value.clone(),
                },
            );
        }

        if stored_entry.end > range_to_remove.end {
            // Insert the piece right of the range to remove.
            self.btm.insert(range_to_remove.end.clone(), stored_entry);
        }
    }

    /// Splits a range in two at the provided key.
    ///
    /// Does nothing if no range exists at the key, or if the key is at a range boundary.
    pub fn split_at(&mut self, key: &K) {
        // Find a range that contains the key, but doesn't start or end with the key.
        let bounds = (Bound::Unbounded, Bound::Excluded(key.clone()));

        if let Some((_start, entry)) = self
            .btm
            .range_mut(bounds)
            .next_back()
            .filter(|(_start, Entry { end, value: _ })| end > key)
        {
            let second_half_entry = entry.clone();
            // Adjust the end of the range.
            entry.end = key.clone();
            // Insert the second half of the range.
            self.btm.insert(key.clone(), second_half_entry);
        }
    }

    /// Gets an iterator over all pairs of key range and value, where the key range overlaps with
    /// the provided range.
    ///
    /// The iterator element type is `(&Range<K>, &V)`.
    pub fn range(&self, range: &Range<K>) -> RangeIter<'_, K, V> {
        let start = self
            .get_key_value(&range.start)
            .map_or(range.start.clone(), |(k, _v)| k.start);
        let end = range.end.clone();

        RangeIter {
            inner: self
                .btm
                .range((Bound::Included(start), Bound::Excluded(end))),
        }
    }

    /// Gets a mutable iterator over all pairs of key range and value, where the key range overlaps
    /// with the provided range.
    ///
    /// The iterator element type is `(&Range<K>, &mut V)`.
    pub fn range_mut(&mut self, range: &Range<K>) -> RangeMutIter<'_, K, V> {
        let start = self
            .get_key_value(&range.start)
            .map_or(range.start.clone(), |(k, _v)| k.start);
        let end = range.end.clone();

        RangeMutIter {
            inner: self
                .btm
                .range_mut((Bound::Included(start), Bound::Excluded(end))),
        }
    }
}

/// An iterator over the entries of a `RangeMap`, ordered by key range.
///
/// The iterator element type is `(&'a Range<K>, &'a V)`.
///
/// This `struct` is created by the [`iter`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`iter`]: RangeMap::iter
pub struct Iter<'a, K, V> {
    inner: btree_map::Iter<'a, K, Entry<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: 'a + Clone,
    V: 'a,
{
    type Item = (Range<K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(start, Entry { end, value })| (start.clone()..end.clone(), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> where K: Ord + Clone {}
impl<K, V> ExactSizeIterator for Iter<'_, K, V> where K: Ord + Clone {}

/// An iterator over the entries of a `RangeMap`, ordered by key range.
///
/// The iterator element type is `(&'a Range<K>, &'a V)`.
///
/// This `struct` is created by the [`iter`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`iter`]: RangeMap::iter
pub struct IterMut<'a, K, V> {
    inner: btree_map::IterMut<'a, K, Entry<K, V>>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V>
where
    K: 'a + Clone,
    V: 'a,
{
    type Item = (Range<K>, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(start, Entry { end, value })| (start.clone()..end.clone(), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> FusedIterator for IterMut<'_, K, V> where K: Ord + Clone {}
impl<K, V> ExactSizeIterator for IterMut<'_, K, V> where K: Ord + Clone {}

/// An owning iterator over the entries of a `RangeMap`, ordered by key range.
///
/// The iterator element type is `(Range<K>, V)`.
///
/// This `struct` is created by the [`into_iter`] method on [`RangeMap`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
pub struct IntoIter<K, V> {
    inner: btree_map::IntoIter<K, Entry<K, V>>,
}

impl<K, V> IntoIterator for RangeMap<K, V> {
    type Item = (Range<K>, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.btm.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (Range<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(start, Entry { end, value })| (start..end, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> where K: Ord + Clone {}
impl<K, V> ExactSizeIterator for IntoIter<K, V> where K: Ord + Clone {}

// We can't just derive this automatically, because that would
// expose irrelevant (and private) implementation details.
// Instead implement it in the same way that the underlying BTreeMap does.
impl<K: Debug, V: Debug> Debug for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V> FromIterator<(Range<K>, V)> for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn from_iter<T: IntoIterator<Item = (Range<K>, V)>>(iter: T) -> Self {
        let mut range_map = RangeMap::new();
        range_map.extend(iter);
        range_map
    }
}

impl<K, V> Extend<(Range<K>, V)> for RangeMap<K, V>
where
    K: Ord + Clone,
    V: Eq + Clone,
{
    fn extend<T: IntoIterator<Item = (Range<K>, V)>>(&mut self, iter: T) {
        iter.into_iter().for_each(move |(k, v)| {
            self.insert(k, v);
        })
    }
}

/// An iterator over entries of a `RangeMap` whose range overlaps with a specified range.
///
/// The iterator element type is `(&'a Range<K>, &'a V)`.
///
/// This `struct` is created by the [`range`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`range`]: RangeMap::range
pub struct RangeIter<'a, K, V> {
    inner: btree_map::Range<'a, K, Entry<K, V>>,
}

impl<K, V> FusedIterator for RangeIter<'_, K, V> where K: Ord + Clone {}

impl<'a, K, V> Iterator for RangeIter<'a, K, V>
where
    K: 'a + Clone,
    V: 'a,
{
    type Item = (Range<K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(start, Entry { end, value })| (start.clone()..end.clone(), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// A mutable iterator over entries of a `RangeMap` whose range overlaps with a specified range.
///
/// The iterator element type is `(&'a Range<K>, &'a mut V)`.
///
/// This `struct` is created by the [`range_mut`] method on [`RangeMap`]. See its
/// documentation for more.
///
/// [`range_mut`]: RangeMap::range_mut
pub struct RangeMutIter<'a, K, V> {
    inner: btree_map::RangeMut<'a, K, Entry<K, V>>,
}

impl<K, V> FusedIterator for RangeMutIter<'_, K, V> where K: Ord + Clone {}

impl<'a, K, V> Iterator for RangeMutIter<'a, K, V>
where
    K: 'a + Clone,
    V: 'a,
{
    type Item = (Range<K>, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(start, Entry { end, value })| (start.clone()..end.clone(), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub trait RangeExt<T> {
    fn overlaps(&self, other: &Self) -> bool;
    fn touches(&self, other: &Self) -> bool;
}

impl<T> RangeExt<T> for Range<T>
where
    T: Ord,
{
    fn overlaps(&self, other: &Self) -> bool {
        // Strictly less than, because ends are excluded.
        cmp::max(&self.start, &other.start) < cmp::min(&self.end, &other.end)
    }

    fn touches(&self, other: &Self) -> bool {
        // Less-than-or-equal-to because if one end is excluded, the other is included.
        // I.e. the two could be joined into a single range, because they're overlapping
        // or immediately adjacent.
        cmp::max(&self.start, &other.start) <= cmp::min(&self.end, &other.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{format, vec, vec::Vec};

    trait RangeMapExt<K, V> {
        fn to_vec(&self) -> Vec<(Range<K>, V)>;
    }

    impl<K, V> RangeMapExt<K, V> for RangeMap<K, V>
    where
        K: Ord + Clone,
        V: Eq + Clone,
    {
        fn to_vec(&self) -> Vec<(Range<K>, V)> {
            self.iter().map(|(kr, v)| (kr, v.clone())).collect()
        }
    }

    //
    // Insertion tests
    //

    #[test]
    fn empty_map_is_empty() {
        let range_map: RangeMap<u32, bool> = RangeMap::new();
        assert_eq!(range_map.to_vec(), vec![]);
    }

    #[test]
    fn insert_into_empty_map() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(0..50, false);
        assert_eq!(range_map.to_vec(), vec![(0..50, false)]);
    }

    #[test]
    fn new_same_value_immediately_following_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_immediately_following_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..3, false), (3..5, true)]);
    }

    #[test]
    fn new_same_value_overlapping_end_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-----◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_overlapping_end_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-----◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..3, false), (3..5, true)]);
    }

    #[test]
    fn new_same_value_immediately_preceding_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_immediately_preceding_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..3, false), (3..5, true)]);
    }

    #[test]
    fn new_same_value_wholly_inside_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        range_map.insert(1..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(2..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    #[test]
    fn new_different_value_wholly_inside_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◆-------◇ ◌ ◌ ◌ ◌
        range_map.insert(1..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(2..4, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-◌ ◌ ◌ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◆---◇ ◌ ◌ ◌ ◌ ◌
        // ◌ ◌ ◌ ◌ ●-◌ ◌ ◌ ◌ ◌
        assert_eq!(
            range_map.to_vec(),
            vec![(1..2, true), (2..4, false), (4..5, true)]
        );
    }

    #[test]
    fn replace_at_end_of_existing_range_should_coalesce() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●---◌ ◌ ◌ ◌ ◌ ◌ ◌
        range_map.insert(1..3, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, true);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ◌ ◌ ●---◌ ◌ ◌ ◌ ◌
        range_map.insert(3..5, false);
        // 0 1 2 3 4 5 6 7 8 9
        // ◌ ●-------◌ ◌ ◌ ◌ ◌
        assert_eq!(range_map.to_vec(), vec![(1..5, false)]);
    }

    //
    // Get* tests
    //

    #[test]
    fn get() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(0..50, false);
        assert_eq!(range_map.get(&49), Some(&false));
        assert_eq!(range_map.get(&50), None);
    }

    #[test]
    fn get_key_value() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(0..50, false);
        assert_eq!(range_map.get_key_value(&49), Some((0..50, &false)));
        assert_eq!(range_map.get_key_value(&50), None);
    }

    //
    // Removal tests
    //

    #[test]
    fn remove_from_empty_map() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.remove(0..50);
        assert_eq!(range_map.to_vec(), vec![]);
    }

    #[test]
    fn remove_non_covered_range_before_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(0..25);
        assert_eq!(range_map.to_vec(), vec![(25..75, false)]);
    }

    #[test]
    fn remove_non_covered_range_after_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(75..100);
        assert_eq!(range_map.to_vec(), vec![(25..75, false)]);
    }

    #[test]
    fn remove_overlapping_start_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(0..30);
        assert_eq!(range_map.to_vec(), vec![(30..75, false)]);
    }

    #[test]
    fn remove_middle_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(30..70);
        assert_eq!(range_map.to_vec(), vec![(25..30, false), (70..75, false)]);
    }

    #[test]
    fn remove_overlapping_end_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(70..100);
        assert_eq!(range_map.to_vec(), vec![(25..70, false)]);
    }

    #[test]
    fn remove_exactly_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(25..75);
        assert_eq!(range_map.to_vec(), vec![]);
    }

    #[test]
    fn remove_superset_of_stored() {
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(25..75, false);
        range_map.remove(0..100);
        assert_eq!(range_map.to_vec(), vec![]);
    }

    // impl Debug

    #[test]
    fn map_debug_repr_looks_right() {
        let mut map: RangeMap<u32, ()> = RangeMap::new();

        // Empty
        assert_eq!(format!("{:?}", map), "{}");

        // One entry
        map.insert(2..5, ());
        assert_eq!(format!("{:?}", map), "{2..5: ()}");

        // Many entries
        map.insert(6..7, ());
        map.insert(8..9, ());
        assert_eq!(format!("{:?}", map), "{2..5: (), 6..7: (), 8..9: ()}");
    }

    // Iterator Tests

    #[test]
    fn into_iter_matches_iter() {
        // Just use vec since that's the same implementation we'd expect
        let mut range_map: RangeMap<u32, bool> = RangeMap::new();
        range_map.insert(1..3, false);
        range_map.insert(3..5, true);

        let cloned = range_map.to_vec();
        let consumed = range_map.into_iter().collect::<Vec<_>>();

        // Correct value
        assert_eq!(cloned, vec![(1..3, false), (3..5, true)]);

        // Equality
        assert_eq!(cloned, consumed);
    }
}
