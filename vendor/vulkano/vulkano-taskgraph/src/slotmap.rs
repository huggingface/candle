use concurrent_slotmap::{Key, SlotId};
use core::slice;
use std::{
    iter::{self, FusedIterator},
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
};

const NIL: u32 = u32::MAX;

const TAG_MASK: u32 = SlotId::TAG_MASK;

const STATE_MASK: u32 = SlotId::STATE_MASK;

const OCCUPIED_TAG: u32 = SlotId::OCCUPIED_TAG;

const GENERATION_MASK: u32 = u32::MAX << (SlotId::TAG_BITS + SlotId::STATE_BITS);

const ONE_GENERATION: u32 = 1 << (SlotId::TAG_BITS + SlotId::STATE_BITS);

pub struct SlotMap<K, V> {
    inner: SlotMapInner<V>,
    marker: PhantomData<fn(K) -> K>,
}

struct SlotMapInner<V> {
    slots: Vec<Slot<V>>,
    len: u32,
    free_list_head: u32,
}

impl<K, V> Default for SlotMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::with_key()
    }
}

impl<V> SlotMap<SlotId, V> {
    #[cfg(test)]
    #[inline]
    pub fn new() -> Self {
        Self::with_key()
    }
}

impl<K, V> SlotMap<K, V> {
    #[inline]
    pub fn with_key() -> Self {
        SlotMap {
            inner: SlotMapInner {
                slots: Vec::new(),
                len: 0,
                free_list_head: NIL,
            },
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn reserved_len(&self) -> u32 {
        self.inner.slots.len() as u32
    }

    #[inline]
    pub fn len(&self) -> u32 {
        self.inner.len
    }
}

impl<K: Key, V> SlotMap<K, V> {
    #[inline]
    pub fn insert(&mut self, value: V) -> K {
        K::from_id(self.inner.insert_with_tag(value, 0))
    }

    #[inline]
    pub fn insert_with_tag(&mut self, value: V, tag: u32) -> K {
        K::from_id(self.inner.insert_with_tag(value, tag))
    }

    #[inline]
    pub fn remove(&mut self, key: K) -> Option<V> {
        self.inner.remove(key.as_id())
    }

    #[inline(always)]
    pub fn get(&self, key: K) -> Option<&V> {
        self.inner.get(key.as_id())
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, key: K) -> &V {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.get_unchecked(key.as_id()) }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.inner.get_mut(key.as_id())
    }

    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, key: K) -> &mut V {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.get_unchecked_mut(key.as_id()) }
    }

    #[inline]
    pub fn get_many_mut<const N: usize>(&mut self, keys: [K; N]) -> Option<[&mut V; N]> {
        self.inner.get_many_mut(keys.map(K::as_id))
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.inner.slots.iter().enumerate(),
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.inner.slots.iter_mut().enumerate(),
            marker: PhantomData,
        }
    }
}

impl<V> SlotMapInner<V> {
    fn insert_with_tag(&mut self, value: V, tag: u32) -> SlotId {
        assert_eq!(tag & !TAG_MASK, 0);

        if self.free_list_head != NIL {
            // SAFETY: We always push indices of existing slots into the free-list and the slots
            // vector never shrinks, therefore the index must have staid in bounds.
            let slot = unsafe { self.slots.get_unchecked_mut(self.free_list_head as usize) };

            let index = self.free_list_head;
            let generation = slot.generation | OCCUPIED_TAG | tag;

            // SAFETY: We always link free slots into the free-list by setting the `next_free`
            // union field.
            self.free_list_head = unsafe { slot.inner.next_free };

            slot.generation = generation;
            slot.inner.value = ManuallyDrop::new(value);

            self.len += 1;

            // SAFETY: The generation's state tag is `OCCUPIED_TAG`.
            unsafe { SlotId::new_unchecked(index, generation) }
        } else {
            if self.slots.len() == (NIL - 1) as usize {
                capacity_overflow();
            }

            let index = self.slots.len() as u32;
            let generation = OCCUPIED_TAG | tag;

            self.slots.push(Slot {
                generation,
                inner: SlotInner {
                    value: ManuallyDrop::new(value),
                },
            });

            self.len += 1;

            // SAFETY: The generation's state tag is `OCCUPIED_TAG`.
            unsafe { SlotId::new_unchecked(index, generation) }
        }
    }

    fn remove(&mut self, id: SlotId) -> Option<V> {
        let slot = self.slots.get_mut(id.index() as usize)?;

        if slot.generation == id.generation() {
            slot.generation = (id.generation() & GENERATION_MASK).wrapping_add(ONE_GENERATION);

            // SAFETY: We checked that `id.generation` matches the slot's generation, which means
            // that the previous state tag of the slot must have been `OCCUPIED_TAG`, which means
            // it must have been initialized in `SlotMap::insert`.
            let value = unsafe { &mut slot.inner.value };

            // SAFETY: We set the slot's state tag to `VACANT_TAG` such that future attempts to
            // access the slot will fail.
            let value = unsafe { ManuallyDrop::take(value) };

            slot.inner.next_free = self.free_list_head;
            self.free_list_head = id.index();

            self.len -= 1;

            Some(value)
        } else {
            None
        }
    }

    #[inline(always)]
    fn get(&self, id: SlotId) -> Option<&V> {
        let slot = self.slots.get(id.index() as usize)?;

        if slot.generation == id.generation() {
            // SAFETY: We checked that `id.generation` matches the slot's generation, which means
            // that the previous state tag of the slot must have been `OCCUPIED_TAG`, which means
            // it must have been initialized in `SlotMap::insert`.
            Some(unsafe { slot.value_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, id: SlotId) -> &V {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked(id.index() as usize) };

        // SAFETY: The caller must ensure that the slot is occupied.
        unsafe { slot.value_unchecked() }
    }

    #[inline(always)]
    fn get_mut(&mut self, id: SlotId) -> Option<&mut V> {
        let slot = self.slots.get_mut(id.index() as usize)?;

        if slot.generation == id.generation() {
            // SAFETY: We checked that `id.generation` matches the slot's generation, which means
            // that the previous state tag of the slot must have been `OCCUPIED_TAG`, which means
            // it must have been initialized in `SlotMap::insert`.
            Some(unsafe { slot.value_unchecked_mut() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, id: SlotId) -> &mut V {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.slots.get_unchecked_mut(id.index() as usize) };

        // SAFETY: The caller must ensure that the slot is occupied.
        unsafe { slot.value_unchecked_mut() }
    }

    #[inline]
    fn get_many_mut<const N: usize>(&mut self, ids: [SlotId; N]) -> Option<[&mut V; N]> {
        #[inline]
        fn get_many_check_valid<const N: usize>(ids: &[SlotId; N], len: u32) -> bool {
            let mut valid = true;

            for (i, id) in ids.iter().enumerate() {
                valid &= id.index() < len;

                for id2 in &ids[..i] {
                    valid &= id.index() != id2.index();
                }
            }

            valid
        }

        if get_many_check_valid(&ids, self.slots.len() as u32) {
            // SAFETY: We checked that all indices are disjunct and in bounds of the slots vector.
            unsafe { self.get_many_unchecked_mut(ids) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_many_unchecked_mut<const N: usize>(
        &mut self,
        ids: [SlotId; N],
    ) -> Option<[&mut V; N]> {
        let slots = self.slots.as_mut_ptr();
        let mut refs = MaybeUninit::<[&mut V; N]>::uninit();
        let refs_ptr = refs.as_mut_ptr().cast::<&mut V>();

        for i in 0..N {
            // SAFETY: `i` is in bounds of the array.
            let id = unsafe { ids.get_unchecked(i) };

            // SAFETY: The caller must ensure that `ids` contains only IDs whose indices are in
            // bounds of the slots vector.
            let slot = unsafe { slots.add(id.index() as usize) };

            // SAFETY: The caller must ensure that `ids` contains only IDs with disjunct indices.
            let slot = unsafe { &mut *slot };

            if slot.generation != id.generation() {
                return None;
            }

            // SAFETY: We checked that `id.generation` matches the slot's generation, which means
            // that the previous state tag of the slot must have been `OCCUPIED_TAG`, which means
            // it must have been initialized in `SlotMap::insert`.
            let value = unsafe { slot.value_unchecked_mut() };

            // SAFETY: `i` is in bounds of the array.
            let ptr = unsafe { refs_ptr.add(i) };

            // SAFETY: The pointer is valid.
            unsafe { *ptr = value };
        }

        // SAFETY: We initialized all the elements.
        Some(unsafe { refs.assume_init() })
    }
}

#[inline(never)]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

struct Slot<V> {
    generation: u32,
    inner: SlotInner<V>,
}

union SlotInner<V> {
    next_free: u32,
    value: ManuallyDrop<V>,
}

impl<V> Slot<V> {
    #[inline(always)]
    unsafe fn value_unchecked(&self) -> &V {
        unsafe { &self.inner.value }
    }

    #[inline(always)]
    unsafe fn value_unchecked_mut(&mut self) -> &mut V {
        unsafe { &mut self.inner.value }
    }
}

impl<V> Drop for Slot<V> {
    #[inline]
    fn drop(&mut self) {
        if is_occupied(self.generation) {
            // SAFETY: We checked that the slot is occupied.
            let value = unsafe { &mut self.inner.value };

            // SAFETY: The fact that the slot is being dropped is proof that the value cannot be
            // accessed again.
            unsafe { ManuallyDrop::drop(value) };
        }
    }
}

macro_rules! declare_key {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident $(;)?
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(transparent)]
        $vis struct $name(concurrent_slotmap::SlotId);

        impl Default for $name {
            #[inline]
            fn default() -> Self {
                Self::INVALID
            }
        }

        impl $name {
            /// An ID that's guaranteed to be invalid.
            pub const INVALID: Self = Self(concurrent_slotmap::SlotId::INVALID);
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        #[doc(hidden)]
        impl concurrent_slotmap::Key for $name {
            #[inline(always)]
            fn from_id(id: concurrent_slotmap::SlotId) -> Self {
                Self(id)
            }

            #[inline(always)]
            fn as_id(self) -> concurrent_slotmap::SlotId {
                self.0
            }
        }
    };
}
pub(crate) use declare_key;

pub struct Iter<'a, K, V> {
    inner: iter::Enumerate<slice::Iter<'a, Slot<V>>>,
    marker: PhantomData<fn(K) -> K>,
}

impl<'a, K: Key, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.inner.next()?;

            if is_occupied(slot.generation) {
                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, slot.generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert`.
                let value = unsafe { slot.value_unchecked() };

                break Some((K::from_id(id), value));
            }
        }
    }
}

impl<K: Key, V> DoubleEndedIterator for Iter<'_, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.inner.next_back()?;

            if is_occupied(slot.generation) {
                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, slot.generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert`.
                let value = unsafe { slot.value_unchecked() };

                break Some((K::from_id(id), value));
            }
        }
    }
}

impl<K: Key, V> FusedIterator for Iter<'_, K, V> {}

pub struct IterMut<'a, K, V> {
    inner: iter::Enumerate<slice::IterMut<'a, Slot<V>>>,
    marker: PhantomData<fn(K) -> K>,
}

impl<'a, K: Key, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.inner.next()?;

            if is_occupied(slot.generation) {
                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, slot.generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert`.
                let value = unsafe { slot.value_unchecked_mut() };

                break Some((K::from_id(id), value));
            }
        }
    }
}

impl<K: Key, V> DoubleEndedIterator for IterMut<'_, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let (index, slot) = self.inner.next_back()?;

            if is_occupied(slot.generation) {
                // SAFETY: We checked that the occupied bit is set.
                let id = unsafe { SlotId::new_unchecked(index as u32, slot.generation) };

                // SAFETY: We checked that the slot is occupied, which means that it must have been
                // initialized in `SlotMap::insert`.
                let value = unsafe { slot.value_unchecked_mut() };

                break Some((K::from_id(id), value));
            }
        }
    }
}

impl<K: Key, V> FusedIterator for IterMut<'_, K, V> {}

fn is_occupied(generation: u32) -> bool {
    generation & STATE_MASK == OCCUPIED_TAG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage1() {
        let mut map = SlotMap::new();

        let x = map.insert(69);
        let y = map.insert(42);

        assert_eq!(map.get(x), Some(&69));
        assert_eq!(map.get(y), Some(&42));

        map.remove(x);

        let x2 = map.insert(12);

        assert_eq!(map.get(x2), Some(&12));
        assert_eq!(map.get(x), None);

        map.remove(y);
        map.remove(x2);

        assert_eq!(map.get(y), None);
        assert_eq!(map.get(x2), None);
    }

    #[test]
    fn basic_usage2() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let y = map.insert(2);
        let z = map.insert(3);

        assert_eq!(map.get(x), Some(&1));
        assert_eq!(map.get(y), Some(&2));
        assert_eq!(map.get(z), Some(&3));

        map.remove(y);

        let y2 = map.insert(20);

        assert_eq!(map.get(y2), Some(&20));
        assert_eq!(map.get(y), None);

        map.remove(x);
        map.remove(z);

        let x2 = map.insert(10);

        assert_eq!(map.get(x2), Some(&10));
        assert_eq!(map.get(x), None);

        let z2 = map.insert(30);

        assert_eq!(map.get(z2), Some(&30));
        assert_eq!(map.get(x), None);

        map.remove(x2);

        assert_eq!(map.get(x2), None);

        map.remove(y2);
        map.remove(z2);

        assert_eq!(map.get(y2), None);
        assert_eq!(map.get(z2), None);
    }

    #[test]
    fn basic_usage3() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let y = map.insert(2);

        assert_eq!(map.get(x), Some(&1));
        assert_eq!(map.get(y), Some(&2));

        let z = map.insert(3);

        assert_eq!(map.get(z), Some(&3));

        map.remove(x);
        map.remove(z);

        let z2 = map.insert(30);
        let x2 = map.insert(10);

        assert_eq!(map.get(x2), Some(&10));
        assert_eq!(map.get(z2), Some(&30));
        assert_eq!(map.get(x), None);
        assert_eq!(map.get(z), None);

        map.remove(x2);
        map.remove(y);
        map.remove(z2);

        assert_eq!(map.get(x2), None);
        assert_eq!(map.get(y), None);
        assert_eq!(map.get(z2), None);
    }

    #[test]
    fn basic_usage_mut1() {
        let mut map = SlotMap::new();

        let x = map.insert(69);
        let y = map.insert(42);

        assert_eq!(map.get_mut(x), Some(&mut 69));
        assert_eq!(map.get_mut(y), Some(&mut 42));

        map.remove(x);

        let x2 = map.insert(12);

        assert_eq!(map.get_mut(x2), Some(&mut 12));
        assert_eq!(map.get_mut(x), None);

        map.remove(y);
        map.remove(x2);

        assert_eq!(map.get_mut(y), None);
        assert_eq!(map.get_mut(x2), None);
    }

    #[test]
    fn basic_usage_mut2() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let y = map.insert(2);
        let z = map.insert(3);

        assert_eq!(map.get_mut(x), Some(&mut 1));
        assert_eq!(map.get_mut(y), Some(&mut 2));
        assert_eq!(map.get_mut(z), Some(&mut 3));

        map.remove(y);

        let y2 = map.insert(20);

        assert_eq!(map.get_mut(y2), Some(&mut 20));
        assert_eq!(map.get_mut(y), None);

        map.remove(x);
        map.remove(z);

        let x2 = map.insert(10);

        assert_eq!(map.get_mut(x2), Some(&mut 10));
        assert_eq!(map.get_mut(x), None);

        let z2 = map.insert(30);

        assert_eq!(map.get_mut(z2), Some(&mut 30));
        assert_eq!(map.get_mut(x), None);

        map.remove(x2);

        assert_eq!(map.get_mut(x2), None);

        map.remove(y2);
        map.remove(z2);

        assert_eq!(map.get_mut(y2), None);
        assert_eq!(map.get_mut(z2), None);
    }

    #[test]
    fn basic_usage_mut3() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let y = map.insert(2);

        assert_eq!(map.get_mut(x), Some(&mut 1));
        assert_eq!(map.get_mut(y), Some(&mut 2));

        let z = map.insert(3);

        assert_eq!(map.get_mut(z), Some(&mut 3));

        map.remove(x);
        map.remove(z);

        let z2 = map.insert(30);
        let x2 = map.insert(10);

        assert_eq!(map.get_mut(x2), Some(&mut 10));
        assert_eq!(map.get_mut(z2), Some(&mut 30));
        assert_eq!(map.get_mut(x), None);
        assert_eq!(map.get_mut(z), None);

        map.remove(x2);
        map.remove(y);
        map.remove(z2);

        assert_eq!(map.get_mut(x2), None);
        assert_eq!(map.get_mut(y), None);
        assert_eq!(map.get_mut(z2), None);
    }

    #[test]
    fn iter1() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let _ = map.insert(2);
        let y = map.insert(3);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(x);
        map.remove(y);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.insert(3);
        map.insert(1);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn iter2() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let y = map.insert(2);
        let z = map.insert(3);

        map.remove(x);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(y);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(z);

        let mut iter = map.iter();

        assert!(iter.next().is_none());
    }

    #[test]
    fn iter3() {
        let mut map = SlotMap::new();

        let _ = map.insert(1);
        let x = map.insert(2);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.remove(x);

        let x = map.insert(2);
        let _ = map.insert(3);
        let y = map.insert(4);

        map.remove(y);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(x);

        let mut iter = map.iter();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_mut1() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let _ = map.insert(2);
        let y = map.insert(3);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(x);
        map.remove(y);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.insert(3);
        map.insert(1);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_mut2() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let y = map.insert(2);
        let z = map.insert(3);

        map.remove(x);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(y);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(z);

        let mut iter = map.iter_mut();

        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_mut3() {
        let mut map = SlotMap::new();

        let _ = map.insert(1);
        let x = map.insert(2);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert!(iter.next().is_none());

        map.remove(x);

        let x = map.insert(2);
        let _ = map.insert(3);
        let y = map.insert(4);

        map.remove(y);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 2);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());

        map.remove(x);

        let mut iter = map.iter_mut();

        assert_eq!(*iter.next().unwrap().1, 1);
        assert_eq!(*iter.next().unwrap().1, 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn reusing_slots1() {
        let mut map = SlotMap::new();

        let x = map.insert(0);
        let y = map.insert(0);

        map.remove(y);

        let y2 = map.insert(0);
        assert_eq!(y2.index(), y.index());
        assert_ne!(y2.generation(), y.generation());

        map.remove(x);

        let x2 = map.insert(0);
        assert_eq!(x2.index(), x.index());
        assert_ne!(x2.generation(), x.generation());

        map.remove(y2);
        map.remove(x2);
    }

    #[test]
    fn reusing_slots2() {
        let mut map = SlotMap::new();

        let x = map.insert(0);

        map.remove(x);

        let x2 = map.insert(0);
        assert_eq!(x.index(), x2.index());
        assert_ne!(x.generation(), x2.generation());

        let y = map.insert(0);
        let z = map.insert(0);

        map.remove(y);
        map.remove(x2);

        let x3 = map.insert(0);
        let y2 = map.insert(0);
        assert_eq!(x3.index(), x2.index());
        assert_ne!(x3.generation(), x2.generation());
        assert_eq!(y2.index(), y.index());
        assert_ne!(y2.generation(), y.generation());

        map.remove(x3);
        map.remove(y2);
        map.remove(z);
    }

    #[test]
    fn reusing_slots3() {
        let mut map = SlotMap::new();

        let x = map.insert(0);
        let y = map.insert(0);

        map.remove(x);
        map.remove(y);

        let y2 = map.insert(0);
        let x2 = map.insert(0);
        let z = map.insert(0);
        assert_eq!(x2.index(), x.index());
        assert_ne!(x2.generation(), x.generation());
        assert_eq!(y2.index(), y.index());
        assert_ne!(y2.generation(), y.generation());

        map.remove(x2);
        map.remove(z);
        map.remove(y2);

        let y3 = map.insert(0);
        let z2 = map.insert(0);
        let x3 = map.insert(0);
        assert_eq!(y3.index(), y2.index());
        assert_ne!(y3.generation(), y2.generation());
        assert_eq!(z2.index(), z.index());
        assert_ne!(z2.generation(), z.generation());
        assert_eq!(x3.index(), x2.index());
        assert_ne!(x3.generation(), x2.generation());

        map.remove(x3);
        map.remove(y3);
        map.remove(z2);
    }

    #[test]
    fn get_many_mut() {
        let mut map = SlotMap::new();

        let x = map.insert(1);
        let y = map.insert(2);
        let z = map.insert(3);

        assert_eq!(map.get_many_mut([x, y]), Some([&mut 1, &mut 2]));
        assert_eq!(map.get_many_mut([y, z]), Some([&mut 2, &mut 3]));
        assert_eq!(map.get_many_mut([z, x]), Some([&mut 3, &mut 1]));

        assert_eq!(map.get_many_mut([x, y, z]), Some([&mut 1, &mut 2, &mut 3]));
        assert_eq!(map.get_many_mut([z, y, x]), Some([&mut 3, &mut 2, &mut 1]));

        assert_eq!(map.get_many_mut([x, x]), None);
        assert_eq!(map.get_many_mut([x, SlotId::new(3, OCCUPIED_TAG)]), None);

        map.remove(y);

        assert_eq!(map.get_many_mut([x, z]), Some([&mut 1, &mut 3]));

        assert_eq!(map.get_many_mut([y]), None);
        assert_eq!(map.get_many_mut([x, y]), None);
        assert_eq!(map.get_many_mut([y, z]), None);

        let y = map.insert(2);

        assert_eq!(map.get_many_mut([x, y, z]), Some([&mut 1, &mut 2, &mut 3]));

        map.remove(x);
        map.remove(z);

        assert_eq!(map.get_many_mut([y]), Some([&mut 2]));

        assert_eq!(map.get_many_mut([x]), None);
        assert_eq!(map.get_many_mut([z]), None);

        map.remove(y);

        assert_eq!(map.get_many_mut([]), Some([]));
    }

    #[test]
    fn tagged() {
        let mut map = SlotMap::new();

        let x = map.insert_with_tag(42, 1);
        assert_eq!(x.generation() & TAG_MASK, 1);
        assert_eq!(map.get(x), Some(&42));
    }

    #[test]
    fn tagged_mut() {
        let mut map = SlotMap::new();

        let x = map.insert_with_tag(42, 1);
        assert_eq!(x.generation() & TAG_MASK, 1);
        assert_eq!(map.get_mut(x), Some(&mut 42));
    }
}
