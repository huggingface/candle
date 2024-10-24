use std::{
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::atomic::AtomicU32,
};

use tracing::{instrument, span};

pub trait ToU32 {
    fn to_u32(self) -> u32;
}

impl ToU32 for i32 {
    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl ToU32 for u32 {
    fn to_u32(self) -> u32 {
        self
    }
}

impl ToU32 for f32 {
    fn to_u32(self) -> u32 {
        f32::to_bits(self)
    }
}

impl ToU32 for usize {
    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl ToU32 for bool {
    fn to_u32(self) -> u32 {
        if self { 1 } else { 0 }
    }
}

pub trait ToU64 {
    fn to_u64(self) -> u64;
}

impl ToU64 for i32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}


impl ToU64 for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}

impl ToU64 for usize {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

pub trait ToF64 {
    fn to_f64(self) -> f64;
}

impl ToF64 for usize {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToF64 for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToF64 for bool {
    fn to_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}

#[derive(Debug)]
pub(crate) struct HashMapMulti<K, V> {
    pub(crate) map: HashMap<K, Vec<V>>,
    pub(crate) _empty: Vec<V>,
}


impl<K: std::cmp::Eq + PartialEq + std::hash::Hash, V> HashMapMulti<K, V> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            _empty: vec![],
        }
    }

    #[instrument(skip(self, key, value))]
    pub fn add_mapping(&mut self, key: K, value: V) {
        self.map.entry(key).or_default().push(value);
    }

    #[instrument(skip(self, key))]
    pub fn get(&self, key: &K) -> &Vec<V> {
        self.map.get(key).unwrap_or(&self._empty)
    }
}

impl<K: std::cmp::Eq + PartialEq + std::hash::Hash, V: PartialEq> HashMapMulti<K, V> {
    

    #[instrument(skip(self, key, value))]
    pub fn remove_mapping(&mut self, key: &K, value: &V) {
        if let Some(vec) = self.map.get_mut(key) {
            if let Some(pos) = vec.iter().position(|x| x == value) {
                vec.remove(pos);
            }

            if vec.is_empty() {
                self.map.remove(key);
            }
        }
    }
}

#[derive(Debug)]
pub struct FixedSizeQueue<T> {
    pub(crate) deque: VecDeque<T>,
    capacity: usize,
}

impl<T> FixedSizeQueue<T> {
    // Create a new FixedSizeQueue with the specified capacity
    pub fn new(capacity: usize) -> Self {
        FixedSizeQueue {
            deque: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    // Push a new element into the queue
    pub fn push(&mut self, item: T) {
        if self.deque.len() == self.capacity {
            self.deque.pop_front(); // Remove the oldest element if the capacity is reached
        }
        self.deque.push_back(item);
    }

    // Iterate over the elements in the queue
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.deque.iter()
    }
}

#[derive(Debug)]
pub struct Counter(AtomicU32);

impl Counter {
    pub fn new(default: u32) -> Self {
        Counter(AtomicU32::new(default))
    }

    pub fn inc(&self) -> u32 {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn get(&self) -> u32 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FixedArray<T, const TSIZE: usize> {
    data: [T; TSIZE],
    len: usize,
}

impl<const TSIZE: usize, T: std::marker::Copy + std::default::Default> FixedArray<T, TSIZE> {
    pub fn new() -> Self {
        FixedArray {
            data: [Default::default(); TSIZE],
            len: 0,
        }
    }

    pub fn get(&self) -> &[T] {
        &self.data[0..self.len]
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        return self.data[0..self.len].iter();
    }

    pub fn from_vec(vec: &[T]) -> Self {
        let mut data = [Default::default(); TSIZE];
        let len = vec.len();
        assert!(len <= TSIZE);
        data[..len].copy_from_slice(&vec[..len]);
        FixedArray { data, len }
    }

    pub fn push(&mut self, data: T) {
        self.data[self.len] = data;
        self.len += 1;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<const TSIZE: usize, T: std::marker::Copy + std::default::Default> Default for FixedArray<T, TSIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const TSIZE: usize, T: Hash> Hash for FixedArray<T, TSIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for x in self.data[0..self.len].iter() {
            x.hash(state);
        }
    }
}

impl<const TSIZE: usize, T: Hash + std::cmp::PartialEq> PartialEq for FixedArray<T, TSIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data[..self.len] == other.data[..other.len]
    }
}

impl<const TSIZE: usize, T: Hash + std::cmp::PartialEq> Eq for FixedArray<T, TSIZE> {}

#[derive(Debug)]
pub struct ObjectToIdMapper<K> {
    map: HashMap<K, usize>,
    pub(crate) next_id: usize,
}

impl<K: std::cmp::Eq + Hash + Clone> ObjectToIdMapper<K> {
    pub fn new() -> Self {
        ObjectToIdMapper {
            map: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn insert_force(&mut self, key: &K, value: usize) {
        self.map.insert(key.clone(), value);
        self.next_id = value;
    }

    pub fn get_or_insert(&mut self, key: &K) -> (usize, bool) {
        if let Some(id) = self.map.get(key) {
            (*id, false)
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.map.insert(key.clone(), id);
            (id, true)
        }
    }
}

impl<K: std::cmp::Eq + Hash + Clone> Default for ObjectToIdMapper<K> {
    fn default() -> Self {
        Self::new()
    }
}

///////////// STORAGE Field Helper Struct:

#[derive(Debug, Clone, Eq, PartialEq, Hash, std::marker::Copy, Default)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Reference {
    #[cfg_attr(
        any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
        serde(skip)
    )]
    id: u32,
    #[cfg_attr(
        any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
        serde(skip)
    )]
    time: ReferenceTime,
}

pub trait ReferenceTrait {
    fn new(id: u32, time: ReferenceTime) -> Self;
    fn id(&self) -> u32;
    fn time(&self) -> ReferenceTime;
    fn is_valid(&self) -> bool {
        self.time() != 0
    }
}

impl Reference {
    pub fn new<T: ToU32>(id: T, time: u32) -> Self {
        Self {
            id: id.to_u32(),
            time,
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn time(&self) -> ReferenceTime {
        self.time
    }

    pub fn is_valid(&self) -> bool {
        self.time != 0
    }
}

type ReferenceTime = u32;

#[derive(Debug)]
struct StorageField<T> {
    time_stamp: ReferenceTime,
    is_used: bool,
    value: T,
}

#[derive(Debug)]
pub struct Storage<T, TREF> {
    data: Vec<StorageField<T>>,
    free: Vec<usize>, //free references
    phantom: PhantomData<TREF>,
}

#[derive(Debug)]
pub struct StorageOptional<T, TREF> {
    data: Vec<StorageField<Option<T>>>,
    free: Vec<usize>, //free references
    phantom: PhantomData<TREF>,
}

pub trait StorageTrait<T, TREF> {
    fn new() -> Self;
    fn insert(&mut self, value: T) -> TREF;
    fn get(&self, id: &TREF) -> Option<&T>;
    fn get_mut(&mut self, id: &TREF) -> Option<&mut T>;
    fn delete(&mut self, id: &TREF) -> bool;

    //returns the Reference at the i-th index in this storage
    fn get_reference(&self, i: u32) -> Option<(TREF, &T)>;
    fn iter<'a>(&'a self) -> impl Iterator<Item = &T>
    where
        T: 'a;
    fn retain(&mut self, keep: impl Fn((&TREF, &T)) -> bool);
    fn retain_mut(&mut self, keep: impl FnMut((&TREF, &T)) -> bool);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, TREF> StorageTrait<T, TREF> for StorageOptional<T, TREF>
where
    TREF: ReferenceTrait,
{
    fn new() -> Self {
        Self {
            data: vec![],
            free: vec![],
            phantom: PhantomData,
        }
    }

    fn insert(&mut self, value: T) -> TREF {
        if let Some(id) = self.free.pop() {
            self.data[id].value = Some(value);
            self.data[id].is_used = true;
            TREF::new(id as u32, self.data[id].time_stamp)
        } else {
            let field = StorageField {
                time_stamp: 1,
                value: Some(value),
                is_used: true,
            };
            self.data.push(field);
            TREF::new((self.data.len() - 1) as u32, 1)
        }
    }

    fn get(&self, id: &TREF) -> Option<&T> {
        if let Some(val) = self.data.get(id.id() as usize) {
            if val.time_stamp == id.time() {
                return val.value.as_ref();
            }
        }
        None
    }

    fn get_mut(&mut self, id: &TREF) -> Option<&mut T> {
        if let Some(val) = self.data.get_mut(id.id() as usize) {
            if val.time_stamp == id.time() {
                return val.value.as_mut();
            }
        }
        None
    }

    fn delete(&mut self, id: &TREF) -> bool {
        if let Some(val) = self.data.get_mut(id.id() as usize) {
            if val.time_stamp == id.time() {
                val.is_used = false;
                val.time_stamp += 1;
                val.value = None;
                self.free.push(id.id() as usize);
                return true;
            }
        }
        false
    }

    fn get_reference(&self, pos: u32) -> Option<(TREF, &T)> {
        if let Some(val) = self.data.get(pos as usize) {
            if val.is_used {
                let value = val.value.as_ref()?;
                return Some((TREF::new(pos, val.time_stamp), value));
            }
        }
        None
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &T>
    where
        T: 'a,
    {
        self.data
            .iter()
            .filter_map(|c| if !c.is_used { None } else { c.value.as_ref() })
    }

    #[instrument(skip(self, keep))]
    fn retain(&mut self, keep: impl Fn((&TREF, &T)) -> bool) {
        for (index, sf) in self.data.iter_mut().enumerate() {
            if let Some(val) = &sf.value {
                if !keep((&TREF::new(index as u32, sf.time_stamp), val)) {
                    sf.is_used = false;
                    sf.time_stamp += 1;
                    sf.value = None;
                    self.free.push(index);
                }
            }
        }
    }

    #[instrument(skip(self, keep))]
    fn retain_mut(&mut self, mut keep: impl FnMut((&TREF, &T)) -> bool) {
        for (index, sf) in self.data.iter_mut().enumerate() {
            if let Some(val) = &sf.value {
                if !keep((&TREF::new(index as u32, sf.time_stamp), val)) {
                    sf.is_used = false;
                    sf.time_stamp += 1;
                    let span1 = span!(tracing::Level::INFO, "Storage Setting to None");
                    let _enter1 = span1.enter();
                    sf.value = None;
                    drop(_enter1);
                    self.free.push(index);
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.data.len() - self.free.len()
    }
}

impl<T, TREF> StorageOptional<T, TREF>
where
    TREF: ReferenceTrait,
{
    pub fn delete_move(&mut self, id: &TREF) -> Option<T> {
        if let Some(val) = self.data.get_mut(id.id() as usize) {
            if val.time_stamp == id.time() {
                val.time_stamp += 1;
                val.is_used = false;
                self.free.push(id.id() as usize);
                return val.value.take();
            }
        }
        None
    }

    pub fn iter_option(&self) -> impl Iterator<Item = &T> {
        self.data
            .iter()
            .filter_map(|c| if !c.is_used { None } else { c.value.as_ref() })
    }

    pub fn iter_mut_option(&mut self) -> impl Iterator<Item = &mut T> {
        self.data
            .iter_mut()
            .filter_map(|c| if !c.is_used { None } else { c.value.as_mut() })
    }

    pub fn enumerate_option(&self) -> impl Iterator<Item = (TREF, &T)> {
        self.data.iter().enumerate().filter_map(|(i, c)| {
            if !c.is_used {
                None
            } else {
                Some((TREF::new(i as u32, c.time_stamp), c.value.as_ref()?))
            }
        })
    }

    pub fn enumerate_mut_option(&mut self) -> impl Iterator<Item = (TREF, &mut T)> {
        self.data.iter_mut().enumerate().filter_map(|(i, c)| {
            if !c.is_used {
                None
            } else {
                Some((TREF::new(i as u32, c.time_stamp), c.value.as_mut()?))
            }
        })
    }
}

impl<T, TREF> StorageTrait<T, TREF> for Storage<T, TREF>
where
    TREF: ReferenceTrait,
{
    fn new() -> Self {
        Self {
            data: vec![],
            free: vec![],
            phantom: PhantomData,
        }
    }

    fn insert(&mut self, referece: T) -> TREF {
        if let Some(id) = self.free.pop() {
            self.data[id].value = referece;
            self.data[id].is_used = true;
            TREF::new(id as u32, self.data[id].time_stamp)
        } else {
            let field = StorageField {
                time_stamp: 1,
                value: referece,
                is_used: true,
            };
            self.data.push(field);
            TREF::new((self.data.len() - 1) as u32, 1)
        }
    }

    fn get(&self, id: &TREF) -> Option<&T> {
        if let Some(val) = self.data.get(id.id() as usize) {
            if val.time_stamp == id.time() {
                return Some(&val.value);
            }
        }
        None
    }

    fn get_mut(&mut self, id: &TREF) -> Option<&mut T> {
        if let Some(val) = self.data.get_mut(id.id() as usize) {
            if val.time_stamp == id.time() {
                return Some(&mut val.value);
            }
        }
        None
    }

    fn delete(&mut self, id: &TREF) -> bool {
        if let Some(val) = self.data.get_mut(id.id() as usize) {
            if val.time_stamp == id.time() {
                val.time_stamp += 1;
                val.is_used = false;
                self.free.push(id.id() as usize);
                return true;
            }
        }
        false
    }

    fn get_reference(&self, pos: u32) -> Option<(TREF, &T)> {
        if let Some(val) = self.data.get(pos as usize) {
            if val.is_used {
                let value = &val.value;
                return Some((TREF::new(pos, val.time_stamp), value));
            }
        }
        None
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &T>
    where
        T: 'a,
    {
        self.data.iter().filter(|c| !c.is_used).map(|c| &c.value)
    }

    fn retain(&mut self, keep: impl Fn((&TREF, &T)) -> bool) {
        for (index, sf) in self.data.iter_mut().enumerate() {
            if !keep((&TREF::new(index as u32, sf.time_stamp), &sf.value)) {
                sf.is_used = false;
                sf.time_stamp += 1;
                self.free.push(index);
            }
        }
    }

    fn retain_mut(&mut self, mut keep: impl FnMut((&TREF, &T)) -> bool) {
        for (index, sf) in self.data.iter_mut().enumerate() {
            if !keep((&TREF::new(index as u32, sf.time_stamp), &sf.value)) {
                sf.is_used = false;
                sf.time_stamp += 1;
                self.free.push(index);
            }
        }
    }

    fn len(&self) -> usize {
        self.data.len() - self.free.len()
    }
}