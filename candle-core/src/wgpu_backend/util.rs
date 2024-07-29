use std::{collections::{BTreeMap, HashMap, VecDeque}, sync::atomic::AtomicU32};

pub(crate) trait ToU32 {
    fn to_u32(self) -> u32;
}

impl ToU32 for i32 {
    fn to_u32(self) -> u32 {
        return self as u32;
    }
}


impl ToU32 for u32 {
    fn to_u32(self) -> u32 {
        return self;
    }
}

impl ToU32 for f32 {
    fn to_u32(self) -> u32 {
        return f32::to_bits(self);
    }
}

impl ToU32 for usize {
    fn to_u32(self) -> u32 {
        return self as u32;
    }
}

pub trait ToU64 {
    fn to_u64(self) -> u64;
}

impl ToU64 for u32 {
    fn to_u64(self) -> u64 {
        return self as u64;
    }
}

impl ToU64 for u64 {
    fn to_u64(self) -> u64 {
        return self;
    }
}

impl ToU64 for usize {
    fn to_u64(self) -> u64 {
        return self as u64;
    }
}


pub trait ToF64 {
    fn to_f64(self) -> f64;
}

impl ToF64 for usize {
    fn to_f64(self) -> f64 {
        return self as f64;
    }
}

impl ToF64 for u32 {
    fn to_f64(self) -> f64 {
        return self as f64;
    }
}

impl ToF64 for bool {
    fn to_f64(self) -> f64 {
        return if self {1.0} else{0.0};
    }
}



#[derive(Debug)]
pub(crate) struct HashMapMulti<K, V> {
    pub(crate) map: HashMap<K, Vec<V>>,
    pub(crate) empty: Vec<V>,
}

#[derive(Debug)]
pub(crate) struct BTreeMulti<K, V> {
    pub(crate) map: BTreeMap<K, Vec<V>>,
    pub(crate) empty: Vec<V>,
}

impl<K: std::cmp::Eq + PartialEq + std::hash::Hash, V> HashMapMulti<K, V> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            empty: vec![],
        }
    }

    pub fn add_mapping(&mut self, key: K, value: V) {
        self.map.entry(key).or_insert_with(Vec::new).push(value);
    }



    // pub fn remove_mapping(&mut self, key : K, value: &V) {
    //      if let Some(vec) = self.map.get_mut(&key) {
    //         if let Some(pos) = vec.iter().position(|x| x == value) {
    //             vec.remove(pos);
    //         }

    //         if vec.is_empty() {
    //             self.map.remove(&key);
    //         }
    //     }
    // }

    pub fn get(&self, key: &K) -> &Vec<V> {
        self.map.get(key).unwrap_or(&self.empty)
    }
}

impl<K: std::cmp::Eq + PartialEq + std::hash::Hash + Ord, V: std::cmp::PartialEq> BTreeMulti<K, V> {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            empty: vec![],
        }
    }

    pub fn add_mapping(&mut self, key: K, value: V) -> bool {
        let values = self.map.entry(key).or_insert_with(Vec::new);
        if !values.contains(&value) {
            values.push(value);
            return true;
        }
        return false;
    }

    // pub fn remove_mapping(&mut self, key : K, value: &V) {
    //      if let Some(vec) = self.map.get_mut(&key) {
    //         if let Some(pos) = vec.iter().position(|x| x == value) {
    //             vec.remove(pos);
    //         }

    //         if vec.is_empty() {
    //             self.map.remove(&key);
    //         }
    //     }
    // }

    pub fn get(&self, key: &K) -> &Vec<V> {
        self.map.get(key).unwrap_or(&self.empty)
    }

    pub fn get_mut(&mut self, key: &K) -> &mut Vec<V> {
        self.map.get_mut(key).unwrap_or(&mut self.empty)
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

impl Counter{
    pub fn new(default : u32)->Self{
        return Counter(AtomicU32::new(default));
    }

    pub fn inc(&self) -> u32{
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn get(&self) -> u32{
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}