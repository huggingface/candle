use std::collections::{BTreeMap, HashMap};


pub(crate) trait ToU32 {
    fn to_u32(self) -> u32;
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




#[derive(Debug)]
pub(crate) struct HashMapMulti<K, V>{
    pub(crate) map : HashMap<K, Vec<V>>,
    pub(crate) empty : Vec<V>,
}

#[derive(Debug)]
pub(crate) struct BTreeMulti<K, V>{
    pub(crate) map : BTreeMap<K, Vec<V>>,
    pub(crate) empty : Vec<V>,
}

impl<K: std::cmp::Eq + PartialEq + std::hash::Hash, V> HashMapMulti<K, V>{
    
    pub fn new() -> Self{
        Self{map : HashMap::new(), empty : vec![]}
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

    pub fn get(&self, key : &K) -> &Vec<V>{
        self.map.get(key).unwrap_or(&self.empty)
    }
}


impl<K: std::cmp::Eq + PartialEq + std::hash::Hash + Ord, V : std::cmp::PartialEq> BTreeMulti<K, V>{
    
    pub fn new() -> Self{
        Self{map : BTreeMap::new(), empty : vec![]}
    } 
    
    pub fn add_mapping(&mut self, key: K, value: V) {
       let values = self.map.entry(key).or_insert_with(Vec::new);
       if !values.contains(&value){
        values.push(value);
       }
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

    pub fn get(&self, key : &K) -> &Vec<V>{
        self.map.get(key).unwrap_or(&self.empty)
    }

    pub fn get_mut(&mut self, key : &K) -> &mut Vec<V>{
        self.map.get_mut(key).unwrap_or(&mut self.empty)
    }
}