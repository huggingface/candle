use std::collections::BTreeSet;

use tracing::instrument;

use crate::{
    wgpu_backend::{
        util::{ReferenceTrait, StorageOptional, StorageTrait},
        wgpu_functions,
    },
    WgpuDevice,
};

use super::CachedBufferId;

////////////////// CACHED BUFFER:
#[derive(Debug)]
pub struct CachedBuffer {
    buffer: wgpu::Buffer,
    is_free: bool, //wheter this buffer is currently free
    last_used_counter: u32,
}

impl CachedBuffer {
    pub fn new(buffer: wgpu::Buffer) -> Self {
        Self {
            buffer,
            is_free: false,
            last_used_counter: 0,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn is_free(&self) -> bool {
        self.is_free
    }
}















// Struct used for ordering by size
#[derive(Debug, Eq, PartialEq)]
struct OrderedIndex<TI, TV> {
    index: TI,
    value: TV,
}

impl<TI, TV> OrderedIndex<TI, TV> {
    fn new(index: TI, value: TV) -> Self {
        OrderedIndex { index, value }
    }
}

// Implementing Ord and PartialOrd for OrderedIndex so it can be stored in BTreeSet
impl<TI: Ord, TV: Ord> Ord for OrderedIndex<TI, TV> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value
            .cmp(&other.value)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl<TI: PartialOrd + Ord, TV: PartialOrd + Ord> PartialOrd for OrderedIndex<TI, TV> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}





/// Cache of all free CachedBuffers
#[derive(Debug)]
pub(crate) struct BufferCacheStorage {
    storage: StorageOptional<CachedBuffer, CachedBufferId>,
    order: BTreeSet<OrderedIndex<u32, u64>>, //contains a ordered list of the currently free buffers in the storage
    //(a buffer may be free if it is currently not used, this does not mean that it was deleted. a deleted buffer was complely removed from the storage and droped)
    buffer_counter: u32,       //total number of buffers created
    buffer_reuse_counter: u32, //total number of buffers created
    buffer_memory: u64,        //total memory allocated
    buffer_memory_free: u64,   //total memory in buffers btree map
    max_memory_allowed: u64,
}

impl BufferCacheStorage {
    pub fn new() -> Self {
        return Self {
            storage: StorageOptional::new(),
            order: BTreeSet::new(),
            buffer_counter: 0,
            buffer_reuse_counter: 0,
            buffer_memory: 0,
            buffer_memory_free: 0,
            max_memory_allowed: 0,
        };
    }

    //creats a Buffer, expect that it will be used and not be part of free memory
    #[instrument(skip(self, dev, command_id, size))]
    pub(crate) fn create_buffer(&mut self, dev: &WgpuDevice, size: u64, command_id: u32) -> CachedBufferId {
        let buffer = wgpu_functions::create_buffer(dev, size);
        let mut buffer = CachedBuffer::new(buffer);
        buffer.last_used_counter = command_id;
        let id = self.storage.insert(buffer);
        self.buffer_memory += size;
        self.buffer_counter += 1;
        tracing::info!("created buffer {:?}, size: {size}", id);
        return id;
    }

    #[instrument(skip(self, id))]
    pub fn delete_buffer(&mut self, id: &CachedBufferId) {
        tracing::info!("delete buffer {:?}", id);
        let value = self.storage.delete_move(id);
        if let Some(val) = value {
            if self
                .order
                .remove(&OrderedIndex::new(id.id(), val.buffer.size()))
            {
                self.buffer_memory_free -= val.buffer.size();
            }
            self.buffer_memory -= val.buffer.size();
        }
    }

    #[instrument(skip(self, id))]
    pub fn get_buffer(&self, id: &CachedBufferId) -> Option<&CachedBuffer> {
        self.storage.get(id)
    }

    //will not delete the buffer, but mark it free
    #[instrument(skip(self, id))]
    pub fn free_buffer(&mut self, id: &CachedBufferId) {
        tracing::info!("free buffer {:?}", id);
        let buffer: Option<&mut CachedBuffer> = self.storage.get_mut(id);
        if let Some(buffer) = buffer {
            if buffer.is_free == false {
                //the buffer is currently not free -> add it into the free order list
                self.order
                    .insert(OrderedIndex::new(id.id(), buffer.buffer.size()));
                buffer.is_free = true;
                self.buffer_memory_free += buffer.buffer.size()
            }
        }
    }

    //will not create a buffer, but mark the buffer as used
    #[instrument(skip(self, command_id, id))]
    pub fn use_buffer(&mut self, id: &CachedBufferId, command_id: u32) {
        tracing::info!("use buffer {:?}", id);
        let buffer: Option<&mut CachedBuffer> = self.storage.get_mut(id);
        if let Some(buffer) = buffer {
            if buffer.is_free == true {
                //the buffer is currently free -> remove it from the free order list
                self.order
                    .remove(&OrderedIndex::new(id.id(), buffer.buffer.size()));
                buffer.is_free = false;
                buffer.last_used_counter = command_id;
                self.buffer_reuse_counter += 1;
                //println!("use_buffer, remove buffer from free buffers");
                self.buffer_memory_free -= buffer.buffer.size()
            }
        }
    }

    // //will save the currentl free buffers and not free buffers
    // pub fn store_usage(&mut self){
    //     for b in self.storage.iter_mut_option(){
    //         b.stored_free  = b.is_free;
    //     }
    // }

    // //will reset all buffers usage to last storage usage
    // pub fn reset_usage(&mut self){
    //     for (id, b) in self.storage.enumerate_mut_option(){
    //         if b.stored_free && !b.is_free{
    //             self.order.insert(OrderedIndex::new(id.id(), b.buffer.size()));
    //             b.is_free = true;
    //             self.buffer_memory_free += b.buffer.size()
    //         }
    //         else if !b.stored_free  && b.is_free{
    //             self.order.remove(&OrderedIndex::new(id.id(), b.buffer.size()));
    //             b.is_free = false;
    //             self.buffer_memory_free -= b.buffer.size()
    //         }
    //     }
    // }

    //the length, this buffer should be used for(if a buffer is only used temporary we may use a way bigger buffer for just one command)
    pub(crate) fn max_cached_size(size: u64, length: u32) -> u64 {
        let length = (length + 1).min(100);
        let i = (300 / (length * length * length)).min(64).max(1) as u64;

        const TRANSITION_POINT: u64 = 1000 * 1024;
        return size + (i * size * TRANSITION_POINT / (TRANSITION_POINT + size));
    }

    //will try to find a free buffer in the cache, or create a new one
    #[instrument(skip(self, dev, command_id, minimum_size,optimal_size, duration))]
    pub fn search_buffer(
        &mut self,
        dev: &WgpuDevice,
        minimum_size: u64,
        optimal_size: u64,
        command_id: u32,
        duration: u32,
    ) -> CachedBufferId {
        //println!("search buffer: size: {size}");
        let max_size = BufferCacheStorage::max_cached_size(minimum_size, duration).max(optimal_size);

        if dev.configuration.use_cache {
            let mut buffer_found = None;
            for id in self.order.range(OrderedIndex::new(0, minimum_size)..) {
                if id.value < minimum_size {
                    panic!("Did not expect size to be smaller, than key");
                }

                if id.value > max_size {
                    break;
                }
                buffer_found = Some(id);
            }

            //remove this buffer from free memory:
            if let Some(buffer_found) = buffer_found {
                if let Some((reference, _)) = self.storage.get_reference(buffer_found.index) {
                    //println!("search buffer: found free buffer, using: {:?}, size: {:?}", reference, buffer.buffer.size());
                    self.use_buffer(&reference, command_id);
                    return reference;
                }
            }
        }
        return self.create_buffer(dev, optimal_size, command_id);
    }

    pub fn max_memory_allowed(&self) -> u64 {
        self.max_memory_allowed
    }

    pub fn set_max_memory_allowed(&mut self, max_memory_allowed: u64) {
        self.max_memory_allowed = max_memory_allowed;
    }

    pub(crate) fn buffer_memory(&self) -> u64 {
        self.buffer_memory
    }

    pub(crate) fn buffer_reuse_counter(&self) -> u32 {
        self.buffer_reuse_counter
    }

    pub(crate) fn buffer_counter(&self) -> u32 {
        self.buffer_counter
    }

    pub fn get_buffer_count(&self) -> usize {
        self.storage.len()
    }

    pub fn has_free_buffers(&self) -> bool {
        !self.order.is_empty()
    }

    #[instrument(skip(self))]
    pub fn get_free_buffers(&self) -> Vec<(CachedBufferId, u32)> {
        return self
            .order
            .iter()
            .map(|entry| {
                let (id, val) = self
                    .storage
                    .get_reference(entry.index)
                    .expect("item in order, that could ne be found in storage");
                return (id, val.last_used_counter);
            }).rev()
            .collect();
    }
}
