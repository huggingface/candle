////////////////// BUFFER REFERENCE:

use tracing::instrument;

use crate::wgpu_backend::util::{ReferenceTrait, Storage, StorageTrait};

use super::{BufferReferenceId, CachedBufferId};

/// Virtual Buffer, used in Compute Graph
#[derive(Debug)]
pub struct BufferReference {
    size: u64,
    referenced_by_candle_storage: bool,
    cached_buffer_id: CachedBufferId,
    first_used: u32,
    last_used: u32, //u32::max means indefitly
}

impl BufferReference {
    pub fn new(size: u64, referenced_by_candle_storage: bool) -> Self {
        Self {
            size,
            cached_buffer_id: CachedBufferId::new(0, 0),
            referenced_by_candle_storage,
            first_used: 0,
            last_used: if referenced_by_candle_storage {
                u32::MAX
            } else {
                0
            },
        }
    }

    pub fn new_with_storage(
        size: u64,
        cached_buffer_id: CachedBufferId,
        referenced_by_candle_storage: bool,
    ) -> Self {
        Self {
            size,
            cached_buffer_id,
            referenced_by_candle_storage,
            first_used: 0,
            last_used: if referenced_by_candle_storage {
                u32::MAX
            } else {
                0
            },
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn set_cached_buffer_id(&mut self, cached_buffer_id: CachedBufferId) {
        self.cached_buffer_id = cached_buffer_id;
    }

    pub fn cached_buffer_id(&self) -> &CachedBufferId {
        &self.cached_buffer_id
    }

    pub fn referenced_by_candle_storage(&self) -> bool {
        self.referenced_by_candle_storage
    }

    pub fn set_referenced_by_candle_storage(&mut self, referenced_by_candle_storage: bool) {
        self.referenced_by_candle_storage = referenced_by_candle_storage;
    }

    pub fn first_used(&self) -> u32 {
        self.first_used
    }

    pub fn set_first_used(&mut self, first_used: u32) {
        self.first_used = first_used;
    }

    pub fn last_used(&self) -> u32 {
        self.last_used
    }

    pub fn set_last_used(&mut self, last_used: u32) {
        self.last_used = last_used;
    }
}

#[derive(Debug)]
pub(crate) struct BufferReferenceStorage {
    storage: Storage<BufferReference, BufferReferenceId>,
    deletion_queue: Vec<BufferReferenceId>, //entires that are marked for deletion
}

impl BufferReferenceStorage {
    pub(crate) fn new() -> Self {
        Self {
            storage: Storage::new(),
            deletion_queue: vec![],
        }
    }

    #[instrument(skip(self, referece))]
    pub(crate) fn insert(&mut self, referece: BufferReference) -> BufferReferenceId {
        let id = self.storage.insert(referece);
        id
    }

    pub fn get(&self, id: &BufferReferenceId) -> Option<&BufferReference> {
        self.storage.get(id)
    }

    pub fn get_mut(&mut self, id: &BufferReferenceId) -> Option<&mut BufferReference> {
        self.storage.get_mut(id)
    }

    pub fn queue_for_deletion(&mut self, id: &BufferReferenceId) {
        self.deletion_queue.push(*id);
    }

    pub fn get_deletion_entries(&mut self) -> Vec<BufferReferenceId> {
        std::mem::take(&mut self.deletion_queue)
    }

    pub fn delete(&mut self, id: &BufferReferenceId) -> bool {
        self.storage.delete(id)
    }

    pub fn get_reference(&self, id: u32) -> Option<(BufferReferenceId, &BufferReference)> {
        self.storage.get_reference(id)
    }
}
