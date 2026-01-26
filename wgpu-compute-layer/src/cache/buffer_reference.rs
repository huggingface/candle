//! Buffer reference helpers used by the cache subsystem.
//!
//! `BufferReference` represents a virtual buffer used in the compute graph.
use tracing::instrument;

use crate::util::{ReferenceTrait, Storage, StorageTrait};

use super::{BufferReferenceId, CachedBufferId};

/// Virtual buffer used in the compute graph.
#[derive(Debug)]
pub struct BufferReference {
    size: u64,
    cached_buffer_id: CachedBufferId,
    first_used: u32,
    last_used: u32, // u32::MAX means indefinitely
}

impl BufferReference {
    pub(crate) fn new(size: u64, referenced_by_wgpu_storage: bool) -> Self {
        Self {
            size,
            cached_buffer_id: CachedBufferId::new(0, 0),
            first_used: 0,
            last_used: if referenced_by_wgpu_storage {
                u32::MAX
            } else {
                0
            },
        }
    }

    pub(crate) fn new_with_storage(
        size: u64,
        cached_buffer_id: CachedBufferId,
        referenced_by_wgpu_storage: bool,
    ) -> Self {
        Self {
            size,
            cached_buffer_id,
            first_used: 0,
            last_used: if referenced_by_wgpu_storage {
                u32::MAX
            } else {
                0
            },
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub(crate) fn set_cached_buffer_id(&mut self, cached_buffer_id: CachedBufferId) {
        self.cached_buffer_id = cached_buffer_id;
    }

    pub(crate) fn cached_buffer_id(&self) -> &CachedBufferId {
        &self.cached_buffer_id
    }

    pub(crate) fn first_used(&self) -> u32 {
        self.first_used
    }

    pub(crate) fn set_first_used(&mut self, first_used: u32) {
        self.first_used = first_used;
    }

    pub(crate) fn last_used(&self) -> u32 {
        self.last_used
    }

    pub(crate) fn set_last_used(&mut self, last_used: u32) {
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

    #[instrument(skip(self, reference))]
    pub(crate) fn insert(&mut self, reference: BufferReference) -> BufferReferenceId {
        self.storage.insert(reference)
    }

    pub(crate) fn get(&self, id: &BufferReferenceId) -> Option<&BufferReference> {
        self.storage.get(id)
    }

    pub(crate) fn get_mut(&mut self, id: &BufferReferenceId) -> Option<&mut BufferReference> {
        self.storage.get_mut(id)
    }

    pub(crate) fn queue_for_deletion(&mut self, id: &BufferReferenceId) {
        self.deletion_queue.push(*id);
    }

    pub(crate) fn get_deletion_entries(&mut self) -> Vec<BufferReferenceId> {
        std::mem::take(&mut self.deletion_queue)
    }

    pub(crate) fn delete(&mut self, id: &BufferReferenceId) -> bool {
        self.storage.delete(id)
    }

    pub(crate) fn get_reference(&self, id: u32) -> Option<(BufferReferenceId, &BufferReference)> {
        self.storage.get_reference(id)
    }
}



#[test]
fn modelcache_buffer_reference_flags() {
    // Small mapping size and max memory size; we won't touch actual GPU buffers.
    let mut cache = crate::cache::ModelCache::new(2, 1024);

    // Create a virtual buffer that is referenced by WgpuStorage (should set last_used = u32::MAX)
    let vr1 = cache.create_buffer_reference(128u64, true);
    let v1 = cache.buffer_reference.get(&vr1).expect("expected buffer ref");
    assert_eq!(v1.size(), 128u64);
    assert_eq!(v1.last_used(), u32::MAX);

    // Create a temporary virtual buffer not referenced by storage (last_used == 0)
    let vr2 = cache.create_buffer_reference(64u64, false);
    let v2 = cache.buffer_reference.get(&vr2).expect("expected buffer ref");
    assert_eq!(v2.size(), 64u64);
    assert_eq!(v2.last_used(), 0u32);
}
