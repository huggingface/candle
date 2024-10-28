use rustc_hash::FxHashMap as HashMap;

use tracing::instrument;

use crate::wgpu_backend::util::{HashMapMulti, StorageOptional, StorageTrait};

use super::{CachedBindgroupFull, CachedBindgroupId, CachedBindgroupInput};

#[derive(Debug)]
pub struct CachedBindgroup {
    bindgroup: wgpu::BindGroup,
    buffer: CachedBindgroupFull,
}

impl CachedBindgroup {
    pub fn new(bindgroup: wgpu::BindGroup, buffer: CachedBindgroupFull) -> Self {
        Self { bindgroup, buffer }
    }

    pub fn bindgroup(&self) -> &wgpu::BindGroup {
        &self.bindgroup
    }

    pub(crate) fn buffer(&self) -> &CachedBindgroupFull {
        &self.buffer
    }
}

/// Cache of all available CachedBindGroups
#[derive(Debug)]
pub(crate) struct BindgroupCacheStorage {
    storage: StorageOptional<CachedBindgroup, CachedBindgroupId>,
    bindgroups: HashMapMulti<CachedBindgroupInput, CachedBindgroupId>, //all bindgroups based on input buffers
    bindgroups_full: HashMap<CachedBindgroupFull, CachedBindgroupId>, //all bindgroups based on input und dest buffers
    bindgroup_counter: u32,
    cached_bindgroup_use_counter: u32,
}

impl BindgroupCacheStorage {
    pub(crate) fn new() -> Self {
        Self {
            storage: StorageOptional::new(),
            bindgroups: HashMapMulti::new(),
            bindgroups_full: HashMap::default(),
            bindgroup_counter: 0,
            cached_bindgroup_use_counter: 0,
        }
    }

    #[instrument(skip(self, keep))]
    pub(crate) fn retain_bindgroups(&mut self, mut keep: impl FnMut(&CachedBindgroup) -> bool) {
        self.storage.retain_mut(|(id, bg)| {
            let keep = keep(bg);

            if !keep {
                let id = *id;
                let buf_reference_input_full = &bg.buffer;
                self.bindgroups
                    .remove_mapping(&buf_reference_input_full.1, &id);
                self.bindgroups_full.remove(buf_reference_input_full);
            }
            keep
        });
    }

    pub fn get_bindgroup(&self, id: &CachedBindgroupId) -> Option<&CachedBindgroup> {
        self.storage.get(id)
    }

    pub(crate) fn get_bindgroup_reference_by_description(
        &self,
        bindgroup_d: &CachedBindgroupFull,
    ) -> Option<&CachedBindgroupId> {
        self.bindgroups_full.get(bindgroup_d)
    }

    pub(crate) fn bindgroup_counter(&self) -> u32 {
        self.bindgroup_counter
    }

    pub(crate) fn cached_bindgroup_use_counter(&self) -> u32 {
        self.cached_bindgroup_use_counter
    }

    pub(crate) fn cached_bindgroup_use_counter_inc(&mut self) {
        self.cached_bindgroup_use_counter += 1;
    }

    pub(crate) fn insert_bindgroup(&mut self, bindgroup: CachedBindgroup) -> CachedBindgroupId {
        let bindgroup_d = bindgroup.buffer.clone();
        let id = self.storage.insert(bindgroup);

        tracing::info!("Create Bindgroup: {id:?}, {:?}", bindgroup_d);

        self.bindgroups.add_mapping(bindgroup_d.1.clone(), id);

        if self.bindgroups_full.contains_key(&bindgroup_d) {
            panic!("bindgroup {:?} was tried to add to the bindgroup cache, but it was already be created", bindgroup_d);
        }

        self.bindgroups_full.insert(bindgroup_d, id);

        self.bindgroup_counter += 1;
        id
    }
}
