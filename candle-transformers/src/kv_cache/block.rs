//! Block allocator and block table for Paged KV Cache.

/// Physical block identifier in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

/// Free-list allocator with reference counting for COW sharing.
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    free_list: Vec<BlockId>,
    ref_counts: Vec<u32>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        let free_list = (0..num_blocks).map(BlockId).rev().collect();
        Self {
            free_list,
            ref_counts: vec![0; num_blocks],
        }
    }

    pub fn alloc(&mut self) -> Option<BlockId> {
        self.free_list.pop().map(|id| {
            self.ref_counts[id.0] = 1;
            id
        })
    }

    pub fn free(&mut self, id: BlockId) {
        let rc = &mut self.ref_counts[id.0];
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            self.free_list.push(id);
        }
    }

    pub fn incr_ref(&mut self, id: BlockId) {
        self.ref_counts[id.0] += 1;
    }

    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }
}

/// Maps a logical sequence position to physical pool blocks.
#[derive(Debug, Clone)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    num_tokens_last: usize,
    block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens_last: 0,
            block_size,
        }
    }

    /// COW fork: share physical blocks, bump ref counts.
    pub fn fork(&self, allocator: &mut BlockAllocator) -> Self {
        for id in &self.blocks {
            allocator.incr_ref(*id);
        }
        Self {
            blocks: self.blocks.clone(),
            num_tokens_last: self.num_tokens_last,
            block_size: self.block_size,
        }
    }

    /// Allocate space for one more token. Returns (BlockId, offset_in_block).
    pub fn append(&mut self, allocator: &mut BlockAllocator) -> Option<(BlockId, usize)> {
        if self.blocks.is_empty() || self.num_tokens_last == self.block_size {
            let block = allocator.alloc()?;
            self.blocks.push(block);
            self.num_tokens_last = 1;
            Some((block, 0))
        } else {
            let offset = self.num_tokens_last;
            self.num_tokens_last += 1;
            Some((*self.blocks.last().unwrap(), offset))
        }
    }

    pub fn seq_len(&self) -> usize {
        if self.blocks.is_empty() {
            0
        } else {
            (self.blocks.len() - 1) * self.block_size + self.num_tokens_last
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn block_ids(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}