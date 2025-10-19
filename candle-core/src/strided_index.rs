use arrayvec::ArrayVec;

use crate::{
    layout::{Stride, MAX_DIMS},
    shape::ShapeVec,
    Layout,
};

type MultiIndex = ArrayVec<(usize, usize, usize), MAX_DIMS>;
/// An iterator over offset position for items of an N-dimensional arrays stored in a
/// flat buffer using some potential strides.
#[derive(Debug)]
pub struct StridedIndex {
    next_storage_index: Option<usize>,
    multi_index: MultiIndex,
}

impl StridedIndex {
    pub(crate) fn new(dims: &ShapeVec, stride: &Stride, start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset)
        };
        // Precompute multi index iterator.
        // For each dim, we have (current_index_in_dim, max_index_in_dim, stride_for_dim)
        let multi_index: MultiIndex = dims
            .iter()
            .zip(stride.iter())
            .rev()
            .map(|(dim, stride)| (0, *dim, *stride))
            .collect();
        StridedIndex {
            next_storage_index,
            multi_index,
        }
    }

    pub(crate) fn from_layout(l: &Layout) -> Self {
        Self::new(l.shape().inner(), l.stride(), l.start_offset())
    }
}

impl Iterator for StridedIndex {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = self.next_storage_index?;
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for (multi_i, max_i, stride_i) in self.multi_index.iter_mut() {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += *stride_i;
                break;
            } else {
                next_storage_index -= *multi_i * *stride_i;
                *multi_i = 0
            }
        }
        self.next_storage_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}

#[derive(Debug)]
pub enum StridedBlocks {
    SingleBlock {
        start_offset: usize,
        len: usize,
    },
    MultipleBlocks {
        block_start_index: StridedIndex,
        block_len: usize,
    },
}
