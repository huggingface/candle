use arrayvec::ArrayVec;

use crate::{layout::MAX_DIMS, Layout};

type MultiIndexCache = ArrayVec<(usize, usize, usize), MAX_DIMS>;
/// An iterator over offset position for items of an N-dimensional arrays stored in a
/// flat buffer using some potential strides.
#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_storage_index: Option<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
    multi_index_cache: MultiIndexCache,
    remaining: usize,
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        let elem_count: usize = dims.into_iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset)
        };
        StridedIndex {
            next_storage_index,
            dims,
            stride,
            multi_index_cache: MultiIndexCache::new(),
            remaining: elem_count,
        }
    }
}

impl<'a> From<&'a Layout> for StridedIndex<'a> {
    fn from(layout: &'a Layout) -> Self {
        StridedIndex::new(
            layout.shape().inner().as_slice(),
            layout.stride().as_slice(),
            layout.start_offset(),
        )
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = self.next_storage_index?;
        let mut updated = false;
        let mut next_storage_index = storage_index;

        // First next call multi_index is uninitialized.
        if self.multi_index_cache.len() == 0 {
            // Precompute multi index iterator.
            let mut ptr = self.multi_index_cache.as_mut_ptr();
            self.dims
                .iter()
                .zip(self.stride.iter())
                .rev()
                .for_each(|(dim, stride)| {
                    unsafe {
                        ptr.write((0, *dim, *stride));
                        ptr = ptr.add(1);
                    };
                });
            // SAFETY: All values are set by the loop above.
            unsafe { self.multi_index_cache.set_len(self.dims.len()) };
        }
        for (current_index_in_dim, max_index_in_dim, stride_for_dim) in
            self.multi_index_cache.iter_mut()
        {
            let next_i = *current_index_in_dim + 1;
            if next_i < *max_index_in_dim {
                *current_index_in_dim = next_i;
                updated = true;
                next_storage_index += *stride_for_dim;
                break;
            } else {
                next_storage_index -= *current_index_in_dim * *stride_for_dim;
                *current_index_in_dim = 0
            }
        }
        self.next_storage_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for StridedIndex<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

#[derive(Debug)]
pub enum StridedBlocks<'a> {
    SingleBlock {
        start_offset: usize,
        len: usize,
    },
    MultipleBlocks {
        block_start_index: StridedIndex<'a>,
        block_len: usize,
    },
}
