/// An iterator over offset position for items of an N-dimensional arrays stored in a
/// flat buffer using some potential strides.
#[derive(Debug)]
pub(crate) struct StridedIndex<'a> {
    next_storage_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(dims: &'a [usize], stride: &'a [usize]) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(0)
        };
        StridedIndex {
            next_storage_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = match self.next_storage_index {
            None => return None,
            Some(storage_index) => storage_index,
        };
        let mut updated = false;
        for (multi_i, max_i) in self.multi_index.iter_mut().zip(self.dims.iter()).rev() {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                break;
            } else {
                *multi_i = 0
            }
        }
        self.next_storage_index = if updated {
            let next_storage_index = self
                .multi_index
                .iter()
                .zip(self.stride.iter())
                .map(|(&x, &y)| x * y)
                .sum();
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}
