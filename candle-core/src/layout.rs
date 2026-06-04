//! Tensor Layouts including contiguous or sparse strides
use std::iter::FusedIterator;

use crate::{Error, Result, Shape};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Layout {
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Self {
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// The dimension size for a specified dimension index.
    pub fn dim<D: crate::shape::Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(&self.shape, "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Returns outer stride along `dim` if valid.
    ///
    /// Two conditions must hold:
    ///  1. Inner dims `[dim..]` has standard contiguous strides.
    ///  2. Outer dims `[..dim]` are contiguous among themselves, i.e.
    ///     `stride[k] == dims[k+1] * stride[k+1]` for `k` in `0..dim-1`.
    ///
    /// When the tensor is fully contiguous this returns `Some(dims[dim..].product())`.
    pub(crate) fn outer_stride_for_dim(&self, dim: usize) -> Option<usize> {
        let dims = self.dims();
        let strides = self.stride();

        // 1. Inner `dims[dim..]` must have contiguous strides.
        let mut expected = 1usize;
        for i in (dim..dims.len()).rev() {
            if strides[i] != expected {
                return None;
            }
            expected *= dims[i];
        }

        if dim == 0 {
            // No outer dims.
            // `expected = dims[dim..].product()`
            return Some(expected);
        }

        // 2. Outer `dims[0..dim]` must be internally contiguous.
        let outer_stride = strides[dim - 1];
        let mut expected_outer = outer_stride;
        for k in (0..dim - 1).rev() {
            expected_outer *= dims[k + 1];
            if strides[k] != expected_outer {
                return None;
            }
        }

        Some(outer_stride)
    }

    /// Returns the appropriate start and stop offset if the data is stored in a C
    /// contiguous (aka row major) way.
    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        if self.is_contiguous() {
            let start_o = self.start_offset;
            Some((start_o, start_o + self.shape.elem_count()))
        } else {
            None
        }
    }

    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    /// Note that this does not implies that the start offset is 0 or that there are no extra
    /// elements at the end of the storage.
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    /// Returns true if the data is stored in a Fortran contiguous (aka column major) way.
    pub fn is_fortran_contiguous(&self) -> bool {
        self.shape.is_fortran_contiguous(&self.stride)
    }

    pub fn is_scalar(&self) -> bool {
        let dims = self.dims();
        dims.is_empty() || dims.iter().all(|d| *d == 1)
    }

    /// Returns true if the data is actually a scalar during broadcast
    pub fn is_scalar_broadcast(&self) -> bool {
        self.stride().iter().all(|s| *s == 0)
    }

    pub fn is_scalar_like(&self) -> bool {
        self.is_scalar() || self.is_scalar_broadcast()
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim as i32,
                op: "narrow",
            }
            .bt())?
        }
        if start + len > dims[dim] {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape.clone(),
                dim,
                start,
                len,
                msg: "start + len > dim_len",
            }
            .bt())?
        }
        let mut dims = dims.to_vec();
        dims[dim] = len;
        Ok(Self {
            shape: Shape::from(dims),
            stride: self.stride.clone(),
            start_offset: self.start_offset + self.stride[dim] * start,
        })
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: usize::max(dim1, dim2),
                got: rank,
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().dims().to_vec();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        Ok(Self {
            shape: Shape::from(dims),
            stride,
            start_offset: self.start_offset,
        })
    }

    pub fn permute(&self, idxs: &[usize]) -> Result<Self> {
        let is_permutation =
            idxs.len() == self.shape.rank() && (0..idxs.len()).all(|i| idxs.contains(&i));
        if !is_permutation {
            crate::bail!(
                "dimension mismatch in permute, tensor {:?}, dims: {:?}",
                self.dims(),
                idxs
            )
        }
        let stride = self.stride();
        let dims = self.shape().dims();
        let mut perm_stride = stride.to_vec();
        let mut perm_dims = dims.to_vec();
        for (i, &idx) in idxs.iter().enumerate() {
            perm_stride[i] = stride[idx];
            perm_dims[i] = dims[idx];
        }
        Ok(Self {
            shape: Shape::from(perm_dims),
            stride: perm_stride,
            start_offset: self.start_offset,
        })
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape().rank() {
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: self.shape().clone(),
                dst_shape: shape,
            }
            .bt());
        }
        let added_dims = shape.rank() - self.shape().rank();
        let mut stride = vec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    src_shape: self.shape().clone(),
                    dst_shape: shape,
                }
                .bt());
            } else {
                0
            };
            stride.push(s)
        }
        Ok(Self {
            shape,
            stride,
            start_offset: self.start_offset,
        })
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex<'_> {
        crate::StridedIndex::from_layout(self)
    }

    pub(crate) fn strided_blocks(&self) -> crate::StridedBlocks<'_> {
        let mut block_len = 1usize;
        let mut contiguous_dims = 0usize; // Counted from the right.
        for (&stride, &dim) in self.stride().iter().zip(self.dims().iter()).rev() {
            // Size-1 dimensions are trivially contiguous regardless of their stride.
            if dim == 1 {
                contiguous_dims += 1;
                continue;
            }
            if stride != block_len {
                break;
            }
            block_len *= dim;
            contiguous_dims += 1;
        }
        let index_dims = self.dims().len() - contiguous_dims;
        match index_dims {
            0 => crate::StridedBlocks::SingleBlock {
                start_offset: self.start_offset,
                len: block_len,
            },
            1 => crate::StridedBlocks::UniformBlocks {
                start_offset: self.start_offset,
                block_len,
                count: self.dims()[0],
                src_stride: self.stride[0],
            },
            _ => {
                let block_start_index = crate::StridedIndex::new(
                    &self.dims()[..index_dims],
                    &self.stride[..index_dims],
                    self.start_offset,
                );
                crate::StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len,
                }
            }
        }
    }
}

/// Maximum tensor rank supported by [`NdIter`].
pub const MAX_DIMS: usize = 8;

/// Multi-dimensional iterator that walks `N` tensor layouts simultaneously.
///
/// Internally, adjacent dimensions are merged whenever their strides allow it,
/// reducing the number of outer iterations. Each call to [`Iterator::next`] yields
/// `[usize; N]`, one base offset per layout, for one inner slice of `inner_size`
/// elements. Callers iterate that slice using `inner_strides`:
///
/// ```ignore
/// let nd_iter = NdIter::new([lhs_l, rhs_l]);
/// let inner_size = nd_iter.inner_size;
/// let [inner_ls, inner_rs] = nd_iter.inner_strides;
/// for [lhs_off, rhs_off] in nd_iter {
///     for i in 0..inner_size {
///         let lhs = lhs_buf[lhs_off + i * inner_ls];
///         let rhs = rhs_buf[rhs_off + i * inner_rs];
///         ...
///     }
/// }
/// ```
///
/// Note: `inner_strides[n]` can be 1 contiguous, but also 0 (broadcast/scalar) or > 1 (non-contiguous),
/// so callers must not assume contiguous element access within a slice.
pub struct NdIter<const N: usize> {
    pub inner_size: usize,
    /// Per-layout element stride within each inner slice: `element_offset = base + i * stride`.
    pub inner_strides: [usize; N],

    outer_dims: [usize; MAX_DIMS],
    outer_strides: [[usize; MAX_DIMS]; N],
    outer_len: usize,

    offsets: [usize; N],
    coords: [usize; MAX_DIMS],
    remaining: usize,
}

impl<const N: usize> NdIter<N> {
    pub fn new(layouts: [&Layout; N]) -> NdIter<N> {
        let dims = layouts[0].dims();
        debug_assert!(
            dims.len() <= MAX_DIMS,
            "rank {} exceeds MAX_DIMS={}",
            dims.len(),
            MAX_DIMS
        );
        #[cfg(debug_assertions)]
        for l in &layouts {
            debug_assert_eq!(l.dims(), dims);
        }

        let rank = dims.len();
        let mut out_dims = [0usize; MAX_DIMS];
        let mut out_strides = [[0usize; MAX_DIMS]; N];
        let mut out_len;

        if rank == 0 {
            out_dims[0] = 1;
            out_len = 1;
        } else {
            out_dims[0] = dims[0];
            for n in 0..N {
                out_strides[n][0] = layouts[n].stride()[0];
            }
            out_len = 1;

            for i in 1..rank {
                let d = dims[i];
                let top = out_len - 1;
                let last_d = out_dims[top];

                let (can_merge, use_inner) = if last_d == 1 {
                    (true, true)
                } else if d == 1 {
                    (true, false)
                } else {
                    let can_merge =
                        (0..N).all(|n| out_strides[n][top] == layouts[n].stride()[i] * d);
                    (can_merge, true)
                };
                if can_merge {
                    out_dims[top] = last_d * d;
                    if use_inner {
                        for n in 0..N {
                            out_strides[n][top] = layouts[n].stride()[i];
                        }
                    }
                } else {
                    out_dims[out_len] = d;
                    for n in 0..N {
                        out_strides[n][out_len] = layouts[n].stride()[i];
                    }
                    out_len += 1;
                }
            }
        }

        // Update inner strides
        let inner_idx = out_len - 1;
        let inner_size = out_dims[inner_idx];
        let mut inner_strides = [0usize; N];
        for n in 0..N {
            inner_strides[n] = out_strides[n][inner_idx];
        }

        // Update outer dims
        let outer_len = inner_idx;
        let mut outer_dims = [0usize; MAX_DIMS];
        outer_dims[..outer_len].copy_from_slice(&out_dims[..outer_len]);

        // Update outer strides
        let mut outer_strides = [[0usize; MAX_DIMS]; N];
        for n in 0..N {
            outer_strides[n][..outer_len].copy_from_slice(&out_strides[n][..outer_len]);
        }

        // Update offsets
        let mut offsets = [0usize; N];
        for n in 0..N {
            offsets[n] = layouts[n].start_offset();
        }

        // Number of outer blocks (product of all dims except the innermost).
        let remaining = out_dims[..outer_len].iter().product();

        NdIter {
            inner_size,
            inner_strides,
            outer_dims,
            outer_strides,
            outer_len,
            offsets,
            coords: [0; MAX_DIMS],
            remaining,
        }
    }
}

impl<const N: usize> Iterator for NdIter<N> {
    type Item = [usize; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = self.offsets;
        self.remaining -= 1;

        for k in (0..self.outer_len).rev() {
            self.coords[k] += 1;
            for n in 0..N {
                self.offsets[n] += self.outer_strides[n][k];
            }
            if self.coords[k] < self.outer_dims[k] {
                break;
            }
            self.coords[k] = 0;
            for n in 0..N {
                self.offsets[n] -= self.outer_dims[k] * self.outer_strides[n][k];
            }
        }

        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<const N: usize> ExactSizeIterator for NdIter<N> {}
impl<const N: usize> FusedIterator for NdIter<N> {}

#[cfg(test)]
mod nd_iter {
    use super::*;

    fn layout(dims: &[usize], strides: &[usize]) -> Layout {
        Layout::new(Shape::from(dims.to_vec()), strides.to_vec(), 0)
    }

    #[test]
    fn rank0_scalar() {
        let l = Layout::contiguous(());
        let mut it = NdIter::new([&l, &l]);
        assert_eq!(it.inner_size, 1);
        assert_eq!(it.inner_strides, [0, 0]);
        assert_eq!(it.len(), 1);
        assert_eq!(it.next(), Some([0, 0]));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn rank1_contiguous_single_block() {
        let l = Layout::contiguous(&[5]);
        let mut it = NdIter::new([&l, &l]);
        assert_eq!(it.inner_size, 5);
        assert_eq!(it.inner_strides, [1, 1]);
        assert_eq!(it.len(), 1);
        assert_eq!(it.next(), Some([0, 0]));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn rank2_contiguous_merges_to_one_block() {
        // [3, 4] strides[4, 1] -> fully merged, inner_size=12
        let l = Layout::contiguous(&[3, 4]);
        let it = NdIter::new([&l, &l]);
        assert_eq!(it.inner_size, 12);
        assert_eq!(it.inner_strides, [1, 1]);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn rank3_contiguous_fully_merged() {
        let l = Layout::contiguous(&[2, 3, 4]);
        let it = NdIter::new([&l]);
        assert_eq!(it.inner_size, 24);
        assert_eq!(it.inner_strides, [1]);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn rank3_outer_gap_partial_merge() {
        // shape [2, 3, 4]
        // strides [24, 4, 1]
        // dims 1 + 2 merge (4 == 1 * 4), dim 0 stays
        let l = layout(&[2, 3, 4], &[24, 4, 1]);
        let it = NdIter::new([&l]);
        assert_eq!(it.inner_size, 12); // 3 * 4 merged
        assert_eq!(it.inner_strides, [1]);
        assert_eq!(it.len(), 2); // outer: dim 0 has size 2
        let offsets: Vec<_> = it.collect();
        assert_eq!(offsets, vec![[0], [24]]);
    }

    #[test]
    fn rank2_no_merge() {
        // shape [3, 4]
        // strides [1, 3]
        // stride[0] = 1 != stride[1] * dim[1] = 12 -> no merge
        let l = layout(&[3, 4], &[1, 3]);
        let it = NdIter::new([&l, &l]);
        assert_eq!(it.inner_size, 4);
        assert_eq!(it.inner_strides, [3, 3]);
        assert_eq!(it.len(), 3);
        let offsets: Vec<_> = it.collect();
        assert_eq!(offsets, vec![[0, 0], [1, 1], [2, 2]]);
    }

    #[test]
    fn broadcast_zeros_merge() {
        // shape [3, 4]
        // strides [0, 0]
        // 0 == 0 * 4 -> dims merge
        let l = layout(&[3, 4], &[0, 0]);
        let it = NdIter::new([&l, &l]);
        assert_eq!(it.inner_size, 12);
        assert_eq!(it.inner_strides, [0, 0]);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn mixed_contiguous_and_broadcast_merge() {
        // lhs [4, 1] contiguous
        // rhs [0,0] broadcast
        // both conditions pass -> merge
        let lhs = Layout::contiguous(&[3, 4]);
        let rhs = layout(&[3, 4], &[0, 0]);
        let it = NdIter::new([&lhs, &rhs]);
        assert_eq!(it.inner_size, 12);
        assert_eq!(it.inner_strides, [1, 0]);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn offsets_lhs_contiguous_rhs_strided() {
        // lhs row-major [2,3] strides [3,1]
        // rhs col-major [2,3] strides [1,2]
        // lhs can merge (3 == 1 * 3)
        // rhs can not (1 ≠ 2 * 3 = 6) -> no merge for the pair
        let lhs = Layout::contiguous(&[2, 3]);
        let rhs = layout(&[2, 3], &[1, 2]);
        let it = NdIter::new([&lhs, &rhs]);
        assert_eq!(it.inner_size, 3);
        assert_eq!(it.inner_strides, [1, 2]);
        assert_eq!(it.len(), 2);
        let offsets: Vec<_> = it.collect();
        assert_eq!(offsets, vec![[0, 0], [3, 1]]);
    }

    #[test]
    fn start_offset_reflected_in_first_iter() {
        let l = Layout::contiguous_with_offset(4, 7);
        let mut it = NdIter::new([&l]);
        assert_eq!(it.next(), Some([7]));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn start_offset_advances_with_outer_dims() {
        // shape [2, 3]
        // strides [4, 1]
        // start_offset=10
        // strides can't merge (4 != 1 * 3)
        // outer blocks at 10 and 14 (10 + 1 * outer_stride)
        let l = Layout::new(Shape::from(vec![2, 3]), vec![4, 1], 10);
        let offsets: Vec<_> = NdIter::new([&l]).collect();
        assert_eq!(offsets, vec![[10], [14]]);
    }
}
