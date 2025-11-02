//! Tensor Layouts including contiguous or sparse strides
use crate::{Error, Result, Shape};
use arrayvec::ArrayVec;
use core::{iter, mem::ManuallyDrop};

pub(crate) const MAX_DIMS: usize = 8;
pub(crate) type IndexArray = ArrayVec<usize, MAX_DIMS>;
pub(crate) type Stride = IndexArray;

pub(crate) trait IndexArrayExt {
    fn from_slice(s: &[usize]) -> Self;

    fn from_arr<const N: usize>(arr: [usize; N]) -> Self;
}

impl IndexArrayExt for IndexArray {
    fn from_slice(s: &[usize]) -> Self {
        debug_assert!(s.len() <= MAX_DIMS);

        let mut index_type = IndexArray::new();
        unsafe {
            index_type
                .as_mut_ptr()
                .copy_from_nonoverlapping(s.as_ptr(), s.len());
            index_type.set_len(s.len());
        }
        index_type
    }

    fn from_arr<const N: usize>(array: [usize; N]) -> Self {
        debug_assert!(N <= MAX_DIMS);
        let array = ManuallyDrop::new(array);
        let mut index_type = IndexArray::new();
        unsafe {
            index_type
                .as_mut_ptr()
                .copy_from_nonoverlapping(array.as_ptr(), N);
            index_type.set_len(N);
        }
        index_type
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Layout {
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: Stride,
    start_offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Stride, start_offset: usize) -> Self {
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

    pub fn dims(&self) -> &IndexArray {
        &self.shape.inner()
    }

    /// The dimension size for a specified dimension index.
    pub fn dim<D: crate::shape::Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(&self.shape, "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
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

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.shape().inner();
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
        let mut dims = dims.clone();
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
        let mut stride = self.stride().clone();
        let mut dims = self.shape().inner().clone();
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
        let dims = self.shape().inner();
        let mut perm_stride = stride.clone();
        let mut perm_dims = dims.clone();
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
        let mut stride = Stride::from_iter(iter::repeat_n(0, added_dims));
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
        crate::StridedIndex::from(self)
    }

    pub(crate) fn strided_blocks<'a>(&'a self) -> crate::StridedBlocks<'a> {
        let mut block_len = 1;
        let mut contiguous_dims = 0; // These are counted from the right.
        for (&stride, &dim) in self.stride().iter().zip(self.dims().iter()).rev() {
            if stride != block_len {
                break;
            }
            block_len *= dim;
            contiguous_dims += 1;
        }
        let index_dims = self.dims().len() - contiguous_dims;
        if index_dims == 0 {
            crate::StridedBlocks::SingleBlock {
                start_offset: self.start_offset,
                len: block_len,
            }
        } else {
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

    // Returns the contiguous offsets with broadcast if applicable.
    pub(crate) fn offsets_b(&self) -> Option<ContiguousOffsetsWithBroadcast> {
        let mut left_broadcast = 1;
        let mut right_broadcast = 1;
        let strides = self.stride();
        let dims = self.dims();
        let mut start_cont = 0;
        let mut end_cont = dims.len();
        for (&s, &d) in strides.iter().zip(dims.iter()) {
            if s != 0 {
                break;
            }
            start_cont += 1;
            left_broadcast *= d;
        }
        if start_cont == dims.len() {
            return Some(ContiguousOffsetsWithBroadcast {
                start: self.start_offset,
                len: 1,
                left_broadcast,
                right_broadcast: 1,
            });
        }
        for (&s, &d) in strides.iter().zip(dims.iter()).rev() {
            if s != 0 {
                break;
            }
            end_cont -= 1;
            right_broadcast *= d;
        }
        // Check that the inner dims are contiguous
        let strides = &strides[start_cont..end_cont];
        let dims = &dims[start_cont..end_cont];
        let mut len = 1;
        for (&stride, &dim) in strides.iter().zip(dims.iter()).rev() {
            if stride != len {
                return None;
            }
            len *= dim;
        }
        Some(ContiguousOffsetsWithBroadcast {
            start: self.start_offset,
            len,
            left_broadcast,
            right_broadcast,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousOffsetsWithBroadcast {
    pub start: usize,
    pub len: usize,
    pub left_broadcast: usize,
    pub right_broadcast: usize,
}
