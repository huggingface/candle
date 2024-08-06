use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::{
    bail,
    op::{BackpropOp, Op},
    shape::Dim,
    tensor::from_storage,
    DType, Error, Result, Tensor,
};

/// Specialization of `std::ops::RangeBounds` for `usize` to allow trait objects.
pub trait RangeBound {
    fn start_bound(&self) -> std::ops::Bound<usize>;
    fn end_bound(&self) -> std::ops::Bound<usize>;
}

macro_rules! range_bound {
    ($name:ident) => {
        impl RangeBound for $name<usize> {
            fn end_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::end_bound(&self).cloned()
            }
            fn start_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::start_bound(&self).cloned()
            }
        }
    };
    // Use the marker to designate no generics
    ($name:ident, $marker:expr) => {
        impl RangeBound for $name {
            fn end_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::end_bound(&self).cloned()
            }
            fn start_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::start_bound(&self).cloned()
            }
        }
    };
    // Use the marker to designate no generics
    ($name:ty) => {
        impl RangeBound for $name {
            fn end_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::end_bound(&self).cloned()
            }
            fn start_bound(&self) -> std::ops::Bound<usize> {
                <Self as std::ops::RangeBounds<usize>>::start_bound(&self).cloned()
            }
        }
    };
}

range_bound!(Range);
range_bound!(RangeFrom);
range_bound!(RangeFull, ());
range_bound!(RangeInclusive);
range_bound!(RangeTo);
range_bound!(RangeToInclusive);
range_bound!((std::ops::Bound<usize>, std::ops::Bound<usize>));

impl RangeBound for usize {
    fn end_bound(&self) -> std::ops::Bound<usize> {
        std::ops::Bound::Excluded(self + 1)
    }
    fn start_bound(&self) -> std::ops::Bound<usize> {
        std::ops::Bound::Included(*self)
    }
}

impl Tensor {
    /// Returns a copy of `self` where the values within `ranges` have been replaced with the
    /// content of `src`. This is analogous to slice asignment in `torch`.
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Device, Tensor};
    ///
    /// let dev = Device::Cpu;
    /// let tensor = Tensor::arange(0u32, 4 * 5, &dev)?.reshape((4, 5))?;
    /// let src = Tensor::arange(100u32, (2 * 3) + 100, &dev)?.reshape((3, 2))?;
    /// let out = tensor.slice_assign(&[&(..3), &(3..5)], &src)?;
    /// assert_eq!(
    ///     out.to_vec2::<u32>()?,
    ///     &[
    ///         [0, 1, 2, 100, 101],
    ///         [5, 6, 7, 102, 103],
    ///         [10, 11, 12, 104, 105],
    ///         [15, 16, 17, 18, 19]
    ///     ]
    /// );
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn slice_assign(&self, ranges: &[&dyn RangeBound], src: &Tensor) -> Result<Self> {
        let src_dims = src.dims();
        let self_dims = self.dims();
        if self_dims.len() != src_dims.len() {
            bail!(
                "slice-assign requires input with the same rank {} <> {}",
                self_dims.len(),
                src_dims.len()
            )
        }
        if self_dims.len() != ranges.len() {
            bail!(
                "slice-assign requires input with the same rank as there are ranges {} <> {}",
                self_dims.len(),
                ranges.len()
            )
        }
        let mut src = src.clone();
        let mut mask = Self::ones(src.shape(), DType::U8, src.device())?;
        for (i, range) in ranges.iter().enumerate() {
            let start_included = match range.start_bound() {
                std::ops::Bound::Unbounded => 0,
                std::ops::Bound::Included(v) => v,
                std::ops::Bound::Excluded(v) => v + 1,
            };
            let end_excluded = match range.end_bound() {
                std::ops::Bound::Unbounded => self_dims[i],
                std::ops::Bound::Included(v) => v + 1,
                std::ops::Bound::Excluded(v) => v,
            };
            if end_excluded <= start_included {
                bail!("slice-assign: empty range for dim {i}, {start_included} {end_excluded}")
            }
            if self_dims[i] < end_excluded {
                bail!(
                    "slice-assign: upper bound is out of range for dim {i}, {end_excluded} {}",
                    self_dims[i]
                )
            }
            if end_excluded - start_included != src_dims[i] {
                bail!(
                    "slice-assign: the range for dim {i} ({start_included}..{end_excluded}) does not match the size of src {}", src_dims[i]
                )
            }
            src = src.pad_with_zeros(i, start_included, self_dims[i] - end_excluded)?;
            mask = mask.pad_with_zeros(i, start_included, self_dims[i] - end_excluded)?
        }
        mask.where_cond(/* on_true= */ &src, /* on_false= */ self)
    }

    pub fn scatter_add<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "scatter-add")?;
        let source_dims = source.dims();
        let self_dims = self.dims();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter-add (self, src)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        if indexes.dims() != source.dims() {
            Err(Error::ShapeMismatchBinaryOp {
                op: "scatter-add (indexes, src)",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        let storage = self.storage().scatter_add(
            self.layout(),
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        let op = BackpropOp::new3(self, indexes, source, |t1, t2, t3| {
            Op::ScatterAdd(t1, t2, t3, dim)
        });
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Embeds the values of the `src` tensor into the `self` tensor on the specified dimension.
    pub fn slice_scatter<D: Dim>(&self, src: &Self, dim: D, start: usize) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "slice-scatter")?;
        if dim == 0 {
            self.slice_scatter0(src, start)
        } else {
            // TODO: Maybe we want to add a more efficient implementation at some point.
            self.transpose(0, dim)?
                .slice_scatter0(&src.transpose(0, dim)?, start)?
                .transpose(0, dim)
        }
    }

    /// Embeds the values of the `src` tensor into the `self` tensor on the first dimension.
    pub fn slice_scatter0(&self, src: &Self, start: usize) -> Result<Self> {
        if self.dtype() != src.dtype() {
            Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: src.dtype(),
                op: "slice-scatter",
            }
            .bt())?
        }
        if self.device().location() != src.device().location() {
            Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device().location(),
                rhs: src.device().location(),
                op: "slice-scatter",
            }
            .bt())?
        }
        if self.rank() != src.rank() {
            Err(Error::UnexpectedNumberOfDims {
                expected: self.rank(),
                got: src.rank(),
                shape: src.shape().clone(),
            }
            .bt())?
        }
        let shape_ok =
            self.dims()
                .iter()
                .zip(src.dims().iter())
                .enumerate()
                .all(|(dim_idx, (&d1, &d2))| {
                    if 0 == dim_idx {
                        d2 + start <= d1
                    } else {
                        d1 == d2
                    }
                });
        if !shape_ok {
            Err(Error::ShapeMismatchBinaryOp {
                op: "slice-scatter (self, src)",
                lhs: self.shape().clone(),
                rhs: src.shape().clone(),
            }
            .bt())?
        }
        let mut storage = unsafe { self.device().alloc_uninit(self.shape(), self.dtype())? };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        let offset = start * src.dims()[1..].iter().product::<usize>();
        src.storage()
            .copy_strided_src(&mut storage, offset, src.layout())?;
        let op = BackpropOp::new2(self, src, |t1, t2| Op::SliceScatter0(t1, t2, start));
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Accumulate element from `source` at indexes `indexes` and add them to `self`.
    pub fn index_add<D: Dim>(&self, indexes: &Self, source: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index-add")?;
        let source_dims = source.dims();
        let self_dims = self.dims();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "index-add (self, source)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        // The number of element in indexes must match the dimension on which the add is
        // performed on the source tensor (and the index values from `indexes` are taken from
        // the target tensor self)
        let indexes_len = indexes.dims1()?;
        if source_dims[dim] != indexes_len {
            Err(Error::ShapeMismatchBinaryOp {
                op: "index-add (ids, source))",
                lhs: indexes.shape().clone(),
                rhs: source.shape().clone(),
            }
            .bt())?
        }
        let storage = self.storage().index_add(
            self.layout(),
            &indexes.storage(),
            indexes.layout(),
            &source.storage(),
            source.layout(),
            dim,
        )?;
        let op = BackpropOp::new3(self, indexes, source, |t1, t2, t3| {
            Op::IndexAdd(t1, t2, t3, dim)
        });
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Gather values across the target dimension.
    ///
    /// # Arguments
    ///
    /// * `self` - The input tensor.
    /// * `indexes` - The indices of elements to gather, this should have the same shape as `self`
    ///   but can have a different number of elements on the target dimension.
    /// * `dim` - the target dimension.
    ///
    /// The resulting tensor has the same shape as `indexes` and use values from `self` indexed on
    /// dimension `dim` by the values in `indexes`.
    pub fn gather<D: Dim>(&self, indexes: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "gather")?;
        let self_dims = self.dims();
        let indexes_dims = indexes.dims();
        let mismatch = if indexes_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(indexes_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(Error::ShapeMismatchBinaryOp {
                op: "gather",
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
            }
            .bt())?
        }
        let storage =
            self.storage()
                .gather(self.layout(), &indexes.storage(), indexes.layout(), dim)?;
        let op = BackpropOp::new2(self, indexes, |t1, t2| Op::Gather(t1, t2, dim));
        Ok(from_storage(storage, indexes.shape(), op, false))
    }

    /// Select values for the input tensor at the target indexes across the specified dimension.
    ///
    /// The `indexes` is argument is an int tensor with a single dimension.
    /// The output has the same number of dimension as the `self` input. The target dimension of
    /// the output has length the length of `indexes` and the values are taken from `self` using
    /// the index from `indexes`. Other dimensions have the same number of elements as the input
    /// tensor.
    pub fn index_select<D: Dim>(&self, indexes: &Self, dim: D) -> Result<Self> {
        let dim = dim.to_index(self.shape(), "index-select")?;
        let indexes_len = match indexes.dims() {
            [l] => *l,
            _ => Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: indexes.shape().clone(),
                op: "index-select",
            }
            .bt())?,
        };
        let storage = self.storage().index_select(
            &indexes.storage(),
            self.layout(),
            indexes.layout(),
            dim,
        )?;
        let mut dims = self.dims().to_vec();
        dims[dim] = indexes_len;
        let op = BackpropOp::new2(self, indexes, |t1, t2| Op::IndexSelect(t1, t2, dim));
        Ok(from_storage(storage, dims, op, false))
    }
}
