use crate::{Error, Tensor};
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

impl Tensor {
    /// Intended to be use by the trait `.i()`
    ///
    /// ```
    /// # use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    ///
    /// let c = a.i(0..1)?;
    /// assert_eq!(c.shape().dims(), &[1, 3]);
    ///
    /// let c = a.i(0)?;
    /// assert_eq!(c.shape().dims(), &[3]);
    ///
    /// let c = a.i((.., ..2) )?;
    /// assert_eq!(c.shape().dims(), &[2, 2]);
    ///
    /// let c = a.i((.., ..=2))?;
    /// assert_eq!(c.shape().dims(), &[2, 3]);
    ///
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn index(&self, indexers: &[TensorIndexer]) -> Result<Self, Error> {
        let mut x = self.clone();
        let dims = self.shape().dims();
        let mut current_dim = 0;
        for (i, indexer) in indexers.iter().enumerate() {
            x = match indexer {
                TensorIndexer::Select(n) => x.narrow(current_dim, *n, 1)?.squeeze(current_dim)?,
                TensorIndexer::Narrow(left_bound, right_bound) => {
                    let start = match left_bound {
                        Bound::Included(n) => *n,
                        Bound::Excluded(n) => *n + 1,
                        Bound::Unbounded => 0,
                    };
                    let stop = match right_bound {
                        Bound::Included(n) => *n + 1,
                        Bound::Excluded(n) => *n,
                        Bound::Unbounded => dims[i],
                    };
                    let out = x.narrow(current_dim, start, stop.saturating_sub(start))?;
                    current_dim += 1;
                    out
                }
            };
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
/// Generic structure used to index a slice of the tensor
pub enum TensorIndexer {
    /// This selects the elemnts for which an index has some specific value.
    Select(usize),
    /// This is a regular slice, purely indexing a chunk of the tensor
    Narrow(Bound<usize>, Bound<usize>),
}

impl From<usize> for TensorIndexer {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

macro_rules! impl_from_range {
    ($range_type:ty) => {
        impl From<$range_type> for TensorIndexer {
            fn from(range: $range_type) -> Self {
                use std::ops::Bound::*;

                let start = match range.start_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                let end = match range.end_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                TensorIndexer::Narrow(start, end)
            }
        }
    };
}

impl_from_range!(Range<usize>);
impl_from_range!(RangeFrom<usize>);
impl_from_range!(RangeFull);
impl_from_range!(RangeInclusive<usize>);
impl_from_range!(RangeTo<usize>);
impl_from_range!(RangeToInclusive<usize>);

/// Trait used to implement multiple signatures for ease of use of the slicing
/// of a tensor
pub trait IndexOp<T> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn i(&self, index: T) -> Result<Tensor, Error>;
}

impl<T> IndexOp<T> for Tensor
where
    T: Into<TensorIndexer>,
{
    fn i(&self, index: T) -> Result<Tensor, Error> {
        self.index(&[index.into()])
    }
}

macro_rules! index_op_tuple {
    ($($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*> IndexOp<($($t,)*)> for Tensor
        where
            $($t: Into<TensorIndexer>,)*
        {
            fn i(&self, ($($t,)*): ($($t,)*)) -> Result<Tensor, Error> {
                self.index(&[$($t.into(),)*])
            }
        }
    };
}
index_op_tuple!(A);
index_op_tuple!(A, B);
index_op_tuple!(A, B, C);
index_op_tuple!(A, B, C, D);
index_op_tuple!(A, B, C, D, E);
index_op_tuple!(A, B, C, D, E, F);
index_op_tuple!(A, B, C, D, E, F, G);
