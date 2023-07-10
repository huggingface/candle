use crate::{Error, Tensor};
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

impl Tensor {
    /// Intended to be use by the trait `.i()`
    ///
    /// ```
    /// # use candle::{Tensor, DType, Device, IndexOp};
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
    /// # Ok::<(), candle::Error>(())
    /// ```
    fn index(&self, indexers: &[TensorIndexer]) -> Result<Self, Error> {
        let mut x = self.clone();
        let dims = self.shape().dims();
        let mut current_dim = 0;
        for (i, indexer) in indexers.iter().enumerate() {
            x = match indexer {
                TensorIndexer::Select(n) => x.get(*n)?,
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
                    let len = stop - start;
                    let out = x.narrow(current_dim, start, stop - start)?;
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
    Select(usize),
    /// This is a regular slice, purely indexing a chunk of the tensor
    Narrow(Bound<usize>, Bound<usize>),
    // IndexSelect(Tensor),
}

impl From<usize> for TensorIndexer {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

// impl From<&[usize]> for TensorIndexer {
//     fn from(index: &[usize]) -> Self {
//         let tensor = index.into();
//         TensorIndexer::IndexSelect(tensor)
//     }
// }
//
// impl From<Vec<usize>> for TensorIndexer {
//     fn from(index: Vec<usize>) -> Self {
//         let tensor = Tensor::of_slice(&index);
//         TensorIndexer::IndexSelect(tensor)
//     }
// }

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

impl<A> IndexOp<(A,)> for Tensor
where
    A: Into<TensorIndexer>,
{
    fn i(&self, index: (A,)) -> Result<Tensor, Error> {
        let idx_a = index.0.into();
        self.index(&[idx_a])
    }
}

impl<A, B> IndexOp<(A, B)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B)) -> Result<Tensor, Error> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        self.index(&[idx_a, idx_b])
    }
}

impl<A, B, C> IndexOp<(A, B, C)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C)) -> Result<Tensor, Error> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        self.index(&[idx_a, idx_b, idx_c])
    }
}

impl<A, B, C, D> IndexOp<(A, B, C, D)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D)) -> Result<Tensor, Error> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        self.index(&[idx_a, idx_b, idx_c, idx_d])
    }
}

impl<A, B, C, D, E> IndexOp<(A, B, C, D, E)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
    E: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D, E)) -> Result<Tensor, Error> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        let idx_e = index.4.into();
        self.index(&[idx_a, idx_b, idx_c, idx_d, idx_e])
    }
}

impl<A, B, C, D, E, F> IndexOp<(A, B, C, D, E, F)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
    E: Into<TensorIndexer>,
    F: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D, E, F)) -> Result<Tensor, Error> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        let idx_e = index.4.into();
        let idx_f = index.5.into();
        self.index(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f])
    }
}

impl<A, B, C, D, E, F, G> IndexOp<(A, B, C, D, E, F, G)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
    E: Into<TensorIndexer>,
    F: Into<TensorIndexer>,
    G: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D, E, F, G)) -> Result<Tensor, Error> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        let idx_e = index.4.into();
        let idx_f = index.5.into();
        let idx_g = index.6.into();
        self.index(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g])
    }
}
