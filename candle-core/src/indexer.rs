use crate::{backend::BackendStorage, Error, Tensor};
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

impl<B: BackendStorage> Tensor<B> {
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
    fn index(&self, indexers: &[TensorIndexer<B>]) -> Result<Self, Error> {
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
                TensorIndexer::IndexSelect(indexes) => {
                    if indexes.rank() != 1 {
                        crate::bail!("multi-dimensional tensor indexing is not supported")
                    }
                    let out = x.index_select(&indexes.to_device(x.device())?, current_dim)?;
                    current_dim += 1;
                    out
                }
                TensorIndexer::Err(e) => crate::bail!("indexing error {e:?}"),
            };
        }
        Ok(x)
    }
}

#[derive(Debug)]
/// Generic structure used to index a slice of the tensor
pub enum TensorIndexer<B: BackendStorage> {
    /// This selects the elements for which an index has some specific value.
    Select(usize),
    /// This is a regular slice, purely indexing a chunk of the tensor
    Narrow(Bound<usize>, Bound<usize>),
    /// Indexing via a 1d tensor
    IndexSelect(Tensor<B>),
    Err(Error),
}

impl<B: BackendStorage> From<usize> for TensorIndexer<B> {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

impl From<&[u32]> for TensorIndexer<crate::CpuStorage> {
    fn from(index: &[u32]) -> Self {
        match Tensor::new(index, &crate::cpu_backend::CpuDevice {}) {
            Ok(tensor) => TensorIndexer::IndexSelect(tensor),
            Err(e) => TensorIndexer::Err(e),
        }
    }
}

impl From<Vec<u32>> for TensorIndexer<crate::CpuStorage> {
    fn from(index: Vec<u32>) -> Self {
        let len = index.len();
        match Tensor::from_vec(index, len, &crate::cpu_backend::CpuDevice {}) {
            Ok(tensor) => TensorIndexer::IndexSelect(tensor),
            Err(e) => TensorIndexer::Err(e),
        }
    }
}

impl<B: BackendStorage> From<&Tensor<B>> for TensorIndexer<B> {
    fn from(tensor: &Tensor<B>) -> Self {
        TensorIndexer::IndexSelect(tensor.clone())
    }
}

trait RB: RangeBounds<usize> {}
impl RB for Range<usize> {}
impl RB for RangeFrom<usize> {}
impl RB for RangeFull {}
impl RB for RangeInclusive<usize> {}
impl RB for RangeTo<usize> {}
impl RB for RangeToInclusive<usize> {}

impl<T: RB, B: BackendStorage> From<T> for TensorIndexer<B> {
    fn from(range: T) -> Self {
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

/// Trait used to implement multiple signatures for ease of use of the slicing
/// of a tensor
pub trait IndexOp<T, B: BackendStorage> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn i(&self, index: T) -> Result<Tensor<B>, Error>;
}

impl<T, B: BackendStorage> IndexOp<T, B> for Tensor<B>
where
    T: Into<TensorIndexer<B>>,
{
    ///```rust
    /// use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::new(&[
    ///     [0., 1.],
    ///     [2., 3.],
    ///     [4., 5.]
    /// ], &Device::Cpu)?;
    ///
    /// let b = a.i(0)?;
    /// assert_eq!(b.shape().dims(), &[2]);
    /// assert_eq!(b.to_vec1::<f64>()?, &[0., 1.]);
    ///
    /// let c = a.i(..2)?;
    /// assert_eq!(c.shape().dims(), &[2, 2]);
    /// assert_eq!(c.to_vec2::<f64>()?, &[
    ///     [0., 1.],
    ///     [2., 3.]
    /// ]);
    ///
    /// let d = a.i(1..)?;
    /// assert_eq!(d.shape().dims(), &[2, 2]);
    /// assert_eq!(d.to_vec2::<f64>()?, &[
    ///     [2., 3.],
    ///     [4., 5.]
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn i(&self, index: T) -> Result<Tensor<B>, Error> {
        self.index(&[index.into()])
    }
}

impl<T, B: BackendStorage> IndexOp<(T,), B> for Tensor<B>
where
    T: Into<TensorIndexer<B>>,
{
    ///```rust
    /// use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::new(&[
    ///     [0f32, 1.],
    ///     [2.  , 3.],
    ///     [4.  , 5.]
    /// ], &Device::Cpu)?;
    ///
    /// let b = a.i((0,))?;
    /// assert_eq!(b.shape().dims(), &[2]);
    /// assert_eq!(b.to_vec1::<f32>()?, &[0., 1.]);
    ///
    /// let c = a.i((..2,))?;
    /// assert_eq!(c.shape().dims(), &[2, 2]);
    /// assert_eq!(c.to_vec2::<f32>()?, &[
    ///     [0., 1.],
    ///     [2., 3.]
    /// ]);
    ///
    /// let d = a.i((1..,))?;
    /// assert_eq!(d.shape().dims(), &[2, 2]);
    /// assert_eq!(d.to_vec2::<f32>()?, &[
    ///     [2., 3.],
    ///     [4., 5.]
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn i(&self, (t,): (T,)) -> Result<Tensor<B>, Error> {
        self.index(&[t.into()])
    }
}
#[allow(non_snake_case)]
impl<T, U, B: BackendStorage> IndexOp<(T, U), B> for Tensor<B>
where
    T: Into<TensorIndexer<B>>,
    U: Into<TensorIndexer<B>>,
{
    ///```rust
    /// use candle_core::{Tensor, DType, Device, IndexOp};
    /// let a = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.], [6., 7., 8.]], &Device::Cpu)?;
    ///
    /// let b = a.i((1, 0))?;
    /// assert_eq!(b.to_vec0::<f32>()?, 3.);
    ///
    /// let c = a.i((..2, 1))?;
    /// assert_eq!(c.shape().dims(), &[2]);
    /// assert_eq!(c.to_vec1::<f32>()?, &[1., 4.]);
    ///
    /// let d = a.i((2.., ..))?;
    /// assert_eq!(d.shape().dims(), &[1, 3]);
    /// assert_eq!(d.to_vec2::<f32>()?, &[[6., 7., 8.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn i(&self, (t, u): (T, U)) -> Result<Tensor<B>, Error> {
        self.index(&[t.into(), u.into()])
    }
}

macro_rules! index_op_tuple {
    ($doc:tt, $($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<Storage: BackendStorage,$($t),*> IndexOp<($($t,)*), Storage> for Tensor<Storage>
        where
            $($t: Into<TensorIndexer<Storage>>,)*
        {
            #[doc=$doc]
            fn i(&self, ($($t,)*): ($($t,)*)) -> Result<Tensor<Storage>, Error> {
                self.index(&[$($t.into(),)*])
            }
        }
    };
}

index_op_tuple!("see [TensorIndex#method.i]", A, B, C);
index_op_tuple!("see [TensorIndex#method.i]", A, B, C, D);
index_op_tuple!("see [TensorIndex#method.i]", A, B, C, D, E);
index_op_tuple!("see [TensorIndex#method.i]", A, B, C, D, E, F);
index_op_tuple!("see [TensorIndex#method.i]", A, B, C, D, E, F, G);
