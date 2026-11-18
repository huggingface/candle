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
                TensorIndexer::IndexSelect(indexes) => {
                    if indexes.rank() != 1 {
                        crate::bail!("multi-dimensional tensor indexing is not supported")
                    }
                    let out = x.index_select(&indexes.to_device(x.device())?, current_dim)?;
                    current_dim += 1;
                    out
                }
                TensorIndexer::Step(range) => {
                    let out = range.index(&x, current_dim)?;
                    current_dim += 1;
                    out
                }
                TensorIndexer::Err(e) => crate::bail!("indexing error {e:?}"),
            };
        }
        Ok(x)
    }
}

/// A range of tensor indices traversed with a nonzero signed step.
///
/// Bounds are nonnegative indices. With a positive step, omitted bounds are
/// zero and the dimension's length. With a negative step, they are the last
/// index and a position before zero. An excluded start moves one position in
/// the direction of the step; an included end permits that position.
/// Bounds are checked against the dimension rather than clipped.
/// A zero step returns an error when the range is used for indexing.
///
/// ```
/// use candle_core::{Device, IndexOp, StepRange, Tensor};
/// let xs = Tensor::arange(0u32, 8, &Device::Cpu)?;
/// assert_eq!(xs.i(StepRange::new(.., 2))?.to_vec1::<u32>()?, [0, 2, 4, 6]);
/// assert_eq!(xs.i(StepRange::new(6..1, -2))?.to_vec1::<u32>()?, [6, 4, 2]);
/// assert_eq!(xs.i(StepRange::new(.., -2))?.to_vec1::<u32>()?, [7, 5, 3, 1]);
/// # Ok::<(), candle_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct StepRange {
    start: Bound<usize>,
    end: Bound<usize>,
    step: isize,
}

impl StepRange {
    /// Creates a stepped range. Validation takes place when indexing a tensor.
    pub fn new<R: RangeBounds<usize>>(range: R, step: isize) -> Self {
        Self {
            start: range.start_bound().cloned(),
            end: range.end_bound().cloned(),
            step,
        }
    }

    fn index(&self, tensor: &Tensor, dim: usize) -> Result<Tensor, Error> {
        if self.step == 0 {
            crate::bail!("step size cannot be zero")
        }
        let len = tensor.dim(dim)? as i128;
        if let Bound::Included(n) = self.end {
            if n as i128 >= len {
                crate::bail!(
                    "step indexing inclusive end {n} out of range for dimension of size {len}"
                )
            }
        }
        let forward = self.step > 0;
        let direction = if forward { 1 } else { -1 };
        let start = match self.start {
            Bound::Included(n) => n as i128,
            Bound::Excluded(n) => n as i128 + direction,
            Bound::Unbounded => {
                if forward {
                    0
                } else {
                    len - 1
                }
            }
        };
        let end = match self.end {
            Bound::Included(n) => n as i128 + direction,
            Bound::Excluded(n) => n as i128,
            Bound::Unbounded => {
                if forward {
                    len
                } else {
                    -1
                }
            }
        };
        let valid = if forward {
            (0..=len).contains(&start) && (0..=len).contains(&end)
        } else {
            (-1..len).contains(&start) && (-1..len).contains(&end)
        };
        if !valid {
            crate::bail!(
                "step indexing bounds {:?}..{:?} out of range for dimension of size {len}",
                self.start,
                self.end
            )
        }
        if (forward && start >= end) || (!forward && start <= end) {
            return tensor.narrow(dim, 0, 0);
        }
        if self.step == 1 {
            return tensor.narrow(dim, start as usize, (end - start) as usize);
        }
        if tensor.elem_count() == 0 {
            let distance = (end - start).abs();
            let count = (distance - 1) / (self.step as i128).abs() + 1;
            return tensor.narrow(dim, 0, count as usize);
        }
        if len > i64::MAX as i128 {
            crate::bail!("step indexing dimension is too large for i64 indices")
        }
        // Widen before adding: even a small tensor may be indexed with
        // isize::MIN/MAX, and the final unused position can exceed i64.
        let mut position = start;
        let mut indices = Vec::new();
        while if forward {
            position < end
        } else {
            position > end
        } {
            indices.push(position as i64);
            position += self.step as i128;
        }
        let count = indices.len();
        let indices = Tensor::from_vec(indices, count, tensor.device())?;
        tensor.contiguous()?.index_select(&indices, dim)
    }
}

impl From<StepRange> for TensorIndexer {
    fn from(range: StepRange) -> Self {
        Self::Step(range)
    }
}

#[derive(Debug)]
/// Generic structure used to index a slice of the tensor
pub enum TensorIndexer {
    /// This selects the elements for which an index has some specific value.
    Select(usize),
    /// This is a regular slice, purely indexing a chunk of the tensor
    Narrow(Bound<usize>, Bound<usize>),
    /// Indexing via a range with a signed step.
    Step(StepRange),
    /// Indexing via a 1d tensor
    IndexSelect(Tensor),
    Err(Error),
}

impl From<usize> for TensorIndexer {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

impl From<&[u32]> for TensorIndexer {
    fn from(index: &[u32]) -> Self {
        match Tensor::new(index, &crate::Device::Cpu) {
            Ok(tensor) => TensorIndexer::IndexSelect(tensor),
            Err(e) => TensorIndexer::Err(e),
        }
    }
}

impl From<Vec<u32>> for TensorIndexer {
    fn from(index: Vec<u32>) -> Self {
        let len = index.len();
        match Tensor::from_vec(index, len, &crate::Device::Cpu) {
            Ok(tensor) => TensorIndexer::IndexSelect(tensor),
            Err(e) => TensorIndexer::Err(e),
        }
    }
}

impl From<&Tensor> for TensorIndexer {
    fn from(tensor: &Tensor) -> Self {
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

impl<T: RB> From<T> for TensorIndexer {
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
pub trait IndexOp<T> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn i(&self, index: T) -> Result<Tensor, Error>;
}

impl<T> IndexOp<T> for Tensor
where
    T: Into<TensorIndexer>,
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
    fn i(&self, index: T) -> Result<Tensor, Error> {
        self.index(&[index.into()])
    }
}

impl<A> IndexOp<(A,)> for Tensor
where
    A: Into<TensorIndexer>,
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
    fn i(&self, (a,): (A,)) -> Result<Tensor, Error> {
        self.index(&[a.into()])
    }
}
#[allow(non_snake_case)]
impl<A, B> IndexOp<(A, B)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
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
    fn i(&self, (a, b): (A, B)) -> Result<Tensor, Error> {
        self.index(&[a.into(), b.into()])
    }
}

macro_rules! index_op_tuple {
    ($doc:tt, $($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*> IndexOp<($($t,)*)> for Tensor
        where
            $($t: Into<TensorIndexer>,)*
        {
            #[doc=$doc]
            fn i(&self, ($($t,)*): ($($t,)*)) -> Result<Tensor, Error> {
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
