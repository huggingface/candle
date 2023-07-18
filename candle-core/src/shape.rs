use crate::{Error, Result};

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

pub const SCALAR: Shape = Shape(vec![]);

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(d12: (usize, usize)) -> Self {
        Self(vec![d12.0, d12.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(d123: (usize, usize, usize)) -> Self {
        Self(vec![d123.0, d123.1, d123.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(d1234: (usize, usize, usize, usize)) -> Self {
        Self(vec![d1234.0, d1234.1, d1234.2, d1234.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(d12345: (usize, usize, usize, usize, usize)) -> Self {
        Self(vec![d12345.0, d12345.1, d12345.2, d12345.3, d12345.4])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        impl Shape {
            pub fn $fn_name(&self) -> Result<$out_type> {
                if self.0.len() != $cnt {
                    Err(Error::UnexpectedNumberOfDims {
                        expected: $cnt,
                        got: self.0.len(),
                        shape: self.clone(),
                    }
                    .bt())
                } else {
                    Ok($dims(&self.0))
                }
            }
        }
        impl std::convert::TryInto<$out_type> for Shape {
            type Error = crate::Error;
            fn try_into(self) -> std::result::Result<$out_type, Self::Error> {
                self.$fn_name()
            }
        }
    };
}

impl Shape {
    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn into_dims(self) -> Vec<usize> {
        self.0
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    /// The strides given in number of elements for a contiguous n-dimensional
    /// arrays using this shape.
    pub(crate) fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride: Vec<_> = self
            .0
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect();
        stride.reverse();
        stride
    }

    /// Returns true if the strides are C contiguous (aka row major).
    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()).rev() {
            if stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    /// Returns true if the strides are Fortran contiguous (aka column major).
    pub fn is_fortran_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()) {
            if stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        self.0.extend(additional_dims);
        self
    }
}

pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.dims().len() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(dim)
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim > shape.dims().len() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(dim)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum D {
    Minus1,
    Minus2,
}

impl D {
    fn out_of_range(&self, shape: &Shape, op: &'static str) -> Error {
        let dim = match self {
            Self::Minus1 => -1,
            Self::Minus2 => -2,
        };
        Error::DimOutOfRange {
            shape: shape.clone(),
            dim,
            op,
        }
        .bt()
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 if rank >= 1 => Ok(rank - 1),
            Self::Minus2 if rank >= 2 => Ok(rank - 2),
            _ => Err(self.out_of_range(shape, op)),
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 => Ok(rank),
            Self::Minus2 if rank >= 1 => Ok(rank - 1),
            _ => Err(self.out_of_range(shape, op)),
        }
    }
}

pub trait Dims: Sized {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>>;

    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dims = self.to_indexes_internal(shape, op)?;
        for (i, &dim) in dims.iter().enumerate() {
            if dims[..i].contains(&dim) {
                Err(Error::DuplicateDimIndex {
                    shape: shape.clone(),
                    dims: dims.clone(),
                    op,
                }
                .bt())?
            }
            if dim >= shape.rank() {
                Err(Error::DimOutOfRange {
                    shape: shape.clone(),
                    dim: dim as i32,
                    op,
                }
                .bt())?
            }
        }
        Ok(dims)
    }
}

impl Dims for Vec<usize> {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self)
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self.to_vec())
    }
}

impl Dims for &[usize] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self.to_vec())
    }
}

impl Dims for () {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(vec![])
    }
}

impl<D: Dim + Sized> Dims for D {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.to_index(shape, op)?;
        Ok(vec![dim])
    }
}

impl<D: Dim> Dims for (D,) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.0.to_index(shape, op)?;
        Ok(vec![dim])
    }
}

impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        Ok(vec![d0, d1])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        Ok(vec![d0, d1, d2])
    }
}

extract_dims!(r0, 0, |_: &Vec<usize>| (), ());
extract_dims!(r1, 1, |d: &[usize]| d[0], usize);
extract_dims!(r2, 2, |d: &[usize]| (d[0], d[1]), (usize, usize));
extract_dims!(
    r3,
    3,
    |d: &[usize]| (d[0], d[1], d[2]),
    (usize, usize, usize)
);
extract_dims!(
    r4,
    4,
    |d: &[usize]| (d[0], d[1], d[2], d[3]),
    (usize, usize, usize, usize)
);
extract_dims!(
    r5,
    5,
    |d: &[usize]| (d[0], d[1], d[2], d[3], d[4]),
    (usize, usize, usize, usize, usize)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride() {
        let shape = Shape::from(());
        assert_eq!(shape.stride_contiguous(), Vec::<usize>::new());
        let shape = Shape::from(42);
        assert_eq!(shape.stride_contiguous(), [1]);
        let shape = Shape::from((42, 1337));
        assert_eq!(shape.stride_contiguous(), [1337, 1]);
        let shape = Shape::from((299, 792, 458));
        assert_eq!(shape.stride_contiguous(), [458 * 792, 458, 1]);
    }
}
