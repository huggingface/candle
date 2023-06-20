use crate::{Error, Result};

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(pub(crate) Vec<usize>);

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}

impl From<&[usize; 1]> for Shape {
    fn from(dims: &[usize; 1]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 2]> for Shape {
    fn from(dims: &[usize; 2]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 3]> for Shape {
    fn from(dims: &[usize; 3]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
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

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(&self) -> Result<$out_type> {
            if self.0.len() != $cnt {
                Err(Error::UnexpectedNumberOfDims {
                    expected: $cnt,
                    got: self.0.len(),
                    shape: self.clone(),
                })
            } else {
                Ok($dims(&self.0))
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

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
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
}

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
