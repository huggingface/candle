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

    pub fn r0(&self) -> Result<()> {
        let shape = &self.0;
        if shape.is_empty() {
            Ok(())
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 0,
                got: shape.len(),
                shape: self.clone(),
            })
        }
    }

    pub fn r1(&self) -> Result<usize> {
        let shape = &self.0;
        if shape.len() == 1 {
            Ok(shape[0])
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: shape.len(),
                shape: self.clone(),
            })
        }
    }

    pub fn r2(&self) -> Result<(usize, usize)> {
        let shape = &self.0;
        if shape.len() == 2 {
            Ok((shape[0], shape[1]))
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: shape.len(),
                shape: self.clone(),
            })
        }
    }

    pub fn r3(&self) -> Result<(usize, usize, usize)> {
        let shape = &self.0;
        if shape.len() == 3 {
            Ok((shape[0], shape[1], shape[2]))
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 3,
                got: shape.len(),
                shape: self.clone(),
            })
        }
    }

    pub fn r4(&self) -> Result<(usize, usize, usize, usize)> {
        let shape = &self.0;
        if shape.len() == 4 {
            Ok((shape[0], shape[1], shape[2], shape[4]))
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 4,
                got: shape.len(),
                shape: self.clone(),
            })
        }
    }

    /// The strides given in number of elements for a contiguous n-dimensional
    /// arrays using this shape.
    pub(crate) fn stride_contiguous(&self) -> Vec<usize> {
        self.0
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect()
    }
}
