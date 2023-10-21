use ::candle::Tensor;
use pyo3::prelude::*;

#[derive(Clone, Debug)]
/// Represents an absolute shape e.g. (1, 2, 3)
pub struct PyShape(Vec<usize>);

impl<'source> pyo3::FromPyObject<'source> for PyShape {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if ob.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shape cannot be None",
            ));
        }

        let tuple = ob.downcast::<pyo3::types::PyTuple>()?;
        if tuple.len() == 1 {
            let first_element = tuple.get_item(0)?;
            let dims: Vec<usize> = pyo3::FromPyObject::extract(first_element)?;
            Ok(PyShape(dims))
        } else {
            let dims: Vec<usize> = pyo3::FromPyObject::extract(tuple)?;
            Ok(PyShape(dims))
        }
    }
}

impl From<PyShape> for ::candle::Shape {
    fn from(val: PyShape) -> Self {
        val.0.into()
    }
}

#[derive(Clone, Debug)]
/// Represents a relative shape e.g. (1, -1, 3)
pub struct PyRelativeShape(Vec<isize>);

impl<'source> pyo3::FromPyObject<'source> for PyRelativeShape {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if ob.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Shape cannot be None",
            ));
        }

        let tuple = ob.downcast::<pyo3::types::PyTuple>()?;
        if tuple.len() == 1 {
            let first_element = tuple.get_item(0)?;
            let dims: Vec<isize> = pyo3::FromPyObject::extract(first_element)?;
            Ok(PyRelativeShape(dims))
        } else {
            let dims: Vec<isize> = pyo3::FromPyObject::extract(tuple)?;
            Ok(PyRelativeShape(dims))
        }
    }
}

impl PyRelativeShape {
    /// Returns `true` if the shape is absolute e.g. (1, 2, 3)
    pub fn is_absolute(&self) -> bool {
        self.0.iter().all(|x| *x > 0)
    }

    /// Convert a relative shape to an absolute shape e.g. (1, -1) -> (1, 12)
    pub fn to_absolute(&self, t: &Tensor) -> PyResult<PyShape> {
        if self.is_absolute() {
            return Ok(PyShape(
                self.0.iter().map(|x| *x as usize).collect::<Vec<usize>>(),
            ));
        }

        let mut elements = t.elem_count();
        let mut new_dims: Vec<usize> = vec![];
        for dim in self.0.iter() {
            if *dim > 0 {
                new_dims.push(*dim as usize);
                elements /= *dim as usize;
            } else if *dim == -1 {
                new_dims.push(elements);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid dimension in shape: {}",
                    dim
                )));
            }
        }
        Ok(PyShape(new_dims))
    }
}
