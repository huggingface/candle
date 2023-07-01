use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use ::candle::{Device::Cpu, Tensor};

pub fn wrap_err(err: ::candle::Error) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

#[derive(Clone)]
#[pyclass(name = "Tensor")]
struct PyTensor(Tensor);

impl std::ops::Deref for PyTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(f: f32) -> PyResult<Self> {
        Ok(Self(Tensor::new(f, &Cpu).map_err(wrap_err)?))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.dims().to_vec()
    }

    #[getter]
    fn rank(&self) -> usize {
        self.0.rank()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __add__(&self, rhs: &PyAny) -> PyResult<Self> {
        let tensor = if let Ok(rhs) = rhs.extract::<Self>() {
            (&self.0 + &rhs.0).map_err(wrap_err)?
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            (&self.0 + rhs).map_err(wrap_err)?
        } else {
            Err(PyTypeError::new_err("unsupported for add"))?
        };
        Ok(Self(tensor))
    }

    fn __radd__(&self, rhs: &PyAny) -> PyResult<Self> {
        self.__add__(rhs)
    }
}

#[pyfunction]
fn add(tensor: &PyTensor, f: f64) -> PyResult<PyTensor> {
    let tensor = (&tensor.0 + f).map_err(wrap_err)?;
    Ok(PyTensor(tensor))
}

#[pymodule]
fn candle(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
