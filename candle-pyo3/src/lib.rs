use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::candle::{Device::Cpu, Tensor};

pub fn wrap_err(err: ::candle::Error) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

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
