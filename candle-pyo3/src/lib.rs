use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use half::{bf16, f16};

use ::candle::{DType, Device::Cpu, Tensor};

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
    // TODO: Handle arbitrary input dtype and shape.
    fn new(f: f32) -> PyResult<Self> {
        Ok(Self(Tensor::new(f, &Cpu).map_err(wrap_err)?))
    }

    fn values(&self, py: Python<'_>) -> PyResult<PyObject> {
        // TODO: Handle arbitrary shapes.
        let v = match self.0.dtype() {
            // TODO: Use the map bits to avoid enumerating the types.
            DType::U8 => self.to_scalar::<u8>().map_err(wrap_err)?.to_object(py),
            DType::U32 => self.to_scalar::<u32>().map_err(wrap_err)?.to_object(py),
            DType::F32 => self.to_scalar::<f32>().map_err(wrap_err)?.to_object(py),
            DType::F64 => self.to_scalar::<f64>().map_err(wrap_err)?.to_object(py),
            DType::BF16 => self
                .to_scalar::<bf16>()
                .map_err(wrap_err)?
                .to_f32()
                .to_object(py),
            DType::F16 => self
                .to_scalar::<f16>()
                .map_err(wrap_err)?
                .to_f32()
                .to_object(py),
        };
        Ok(v)
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, self.0.dims()).to_object(py)
    }

    #[getter]
    fn stride(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, self.0.stride()).to_object(py)
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyObject {
        PyString::new(py, self.0.dtype().as_str()).to_object(py)
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

    fn __mul__(&self, rhs: &PyAny) -> PyResult<Self> {
        let tensor = if let Ok(rhs) = rhs.extract::<Self>() {
            (&self.0 * &rhs.0).map_err(wrap_err)?
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            (&self.0 * rhs).map_err(wrap_err)?
        } else {
            Err(PyTypeError::new_err("unsupported for mul"))?
        };
        Ok(Self(tensor))
    }

    fn __rmul__(&self, rhs: &PyAny) -> PyResult<Self> {
        self.__mul__(rhs)
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
