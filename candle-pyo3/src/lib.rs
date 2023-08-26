#![allow(clippy::redundant_closure_call)]
// TODO: Handle negative dimension indexes.
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use half::{bf16, f16};

use ::candle::{DType, Device, Tensor, WithDType};

pub fn wrap_err(err: ::candle::Error) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

#[derive(Clone, Debug)]
struct PyShape(Vec<usize>);

impl<'source> pyo3::FromPyObject<'source> for PyShape {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let dims: Vec<usize> = pyo3::FromPyObject::extract(ob)?;
        Ok(PyShape(dims))
    }
}

impl From<PyShape> for ::candle::Shape {
    fn from(val: PyShape) -> Self {
        val.0.into()
    }
}

#[derive(Clone, Debug)]
#[pyclass(name = "Tensor")]
struct PyTensor(Tensor);

impl std::ops::Deref for PyTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass(name = "DType")]
struct PyDType(DType);

#[pymethods]
impl PyDType {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyDType {
    fn from_pyobject(ob: PyObject, py: Python<'_>) -> PyResult<Self> {
        use std::str::FromStr;
        if let Ok(dtype) = ob.extract::<&str>(py) {
            let dtype = DType::from_str(dtype)
                .map_err(|_| PyTypeError::new_err(format!("invalid dtype '{dtype}'")))?;
            Ok(Self(dtype))
        } else {
            ob.extract(py)
        }
    }
}

static CUDA_DEVICE: std::sync::Mutex<Option<Device>> = std::sync::Mutex::new(None);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyDevice {
    Cpu,
    Cuda,
}

impl PyDevice {
    fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Cuda(_) => Self::Cuda,
        }
    }

    fn as_device(&self) -> PyResult<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => {
                let mut device = CUDA_DEVICE.lock().unwrap();
                if let Some(device) = device.as_ref() {
                    return Ok(device.clone());
                };
                let d = Device::new_cuda(0).map_err(wrap_err)?;
                *device = Some(d.clone());
                Ok(d)
            }
        }
    }
}

impl<'source> FromPyObject<'source> for PyDevice {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let device: &str = ob.extract()?;
        let device = match device {
            "cpu" => PyDevice::Cpu,
            "cuda" => PyDevice::Cuda,
            _ => Err(PyTypeError::new_err(format!("invalid device '{device}'")))?,
        };
        Ok(device)
    }
}

impl ToPyObject for PyDevice {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let str = match self {
            PyDevice::Cpu => "cpu",
            PyDevice::Cuda => "cuda",
        };
        str.to_object(py)
    }
}

trait PyWithDType: WithDType {
    fn to_py(&self, py: Python<'_>) -> PyObject;
}

macro_rules! pydtype {
    ($ty:ty, $conv:expr) => {
        impl PyWithDType for $ty {
            fn to_py(&self, py: Python<'_>) -> PyObject {
                $conv(*self).to_object(py)
            }
        }
    };
}
pydtype!(u8, |v| v);
pydtype!(u32, |v| v);
pydtype!(i64, |v| v);
pydtype!(f16, f32::from);
pydtype!(bf16, f32::from);
pydtype!(f32, |v| v);
pydtype!(f64, |v| v);

// TODO: Something similar to this should probably be a part of candle core.
trait MapDType {
    type Output;
    fn f<T: PyWithDType>(&self, t: &Tensor) -> PyResult<Self::Output>;

    fn map(&self, t: &Tensor) -> PyResult<Self::Output> {
        match t.dtype() {
            DType::U8 => self.f::<u8>(t),
            DType::U32 => self.f::<u32>(t),
            DType::I64 => self.f::<i64>(t),
            DType::BF16 => self.f::<bf16>(t),
            DType::F16 => self.f::<f16>(t),
            DType::F32 => self.f::<f32>(t),
            DType::F64 => self.f::<f64>(t),
        }
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    // TODO: Handle arbitrary input dtype and shape.
    fn new(py: Python<'_>, vs: PyObject) -> PyResult<Self> {
        use Device::Cpu;
        let tensor = if let Ok(vs) = vs.extract::<u32>(py) {
            Tensor::new(vs, &Cpu).map_err(wrap_err)?
        } else if let Ok(vs) = vs.extract::<i64>(py) {
            Tensor::new(vs, &Cpu).map_err(wrap_err)?
        } else if let Ok(vs) = vs.extract::<Vec<u32>>(py) {
            Tensor::new(vs.as_slice(), &Cpu).map_err(wrap_err)?
        } else if let Ok(vs) = vs.extract::<Vec<i64>>(py) {
            Tensor::new(vs.as_slice(), &Cpu).map_err(wrap_err)?
        } else if let Ok(vs) = vs.extract::<f32>(py) {
            Tensor::new(vs, &Cpu).map_err(wrap_err)?
        } else if let Ok(vs) = vs.extract::<Vec<f32>>(py) {
            Tensor::new(vs.as_slice(), &Cpu).map_err(wrap_err)?
        } else {
            Err(PyTypeError::new_err("incorrect type for tensor"))?
        };
        Ok(Self(tensor))
    }

    /// Gets the tensor data as a Python value/array/array of array/...
    fn values(&self, py: Python<'_>) -> PyResult<PyObject> {
        struct M<'a>(Python<'a>);
        impl<'a> MapDType for M<'a> {
            type Output = PyObject;
            fn f<T: PyWithDType>(&self, t: &Tensor) -> PyResult<Self::Output> {
                match t.rank() {
                    0 => Ok(t.to_scalar::<T>().map_err(wrap_err)?.to_py(self.0)),
                    1 => {
                        let v = t.to_vec1::<T>().map_err(wrap_err)?;
                        let v = v.iter().map(|v| v.to_py(self.0)).collect::<Vec<_>>();
                        Ok(v.to_object(self.0))
                    }
                    2 => {
                        let v = t.to_vec2::<T>().map_err(wrap_err)?;
                        let v = v
                            .iter()
                            .map(|v| v.iter().map(|v| v.to_py(self.0)).collect())
                            .collect::<Vec<Vec<_>>>();
                        Ok(v.to_object(self.0))
                    }
                    3 => {
                        let v = t.to_vec3::<T>().map_err(wrap_err)?;
                        let v = v
                            .iter()
                            .map(|v| {
                                v.iter()
                                    .map(|v| v.iter().map(|v| v.to_py(self.0)).collect())
                                    .collect()
                            })
                            .collect::<Vec<Vec<Vec<_>>>>();
                        Ok(v.to_object(self.0))
                    }
                    n => Err(PyTypeError::new_err(format!(
                        "TODO: conversion to PyObject is not handled for rank {n}"
                    )))?,
                }
            }
        }
        // TODO: Handle arbitrary shapes.
        M(py).map(self)
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
    fn dtype(&self) -> PyDType {
        PyDType(self.0.dtype())
    }

    #[getter]
    fn device(&self, py: Python<'_>) -> PyObject {
        PyDevice::from_device(self.0.device()).to_object(py)
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

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        Ok(PyTensor(self.0.matmul(rhs).map_err(wrap_err)?))
    }

    fn where_cond(&self, on_true: &Self, on_false: &Self) -> PyResult<Self> {
        Ok(PyTensor(
            self.0.where_cond(on_true, on_false).map_err(wrap_err)?,
        ))
    }

    fn __add__(&self, rhs: &PyAny) -> PyResult<Self> {
        let tensor = if let Ok(rhs) = rhs.extract::<Self>() {
            (&self.0 + &rhs.0).map_err(wrap_err)?
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            (&self.0 + rhs).map_err(wrap_err)?
        } else {
            Err(PyTypeError::new_err("unsupported rhs for add"))?
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
            Err(PyTypeError::new_err("unsupported rhs for mul"))?
        };
        Ok(Self(tensor))
    }

    fn __rmul__(&self, rhs: &PyAny) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    fn __sub__(&self, rhs: &PyAny) -> PyResult<Self> {
        let tensor = if let Ok(rhs) = rhs.extract::<Self>() {
            (&self.0 - &rhs.0).map_err(wrap_err)?
        } else if let Ok(rhs) = rhs.extract::<f64>() {
            (&self.0 - rhs).map_err(wrap_err)?
        } else {
            Err(PyTypeError::new_err("unsupported rhs for sub"))?
        };
        Ok(Self(tensor))
    }

    fn reshape(&self, shape: PyShape) -> PyResult<Self> {
        Ok(PyTensor(self.0.reshape(shape).map_err(wrap_err)?))
    }

    fn broadcast_as(&self, shape: PyShape) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_as(shape).map_err(wrap_err)?))
    }

    fn broadcast_left(&self, shape: PyShape) -> PyResult<Self> {
        Ok(PyTensor(self.0.broadcast_left(shape).map_err(wrap_err)?))
    }

    fn squeeze(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor(self.0.squeeze(dim).map_err(wrap_err)?))
    }

    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        Ok(PyTensor(self.0.unsqueeze(dim).map_err(wrap_err)?))
    }

    fn get(&self, index: usize) -> PyResult<Self> {
        Ok(PyTensor(self.0.get(index).map_err(wrap_err)?))
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> PyResult<Self> {
        Ok(PyTensor(self.0.transpose(dim1, dim2).map_err(wrap_err)?))
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> PyResult<Self> {
        Ok(PyTensor(self.0.narrow(dim, start, len).map_err(wrap_err)?))
    }

    fn sum_keepdim(&self, dims: Vec<usize>) -> PyResult<Self> {
        // TODO: Support a single dim as input?
        Ok(PyTensor(
            self.0.sum_keepdim(dims.as_slice()).map_err(wrap_err)?,
        ))
    }

    fn sum_all(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.sum_all().map_err(wrap_err)?))
    }

    fn flatten_all(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.flatten_all().map_err(wrap_err)?))
    }

    fn t(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.t().map_err(wrap_err)?))
    }

    fn contiguous(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.contiguous().map_err(wrap_err)?))
    }

    fn is_contiguous(&self) -> bool {
        self.0.is_contiguous()
    }

    fn is_fortran_contiguous(&self) -> bool {
        self.0.is_fortran_contiguous()
    }

    fn detach(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.detach().map_err(wrap_err)?))
    }

    fn copy(&self) -> PyResult<Self> {
        Ok(PyTensor(self.0.copy().map_err(wrap_err)?))
    }

    fn to_dtype(&self, dtype: PyObject, py: Python<'_>) -> PyResult<Self> {
        let dtype = PyDType::from_pyobject(dtype, py)?;
        Ok(PyTensor(self.0.to_dtype(dtype.0).map_err(wrap_err)?))
    }

    fn to_device(&self, device: PyDevice) -> PyResult<Self> {
        let device = device.as_device()?;
        Ok(PyTensor(self.0.to_device(&device).map_err(wrap_err)?))
    }
}

/// Concatenate the tensors across one axis.
#[pyfunction]
fn cat(tensors: Vec<PyTensor>, dim: usize) -> PyResult<PyTensor> {
    let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    let tensor = Tensor::cat(&tensors, dim).map_err(wrap_err)?;
    Ok(PyTensor(tensor))
}

#[pyfunction]
fn stack(tensors: Vec<PyTensor>, dim: usize) -> PyResult<PyTensor> {
    let tensors = tensors.into_iter().map(|t| t.0).collect::<Vec<_>>();
    let tensor = Tensor::stack(&tensors, dim).map_err(wrap_err)?;
    Ok(PyTensor(tensor))
}

#[pyfunction]
fn tensor(py: Python<'_>, vs: PyObject) -> PyResult<PyTensor> {
    PyTensor::new(py, vs)
}

#[pyfunction]
#[pyo3(signature = (shape, *, device=None))]
fn rand(_py: Python<'_>, shape: PyShape, device: Option<PyDevice>) -> PyResult<PyTensor> {
    let device = device.unwrap_or(PyDevice::Cpu).as_device()?;
    let tensor = Tensor::rand(0f32, 1f32, shape.0, &device).map_err(wrap_err)?;
    Ok(PyTensor(tensor))
}

#[pyfunction]
#[pyo3(signature = (shape, *, device=None))]
fn randn(_py: Python<'_>, shape: PyShape, device: Option<PyDevice>) -> PyResult<PyTensor> {
    let device = device.unwrap_or(PyDevice::Cpu).as_device()?;
    let tensor = Tensor::randn(0f32, 1f32, shape.0, &device).map_err(wrap_err)?;
    Ok(PyTensor(tensor))
}

#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None, device=None))]
fn ones(
    py: Python<'_>,
    shape: PyShape,
    dtype: Option<PyObject>,
    device: Option<PyDevice>,
) -> PyResult<PyTensor> {
    let dtype = match dtype {
        None => DType::F32,
        Some(dtype) => PyDType::from_pyobject(dtype, py)?.0,
    };
    let device = device.unwrap_or(PyDevice::Cpu).as_device()?;
    let tensor = Tensor::ones(shape.0, dtype, &device).map_err(wrap_err)?;
    Ok(PyTensor(tensor))
}

#[pyfunction]
#[pyo3(signature = (shape, *, dtype=None, device=None))]
fn zeros(
    py: Python<'_>,
    shape: PyShape,
    dtype: Option<PyObject>,
    device: Option<PyDevice>,
) -> PyResult<PyTensor> {
    let dtype = match dtype {
        None => DType::F32,
        Some(dtype) => PyDType::from_pyobject(dtype, py)?.0,
    };
    let device = device.unwrap_or(PyDevice::Cpu).as_device()?;
    let tensor = Tensor::zeros(shape.0, dtype, &device).map_err(wrap_err)?;
    Ok(PyTensor(tensor))
}

#[pymodule]
fn candle(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyDType>()?;
    m.add("u8", PyDType(DType::U8))?;
    m.add("u32", PyDType(DType::U32))?;
    m.add("bf16", PyDType(DType::BF16))?;
    m.add("f16", PyDType(DType::F16))?;
    m.add("f32", PyDType(DType::F32))?;
    m.add("f64", PyDType(DType::F64))?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    Ok(())
}
