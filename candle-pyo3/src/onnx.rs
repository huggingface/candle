use std::collections::HashMap;

use crate::utils::wrap_err;
use crate::{PyDType, PyTensor};
use candle_onnx::eval::{dtype, get_tensor, simple_eval};
use candle_onnx::onnx::tensor_proto::DataType;
use candle_onnx::onnx::tensor_shape_proto::dimension::Value;
use candle_onnx::onnx::type_proto::{Tensor as ONNXTensor, Value as ONNXValue};
use candle_onnx::onnx::{ModelProto, ValueInfoProto};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

#[derive(Clone, Debug)]
#[pyclass(name = "ONNXTensorDescription")]
/// A wrapper around an ONNX tensor description.
pub struct PyONNXTensorDescriptor(ONNXTensor);

#[pymethods]
impl PyONNXTensorDescriptor {
    #[getter]
    /// The data type of the tensor.
    /// &RETURNS&: DType
    fn dtype(&self) -> PyResult<PyDType> {
        match DataType::try_from(self.0.elem_type) {
            Ok(dt) => match dtype(dt) {
                Some(dt) => Ok(PyDType(dt)),
                None => Err(PyValueError::new_err(format!(
                    "unsupported 'value' data-type {dt:?}"
                ))),
            },
            type_ => Err(PyValueError::new_err(format!(
                "unsupported input type {type_:?}"
            ))),
        }
    }

    #[getter]
    /// The shape of the tensor.
    /// &RETURNS&: Tuple[Union[int,str,Any]]
    fn shape(&self, py: Python) -> PyResult<Py<PyTuple>> {
        let shape = PyList::empty_bound(py);
        if let Some(d) = &self.0.shape {
            for dim in d.dim.iter() {
                if let Some(value) = &dim.value {
                    match value {
                        Value::DimValue(v) => shape.append(*v)?,
                        Value::DimParam(s) => shape.append(s.clone())?,
                    };
                } else {
                    return Err(PyValueError::new_err("None value in shape"));
                }
            }
        }
        Ok(shape.to_tuple().into())
    }

    fn __repr__(&self, py: Python) -> String {
        match (self.shape(py), self.dtype()) {
            (Ok(shape), Ok(dtype)) => format!(
                "TensorDescriptor[shape: {:?}, dtype: {:?}]",
                shape.to_string(),
                dtype.__str__()
            ),
            (Err(_), Err(_)) => "TensorDescriptor[shape: unknown, dtype: unknown]".to_string(),
            (Err(_), Ok(dtype)) => format!(
                "TensorDescriptor[shape: unknown, dtype: {:?}]",
                dtype.__str__()
            ),
            (Ok(shape), Err(_)) => format!(
                "TensorDescriptor[shape: {:?}, dtype: unknown]",
                shape.to_string()
            ),
        }
    }

    fn __str__(&self, py: Python) -> String {
        self.__repr__(py)
    }
}

#[derive(Clone, Debug)]
#[pyclass(name = "ONNXModel")]
/// A wrapper around an ONNX model.
pub struct PyONNXModel(ModelProto);

fn extract_tensor_descriptions(
    value_infos: &[ValueInfoProto],
) -> HashMap<String, PyONNXTensorDescriptor> {
    let mut map = HashMap::new();
    for value_info in value_infos.iter() {
        let input_type = match &value_info.r#type {
            Some(input_type) => input_type,
            None => continue,
        };
        let input_type = match &input_type.value {
            Some(input_type) => input_type,
            None => continue,
        };

        let tensor_type: &ONNXTensor = match input_type {
            ONNXValue::TensorType(tt) => tt,
            _ => continue,
        };
        map.insert(
            value_info.name.to_string(),
            PyONNXTensorDescriptor(tensor_type.clone()),
        );
    }
    map
}

#[pymethods]
impl PyONNXModel {
    #[new]
    #[pyo3(text_signature = "(self, path:str)")]
    /// Load an ONNX model from the given path.
    fn new(path: String) -> PyResult<Self> {
        let model: ModelProto = candle_onnx::read_file(path).map_err(wrap_err)?;
        Ok(PyONNXModel(model))
    }

    #[getter]
    /// The version of the IR this model targets.
    /// &RETURNS&: int
    fn ir_version(&self) -> i64 {
        self.0.ir_version
    }

    #[getter]
    /// The producer of the model.  
    /// &RETURNS&: str      
    fn producer_name(&self) -> String {
        self.0.producer_name.clone()
    }

    #[getter]
    /// The version of the producer of the model.       
    /// &RETURNS&: str
    fn producer_version(&self) -> String {
        self.0.producer_version.clone()
    }

    #[getter]
    /// The domain of the operator set of the model.
    /// &RETURNS&: str
    fn domain(&self) -> String {
        self.0.domain.clone()
    }

    #[getter]
    /// The version of the model.
    /// &RETURNS&: int
    fn model_version(&self) -> i64 {
        self.0.model_version
    }

    #[getter]
    /// The doc string of the model.
    /// &RETURNS&: str
    fn doc_string(&self) -> String {
        self.0.doc_string.clone()
    }

    /// Get the weights of the model.
    /// &RETURNS&: Dict[str, Tensor]
    fn initializers(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut map = HashMap::new();
        if let Some(graph) = self.0.graph.as_ref() {
            for tensor_description in graph.initializer.iter() {
                let tensor = get_tensor(tensor_description, tensor_description.name.as_str())
                    .map_err(wrap_err)?;
                map.insert(tensor_description.name.to_string(), PyTensor(tensor));
            }
        }
        Ok(map)
    }

    #[getter]
    /// The inputs of the model.
    /// &RETURNS&: Optional[Dict[str, ONNXTensorDescription]]
    fn inputs(&self) -> Option<HashMap<String, PyONNXTensorDescriptor>> {
        if let Some(graph) = self.0.graph.as_ref() {
            return Some(extract_tensor_descriptions(&graph.input));
        }
        None
    }

    #[getter]
    /// The outputs of the model.
    /// &RETURNS&: Optional[Dict[str, ONNXTensorDescription]]
    fn outputs(&self) -> Option<HashMap<String, PyONNXTensorDescriptor>> {
        if let Some(graph) = self.0.graph.as_ref() {
            return Some(extract_tensor_descriptions(&graph.output));
        }
        None
    }

    #[pyo3(text_signature = "(self, inputs:Dict[str,Tensor])")]
    /// Run the model on the given inputs.
    /// &RETURNS&: Dict[str,Tensor]
    fn run(&self, inputs: HashMap<String, PyTensor>) -> PyResult<HashMap<String, PyTensor>> {
        let unwrapped_tensors = inputs.into_iter().map(|(k, v)| (k.clone(), v.0)).collect();

        let result = simple_eval(&self.0, unwrapped_tensors).map_err(wrap_err)?;

        Ok(result
            .into_iter()
            .map(|(k, v)| (k.clone(), PyTensor(v)))
            .collect())
    }
}
