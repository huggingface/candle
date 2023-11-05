use pyo3::prelude::*;
use candle_onnx;
use candle_onnx::onnx::{ModelProto};
use crate::utils::wrap_err;


#[derive(Clone, Debug)]
#[pyclass(name = "ONNXModel")]
pub struct PyONNXModel(ModelProto);



#[pymethods]
impl PyONNXModel {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let model: ModelProto = candle_onnx::read_file(path).map_err(wrap_err)?;
        Ok(PyONNXModel(model))
    }

    #[getter]
    /// The version of the IR this model targets.
    /// &RETURNS&: int
    fn ir_version(&self)-> i64 {
       self.0.ir_version
    }

    #[getter]
    /// The producer of the model.  
    /// &RETURNS&: str      
    fn producer_name(&self)-> String {
       self.0.producer_name.clone()
    }

    #[getter]
    /// The version of the producer of the model.       
    /// &RETURNS&: str
    fn producer_version(&self)-> String {
       self.0.producer_version.clone()
    }

    #[getter]
    /// The domain of the operator set of the model.
    /// &RETURNS&: str
    fn domain(&self)-> String {
       self.0.domain.clone()
    }

    #[getter]
    /// The version of the model.
    /// &RETURNS&: int
    fn model_version(&self)-> i64 {
       self.0.model_version
    }   

    #[getter]
    /// The doc string of the model.
    /// &RETURNS&: str
    fn doc_string(&self)-> String {
       self.0.doc_string.clone()
    }
}