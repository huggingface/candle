use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

struct SafeTensorWithRouting<'a> {
    routing: HashMap<String, usize>,
    safetensors: Vec<SafeTensors<'a>>,
}

struct TensorData<'a> {
    // TODO: Make this part generic, probably via some Box<dyn> to avoid too much generics.
    safetensors: Option<SafeTensorWithRouting<'a>>,
    pub dtype: DType,
    pub device: Device,
}

impl<'a> TensorData<'a> {
    fn from_safetensors(safetensors: Vec<SafeTensors<'a>>, dtype: DType, device: &Device) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }
        let safetensors = SafeTensorWithRouting {
            routing,
            safetensors,
        };
        Self {
            safetensors: Some(safetensors),
            device: device.clone(),
            dtype,
        }
    }

    fn zeros(dtype: DType, device: &Device) -> Self {
        Self {
            safetensors: None,
            device: device.clone(),
            dtype,
        }
    }
}

#[derive(Clone)]
pub struct VarBuilder<'a> {
    data: Arc<TensorData<'a>>,
    path: Vec<String>,
}

impl<'a> VarBuilder<'a> {
    /// Create a `VarBuilder` accessing data frome the safetensors storage. The initial path is
    /// set to the root path and sub-paths can be created via the `push_prefix` method.
    pub fn from_safetensors(st: Vec<SafeTensors<'a>>, dtype: DType, device: &Device) -> Self {
        let data = TensorData::from_safetensors(st, dtype, device);
        Self {
            data: Arc::new(data),
            path: vec![],
        }
    }

    pub fn zeros(dtype: DType, device: &Device) -> Self {
        let data = TensorData::zeros(dtype, device);
        Self {
            data: Arc::new(data),
            path: vec![],
        }
    }

    pub fn push_prefix(&self, s: &str) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
        }
    }

    /// Short alias for `push_prefix`.
    pub fn pp(&self, s: &str) -> Self {
        self.push_prefix(s)
    }

    pub fn device(&self) -> &Device {
        &self.data.device
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype
    }
}

impl<'a> VarBuilder<'a> {
    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> candle::Result<Tensor> {
        let data = self.data.as_ref();
        let s: Shape = s.into();
        match &self.data.safetensors {
            None => Tensor::zeros(s, data.dtype, &data.device),
            Some(SafeTensorWithRouting {
                routing,
                safetensors,
            }) => {
                let path = [&self.path.join("."), tensor_name].join(".");
                // Unwrap or 0 just to let the proper error flow.
                let index = routing.get(&path).unwrap_or(&0);
                let tensor = safetensors[*index]
                    .tensor(&path, &data.device)?
                    .to_dtype(data.dtype)?;
                if *tensor.shape() != s {
                    Err(candle::Error::UnexpectedShape {
                        msg: format!("shape mismatch for {path}"),
                        expected: s,
                        got: tensor.shape().clone(),
                    })?
                }
                Ok(tensor)
            }
        }
    }
}
