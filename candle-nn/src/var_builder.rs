use candle::{safetensors::Load, DType, Device, Error, Result, Shape, Tensor};
use safetensors::slice::IndexOp;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::sync::Arc;

// TODO: Maybe we would want the storage to be generic, e.g. with Box<dyn> to avoid too many
// generics.
enum Tensors<'a> {
    SafeTensorWithRouting {
        routing: HashMap<String, usize>,
        safetensors: Vec<SafeTensors<'a>>,
    },
    Npz(candle::npy::NpzTensors),
    TensorMap(HashMap<String, Tensor>),
    Zeros,
}

struct TensorData<'a> {
    tensors: Tensors<'a>,
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
        let tensors = Tensors::SafeTensorWithRouting {
            routing,
            safetensors,
        };
        Self {
            tensors,
            device: device.clone(),
            dtype,
        }
    }

    fn zeros(dtype: DType, device: &Device) -> Self {
        Self {
            tensors: Tensors::Zeros,
            device: device.clone(),
            dtype,
        }
    }

    fn from_tensors(tensors: HashMap<String, Tensor>, dtype: DType, device: &Device) -> Self {
        Self {
            tensors: Tensors::TensorMap(tensors),
            device: device.clone(),
            dtype,
        }
    }

    fn from_npz<P: AsRef<std::path::Path>>(file: P, dtype: DType, device: &Device) -> Result<Self> {
        let npz = candle::npy::NpzTensors::new(file)?;
        Ok(Self {
            tensors: Tensors::Npz(npz),
            device: device.clone(),
            dtype,
        })
    }
}

#[derive(Clone)]
pub struct VarBuilder<'a> {
    data: Arc<TensorData<'a>>,
    pub path: Vec<String>,
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

    pub fn from_tensors(ts: HashMap<String, Tensor>, dtype: DType, device: &Device) -> Self {
        let data = TensorData::from_tensors(ts, dtype, device);
        Self {
            data: Arc::new(data),
            path: vec![],
        }
    }

    pub fn from_npz<P: AsRef<std::path::Path>>(
        file: P,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let data = TensorData::from_npz(file, dtype, device)?;
        Ok(Self {
            data: Arc::new(data),
            path: vec![],
        })
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
    pub fn get_sharded(
        &self,
        tensor_name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Tensor> {
        let data = self.data.as_ref();
        let path = if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        };
        let tensor = match &self.data.tensors {
            Tensors::SafeTensorWithRouting {
                routing,
                safetensors,
            } => {
                let index = routing.get(&path).ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: path.to_string(),
                    }
                    .bt()
                })?;

                let view = safetensors[*index].tensor(&path)?;
                let dtype = view.dtype();
                let mut shape = view.shape().to_vec();
                let size = shape[dim];
                let block_size = size / world_size;
                let start = rank * block_size;
                let stop = (rank + 1) * block_size;

                let iterator = if dim == 0 {
                    view.slice(start..stop).unwrap()
                } else if dim == 1 {
                    view.slice((.., start..stop)).unwrap()
                } else {
                    unimplemented!("Get sharded on dimensions != 0 or 1");
                };

                shape[dim] = block_size;

                Tensor::from_safetensors_slice(iterator, dtype, &shape, &data.device)?
            }
            _ => unimplemented!(),
        };
        Ok(tensor)
    }
    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> Result<Tensor> {
        let data = self.data.as_ref();
        let s: Shape = s.into();
        let path = if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        };
        let tensor = match &self.data.tensors {
            Tensors::Zeros => Tensor::zeros(&s, data.dtype, &data.device)?.contiguous()?,
            Tensors::TensorMap(ts) => ts
                .get(&path)
                .ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: path.to_string(),
                    }
                    .bt()
                })?
                .clone(),
            Tensors::Npz(npz) => npz.get(&path)?.ok_or_else(|| {
                Error::CannotFindTensor {
                    path: path.to_string(),
                }
                .bt()
            })?,
            Tensors::SafeTensorWithRouting {
                routing,
                safetensors,
            } => {
                let index = routing.get(&path).ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: path.to_string(),
                    }
                    .bt()
                })?;
                safetensors[*index]
                    .tensor(&path)?
                    .load(&data.device)?
                    .to_dtype(data.dtype)?
            }
        };
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }
}
