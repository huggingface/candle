use std::{collections::HashMap, path::Path};

use candle::{DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, VarBuilder};

use crate::{
    generic_error::GenericResult,
    opfs::get_rust_blob,
    safetensors::{SafeTensors, TensorView},
};

pub struct MmapedSafetensors {
    data: HashMap<String, TensorView>,
}

impl MmapedSafetensors {
    pub async fn new<P: AsRef<Path>>(p: P) -> GenericResult<Self> {
        let blob = get_rust_blob(p).await?;
        let data = SafeTensors::deserialize(blob).await?;
        let data = data.tensors().await?;
        let mut hashmap = HashMap::new();
        for (key, value) in data {
            hashmap.insert(key, value);
        }
        Ok(Self { data: hashmap })
    }

    pub async fn multi<P: AsRef<Path>>(paths: &[P]) -> GenericResult<Self> {
        let mut hashmap = HashMap::new();
        for p in paths.iter() {
            let blob = get_rust_blob(p).await?;
            let data = SafeTensors::deserialize(blob).await?;
            let data = data.tensors().await?;
            for (key, value) in data {
                hashmap.insert(key, value);
            }
        }
        Ok(Self { data: hashmap })
    }

    pub fn load(&self, name: &str, dev: &Device) -> GenericResult<Tensor> {
        let tensor_view = self.get(name)?;
        let dtype: candle::DType = match tensor_view.dtype() {
            safetensors::Dtype::U8 => candle::DType::U8,
            safetensors::Dtype::F16 => candle::DType::F16,
            safetensors::Dtype::BF16 => candle::DType::BF16,
            safetensors::Dtype::U32 => candle::DType::U32,
            safetensors::Dtype::F32 => candle::DType::F32,
            safetensors::Dtype::F64 => candle::DType::F64,
            safetensors::Dtype::I64 => candle::DType::I64,
            t => panic!("type {:?} not supported by candle", t),
        };

        Ok(Tensor::from_raw_buffer(
            tensor_view.data(),
            dtype,
            tensor_view.shape(),
            dev,
        )?)
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&String, &TensorView)> {
        self.data.iter()
    }

    pub fn get(&self, name: &str) -> GenericResult<&TensorView> {
        let data = self.data.get(name).ok_or_else(|| {
            candle::Error::CannotFindTensor {
                path: name.to_string(),
            }
            .bt()
        })?;
        Ok(data)
    }
}

impl SimpleBackend for MmapedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        let tensor = self
            .load(name, dev)
            .map_err(candle::Error::msg)?
            .to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

/// Initializes a `VarBuilder` from a binary builder in the safetensor format.
pub async fn var_builder_from_opfs_safetensors<P: AsRef<Path>>(
    p: P,
    dtype: DType,
    dev: &Device,
) -> GenericResult<VarBuilder> {
    let tensors = MmapedSafetensors::new(p).await?;

    Ok(VarBuilder::from_backend(
        Box::new(tensors),
        dtype,
        dev.clone(),
    ))
}
