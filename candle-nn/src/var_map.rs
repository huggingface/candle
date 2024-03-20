use candle::{DType, Device, Result, Shape, Tensor, Var};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A `VarMap` is a store that holds named variables. Variables can be retrieved from the stores
/// and new variables can be added by providing some initialization config in case they are
/// missing.
/// `VarMap` structures can be serialized in the safetensors format.
#[derive(Clone)]
pub struct VarMap {
    data: Arc<Mutex<HashMap<String, Var>>>,
}

impl VarMap {
    /// Create a new empty `VarMap`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let data = Arc::new(Mutex::new(HashMap::new()));
        Self { data }
    }

    /// Retrieve all the variables currently stored in the map.
    pub fn all_vars(&self) -> Vec<Var> {
        let tensor_data = self.data.lock().unwrap();
        #[allow(clippy::map_clone)]
        tensor_data.values().map(|c| c.clone()).collect::<Vec<_>>()
    }

    /// Save the map in the safetensors format.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let tensor_data = self.data.lock().unwrap();
        let data = tensor_data.iter().map(|(k, v)| (k, v.as_tensor()));
        safetensors::tensor::serialize_to_file(data, &None, path.as_ref())?;
        Ok(())
    }

    /// Load some values from a safetensors file and modify the existing variables to have these
    /// values.
    ///
    /// Note that values for variables that are currently not in the map are not kept.
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let data = unsafe { candle::safetensors::MmapedSafetensors::new(path)? };
        let mut tensor_data = self.data.lock().unwrap();
        for (name, var) in tensor_data.iter_mut() {
            let data = data.load(name, var.device())?;
            if let Err(err) = var.set(&data) {
                candle::bail!("error setting {name} using data from {path:?}: {err}",)
            }
        }
        Ok(())
    }

    /// Set a named variable to some value.
    pub fn set_one<K: AsRef<str>, V: AsRef<Tensor>>(&mut self, name: K, value: V) -> Result<()> {
        let tensor_data = self.data.lock().unwrap();
        let name = name.as_ref();
        match tensor_data.get(name) {
            None => candle::bail!("cannot find {name} in VarMap"),
            Some(var) => {
                if let Err(err) = var.set(value.as_ref()) {
                    candle::bail!("error setting {name}: {err}",)
                }
            }
        }
        Ok(())
    }

    /// Set some named variables to some values.
    ///
    /// If an error is returned, some of the variables might have already been set to their new
    /// values.
    pub fn set<I: Iterator<Item = (K, V)>, K: AsRef<str>, V: AsRef<Tensor>>(
        &mut self,
        iter: I,
    ) -> Result<()> {
        let tensor_data = self.data.lock().unwrap();
        for (name, value) in iter {
            let name = name.as_ref();
            match tensor_data.get(name) {
                None => candle::bail!("cannot find {name} in VarMap"),
                Some(var) => {
                    if let Err(err) = var.set(value.as_ref()) {
                        candle::bail!("error setting {name}: {err}",)
                    }
                }
            }
        }
        Ok(())
    }

    /// Retrieve or add a new variable.
    pub fn get<S: Into<Shape>>(
        &self,
        shape: S,
        path: &str,
        init: crate::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let mut tensor_data = self.data.lock().unwrap();
        if let Some(tensor) = tensor_data.get(path) {
            let tensor_shape = tensor.shape();
            if &shape != tensor_shape {
                candle::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
            }
            return Ok(tensor.as_tensor().clone());
        }
        let var = init.var(shape, dtype, device)?;
        let tensor = var.as_tensor().clone();
        tensor_data.insert(path.to_string(), var);
        Ok(tensor)
    }

    pub fn data(&self) -> &Mutex<HashMap<String, Var>> {
        &self.data
    }
}
