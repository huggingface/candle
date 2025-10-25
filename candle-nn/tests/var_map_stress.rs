//! A `VarMap` is a store that holds named variables.
//!
use candle::{DType, Device, Result, Shape, Tensor, Var};
use candle_nn::Init;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Storage backend trait for VarMap - allows different synchronization strategies
pub trait VarStorage: Send + Sync + Clone {
    fn new() -> Self;
    fn get_var(&self, name: &str) -> Option<Var>;
    fn all_vars(&self) -> Vec<Var>;
    fn insert_var(&self, name: String, var: Var);
    fn contains_key(&self, name: &str) -> bool;
    fn len(&self) -> usize;
    fn iter_for_save(&self) -> Vec<(String, Var)>;
    fn iter_for_load(&self) -> Vec<(String, Var)>;
    fn iter_mut_for_load(&self) -> Vec<(String, Var)>;
}

/// Original Mutex-based storage (for training)
#[derive(Clone)]
pub struct MutexStorage {
    data: Arc<Mutex<HashMap<String, Var>>>,
}

/// New RwLock-based storage (for concurrent inference)
#[derive(Clone)]
pub struct RwLockStorage {
    data: Arc<RwLock<HashMap<String, Var>>>,
}
// Implementation for existing Mutex storage - maintains exact original behavior
impl VarStorage for MutexStorage {
    fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn get_var(&self, name: &str) -> Option<Var> {
        let data = self.data.lock().unwrap();
        data.get(name).cloned()
    }

    fn all_vars(&self) -> Vec<Var> {
        let data = self.data.lock().unwrap();
        #[allow(clippy::map_clone)]
        data.values().map(|c| c.clone()).collect::<Vec<_>>()
    }

    fn insert_var(&self, name: String, var: Var) {
        let mut data = self.data.lock().unwrap();
        data.insert(name, var);
    }

    fn contains_key(&self, name: &str) -> bool {
        let data = self.data.lock().unwrap();
        data.contains_key(name)
    }

    fn len(&self) -> usize {
        let data = self.data.lock().unwrap();
        data.len()
    }

    fn iter_for_save(&self) -> Vec<(String, Var)> {
        let data = self.data.lock().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    fn iter_for_load(&self) -> Vec<(String, Var)> {
        let data = self.data.lock().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    fn iter_mut_for_load(&self) -> Vec<(String, Var)> {
        let data = self.data.lock().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

// Implementation for RwLock storage
impl VarStorage for RwLockStorage {
    fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn get_var(&self, name: &str) -> Option<Var> {
        let data = self.data.read().unwrap();
        data.get(name).cloned()
    }

    fn all_vars(&self) -> Vec<Var> {
        let data = self.data.read().unwrap();
        #[allow(clippy::map_clone)]
        data.values().map(|c| c.clone()).collect::<Vec<_>>()
    }

    fn insert_var(&self, name: String, var: Var) {
        let mut data = self.data.write().unwrap();
        data.insert(name, var);
    }

    fn contains_key(&self, name: &str) -> bool {
        let data = self.data.read().unwrap();
        data.contains_key(name)
    }

    fn len(&self) -> usize {
        let data = self.data.read().unwrap();
        data.len()
    }

    fn iter_for_save(&self) -> Vec<(String, Var)> {
        let data = self.data.read().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    fn iter_for_load(&self) -> Vec<(String, Var)> {
        let data = self.data.read().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    fn iter_mut_for_load(&self) -> Vec<(String, Var)> {
        let data = self.data.read().unwrap();
        data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

// Generic VarMap implementation
#[derive(Clone)]
pub struct VarMapGeneric<Storage: VarStorage> {
    storage: Storage,
}
// Type aliases for easy usage
/// A `VarMap` is a store that holds named variables. Variables can be retrieved from the stores
/// and new variables can be added by providing some initialization config in case they are
/// missing.
/// `VarMap` structures can be serialized in the safetensors format.
pub type VarMap = VarMapGeneric<MutexStorage>; // Original (for training)

/// Concurrent version of VarMap using RwLock for better read performance in inference scenarios
pub type ConcurrentVarMap = VarMapGeneric<RwLockStorage>;

impl<Storage: VarStorage> VarMapGeneric<Storage> {
    /// Create a new empty `VarMap`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
        }
    }

    /// Retrieve all the variables currently stored in the map.
    pub fn all_vars(&self) -> Vec<Var> {
        self.storage.all_vars()
    }

    /// Save the map in the safetensors format.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let data = self.storage.iter_for_save();
        let data = data.iter().map(|(k, v)| (k, v.as_tensor()));
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
        let vars = self.storage.iter_mut_for_load();

        for (name, var) in vars {
            let tensor_data = data.load(&name, var.device())?;
            if let Err(err) = var.set(&tensor_data) {
                candle::bail!("error setting {name} using data from {path:?}: {err}")
            }
        }
        Ok(())
    }

    /// Set a named variable to some value.
    pub fn set_one<K: AsRef<str>, V: AsRef<Tensor>>(&mut self, name: K, value: V) -> Result<()> {
        let name = name.as_ref();
        match self.storage.get_var(name) {
            None => candle::bail!("cannot find {name} in VarMap"),
            Some(var) => {
                if let Err(err) = var.set(value.as_ref()) {
                    candle::bail!("error setting {name}: {err}")
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
        for (name, value) in iter {
            let name = name.as_ref();
            match self.storage.get_var(name) {
                None => candle::bail!("cannot find {name} in VarMap"),
                Some(var) => {
                    if let Err(err) = var.set(value.as_ref()) {
                        candle::bail!("error setting {name}: {err}")
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
        if let Some(existing_var) = self.storage.get_var(path) {
            let tensor_shape = existing_var.shape();
            if &shape != tensor_shape {
                candle::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
            }
            return Ok(existing_var.as_tensor().clone());
        }
        let var = init.var(shape, dtype, device)?;
        let tensor = var.as_tensor().clone();
        self.storage.insert_var(path.to_string(), var);
        Ok(tensor)
    }

    /// Get a variable by name (method for compatibility).
    pub fn get_var(&self, name: &str) -> Option<Var> {
        self.storage.get_var(name)
    }

    /// Insert a new variable (method for compatibility).
    pub fn insert(&self, name: String, var: Var) {
        self.storage.insert_var(name, var);
    }

    /// Check if a variable exists (method for compatibility).
    pub fn contains_key(&self, name: &str) -> bool {
        self.storage.contains_key(name)
    }

    /// Convert to the other storage type (for migration)
    pub fn into_concurrent(self) -> ConcurrentVarMap
    where
        Storage: VarStorage,
    {
        let concurrent = ConcurrentVarMap::new();

        // Transfer all variables
        for (name, var) in self.storage.iter_for_save() {
            concurrent.insert(name, var);
        }

        concurrent
    }
}

impl VarMap {
    pub fn data(&self) -> &Arc<Mutex<HashMap<String, Var>>> {
        &self.storage.data
    }
}
impl ConcurrentVarMap {
    pub fn read_data(&self) -> std::sync::RwLockReadGuard<HashMap<String, Var>> {
        self.storage.data.read().unwrap()
    }
    pub fn write_data(&self) -> std::sync::RwLockWriteGuard<HashMap<String, Var>> {
        self.storage.data.write().unwrap()
    }

    pub fn get_vars_batch(&self, names: &[&str]) -> HashMap<String, Var> {
        let data = self.storage.data.read().unwrap();
        names
            .iter()
            .filter_map(|&name| data.get(name).map(|v| (name.to_string(), v.clone())))
            .collect()
    }

    pub fn data(&self) -> &Arc<RwLock<HashMap<String, Var>>> {
        &self.storage.data
    }
}
