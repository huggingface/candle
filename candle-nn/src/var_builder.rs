//! A `VarBuilder` for variable retrieval from models
//!
//! A `VarBuilder` is used to retrieve variables used by a model. These variables can either come
//! from a pre-trained checkpoint, e.g. using `VarBuilder::from_mmaped_safetensors`, or initialized
//! for training, e.g. using `VarBuilder::from_varmap`.
use crate::VarMap;
use candle::{safetensors::Load, DType, Error, Result, Shape, Tensor};
use candle::{BackendStorage, CpuDevice, CpuStorage, TryConvertStorage};
use safetensors::{slice::IndexOp, tensor::SafeTensors};
use std::collections::HashMap;
use std::sync::Arc;

type CpuTensor = Tensor<CpuStorage>;

/// A structure used to retrieve variables, these variables can either come from storage or be
/// generated via some form of initialization.
///
/// The way to retrieve variables is defined in the backend embedded in the `VarBuilder`.
pub struct VarBuilderArgs<'a, B, BS>
where
    B: Backend<BS>,
    BS: BackendStorage,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    data: Arc<TensorData<B, BS>>,
    path: Vec<String>,
    pub dtype: DType,
    _phantom: std::marker::PhantomData<&'a B>,
}

impl<B, BS> Clone for VarBuilderArgs<'_, B, BS>
where
    B: Backend<BS>,
    BS: BackendStorage,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: self.path.clone(),
            dtype: self.dtype,
            _phantom: self._phantom,
        }
    }
}

/// A simple `VarBuilder`, this is less generic than `VarBuilderArgs` but should cover most common
/// use cases.
#[allow(type_alias_bounds)]
pub type VarBuilder<'a, BS: BackendStorage> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>, BS>;

struct TensorData<B, BS>
where
    B: Backend<BS>,
    BS: BackendStorage,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    backend: B,
    pub device: BS::Device,
}

/// A trait that defines how tensor data is retrieved.
///
/// Typically this would use disk storage in some specific format, or random initialization.
/// Note that there is a specialized version of this trait (`SimpleBackend`) that can be used most
/// of the time. The main restriction is that it doesn't allow for specific args (besides
/// initialization hints).
pub trait Backend<B>
where
    Self: Send + Sync,
    B: BackendStorage,
    B::Device: TryConvertStorage<CpuStorage, B>,
{
    type Hints: Default;

    /// Retrieve a tensor with some target shape.
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &B::Device,
    ) -> Result<Tensor<B>>;

    fn contains_tensor(&self, name: &str) -> bool;
}

pub trait SimpleBackend: Send + Sync {
    /// Retrieve a tensor based on a target name and shape.
    fn get(&self, s: Shape, name: &str, h: crate::Init, dtype: DType) -> Result<CpuTensor>;

    fn contains_tensor(&self, name: &str) -> bool;
}

impl<B> Backend<B> for Box<dyn SimpleBackend + '_>
where
    B: BackendStorage,
    B::Device: TryConvertStorage<CpuStorage, B>,
{
    type Hints = crate::Init;
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &B::Device,
    ) -> Result<Tensor<B>> {
        self.as_ref().get(s, name, h, dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.as_ref().contains_tensor(name)
    }
}

impl<B, BS> VarBuilderArgs<'_, B, BS>
where
    B: Backend<BS>,
    BS: BackendStorage,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    pub fn new_with_args(backend: B, dtype: DType, dev: &BS::Device) -> Self {
        let data = TensorData {
            backend,
            device: dev.clone(),
        };
        Self {
            data: Arc::new(data),
            path: vec![],
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the prefix of the `VarBuilder`.
    pub fn prefix(&self) -> String {
        self.path.join(".")
    }

    /// Returns a new `VarBuilder` using the root path.
    pub fn root(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![],
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a new `VarBuilder` with the prefix set to `prefix`.
    pub fn set_prefix(&self, prefix: impl ToString) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![prefix.to_string()],
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Return a new `VarBuilder` adding `s` to the current prefix. This can be think of as `cd`
    /// into a directory.
    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Short alias for `push_prefix`.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        self.push_prefix(s)
    }

    /// The device used by default.
    pub fn device(&self) -> &BS::Device {
        &self.data.device
    }

    /// The dtype used by default.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Clone the VarBuilder tweaking its dtype
    pub fn to_dtype(&self, dtype: DType) -> Self {
        Self {
            data: self.data.clone(),
            path: self.path.clone(),
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    /// This returns true only if a tensor with the passed in name is available. E.g. when passed
    /// `a`, true is returned if `prefix.a` exists but false is returned if only `prefix.a.b`
    /// exists.
    pub fn contains_tensor(&self, tensor_name: &str) -> bool {
        let path = self.path(tensor_name);
        self.data.backend.contains_tensor(&path)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get_with_hints<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: B::Hints,
    ) -> Result<Tensor<BS>> {
        self.get_with_hints_dtype(s, name, hints, self.dtype)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Tensor<BS>> {
        self.get_with_hints(s, name, Default::default())
    }

    /// Retrieve the tensor associated with the given name & dtype at the current path.
    pub fn get_with_hints_dtype<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: B::Hints,
        dtype: DType,
    ) -> Result<Tensor<BS>> {
        let path = self.path(name);
        self.data
            .backend
            .get(s.into(), &path, hints, dtype, &self.data.device)
    }
}

struct Zeros;

impl SimpleBackend for Zeros {
    fn get(&self, s: Shape, _: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        Tensor::zeros(s, dtype, &CpuDevice)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

impl SimpleBackend for HashMap<String, CpuTensor> {
    fn get(&self, s: Shape, name: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let tensor = self
            .get(name)
            .ok_or_else(|| {
                Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?
            .clone();
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        tensor.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.contains_key(name)
    }
}

impl SimpleBackend for VarMap<CpuStorage> {
    fn get(&self, s: Shape, name: &str, h: crate::Init, dtype: DType) -> Result<CpuTensor> {
        VarMap::get(self, s, name, h, dtype, &CpuDevice)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.data().lock().unwrap().contains_key(name)
    }
}

#[allow(dead_code)]
pub struct SafeTensorWithRouting<'a> {
    routing: HashMap<String, usize>,
    safetensors: Vec<SafeTensors<'a>>,
}

impl SimpleBackend for SafeTensorWithRouting<'_> {
    fn get(&self, s: Shape, path: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let index = self.routing.get(path).ok_or_else(|| {
            Error::CannotFindTensor {
                path: path.to_string(),
            }
            .bt()
        })?;
        let tensor = self.safetensors[*index]
            .tensor(path)?
            .load(&CpuDevice)?
            .to_dtype(dtype)?;
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

    fn contains_tensor(&self, name: &str) -> bool {
        self.routing.contains_key(name)
    }
}

impl SimpleBackend for candle::npy::NpzTensors {
    fn get(&self, s: Shape, path: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let tensor = match self.get(path)? {
            None => Err(Error::CannotFindTensor {
                path: path.to_string(),
            }
            .bt())?,
            Some(tensor) => tensor,
        };
        let tensor = tensor.to_dtype(dtype)?;
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

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok_and(|v| v.is_some())
    }
}

impl SimpleBackend for candle::pickle::PthTensors {
    fn get(&self, s: Shape, path: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let tensor = match self.get(path)? {
            None => Err(Error::CannotFindTensor {
                path: path.to_string(),
            }
            .bt())?,
            Some(tensor) => tensor,
        };
        let tensor = tensor.to_dtype(dtype)?;
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

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok_and(|v| v.is_some())
    }
}

impl SimpleBackend for candle::safetensors::MmapedSafetensors {
    fn get(&self, s: Shape, name: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let tensor = self.load(name, &CpuDevice)?.to_dtype(dtype)?;
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

impl SimpleBackend for candle::safetensors::BufferedSafetensors {
    fn get(&self, s: Shape, name: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let tensor = self.load(name, &CpuDevice)?.to_dtype(dtype)?;
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

impl SimpleBackend for candle::safetensors::SliceSafetensors<'_> {
    fn get(&self, s: Shape, name: &str, _: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let tensor = self.load(name, &CpuDevice)?.to_dtype(dtype)?;
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

impl<'a, BS> VarBuilder<'a, BS>
where
    BS: BackendStorage + 'a,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    /// Initializes a `VarBuilder` using a custom backend.
    ///
    /// It is preferred to use one of the more specific constructors. This
    /// constructor is provided to allow downstream users to define their own
    /// backends.
    pub fn from_backend(
        backend: Box<dyn SimpleBackend + 'a>,
        dtype: DType,
        device: &BS::Device,
    ) -> Self {
        let data = TensorData {
            backend,
            device: device.clone(),
        };
        VarBuilder {
            data: Arc::new(data),
            path: vec![],
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Initializes a `VarBuilder` that uses zeros for any tensor.
    pub fn zeros(dtype: DType, device: &BS::Device) -> Self {
        Self::from_backend(Box::new(Zeros), dtype, device)
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a hashtable. An error is
    /// returned if no tensor is available under the requested path or on shape mismatches.
    pub fn from_tensors(ts: HashMap<String, Tensor<BS>>, dtype: DType, device: &BS::Device) -> Self
    where
        HashMap<String, Tensor<BS>>: SimpleBackend,
    {
        Self::from_backend(Box::new(ts), dtype, device)
    }

    /// Initializes a `VarBuilder` using a `VarMap`. The requested tensors are created and
    /// initialized on new paths, the same tensor is used if the same path is requested multiple
    /// times. This is commonly used when initializing a model before training.
    ///
    /// Note that it is possible to load the tensor values after model creation using the `load`
    /// method on `varmap`, this can be used to start model training from an existing checkpoint.
    pub fn from_varmap(varmap: &VarMap<BS>, dtype: DType, device: &BS::Device) -> Self
    where
        VarMap<BS>: SimpleBackend,
    {
        Self::from_backend(Box::new(varmap.clone()), dtype, device)
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a collection of safetensors
    /// files.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn from_mmaped_safetensors<P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        device: &BS::Device,
    ) -> Result<Self> {
        let tensors = candle::safetensors::MmapedSafetensors::multi(paths)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, device))
    }

    /// Initializes a `VarBuilder` from a binary buffer in the safetensor format.
    pub fn from_buffered_safetensors(
        data: Vec<u8>,
        dtype: DType,
        device: &BS::Device,
    ) -> Result<Self> {
        let tensors = candle::safetensors::BufferedSafetensors::new(data)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, device))
    }

    /// Initializes a `VarBuilder` from a binary slice in the safetensor format.
    pub fn from_slice_safetensors(
        data: &'a [u8],
        dtype: DType,
        device: &BS::Device,
    ) -> Result<Self> {
        let tensors = candle::safetensors::SliceSafetensors::new(data)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, device))
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a numpy npz file.
    pub fn from_npz<P: AsRef<std::path::Path>>(
        p: P,
        dtype: DType,
        device: &BS::Device,
    ) -> Result<Self> {
        let npz = candle::npy::NpzTensors::new(p)?;
        Ok(Self::from_backend(Box::new(npz), dtype, device))
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a pytorch pth file.
    pub fn from_pth<P: AsRef<std::path::Path>>(
        p: P,
        dtype: DType,
        device: &BS::Device,
    ) -> Result<Self> {
        let pth = candle::pickle::PthTensors::new(p, None)?;
        Ok(Self::from_backend(Box::new(pth), dtype, device))
    }
    /// Initializes a `VarBuilder` that retrieves tensors stored in a pytorch pth file.
    /// similar to [`from_pth`] but requires a `state_key`.
    pub fn from_pth_with_state<P: AsRef<std::path::Path>>(
        p: P,
        dtype: DType,
        state_key: &str,
        device: &BS::Device,
    ) -> Result<Self> {
        let pth = candle::pickle::PthTensors::new(p, Some(state_key))?;
        Ok(Self::from_backend(Box::new(pth), dtype, device))
    }
    /// Gets a VarBuilder that applies some renaming function on tensor it gets queried for before
    /// passing the new names to the inner VarBuilder.
    ///
    /// ```rust
    /// use candle::{DType, CpuDevice, CpuStorage};
    /// type Tensor = candle::Tensor<CpuStorage>;
    ///
    /// let a = Tensor::arange(0f32, 6f32, &CpuDevice)?.reshape((2, 3))?;
    /// let tensors: std::collections::HashMap<_, _> = [
    ///     ("foo".to_string(), a),
    /// ]
    /// .into_iter()
    /// .collect();
    /// let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &CpuDevice);
    /// assert!(vb.contains_tensor("foo"));
    /// assert!(vb.get((2, 3), "foo").is_ok());
    /// assert!(!vb.contains_tensor("bar"));
    /// let vb = vb.rename_f(|f: &str| if f == "bar" { "foo".to_string() } else { f.to_string() });
    /// assert!(vb.contains_tensor("bar"));
    /// assert!(vb.contains_tensor("foo"));
    /// assert!(vb.get((2, 3), "bar").is_ok());
    /// assert!(vb.get((2, 3), "foo").is_ok());
    /// assert!(!vb.contains_tensor("baz"));
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn rename_f<F: Fn(&str) -> String + Sync + Send + 'static>(self, f: F) -> Self
    where
        CpuDevice: TryConvertStorage<BS, CpuStorage>,
    {
        let f: Box<dyn Fn(&str) -> String + Sync + Send + 'static> = Box::new(f);
        self.rename(f)
    }

    pub fn rename<R: Renamer + Send + Sync + 'a>(self, renamer: R) -> Self
    where
        CpuDevice: TryConvertStorage<BS, CpuStorage>,
    {
        let dtype = self.dtype();
        let device: BS::Device = self.device().clone();
        let path = self.path.clone();
        let backend = Rename::new(self, renamer);
        let backend: Box<dyn SimpleBackend + 'a> = Box::new(backend);
        let data = TensorData { backend, device };
        Self {
            data: Arc::new(data),
            dtype,
            path,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct ShardedSafeTensors(candle::safetensors::MmapedSafetensors);

#[allow(type_alias_bounds)]
pub type ShardedVarBuilder<'a, BS: BackendStorage> = VarBuilderArgs<'a, ShardedSafeTensors, BS>;

impl ShardedSafeTensors {
    /// Initializes a `VarBuilder` that retrieves tensors stored in a collection of safetensors
    /// files and make them usable in a sharded way.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn var_builder<BS, P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        dev: &BS::Device,
    ) -> Result<ShardedVarBuilder<'static, BS>>
    where
        BS: BackendStorage,
        BS::Device: TryConvertStorage<BS, CpuStorage>,
    {
        let tensors = candle::safetensors::MmapedSafetensors::multi(paths)?;
        let backend = ShardedSafeTensors(tensors);
        Ok(VarBuilderArgs::new_with_args(backend, dtype, dev))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Shard {
    pub dim: usize,
    pub rank: usize,
    pub world_size: usize,
}

impl Default for Shard {
    fn default() -> Self {
        Self {
            dim: 0,
            rank: 0,
            world_size: 1,
        }
    }
}

/// Get part of a tensor, typically used to do Tensor Parallelism sharding.
///
/// If the tensor is of size (1024, 1024).
///
/// `dim` corresponds to the dimension to slice into
/// `rank` is the rank of the current process
/// `world_size` is the total number of ranks in the process group
///
/// `get_sharded("tensor", 0, 0, 2)` means `tensor.i((..512))`
/// `get_sharded("tensor", 0, 1, 2)` means `tensor.i((512..))`
/// `get_sharded("tensor", 1, 0, 2)` means `tensor.i((.., ..512))`
impl<B> Backend<B> for ShardedSafeTensors
where
    B: BackendStorage,
    B::Device: TryConvertStorage<CpuStorage, B>,
{
    type Hints = Shard;

    fn get(
        &self,
        target_shape: Shape, // The size is only checked when the world size is 1.
        path: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &B::Device,
    ) -> Result<Tensor<B>> {
        if h.world_size == 1 {
            // There is no sharding to be applied here so we use the default backend to speed
            // things up.
            return SimpleBackend::get(&self.0, target_shape, path, Default::default(), dtype)?
                .to_device(dev);
        }

        let Shard {
            dim,
            rank,
            world_size,
        } = h;
        let view = self.0.get(path)?;
        let view_dtype = view.dtype();
        let mut shape = view.shape().to_vec();
        let size = shape[dim];

        if size % world_size != 0 {
            return Err(Error::ShapeMismatchSplit {
                shape: shape.into(),
                dim,
                n_parts: world_size,
            });
        }
        let block_size = size / world_size;
        let start = rank * block_size;
        let stop = (rank + 1) * block_size;

        // Everything is expressed in tensor dimension
        // bytes offsets is handled automatically for safetensors.

        let iterator = if dim == 0 {
            view.slice(start..stop).map_err(|_| {
                Error::Msg(format!(
                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                ))
            })?
        } else if dim == 1 {
            view.slice((.., start..stop)).map_err(|_| {
                Error::Msg(format!(
                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                ))
            })?
        } else {
            candle::bail!("Get sharded on dimensions != 0 or 1")
        };

        shape[dim] = block_size;

        let view_dtype: DType = view_dtype.try_into()?;
        let raw: Vec<u8> = iterator.into_iter().flatten().cloned().collect();
        Tensor::from_raw_buffer(&raw, view_dtype, &shape, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.get(name).is_ok()
    }
}

/// This traits specifies a way to rename the queried names into names that are stored in an inner
/// VarBuilder.
pub trait Renamer {
    /// This is applied to the name obtained by a name call and the resulting name is passed to the
    /// inner VarBuilder.
    fn rename(&self, v: &str) -> std::borrow::Cow<'_, str>;
}

pub struct Rename<'a, R, BS>
where
    R: Renamer,
    BS: BackendStorage,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    inner: VarBuilder<'a, BS>,
    renamer: R,
}

impl<R, BS> SimpleBackend for Rename<'_, R, BS>
where
    R: Renamer + Sync + Send,
    BS: BackendStorage,
    CpuDevice: TryConvertStorage<BS, CpuStorage>,
{
    fn get(&self, s: Shape, name: &str, h: crate::Init, dtype: DType) -> Result<CpuTensor> {
        let name = self.renamer.rename(name);
        Ok(self
            .inner
            .get_with_hints_dtype(s, &name, h, dtype)?
            .to_device(&CpuDevice)?)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let name = self.renamer.rename(name);
        self.inner.contains_tensor(&name)
    }
}

impl<R, BS> Backend<BS> for Rename<'_, R, BS>
where
    R: Renamer + Sync + Send,
    BS: BackendStorage,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    type Hints = crate::Init;

    fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dtype: DType,
        _: &BS::Device,
    ) -> Result<Tensor<BS>> {
        let name = self.renamer.rename(name);
        Ok(self.inner.get_with_hints_dtype(s, &name, h, dtype)?)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let name = self.renamer.rename(name);
        self.inner.contains_tensor(&name)
    }
}

impl<'a, R, BS> Rename<'a, R, BS>
where
    R: Renamer,
    BS: BackendStorage,
    BS::Device: TryConvertStorage<CpuStorage, BS>,
{
    pub fn new(inner: VarBuilder<'a, BS>, renamer: R) -> Self {
        Self { inner, renamer }
    }
}

impl Renamer for Box<dyn Fn(&str) -> String + Sync + Send> {
    fn rename(&self, v: &str) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Owned(self(v))
    }
}
