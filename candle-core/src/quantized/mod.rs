//! Code for GGML and GGUF files
use crate::{
    BackendDevice, BackendStorage, Context, CpuDevice, CpuStorage, CustomOp1, DType, Device,
    MetalStorage, Result, Shape, Storage, Tensor,
};
use k_quants::*;
use std::borrow::Cow;
use std::fmt::Debug;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod k_quants;

#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use cuda::QCudaStorage;
pub use k_quants::GgmlType;
pub use metal::QMetalStorage;

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub type DefaultQStorage = QCpuStorage;
#[cfg(feature = "cuda")]
pub type DefaultQStorage = cuda::QCudaStorage;
#[cfg(feature = "metal")]
pub type DefaultQStorage = metal::QMetalStorage;

pub struct QTensor<B = DefaultQStorage>
where
    B: QuantizedBackend,
{
    pub storage: B,
    pub shape: Shape,
}

#[derive(Debug)]
pub struct QCpuStorage(Box<dyn QuantizedType>);

impl QCpuStorage {
    pub fn new(data: Box<dyn QuantizedType>) -> Self {
        QCpuStorage(data)
    }
}

#[derive(Debug)]
pub enum QStorage {
    Cpu(QCpuStorage),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl From<QCpuStorage> for QStorage {
    fn from(storage: QCpuStorage) -> Self {
        QStorage::Cpu(storage)
    }
}

impl From<metal::QMetalStorage> for QStorage {
    fn from(storage: metal::QMetalStorage) -> Self {
        QStorage::Metal(storage)
    }
}

impl From<cuda::QCudaStorage> for QStorage {
    fn from(storage: cuda::QCudaStorage) -> Self {
        QStorage::Cuda(storage)
    }
}

pub trait QuantizedDevice<B: QuantizedBackend> {
    type Storage: BackendStorage;

    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<B>;

    fn load_quantized<T: GgmlType + Send + Sync + Debug + 'static>(&self, _: &[T]) -> Result<B>;
}

impl QuantizedDevice<QCpuStorage> for CpuDevice {
    type Storage = CpuStorage;

    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QCpuStorage> {
        let storage = dtype.cpu_zeros(elem_count);
        Ok(QCpuStorage(storage))
    }

    fn load_quantized<T: GgmlType + Send + Sync + Debug + 'static>(
        &self,
        data: &[T],
    ) -> Result<QCpuStorage> {
        Ok(QCpuStorage(Box::new(data.to_vec())))
    }
}

impl QuantizedDevice<QStorage> for Device {
    type Storage = Storage;

    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(QCpuStorage(storage)))
            }
            Device::Metal(metal) => {
                let storage = metal.qzeros(elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda.qzeros(elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
        }
    }

    fn load_quantized<T: GgmlType + Send + Sync + Debug + 'static>(
        &self,
        data: &[T],
    ) -> Result<QStorage> {
        match self {
            Device::Cpu => Ok(QStorage::Cpu(CpuDevice {}.load_quantized(data)?)),
            Device::Metal(device) => Ok(QStorage::Metal(device.load_quantized(data)?)),
            Device::Cuda(device) => Ok(QStorage::Cuda(device.load_quantized(data)?)),
        }
    }
}

pub trait QuantizedBackend: Sized + Send {
    type Storage: BackendStorage<Device = Self::Device>;
    type Device: QuantizedDevice<Self> + BackendDevice<Self::Storage>;

    fn block_size(&self) -> usize;
    fn dtype(&self) -> GgmlDType;
    fn storage_size_in_bytes(&self) -> usize;
    fn quantize(&mut self, src: &Self::Storage) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<Self::Storage>;
    fn data(&self) -> Result<Cow<'_, [u8]>>;

    fn device(&self) -> impl AsRef<Self::Device>;
}

impl QuantizedBackend for QCpuStorage {
    type Storage = CpuStorage;
    type Device = CpuDevice;

    fn block_size(&self) -> usize {
        self.0.block_size()
    }

    fn dtype(&self) -> GgmlDType {
        self.0.dtype()
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.0.storage_size_in_bytes()
    }

    fn quantize(&mut self, src: &Self::Storage) -> Result<()> {
        self.0.from_float(src.as_slice::<f32>()?)?;
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Self::Storage> {
        self.0.dequantize(elem_count)
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        let data_ptr = self.0.as_ptr();
        let size_in_bytes = self.0.storage_size_in_bytes();
        let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
        Ok(Cow::from(data))
    }

    fn device(&self) -> impl AsRef<Self::Device> {
        &CpuDevice {}
    }
}

impl QuantizedBackend for QStorage {
    type Storage = Storage;
    type Device = Device;

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.0.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> impl AsRef<Device> {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().as_ref().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().as_ref().clone()),
        }
    }

    fn storage_size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => storage.quantize(src)?,
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid dequantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Self::Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.0.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Metal(_) | QStorage::Cuda(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
        }
    }
    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync + std::fmt::Debug {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]) -> Result<()>;
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync + std::fmt::Debug> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) -> Result<()> {
        T::from_float(xs, self)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys)?;
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl<B: QuantizedBackend> std::fmt::Debug for QTensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if dims[dims.len() - 1] % block_size != 0 {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl<QB: QuantizedBackend> QTensor<QB> {
    pub fn new<S: Into<Shape>>(storage: QB, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self { storage, shape })
    }

    pub fn quantize(src: &Tensor<QB::Storage>, dtype: GgmlDType) -> Result<Self>
    where
        QB::Storage: BackendStorage<Device = QB::Device>,
    {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if elem_count % block_size != 0 {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> QB::Device {
        self.storage.device().as_ref().clone()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    // TODO: fix qtensor.dequantize impl
    pub fn dequantize(&self, _: &QB::Device) -> Result<Tensor<QB::Storage>>
    where
        QB::Storage: BackendStorage<Device = QB::Device>,
    {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        //let storage: QB::Storage = device.convert(storage)?;
        let none = crate::op::BackpropOp::none();
        Ok(crate::tensor::from_storage(
            storage,
            self.shape.clone(),
            none,
            false,
        ))
    }

    pub fn dequantize_f16(&self, _: &QB::Device) -> Result<Tensor<QB::Storage>> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136

        let s = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();

        crate::tensor::from_storage(s, self.shape.clone(), none, false).to_dtype(crate::DType::F16)

        /*
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                Ok(
                    crate::tensor::from_storage(s.into(), self.shape.clone(), none, false)
                        .to_device(device)?,
                )
            }
        */
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.storage_size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul<QB>
where
    QB: QuantizedBackend,
{
    QTensor(std::sync::Arc<QTensor<QB>>),
    Tensor(Tensor<QB::Storage>),
    TensorF16(Tensor<QB::Storage>),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl<B: QuantizedBackend> QMatMul<B> {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor<B>>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            // TODO: fix qtensor.dequantize impl
            let storage = qtensor.storage.dequantize(qtensor.shape.elem_count())?;
            let tensor = Tensor::from_storage(
                storage,
                qtensor.shape.clone(),
                crate::op::BackpropOp::none(),
                false,
            );
            Self::Tensor(tensor)
            //let device: B::Device = qtensor.device();
            //let tensor: Tensor<B::Storage> = qtensor.dequantize(&device)?;
            //Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor<B>) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    pub fn dequantize_f16(&self) -> Result<Tensor<B::Storage>> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor<B::Storage>) -> Result<Tensor<B::Storage>> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }
}

impl crate::CustomOp1<CpuStorage> for QTensor<QCpuStorage> {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().context("empty dst_shape")?;
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);

        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];
        self.storage
            .0
            .matmul_t((dst_shape.elem_count() / n, k, n), slice, &mut dst_storage)?;
        Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
    }

    fn metal_fwd(
        &self,
        _: &crate::MetalStorage,
        _: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        crate::bail!("Invalid storage")
    }

    fn cuda_fwd(
        &self,
        _: &crate::CudaStorage,
        _: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        crate::bail!("Invalid storage")
    }
}

impl crate::CustomOp1<MetalStorage> for QTensor<QMetalStorage> {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        _: &crate::CpuStorage,
        _: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        crate::bail!("Invalid storage")
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        self.storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        _: &crate::CudaStorage,
        _: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        crate::bail!("Invalid storage")
    }
}

impl<QB: QuantizedBackend> crate::Module<QB::Storage> for QMatMul<QB>
where
    QTensor<QB>: CustomOp1<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}
