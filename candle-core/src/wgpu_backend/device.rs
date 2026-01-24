
use rand::SeedableRng;
use tracing::instrument;
use wgpu_compute_layer::util::ToU64;

use crate::backend::{BackendDevice, BackendStorage};
use crate::wgpu_backend::MatmulAlgorithm;
use crate::{notImplemented, wrongType, DType, Layout};

use wgpu_compute_layer::cache::{
    BindGroupReference, BindgroupAlignment, BindgroupAlignmentLayout, BindgroupInputBase,
    BufferReferenceId,
};
use wgpu_compute_layer::queue_buffer::QueueBuffer;
use super::wgpu_functions::{self, unary::UnaryOperation};
use super::WgpuStorage;

static DEVICE_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[derive(Debug, Clone)]
pub struct WgpuDevice{
    inner_device : wgpu_compute_layer::WgpuDevice,
    device_id : u32,
    pub(crate) matmul_alg : std::sync::Arc<std::sync::Mutex<MatmulAlgorithm>>
}

impl WgpuDevice {
    #[instrument]
    pub(crate) async fn create(
        index: usize,
        configuration: crate::WgpuDeviceConfig,
    ) -> crate::Result<Self> {
        let device = wgpu_compute_layer::WgpuDevice::create(index, configuration.into()).await?;
        device.add_wgpu_shader_loader(candle_wgpu_kernels::DefaultWgpuShader::LOADER_INDEX, || {
            candle_wgpu_kernels::DefaultWgpuShader{}
        });
        device.add_wgpu_shader_loader(candle_wgpu_kernels::DefaultWgpuDynamicShader::LOADER_INDEX, || {
            candle_wgpu_kernels::DefaultWgpuDynamicShader::new()
        });
        device.set_extension(rand::rngs::StdRng::from_os_rng());
        Ok(WgpuDevice{inner_device: device, device_id: DEVICE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst), matmul_alg: std::sync::Arc::new(std::sync::Mutex::new(MatmulAlgorithm::MatmulX))})
    }

    pub fn inner_device(&self) -> &wgpu_compute_layer::WgpuDevice {
        &self.inner_device
    }

    pub fn add_wgpu_shader_loader<T: wgpu_compute_layer::shader_loader::ShaderLoader + 'static + Send + Sync>(
        &self,
        index: wgpu_compute_layer::shader_loader::LoaderIndex,
        shader_loader: impl Fn() -> T,
    ) {
        self.inner_device().add_wgpu_shader_loader(index, shader_loader);
    }

    pub fn simulate_command(
        &self,
        command: &wgpu_compute_layer::DebugPipelineRecording,
        dest_buffer: &WgpuStorage,
        input1_buffer: &WgpuStorage,
        input2_buffer: &WgpuStorage,
        input3_buffer: &WgpuStorage,
    ) {
        self.inner_device().simulate_command(command, &dest_buffer.0, &input1_buffer.0, &input2_buffer.0, &input3_buffer.0);
    }

    pub fn get_dtype(&self, dtype: crate::DType) -> crate::Result<candle_wgpu_kernels::DType> {
        Ok(self.inner_device().get_dtype(dtype.into())?)
    }

    pub fn get_queue<'a>(&'a self) -> QueueBuffer<'a> 
    {
        self.inner_device().get_queue()
    }

    pub async fn synchronize_async(&self) -> crate::Result<()> {
        Ok(self.inner_device().synchronize_async().await?)
    }

    pub fn is_dtype_available(&self, dtype: DType) -> bool {
        match dtype {
            DType::U32 => true,
            DType::F32 => true,
            DType::U8 => false,
            DType::I64 => self.inner_device().is_dtype_available(dtype.into()),
            DType::F64 => self.inner_device().is_dtype_available(dtype.into()),
            DType::F16 => self.inner_device().is_dtype_available(dtype.into()),
            DType::BF16 => false,
            DType::F8E4M3 => false,
            DType::I16 => false,
            DType::I32 => false,
            DType::F6E2M3 => false,
            DType::F6E3M2 => false,
            DType::F4 => false,
            DType::F8E8M0 => false,
        }
    }

    #[instrument(skip(self, size))]
    pub fn alloc_uninit_size<T: ToU64>(&self, dtype: crate::DType, size: T) -> WgpuStorage {
        let wgpu_storage = self.inner_device().alloc_uninit_size(dtype.into(), size);
        WgpuStorage(wgpu_storage, self.clone())
    }

    #[instrument(skip(self, data))]
    pub fn alloc_from_slice<T: bytemuck::Pod>(
        &self,
        dtype: crate::DType,
        data: &[T],
    ) -> crate::Result<WgpuStorage> {
        let data: &[u8] = bytemuck::cast_slice(data);
        self.alloc_from_bytes(dtype, data)
    }

    #[instrument(skip(self, data))]
    pub fn alloc_from_bytes(&self, dtype: crate::DType, data: &[u8]) -> crate::Result<WgpuStorage> {
        let wgpu_storage = self.inner_device().alloc_from_bytes(dtype.into(), data)?;
        Ok(WgpuStorage(wgpu_storage, self.clone()))
    }

    pub fn allocate_zeros(&self, size_in_bytes: u32) -> crate::Result<WgpuStorage> {
        self.zeros_impl(&((size_in_bytes / 4) as usize,).into(), DType::U32)
    }

    /**************** Virtual Bindgroups: ****************/
    pub fn create_bind_group_input0(
        &self,
        buffer_dest: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        let alignment = BindgroupAlignmentLayout::Bindgroup0(alignment);
        alignment.validate();
        BindGroupReference::new(buffer_dest, BindgroupInputBase::Bindgroup0(alignment))
    }

    pub fn create_bind_group_input1(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        self.create_bind_group_input1_with_alignment(
            buffer_dest,
            buffer_input1,
            BindgroupAlignmentLayout::Bindgroup1(alignment, alignment),
        )
    }

    pub fn create_bind_group_input1_with_alignment(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup1(buffer_input1, alignment),
        )
    }

    pub fn create_bind_group_input2(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        self.create_bind_group_input2_with_alignment(
            buffer_dest,
            buffer_input1,
            buffer_input2,
            BindgroupAlignmentLayout::Bindgroup2(alignment, alignment, alignment),
        )
    }

    pub fn create_bind_group_input2_with_alignment(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup2(buffer_input1, buffer_input2, alignment),
        )
    }

    pub fn create_bind_group_input3(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        buffer_input3: BufferReferenceId,
        alignment: BindgroupAlignment,
    ) -> BindGroupReference {
        self.create_bind_group_input3_with_alignment(
            buffer_dest,
            buffer_input1,
            buffer_input2,
            buffer_input3,
            BindgroupAlignmentLayout::Bindgroup3(alignment, alignment, alignment, alignment),
        )
    }

    pub fn create_bind_group_input3_with_alignment(
        &self,
        buffer_dest: BufferReferenceId,
        buffer_input1: BufferReferenceId,
        buffer_input2: BufferReferenceId,
        buffer_input3: BufferReferenceId,
        alignment: BindgroupAlignmentLayout,
    ) -> BindGroupReference {
        alignment.validate();
        BindGroupReference::new(
            buffer_dest,
            BindgroupInputBase::Bindgroup3(buffer_input1, buffer_input2, buffer_input3, alignment),
        )
    }
}

impl WgpuDevice {
    #[cfg(feature = "wgpu_debug")]
    pub fn start_recording_commands(&self) {
        self.inner_device().start_recording_commands();
    }

    #[cfg(feature = "wgpu_debug")]
    pub fn stop_recording_commands(&self, output_path: &str) -> crate::Result<()> {
        Ok(self.inner_device().stop_recording_commands(output_path)?)
    }
}

impl crate::backend::BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(_: usize) -> crate::Result<Self> {
        Err(crate::Error::Wgpu(
            "A WgpuDevice must be created using the asynchronous create method"
                .to_owned()
                .into(),
        ))
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Wgpu { gpu_id: 0 }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.device_id == other.device_id
    }

    fn zeros_impl(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        let buffer = self.alloc_uninit_size(dtype, shape.elem_count());
        if shape.elem_count() > 0 {
            wgpu_functions::queue_unary_inplace_op(
                self,
                buffer.buffer(),
                UnaryOperation::SetZero,
                0.0,
                0.0,
                dtype,
                &Layout::contiguous(shape),
            )?;
        }

        Ok(buffer)
    }

    unsafe fn alloc_uninit(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        if self.is_dtype_available(dtype) {
            Ok(self.alloc_uninit_size(dtype, shape.elem_count()))
        } else {
            wrongType!(alloc_uninit, dtype);
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> crate::Result<Self::Storage> {
        let data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * T::DTYPE.size_in_bytes(),
            )
        };
        let buffer = self.alloc_from_bytes(T::DTYPE, data)?;
        Ok(buffer)
    }

    fn storage_from_cpu_storage(
        &self,
        storage: &crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        match storage {
            crate::CpuStorage::F32(data) => self.alloc_from_slice(crate::DType::F32, data),
            crate::CpuStorage::U32(data) => self.alloc_from_slice(crate::DType::U32, data),
            crate::CpuStorage::F16(data) => self.alloc_from_slice(crate::DType::F16, data),
            crate::CpuStorage::F64(data) => self.alloc_from_slice(crate::DType::F64, data),
            crate::CpuStorage::I64(data) => self.alloc_from_slice(crate::DType::I64, data),
            _ => wrongType!(storage_from_cpu_storage, storage.dtype()),
        }
    }

    fn storage_from_cpu_storage_owned(
        &self,
        storage: crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        match storage {
            crate::CpuStorage::F32(data) => self.alloc_from_slice(crate::DType::F32, &data),
            crate::CpuStorage::U32(data) => self.alloc_from_slice(crate::DType::U32, &data),
            crate::CpuStorage::I64(data) => self.alloc_from_slice(crate::DType::I64, &data),
            crate::CpuStorage::F64(data) => self.alloc_from_slice(crate::DType::F64, &data),
            _ => wrongType!(storage_from_cpu_storage_owned, storage.dtype()),
        }
    }

    fn rand_uniform(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
        lo: f64,
        up: f64,
    ) -> crate::Result<Self::Storage> {
        let buffer = self.alloc_uninit_size(dtype, shape.elem_count());
        wgpu_functions::queue_unary_inplace_op(
            self,
            buffer.buffer(),
            UnaryOperation::RandUniform,
            lo as f32,
            up as f32,
            dtype,
            &Layout::contiguous(shape),
        )?;
        Ok(buffer)
    }

    fn rand_normal(
        &self,
        shape: &crate::Shape,
        dtype: crate::DType,
        mean: f64,
        std: f64,
    ) -> crate::Result<Self::Storage> {
        let buffer = self.alloc_uninit_size(dtype, shape.elem_count());
        wgpu_functions::queue_unary_inplace_op(
            self,
            buffer.buffer(),
            UnaryOperation::RandNormal,
            mean as f32,
            std as f32,
            dtype,
            &Layout::contiguous(shape),
        )?;
        Ok(buffer)
    }

    fn set_seed(&self, seed: u64) -> crate::Result<()> {
        self.inner_device().set_extension(rand::rngs::StdRng::seed_from_u64(seed));
        Ok(())
    }

    fn get_current_seed(&self) -> crate::Result<u64> {
        notImplemented!(get_current_seed)
    }

    fn synchronize(&self) -> crate::Result<()> {
        Ok(self.inner_device().synchronize()?)
    }
}
