use crate::{ConstantValues, MetalKernelError};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCounterSet,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunction, MTLFunctionConstantValues,
    MTLLibrary, MTLResource, MTLResourceUsage, MTLSize,
};
use std::{collections::HashMap, ffi::c_void, ptr, sync::Arc};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type CounterSet = Retained<ProtocolObject<dyn MTLCounterSet>>;

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

#[derive(Clone, Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl Device {
    pub fn as_ref(&self) -> &ProtocolObject<dyn MTLDevice> {
        &*self.raw
    }

    pub fn registry_id(&self) -> u64 {
        self.as_ref().registryID()
    }

    pub fn all() -> Vec<Self> {
        MTLCreateSystemDefaultDevice()
            .into_iter()
            .map(|raw| Device { raw })
            .collect()
    }

    pub fn system_default() -> Option<Self> {
        MTLCreateSystemDefaultDevice().map(|raw| Device { raw })
    }

    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        self.as_ref()
            .newBufferWithLength_options(length, options)
            .map(|b| Buffer {
                raw: Retained::into_raw(b),
            })
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Buffer".to_string(),
            ))
    }

    pub fn new_buffer_with_data(
        &self,
        pointer: *const c_void,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        let pointer = ptr::NonNull::new(pointer as *mut c_void).unwrap();
        unsafe {
            self.as_ref()
                .newBufferWithBytes_length_options(pointer, length, options)
                .map(|b| Buffer {
                    raw: Retained::into_raw(b),
                })
                .ok_or(MetalKernelError::FailedToCreateResource(
                    "Buffer".to_string(),
                ))
        }
    }

    pub fn new_library_with_source(
        &self,
        source: &str,
        options: Option<&MTLCompileOptions>,
    ) -> Result<Library, MetalKernelError> {
        let raw = self
            .as_ref()
            .newLibraryWithSource_options_error(&NSString::from_str(source), options)
            .unwrap();

        Ok(Library { raw })
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let raw = self
            .as_ref()
            .newComputePipelineStateWithFunction_error(&function.raw)
            .unwrap();
        Ok(ComputePipeline { raw })
    }

    pub fn new_command_queue(&self) -> Result<CommandQueue, MetalKernelError> {
        let raw = self.as_ref().newCommandQueue().unwrap();
        Ok(raw)
    }
}

#[derive(Clone, Debug)]
pub struct Library {
    raw: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl Library {
    pub fn get_function(
        &self,
        name: &str,
        constant_values: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let function = match constant_values {
            Some(constant_values) => self
                .raw
                .newFunctionWithName_constantValues_error(
                    &NSString::from_str(name),
                    &constant_values.function_constant_values().raw,
                )
                .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?,
            None => self
                .raw
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(MetalKernelError::LoadFunctionError("".to_string()))?,
        };

        Ok(Function { raw: function })
    }
}

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &*self.raw
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        self.as_ref()
            .computeCommandEncoder()
            .map(|raw| ComputeCommandEncoder { raw })
            .unwrap()
    }

    pub fn blit_command_encoder(&self) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder { raw })
            .unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(&label)))
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn wait_until_completed(&self) {
        unsafe { self.raw.waitUntilCompleted() }
    }
}

pub struct Function {
    raw: Retained<ProtocolObject<dyn MTLFunction>>,
}

pub struct FunctionConstantValues {
    raw: Retained<MTLFunctionConstantValues>,
}

impl FunctionConstantValues {
    pub fn new() -> FunctionConstantValues {
        FunctionConstantValues {
            raw: MTLFunctionConstantValues::new(),
        }
    }

    pub fn set_constant_value_at_index<T>(&self, value: T, dtype: MTLDataType, index: usize) {
        let value = ptr::NonNull::new(&value as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setConstantValue_type_atIndex(value, dtype, index) }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: *mut ProtocolObject<dyn objc2_metal::MTLBuffer>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            raw: ptr::null_mut(),
        }
    }
}

impl Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        unsafe { &*self.raw }
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    pub fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        self.as_ref().contents().as_ptr() as *mut u8
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }
}

impl<'a> Into<&'a MetalResource> for &'a Buffer {
    fn into(self) -> &'a MetalResource {
        &ProtocolObject::from_ref(self.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}
impl ComputePipeline {
    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}
impl ComputeCommandEncoder {
    pub fn set_threadgroup_memory_length(&self, length: usize, index: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| &*b.raw), offset, index)
        }
    }
    pub fn set_bytes(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
    }
    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(&pipeline.raw);
    }

    pub fn use_resource<'a>(
        &self,
        resource: impl Into<&'a MetalResource>,
        resource_usage: MTLResourceUsage,
    ) {
        self.raw.useResource_usage(resource.into(), resource_usage)
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding()
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        use MTLComputeCommandEncoder as _;
        self.raw.setComputePipelineState(&pipeline.raw);
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

pub struct BlitCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}

impl BlitCommandEncoder {
    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding()
    }

    pub fn set_label(&self, label: &str) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.setLabel(Some(&NSString::from_str(&label)))
    }

    pub fn copy_from_buffer(
        &self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                )
        }
    }

    pub fn fill_buffer(&self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        )
    }
}

pub type BufferMap = HashMap<(usize, MTLResourceOptions), Vec<Arc<Buffer>>>;
pub struct Commands {
    device: Device,
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// One command buffer at a time.
    /// The scheduler works by allowing multiple
    /// [ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// on a single command buffer. Using a single command buffer would be fastest on the GPU but
    /// prevents overlapping of CPU and GPU commands (because command buffer needs to be committed
    /// to start to work).
    /// Despite what the documentation says, command buffers are NOT ordered. They are ordered
    /// for their START time, but there's no guarantee that command buffer1 will finish before
    /// command buffer2 starts (or there are metal bugs there)
    command_buffer: CommandBuffer,
    /// Keeps track of the current amount of compute command encoders on the current
    /// command buffer
    /// Arc, RwLock because of the interior mutability.
    command_buffer_index: usize,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    //capture: Option<Retained<objc2_metal::MTLCaptureManager>>,
    //timestamp_counter_set: Option<CounterSet>,
}

// needed for `capture` and `timestamp_counter_set`
unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

pub fn create_command_buffer(
    command_queue: &CommandQueue,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer { raw })
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let raw_device = MTLCreateSystemDefaultDevice().ok_or(
            MetalKernelError::LoadLibraryError(String::from("Could not get default device")),
        )?;
        let device = Device { raw: raw_device };
        let command_buffer = create_command_buffer(&command_queue)?;
        command_buffer.enqueue();
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val.parse().unwrap_or(50),
            _ => 50,
        };
        Ok(Self {
            device,
            command_queue,
            command_buffer,
            command_buffer_index: 0,
            compute_per_buffer,
        })
    }

    pub fn command_buffer(&mut self) -> Result<(bool, CommandBuffer), MetalKernelError> {
        let mut command_buffer = self.command_buffer.to_owned();
        let mut flushed = false;
        if self.command_buffer_index > self.compute_per_buffer {
            self.command_buffer.commit();
            command_buffer = create_command_buffer(&self.command_queue)?;
            self.command_buffer = command_buffer.clone();
            self.command_buffer_index = 0;
            flushed = true;
        }
        self.command_buffer_index += 1;
        Ok((flushed, command_buffer))
    }

    pub fn wait_until_completed(&mut self) -> Result<(), MetalKernelError> {
        match self.command_buffer.status() {
            objc2_metal::MTLCommandBufferStatus::Committed
            | objc2_metal::MTLCommandBufferStatus::Scheduled
            | objc2_metal::MTLCommandBufferStatus::Completed => {
                panic!("Already committed");
            }
            _ => {}
        }
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
        self.command_buffer = create_command_buffer(&self.command_queue)?;

        Ok(())
    }
}
