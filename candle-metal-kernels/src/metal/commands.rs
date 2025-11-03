use crate::metal::CommandBuffer;
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue, MTLCounterSet};
use std::sync::{Arc, Mutex};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type CounterSet = Retained<ProtocolObject<dyn MTLCounterSet>>;

pub struct Commands {
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
    command_buffer: Arc<Mutex<CommandBuffer>>,
    /// Keeps track of the current amount of compute command encoders on the current
    /// command buffer
    /// Arc, RwLock because of the interior mutability.
    command_buffer_index: usize,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    //capture: Option<Retained<MTLCaptureManager>>,
    //timestamp_counter_set: Option<CounterSet>,
}
unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

pub fn create_command_buffer(
    command_queue: &CommandQueue,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue.commandBuffer().map(CommandBuffer::new).ok_or(
        MetalKernelError::FailedToCreateResource("CommandBuffer".to_string()),
    )
}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let command_buffer = create_command_buffer(&command_queue)?;
        command_buffer.enqueue();
        let command_buffer = Arc::new(Mutex::new(command_buffer));

        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val.parse().unwrap_or(50),
            _ => 50,
        };
        Ok(Self {
            command_queue,
            command_buffer,
            command_buffer_index: 0,
            compute_per_buffer,
        })
    }

    pub fn command_buffer(&mut self) -> Result<(bool, CommandBuffer), MetalKernelError> {
        let mut current = self.command_buffer.lock()?;

        let mut flushed = false;
        if self.command_buffer_index > self.compute_per_buffer {
            current.commit();
            *current = create_command_buffer(&self.command_queue)?;
            self.command_buffer_index = 0;
            flushed = true;
        }
        self.command_buffer_index += 1;
        Ok((flushed, current.clone()))
    }

    pub fn wait_until_completed(&mut self) -> Result<(), MetalKernelError> {
        let command_buffer = {
            let mut current = self.command_buffer.lock()?;
            let current_clone = current.clone();
            *current = create_command_buffer(&self.command_queue)?;
            current_clone
        };

        match command_buffer.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                command_buffer.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {} // No action needed
            MTLCommandBufferStatus::Error => {
                if let Some(error) = command_buffer.error() {
                    return Err(MetalKernelError::CommandBufferError(error.to_string()));
                }
            }
            // All status variants covered.
            // We need this final match arm because the statuses are implemented as integers, not an enum, in the objc2 framework.
            _ => unreachable!(),
        }
        Ok(())
    }
}
