use crate::metal::CommandBuffer;
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue, MTLCounterSet};

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
    command_buffer: CommandBuffer,
    /// Keeps track of the current amount of compute command encoders on the current
    /// command buffer
    /// Arc, RwLock because of the interior mutability.
    command_buffer_index: usize,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
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
            MTLCommandBufferStatus::Committed
            | MTLCommandBufferStatus::Scheduled
            | MTLCommandBufferStatus::Completed => {
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
