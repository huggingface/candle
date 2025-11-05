use crate::metal::{
    BlitCommandEncoder, CommandBuffer, CommandSemaphore, CommandStatus, ComputeCommandEncoder,
};
use crate::{utils::RwLockGuard, MetalKernelError};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue, MTLCounterSet};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard, RwLock, RwLockWriteGuard,
};

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
    /// Arc, RwLock because of the interior mutability.
    command_buffer: Arc<RwLock<CommandBuffer>>,
    /// Keeps track of the current amount of compute command encoders on the current
    /// command buffer
    compute_count: AtomicUsize,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    semaphore: Arc<CommandSemaphore>,
    //capture: Option<Retained<MTLCaptureManager>>,
    //timestamp_counter_set: Option<CounterSet>,
}
unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let command_buffer = create_command_buffer(&command_queue, Arc::clone(&semaphore))?;
        let command_buffer = Arc::new(RwLock::new(command_buffer));

        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val.parse().unwrap_or(50),
            _ => 50,
        };
        Ok(Self {
            command_queue,
            command_buffer,
            compute_count: AtomicUsize::new(0),
            compute_per_buffer,
            semaphore,
        })
    }

    pub fn create_command_buffer(&self) -> Result<CommandBuffer, MetalKernelError> {
        create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))
    }

    pub fn command_buffer(
        &self,
    ) -> Result<(bool, RwLockGuard<'_, CommandBuffer>), MetalKernelError> {
        // If compute count > compute per buffer then commit current command buffer and
        // replace it with a new one.
        if self.compute_count.load(Ordering::Relaxed) > self.compute_per_buffer {
            let mut command_buffer = self.command_buffer.write()?;
            command_buffer.commit();
            *command_buffer = self.create_command_buffer()?;
            self.compute_count.store(1, Ordering::Relaxed);
            Ok((true, command_buffer.into()))
        } else {
            self.compute_count.fetch_add(1, Ordering::Relaxed);
            Ok((false, self.command_buffer.read()?.into()))
        }
    }

    pub fn command_encoder(&mut self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        {
            // Ensure command buffer available.
            let mut guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            // Set status as encoding to block other threads from encoding to this commmand buffer
            *guard = CommandStatus::Encoding;
        }
        // Notify after command status lock is released
        self.semaphore.cond.notify_one();

        let (flush, command_buffer) = self.command_buffer()?;
        let command_encoder = command_buffer.compute_command_encoder();

        Ok((flush, command_encoder))
    }

    pub fn blit_command_encoder(&mut self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        {
            // Ensure command buffer available.
            let mut guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            // Set status as encoding to block other threads from encoding to this commmand buffer
            *guard = CommandStatus::Encoding;
        }
        // Notify after command status lock is released
        self.semaphore.cond.notify_one();

        let (flush, command_buffer) = self.command_buffer()?;
        let blit_command_encoder = command_buffer.blit_command_encoder();

        Ok((flush, blit_command_encoder))
    }

    pub fn wait_until_completed(&mut self) -> Result<(), MetalKernelError> {
        let current = {
            // Ensure command buffer not encoding.
            let mut guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available | CommandStatus::Done));

            // Extract current command buffer, create new in its place
            let current = {
                // Scope drops write lock
                let mut command_buffer = self.command_buffer.write()?;
                let current = command_buffer.clone();
                *command_buffer = self.create_command_buffer()?;
                // Update compute count
                self.compute_count.store(0, Ordering::Relaxed);
                current
            };
            // After replacing the command buffer it is now safe to continue encoding new commands.
            *guard = CommandStatus::Available;

            current
        };
        // Notify after command status lock is released
        self.semaphore.cond.notify_one();

        // Only commit and wait if it needed
        match current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                current.commit();
                current.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                current.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {} // No action needed
            MTLCommandBufferStatus::Error => {
                if let Some(error) = current.error() {
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
