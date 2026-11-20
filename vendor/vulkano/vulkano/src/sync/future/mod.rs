//! Represents an event that will happen on the GPU in the future.
//!
//! Whenever you ask the GPU to start an operation by using a function of the vulkano library (for
//! example executing a command buffer), this function will return a *future*. A future is an
//! object that implements [the `GpuFuture` trait](GpuFuture) and that represents the
//! point in time when this operation is over.
//!
//! No function in vulkano immediately sends an operation to the GPU (with the exception of some
//! unsafe low-level functions). Instead they return a future that is in the pending state. Before
//! the GPU actually starts doing anything, you have to *flush* the future by calling the `flush()`
//! method or one of its derivatives.
//!
//! Futures serve several roles:
//!
//! - Futures can be used to build dependencies between operations and makes it possible to ask
//!   that an operation starts only after a previous operation is finished.
//! - Submitting an operation to the GPU is a costly operation. By chaining multiple operations
//!   with futures you will submit them all at once instead of one by one, thereby reducing this
//!   cost.
//! - Futures keep alive the resources and objects used by the GPU so that they don't get destroyed
//!   while they are still in use.
//!
//! The last point means that you should keep futures alive in your program for as long as their
//! corresponding operation is potentially still being executed by the GPU. Dropping a future
//! earlier will block the current thread (after flushing, if necessary) until the GPU has finished
//! the operation, which is usually not what you want.
//!
//! If you write a function that submits an operation to the GPU in your program, you are
//! encouraged to let this function return the corresponding future and let the caller handle it.
//! This way the caller will be able to chain multiple futures together and decide when it wants to
//! keep the future alive or drop it.
//!
//! # Executing an operation after a future
//!
//! Respecting the order of operations on the GPU is important, as it is what *proves* vulkano that
//! what you are doing is indeed safe. For example if you submit two operations that modify the
//! same buffer, then you need to execute one after the other instead of submitting them
//! independently. Failing to do so would mean that these two operations could potentially execute
//! simultaneously on the GPU, which would be unsafe.
//!
//! This is done by calling one of the methods of the `GpuFuture` trait. For example calling
//! `prev_future.then_execute(command_buffer)` takes ownership of `prev_future` and will make sure
//! to only start executing `command_buffer` after the moment corresponding to `prev_future`
//! happens. The object returned by the `then_execute` function is itself a future that corresponds
//! to the moment when the execution of `command_buffer` ends.
//!
//! ## Between two different GPU queues
//!
//! When you want to perform an operation after another operation on two different queues, you
//! **must** put a *semaphore* between them. Failure to do so would result in a runtime error.
//! Adding a semaphore is a simple as replacing `prev_future.then_execute(...)` with
//! `prev_future.then_signal_semaphore().then_execute(...)`.
//!
//! <div class="vulkano-alert-note">
//!
//! > Note
//! >
//! > A common use-case is using a transfer queue (i.e. a queue that is only capable of performing
//! > transfer operations) to write data to a buffer, then read that data from the rendering queue.
//!
//! </div>
//!
//! What happens when you do so is that the first queue will execute the first set of operations
//! (represented by `prev_future` in the example), then put a semaphore in the signaled state.
//! Meanwhile the second queue blocks (if necessary) until that same semaphore gets signaled, and
//! then only will execute the second set of operations.
//!
//! Since you want to avoid blocking the second queue as much as possible, you probably want to
//! flush the operation to the first queue as soon as possible. This can easily be done by calling
//! `then_signal_semaphore_and_flush()` instead of `then_signal_semaphore()`.
//!
//! ## Between several different GPU queues
//!
//! The `then_signal_semaphore()` method is appropriate when you perform an operation in one queue,
//! and want to see the result in another queue. However in some situations you want to start
//! multiple operations on several different queues.
//!
//! TODO: this is not yet implemented
//!
//! # Fences
//!
//! A `Fence` is an object that is used to signal the CPU when an operation on the GPU is finished.
//!
//! Signaling a fence is done by calling `then_signal_fence()` on a future. Just like semaphores,
//! you are encouraged to use `then_signal_fence_and_flush()` instead.
//!
//! Signaling a fence is kind of a "terminator" to a chain of futures

pub use self::{
    fence_signal::{FenceSignalFuture, FenceSignalFutureBehavior},
    join::JoinFuture,
    now::{now, NowFuture},
    semaphore_signal::SemaphoreSignalFuture,
};
use super::{fence::Fence, semaphore::Semaphore};
use crate::{
    buffer::{Buffer, BufferState},
    command_buffer::{
        CommandBufferExecError, CommandBufferExecFuture, CommandBufferResourcesUsage,
        CommandBufferState, CommandBufferSubmitInfo, CommandBufferUsage, OldSubmitInfo,
        PrimaryCommandBufferAbstract, SemaphoreSubmitInfo, SubmitInfo,
    },
    device::{DeviceOwned, Queue},
    image::{Image, ImageLayout, ImageState},
    swapchain::{self, PresentFuture, PresentInfo, Swapchain, SwapchainPresentInfo},
    DeviceSize, Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use foldhash::HashMap;
use parking_lot::MutexGuard;
use smallvec::{smallvec, SmallVec};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    ops::Range,
    sync::{atomic::Ordering, Arc},
};

mod fence_signal;
mod join;
mod now;
mod semaphore_signal;

/// Represents an event that will happen on the GPU in the future.
///
/// See the documentation of the `sync` module for explanations about futures.
// TODO: consider switching all methods to take `&mut self` for optimization purposes
pub unsafe trait GpuFuture: DeviceOwned {
    /// If possible, checks whether the submission has finished. If so, gives up ownership of the
    /// resources used by these submissions.
    ///
    /// It is highly recommended to call `cleanup_finished` from time to time. Doing so will
    /// prevent memory usage from increasing over time, and will also destroy the locks on
    /// resources used by the GPU.
    fn cleanup_finished(&mut self);

    /// Builds a submission that, if submitted, makes sure that the event represented by this
    /// `GpuFuture` will happen, and possibly contains extra elements (e.g.. a semaphore wait or an
    /// event wait) that makes the dependency with subsequent operations work.
    ///
    /// It is the responsibility of the caller to ensure that the submission is going to be
    /// submitted only once. However keep in mind that this function can perfectly be called
    /// multiple times (as long as the returned object is only submitted once).
    /// Also note that calling `flush()` on the future  may change the value returned by
    /// `build_submission()`.
    ///
    /// It is however the responsibility of the implementation to not return the same submission
    /// from multiple different future objects. For example if you implement `GpuFuture` on
    /// `Arc<Foo>` then `build_submission()` must always return `SubmitAnyBuilder::Empty`,
    /// otherwise it would be possible for the user to clone the `Arc` and make the same
    /// submission be submitted multiple times.
    ///
    /// It is also the responsibility of the implementation to ensure that it works if you call
    /// `build_submission()` and submits the returned value without calling `flush()` first. In
    /// other words, `build_submission()` should perform an implicit flush if necessary.
    ///
    /// Once the caller has submitted the submission and has determined that the GPU has finished
    /// executing it, it should call `signal_finished`. Failure to do so will incur a large runtime
    /// overhead, as the future will have to block to make sure that it is finished.
    #[doc(hidden)]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>>;

    /// Flushes the future and submits to the GPU the actions that will permit this future to
    /// occur.
    ///
    /// The implementation must remember that it was flushed. If the function is called multiple
    /// times, only the first time must result in a flush.
    fn flush(&self) -> Result<(), Validated<VulkanError>>;

    /// Sets the future to its "complete" state, meaning that it can safely be destroyed.
    ///
    /// This must only be done if you called `build_submission()`, submitted the returned
    /// submission, and determined that it was finished.
    ///
    /// The implementation must be aware that this function can be called multiple times on the
    /// same future.
    unsafe fn signal_finished(&self);

    /// Returns the queue that triggers the event. Returns `None` if unknown or irrelevant.
    ///
    /// If this function returns `None` and `queue_change_allowed` returns `false`, then a panic
    /// is likely to occur if you use this future. This is only a problem if you implement
    /// the `GpuFuture` trait yourself for a type outside of vulkano.
    fn queue(&self) -> Option<Arc<Queue>>;

    /// Returns `true` if elements submitted after this future can be submitted to a different
    /// queue than the other returned by `queue()`.
    fn queue_change_allowed(&self) -> bool;

    /// Checks whether submitting something after this future grants access (exclusive or shared,
    /// depending on the parameter) to the given buffer on the given queue.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > Returning `Ok` means "access granted", while returning `Err` means "don't know".
    /// > Therefore returning `Err` is never unsafe.
    ///
    /// </div>
    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError>;

    /// Checks whether submitting something after this future grants access (exclusive or shared,
    /// depending on the parameter) to the given image on the given queue.
    ///
    /// Implementations must ensure that the image is in the given layout. However if the `layout`
    /// is `Undefined` then the implementation should accept any actual layout.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > Returning `Ok` means "access granted", while returning `Err` means "don't know".
    /// > Therefore returning `Err` is never unsafe.
    ///
    /// </div>
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > Keep in mind that changing the layout of an image also requires exclusive access.
    ///
    /// </div>
    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError>;

    /// Checks whether accessing a swapchain image is permitted.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > Setting `before` to `true` should skip checking the current future and always forward the
    /// > call to the future before.
    ///
    /// </div>
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError>;

    /// Joins this future with another one, representing the moment when both events have happened.
    // TODO: handle errors
    fn join<F>(self, other: F) -> JoinFuture<Self, F>
    where
        Self: Sized,
        F: GpuFuture,
    {
        join::join(self, other)
    }

    /// Executes a command buffer after this future.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > This is just a shortcut function. The actual implementation is in the `CommandBuffer`
    /// > trait.
    ///
    /// </div>
    fn then_execute(
        self,
        queue: Arc<Queue>,
        command_buffer: Arc<impl PrimaryCommandBufferAbstract + 'static>,
    ) -> Result<CommandBufferExecFuture<Self>, CommandBufferExecError>
    where
        Self: Sized,
    {
        command_buffer.execute_after(self, queue)
    }

    /// Executes a command buffer after this future, on the same queue as the future.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > This is just a shortcut function. The actual implementation is in the `CommandBuffer`
    /// > trait.
    ///
    /// </div>
    fn then_execute_same_queue(
        self,
        command_buffer: Arc<impl PrimaryCommandBufferAbstract + 'static>,
    ) -> Result<CommandBufferExecFuture<Self>, CommandBufferExecError>
    where
        Self: Sized,
    {
        let queue = self.queue().unwrap();
        command_buffer.execute_after(self, queue)
    }

    /// Signals a semaphore after this future. Returns another future that represents the signal.
    ///
    /// Call this function when you want to execute some operations on a queue and want to see the
    /// result on another queue.
    #[inline]
    fn then_signal_semaphore(self) -> SemaphoreSignalFuture<Self>
    where
        Self: Sized,
    {
        semaphore_signal::then_signal_semaphore(self)
    }

    /// Signals a semaphore after this future and flushes it. Returns another future that
    /// represents the moment when the semaphore is signaled.
    ///
    /// This is a just a shortcut for `then_signal_semaphore()` followed with `flush()`.
    ///
    /// When you want to execute some operations A on a queue and some operations B on another
    /// queue that need to see the results of A, it can be a good idea to submit A as soon as
    /// possible while you're preparing B.
    ///
    /// If you ran A and B on the same queue, you would have to decide between submitting A then
    /// B, or A and B simultaneously. Both approaches have their trade-offs. But if A and B are
    /// on two different queues, then you would need two submits anyway and it is always
    /// advantageous to submit A as soon as possible.
    #[inline]
    fn then_signal_semaphore_and_flush(
        self,
    ) -> Result<SemaphoreSignalFuture<Self>, Validated<VulkanError>>
    where
        Self: Sized,
    {
        let f = self.then_signal_semaphore();
        f.flush()?;

        Ok(f)
    }

    /// Signals a fence after this future. Returns another future that represents the signal.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > More often than not you want to immediately flush the future after calling this function.
    /// > If so, consider using `then_signal_fence_and_flush`.
    ///
    /// </div>
    #[inline]
    fn then_signal_fence(self) -> FenceSignalFuture<Self>
    where
        Self: Sized,
    {
        fence_signal::then_signal_fence(self, FenceSignalFutureBehavior::Continue)
    }

    /// Signals a fence after this future. Returns another future that represents the signal.
    ///
    /// This is a just a shortcut for `then_signal_fence()` followed with `flush()`.
    #[inline]
    fn then_signal_fence_and_flush(self) -> Result<FenceSignalFuture<Self>, Validated<VulkanError>>
    where
        Self: Sized,
    {
        let f = self.then_signal_fence();
        f.flush()?;

        Ok(f)
    }

    /// Presents a swapchain image after this future.
    ///
    /// You should only ever do this indirectly after a `SwapchainAcquireFuture` of the same image,
    /// otherwise an error will occur when flushing.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > This is just a shortcut for the `Swapchain::present()` function.
    ///
    /// </div>
    #[inline]
    fn then_swapchain_present(
        self,
        queue: Arc<Queue>,
        swapchain_info: SwapchainPresentInfo,
    ) -> PresentFuture<Self>
    where
        Self: Sized,
    {
        swapchain::present(self, queue, swapchain_info)
    }

    /// Turn the current future into a `Box<dyn GpuFuture>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture>`.
    #[inline]
    fn boxed(self) -> Box<dyn GpuFuture>
    where
        Self: Sized + 'static,
    {
        Box::new(self) as _
    }

    /// Turn the current future into a `Box<dyn GpuFuture + Send>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture + Send>`.
    #[inline]
    fn boxed_send(self) -> Box<dyn GpuFuture + Send>
    where
        Self: Sized + Send + 'static,
    {
        Box::new(self) as _
    }

    /// Turn the current future into a `Box<dyn GpuFuture + Sync>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture + Sync>`.
    #[inline]
    fn boxed_sync(self) -> Box<dyn GpuFuture + Sync>
    where
        Self: Sized + Sync + 'static,
    {
        Box::new(self) as _
    }

    /// Turn the current future into a `Box<dyn GpuFuture + Send + Sync>`.
    ///
    /// This is a helper function that calls `Box::new(yourFuture) as Box<dyn GpuFuture + Send +
    /// Sync>`.
    #[inline]
    fn boxed_send_sync(self) -> Box<dyn GpuFuture + Send + Sync>
    where
        Self: Sized + Send + Sync + 'static,
    {
        Box::new(self) as _
    }
}

unsafe impl<F: ?Sized> GpuFuture for Box<F>
where
    F: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        (**self).cleanup_finished()
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        unsafe { (**self).build_submission() }
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        (**self).flush()
    }

    unsafe fn signal_finished(&self) {
        unsafe { (**self).signal_finished() }
    }

    fn queue_change_allowed(&self) -> bool {
        (**self).queue_change_allowed()
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        (**self).queue()
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        (**self).check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        (**self).check_image_access(image, range, exclusive, expected_layout, queue)
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        (**self).check_swapchain_image_acquired(swapchain, image_index, before)
    }
}

/// Contains all the possible submission builders.
#[derive(Debug)]
#[doc(hidden)]
pub enum SubmitAnyBuilder {
    Empty,
    SemaphoresWait(SmallVec<[Arc<Semaphore>; 8]>),
    CommandBuffer(OldSubmitInfo, Option<Arc<Fence>>),
    QueuePresent(PresentInfo),
}

impl SubmitAnyBuilder {
    /// Returns true if equal to `SubmitAnyBuilder::Empty`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, SubmitAnyBuilder::Empty)
    }
}

/// Access to a resource was denied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessError {
    /// The resource is already in use, and there is no tracking of concurrent usages.
    AlreadyInUse,

    UnexpectedImageLayout {
        allowed: ImageLayout,
        requested: ImageLayout,
    },

    /// Trying to use an image without transitioning it from the "undefined" or "preinitialized"
    /// layouts first.
    ImageNotInitialized {
        /// The layout that was requested for the image.
        requested: ImageLayout,
    },

    /// Trying to use a swapchain image without depending on a corresponding acquire image future.
    SwapchainImageNotAcquired,
}

impl Error for AccessError {}

impl Display for AccessError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let value = match self {
            AccessError::AlreadyInUse => {
                "the resource is already in use, and there is no tracking of concurrent usages"
            }
            AccessError::UnexpectedImageLayout { allowed, requested } => {
                return write!(
                    f,
                    "unexpected image layout: requested {:?}, allowed {:?}",
                    allowed, requested
                )
            }
            AccessError::ImageNotInitialized { .. } => {
                "trying to use an image without transitioning it from the undefined or \
                preinitialized layouts first"
            }
            AccessError::SwapchainImageNotAcquired => {
                "trying to use a swapchain image without depending on a corresponding acquire \
                image future"
            }
        };

        write!(f, "{}", value,)
    }
}

/// Error that can happen when checking whether we have access to a resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessCheckError {
    /// Access to the resource has been denied.
    Denied(AccessError),
    /// The resource is unknown, therefore we cannot possibly answer whether we have access or not.
    Unknown,
}

impl Error for AccessCheckError {}

impl Display for AccessCheckError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            AccessCheckError::Denied(err) => {
                write!(f, "access to the resource has been denied: {}", err)
            }
            AccessCheckError::Unknown => write!(f, "the resource is unknown"),
        }
    }
}

impl From<AccessError> for AccessCheckError {
    fn from(err: AccessError) -> AccessCheckError {
        AccessCheckError::Denied(err)
    }
}

pub(crate) unsafe fn queue_present(
    queue: &Arc<Queue>,
    present_info: PresentInfo,
) -> Result<impl ExactSizeIterator<Item = Result<bool, VulkanError>> + use<>, Validated<VulkanError>>
{
    let results = queue.with(|mut queue_guard| unsafe { queue_guard.present(&present_info) })?;

    let PresentInfo {
        wait_semaphores: _,
        swapchain_infos: swapchains,
        _ne: _,
    } = &present_info;

    // If a presentation results in a loss of full-screen exclusive mode,
    // signal that to the relevant swapchain.
    for (&result, swapchain_info) in results.iter().zip(swapchains) {
        if result == Err(VulkanError::FullScreenExclusiveModeLost) {
            unsafe { swapchain_info.swapchain.full_screen_exclusive_held() }
                .store(false, Ordering::SeqCst);
        }
    }

    Ok(results.into_iter())
}

pub(crate) unsafe fn queue_submit(
    queue: &Arc<Queue>,
    submit_info: OldSubmitInfo,
    fence: Option<Arc<Fence>>,
    future: &dyn GpuFuture,
) -> Result<(), Validated<VulkanError>> {
    let submit_infos: SmallVec<[_; 4]> = smallvec![submit_info];
    let mut states = States::from_submit_infos(&submit_infos);

    for submit_info in &submit_infos {
        for command_buffer in &submit_info.command_buffers {
            let state = states
                .command_buffers
                .get(&command_buffer.handle())
                .unwrap();

            match command_buffer.usage() {
                CommandBufferUsage::OneTimeSubmit => {
                    if state.has_been_submitted() {
                        return Err(Box::new(ValidationError {
                            problem: "a command buffer, or one of the secondary \
                                command buffers it executes, was created with the \
                                `CommandBufferUsage::OneTimeSubmit` usage, but \
                                it has already been submitted in the past"
                                .into(),
                            vuids: &["VUID-vkQueueSubmit2-commandBuffer-03874"],
                            ..Default::default()
                        })
                        .into());
                    }
                }
                CommandBufferUsage::MultipleSubmit => {
                    if state.is_submit_pending() {
                        return Err(Box::new(ValidationError {
                            problem: "a command buffer, or one of the secondary \
                                command buffers it executes, was not created with the \
                                `CommandBufferUsage::SimultaneousUse` usage, but \
                                it is already in use by the device"
                                .into(),
                            vuids: &["VUID-vkQueueSubmit2-commandBuffer-03875"],
                            ..Default::default()
                        })
                        .into());
                    }
                }
                CommandBufferUsage::SimultaneousUse => (),
            }

            let CommandBufferResourcesUsage {
                buffers,
                images,
                buffer_indices: _,
                image_indices: _,
            } = command_buffer.resources_usage();

            for usage in buffers {
                let state = states.buffers.get_mut(&usage.buffer.handle()).unwrap();

                for (range, range_usage) in usage.ranges.iter() {
                    match future.check_buffer_access(
                        &usage.buffer,
                        range.clone(),
                        range_usage.mutable,
                        queue,
                    ) {
                        Err(AccessCheckError::Denied(error)) => {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "access to a resource has been denied \
                                    (resource use: {:?}, error: {})",
                                    range_usage.first_use, error
                                )
                                .into(),
                                ..Default::default()
                            })
                            .into());
                        }
                        Err(AccessCheckError::Unknown) => {
                            let result = if range_usage.mutable {
                                state.check_gpu_write(range.clone())
                            } else {
                                state.check_gpu_read(range.clone())
                            };

                            if let Err(error) = result {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "access to a resource has been denied \
                                        (resource use: {:?}, error: {})",
                                        range_usage.first_use, error
                                    )
                                    .into(),
                                    ..Default::default()
                                })
                                .into());
                            }
                        }
                        _ => (),
                    }
                }
            }

            for usage in images {
                let state = states.images.get_mut(&usage.image.handle()).unwrap();

                for (range, range_usage) in usage.ranges.iter() {
                    match future.check_image_access(
                        &usage.image,
                        range.clone(),
                        range_usage.mutable,
                        range_usage.expected_layout,
                        queue,
                    ) {
                        Err(AccessCheckError::Denied(error)) => {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "access to a resource has been denied \
                                    (resource use: {:?}, error: {})",
                                    range_usage.first_use, error
                                )
                                .into(),
                                ..Default::default()
                            })
                            .into());
                        }
                        Err(AccessCheckError::Unknown) => {
                            let result = if range_usage.mutable {
                                state.check_gpu_write(range.clone(), range_usage.expected_layout)
                            } else {
                                state.check_gpu_read(range.clone(), range_usage.expected_layout)
                            };

                            if let Err(error) = result {
                                return Err(Box::new(ValidationError {
                                    problem: format!(
                                        "access to a resource has been denied \
                                        (resource use: {:?}, error: {})",
                                        range_usage.first_use, error
                                    )
                                    .into(),
                                    ..Default::default()
                                })
                                .into());
                            }
                        }
                        _ => (),
                    };
                }
            }
        }
    }

    let new_submit_infos = submit_infos
        .iter()
        .map(|submit_info| {
            (
                submit_info
                    .wait_semaphores
                    .iter()
                    .map(|semaphore| SemaphoreSubmitInfo::new(semaphore))
                    .collect::<Vec<_>>(),
                submit_info
                    .command_buffers
                    .iter()
                    .map(|command_buffer| CommandBufferSubmitInfo::new(command_buffer.as_raw()))
                    .collect::<Vec<_>>(),
                submit_info
                    .signal_semaphores
                    .iter()
                    .map(|semaphore| SemaphoreSubmitInfo::new(semaphore))
                    .collect::<Vec<_>>(),
                submit_info.protected_submit,
            )
        })
        .collect::<Vec<_>>();
    let new_submit_infos = new_submit_infos
        .iter()
        .map(
            |(wait_semaphores, command_buffers, signal_semaphores, protected_submit)| SubmitInfo {
                wait_semaphores,
                command_buffers,
                signal_semaphores,
                protected_submit: *protected_submit,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>();

    queue
        .with(|mut queue_guard| unsafe { queue_guard.submit(&new_submit_infos, fence.as_ref()) })?;

    for submit_info in &submit_infos {
        let OldSubmitInfo {
            wait_semaphores: _,
            command_buffers,
            signal_semaphores: _,
            protected_submit: _,
        } = submit_info;

        for command_buffer in command_buffers {
            let state = states
                .command_buffers
                .get_mut(&command_buffer.handle())
                .unwrap();
            unsafe { state.add_queue_submit() };

            let CommandBufferResourcesUsage {
                buffers,
                images,
                buffer_indices: _,
                image_indices: _,
            } = command_buffer.resources_usage();

            for usage in buffers {
                let state = states.buffers.get_mut(&usage.buffer.handle()).unwrap();

                for (range, range_usage) in usage.ranges.iter() {
                    if range_usage.mutable {
                        unsafe { state.gpu_write_lock(range.clone()) };
                    } else {
                        unsafe { state.gpu_read_lock(range.clone()) };
                    }
                }
            }

            for usage in images {
                let state = states.images.get_mut(&usage.image.handle()).unwrap();

                for (range, range_usage) in usage.ranges.iter() {
                    if range_usage.mutable {
                        unsafe { state.gpu_write_lock(range.clone(), range_usage.final_layout) };
                    } else {
                        unsafe { state.gpu_read_lock(range.clone()) };
                    }
                }
            }
        }
    }

    Ok(())
}

// This struct exists to ensure that every object gets locked exactly once.
// Otherwise we get deadlocks.
#[derive(Debug)]
struct States<'a> {
    buffers: HashMap<vk::Buffer, MutexGuard<'a, BufferState>>,
    command_buffers: HashMap<vk::CommandBuffer, MutexGuard<'a, CommandBufferState>>,
    images: HashMap<vk::Image, MutexGuard<'a, ImageState>>,
}

impl<'a> States<'a> {
    fn from_submit_infos(submit_infos: &'a [OldSubmitInfo]) -> Self {
        let mut buffers = HashMap::default();
        let mut command_buffers = HashMap::default();
        let mut images = HashMap::default();

        for submit_info in submit_infos {
            let OldSubmitInfo {
                wait_semaphores: _,
                command_buffers: info_command_buffers,
                signal_semaphores: _,
                protected_submit: _,
            } = submit_info;

            for command_buffer in info_command_buffers {
                command_buffers
                    .entry(command_buffer.handle())
                    .or_insert_with(|| command_buffer.state());

                let CommandBufferResourcesUsage {
                    buffers: buffers_usage,
                    images: images_usage,
                    buffer_indices: _,
                    image_indices: _,
                } = command_buffer.resources_usage();

                for usage in buffers_usage {
                    let buffer = &usage.buffer;
                    buffers
                        .entry(buffer.handle())
                        .or_insert_with(|| buffer.state());
                }

                for usage in images_usage {
                    let image = &usage.image;
                    images
                        .entry(image.handle())
                        .or_insert_with(|| image.state());
                }
            }
        }

        Self {
            buffers,
            command_buffers,
            images,
        }
    }
}
