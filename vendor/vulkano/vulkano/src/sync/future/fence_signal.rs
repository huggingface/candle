use super::{AccessCheckError, GpuFuture};
use crate::{
    buffer::Buffer,
    command_buffer::OldSubmitInfo,
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageLayout},
    swapchain::Swapchain,
    sync::{
        fence::Fence,
        future::{queue_present, queue_submit, AccessError, SubmitAnyBuilder},
    },
    DeviceSize, Validated, ValidationError, VulkanError,
};
use parking_lot::{Mutex, MutexGuard};
use std::{
    future::Future,
    mem::replace,
    ops::Range,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    thread,
    time::Duration,
};

/// Builds a new fence signal future.
pub fn then_signal_fence<F>(future: F, behavior: FenceSignalFutureBehavior) -> FenceSignalFuture<F>
where
    F: GpuFuture,
{
    let device = future.device().clone();

    assert!(future.queue().is_some()); // TODO: document

    let fence = Arc::new(Fence::from_pool(&device).unwrap());
    FenceSignalFuture {
        device,
        state: Mutex::new(FenceSignalFutureState::Pending(future, fence)),
        behavior,
    }
}

/// Describes the behavior of the future if you submit something after it.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FenceSignalFutureBehavior {
    /// Continue execution on the same queue.
    Continue,
    /// Wait for the fence to be signaled before submitting any further operation.
    #[allow(dead_code)] // TODO: why is this never constructed?
    Block {
        /// How long to block the current thread.
        timeout: Option<Duration>,
    },
}

/// Represents a fence being signaled after a previous event.
///
/// Contrary to most other future types, it is possible to block the current thread until the event
/// happens. This is done by calling the `wait()` function.
///
/// This can also be done through Rust's Async system by simply `.await`ing this object. Note
/// though that (due to the Vulkan API fence design) this will spin to check the fence, rather than
/// blocking in the driver. Therefore if you have a long-running task, blocking may be less
/// CPU intense (depending on the driver's implementation).
///
/// Also note that the `GpuFuture` trait is implemented on `Arc<FenceSignalFuture<_>>`.
/// This means that you can put this future in an `Arc` and keep a copy of it somewhere in order
/// to know when the execution reached that point.
///
/// ```
/// use std::sync::Arc;
/// use vulkano::sync::GpuFuture;
///
/// # let future: Box<dyn GpuFuture> = return;
/// // Assuming you have a chain of operations, like this:
/// // let future = ...
/// //      .then_execute(foo)
/// //      .then_execute(bar)
///
/// // You can signal a fence at this point of the chain, and put the future in an `Arc`.
/// let fence_signal = Arc::new(future.then_signal_fence());
///
/// // And then continue the chain:
/// // fence_signal.clone()
/// //      .then_execute(baz)
/// //      .then_execute(qux)
///
/// // Later you can wait until you reach the point of `fence_signal`:
/// fence_signal.wait(None).unwrap();
/// ```
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished \
              processing the submission"]
pub struct FenceSignalFuture<F>
where
    F: GpuFuture,
{
    // Current state. See the docs of `FenceSignalFutureState`.
    state: Mutex<FenceSignalFutureState<F>>,
    // The device of the future.
    device: Arc<Device>,
    behavior: FenceSignalFutureBehavior,
}

// This future can be in three different states: pending (ie. newly-created), submitted (ie. the
// command that submits the fence has been submitted), or cleaned (ie. the previous future has
// been dropped).
enum FenceSignalFutureState<F> {
    // Newly-created. Not submitted yet.
    Pending(F, Arc<Fence>),

    // Partially submitted to the queue. Only happens in situations where submitting requires two
    // steps, and when the first step succeeded while the second step failed.
    //
    // Note that if there's ever a submit operation that needs three steps we will need to rework
    // this code, as it was designed for two-step operations only.
    PartiallyFlushed(F, Arc<Fence>),

    // Submitted to the queue.
    Flushed(F, Arc<Fence>),

    // The submission is finished. The previous future and the fence have been cleaned.
    Cleaned,

    // A function panicked while the state was being modified. Should never happen.
    Poisoned,
}

impl<F> FenceSignalFuture<F>
where
    F: GpuFuture,
{
    /// Returns true if the fence is signaled by the GPU.
    pub fn is_signaled(&self) -> Result<bool, VulkanError> {
        let state = self.state.lock();

        match &*state {
            FenceSignalFutureState::Pending(_, fence)
            | FenceSignalFutureState::PartiallyFlushed(_, fence)
            | FenceSignalFutureState::Flushed(_, fence) => unsafe { fence.is_signaled_unchecked() },
            FenceSignalFutureState::Cleaned => Ok(true),
            FenceSignalFutureState::Poisoned => unreachable!(),
        }
    }

    /// Blocks the current thread until the fence is signaled by the GPU. Performs a flush if
    /// necessary.
    ///
    /// If `timeout` is `None`, then the wait is infinite. Otherwise the thread will unblock after
    /// the specified timeout has elapsed and an error will be returned.
    ///
    /// If the wait is successful, this function also cleans any resource locked by previous
    /// submissions.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), Validated<VulkanError>> {
        let mut state = self.state.lock();

        self.flush_impl(&mut state)?;

        match replace(&mut *state, FenceSignalFutureState::Cleaned) {
            FenceSignalFutureState::Flushed(previous, fence) => {
                unsafe { fence.wait_unchecked(timeout) }?;
                unsafe { previous.signal_finished() };
                Ok(())
            }
            FenceSignalFutureState::Cleaned => Ok(()),
            _ => unreachable!(),
        }
    }
}

impl<F> FenceSignalFuture<F>
where
    F: GpuFuture,
{
    // Implementation of `cleanup_finished`, but takes a `&self` instead of a `&mut self`.
    // This is an external function so that we can also call it from an `Arc<FenceSignalFuture>`.
    fn cleanup_finished_impl(&self) {
        let mut state = self.state.lock();

        match *state {
            FenceSignalFutureState::Flushed(ref mut prev, ref fence) => {
                if unsafe { fence.wait_unchecked(Some(Duration::from_secs(0))) }.is_ok() {
                    unsafe { prev.signal_finished() };
                    *state = FenceSignalFutureState::Cleaned;
                } else {
                    prev.cleanup_finished();
                }
            }
            FenceSignalFutureState::Pending(ref mut prev, _) => {
                prev.cleanup_finished();
            }
            FenceSignalFutureState::PartiallyFlushed(ref mut prev, _) => {
                prev.cleanup_finished();
            }
            _ => (),
        }
    }

    // Implementation of `flush`. You must lock the state and pass the mutex guard here.
    fn flush_impl(
        &self,
        state: &mut MutexGuard<'_, FenceSignalFutureState<F>>,
    ) -> Result<(), Validated<VulkanError>> {
        // In this function we temporarily replace the current state with `Poisoned` at the
        // beginning, and we take care to always put back a value into `state` before
        // returning (even in case of error).
        let old_state = replace(&mut **state, FenceSignalFutureState::Poisoned);

        let (previous, new_fence, partially_flushed) = match old_state {
            FenceSignalFutureState::Pending(prev, fence) => (prev, fence, false),
            FenceSignalFutureState::PartiallyFlushed(prev, fence) => (prev, fence, true),
            other => {
                // We were already flushed in the past, or we're already poisoned. Don't do
                // anything.
                **state = other;
                return Ok(());
            }
        };

        // TODO: meh for unwrap
        let queue = previous.queue().unwrap();

        // There are three possible outcomes for the flush operation: success, partial success
        // in which case `result` will contain `Err(OutcomeErr::Partial)`, or total failure
        // in which case `result` will contain `Err(OutcomeErr::Full)`.
        enum OutcomeErr<E> {
            Partial(E),
            Full(E),
        }
        let result = match unsafe { previous.build_submission() }? {
            SubmitAnyBuilder::Empty => {
                debug_assert!(!partially_flushed);

                unsafe {
                    queue_submit(
                        &queue,
                        Default::default(),
                        Some(new_fence.clone()),
                        &previous,
                    )
                }
                .map_err(OutcomeErr::Full)
            }
            SubmitAnyBuilder::SemaphoresWait(semaphores) => {
                debug_assert!(!partially_flushed);

                unsafe {
                    queue_submit(
                        &queue,
                        OldSubmitInfo {
                            wait_semaphores: semaphores.into_iter().collect(),
                            ..Default::default()
                        },
                        None,
                        &previous,
                    )
                }
                .map_err(OutcomeErr::Full)
            }
            SubmitAnyBuilder::CommandBuffer(submit_info, fence) => {
                debug_assert!(!partially_flushed);
                // The assert below could technically be a debug assertion as it is part of the
                // safety contract of the trait. However it is easy to get this wrong if you
                // write a custom implementation, and if so the consequences would be
                // disastrous and hard to debug. Therefore we prefer to just use a regular
                // assertion.
                assert!(fence.is_none());

                unsafe { queue_submit(&queue, submit_info, Some(new_fence.clone()), &previous) }
                    .map_err(OutcomeErr::Full)
            }
            SubmitAnyBuilder::QueuePresent(present_info) => {
                if partially_flushed {
                    unsafe {
                        queue_submit(
                            &queue,
                            Default::default(),
                            Some(new_fence.clone()),
                            &previous,
                        )
                    }
                    .map_err(OutcomeErr::Partial)
                } else {
                    for swapchain_info in &present_info.swapchain_infos {
                        if swapchain_info.present_id.is_some_and(|present_id| !unsafe {
                            swapchain_info.swapchain.try_claim_present_id(present_id)
                        }) {
                            return Err(Box::new(ValidationError {
                                problem: "the provided `present_id` was not greater than any \
                                        `present_id` passed previously for the same swapchain"
                                    .into(),
                                vuids: &["VUID-VkPresentIdKHR-presentIds-04999"],
                                ..Default::default()
                            })
                            .into());
                        }

                        match previous.check_swapchain_image_acquired(
                            &swapchain_info.swapchain,
                            swapchain_info.image_index,
                            true,
                        ) {
                            Ok(_) => (),
                            Err(AccessCheckError::Unknown) => {
                                return Err(Box::new(ValidationError::from_error(
                                    AccessError::SwapchainImageNotAcquired,
                                ))
                                .into());
                            }
                            Err(AccessCheckError::Denied(err)) => {
                                return Err(Box::new(ValidationError::from_error(err)).into());
                            }
                        }
                    }

                    let intermediary_result = unsafe { queue_present(&queue, present_info) }?
                        .map(|r| r.map(|_| ()))
                        .fold(Ok(()), Result::and);

                    match intermediary_result {
                        Ok(()) => unsafe {
                            queue_submit(
                                &queue,
                                Default::default(),
                                Some(new_fence.clone()),
                                &previous,
                            )
                        }
                        .map_err(OutcomeErr::Partial),
                        Err(err) => Err(OutcomeErr::Full(err.into())),
                    }
                }
            }
        };

        // Restore the state before returning.
        match result {
            Ok(()) => {
                **state = FenceSignalFutureState::Flushed(previous, new_fence);
                Ok(())
            }
            Err(OutcomeErr::Partial(err)) => {
                **state = FenceSignalFutureState::PartiallyFlushed(previous, new_fence);
                Err(err)
            }
            Err(OutcomeErr::Full(err)) => {
                **state = FenceSignalFutureState::Pending(previous, new_fence);
                Err(err)
            }
        }
    }
}

impl<F> Future for FenceSignalFuture<F>
where
    F: GpuFuture,
{
    type Output = Result<(), VulkanError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Implement through fence
        let state = self.state.lock();

        match &*state {
            FenceSignalFutureState::Pending(_, fence)
            | FenceSignalFutureState::PartiallyFlushed(_, fence)
            | FenceSignalFutureState::Flushed(_, fence) => fence.poll_impl(cx),
            FenceSignalFutureState::Cleaned => Poll::Ready(Ok(())),
            FenceSignalFutureState::Poisoned => unreachable!(),
        }
    }
}

impl<F> FenceSignalFutureState<F> {
    fn get_prev(&self) -> Option<&F> {
        match self {
            FenceSignalFutureState::Pending(prev, _) => Some(prev),
            FenceSignalFutureState::PartiallyFlushed(prev, _) => Some(prev),
            FenceSignalFutureState::Flushed(prev, _) => Some(prev),
            FenceSignalFutureState::Cleaned => None,
            FenceSignalFutureState::Poisoned => None,
        }
    }
}

unsafe impl<F> GpuFuture for FenceSignalFuture<F>
where
    F: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.cleanup_finished_impl()
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        let mut state = self.state.lock();
        self.flush_impl(&mut state)?;

        match &*state {
            FenceSignalFutureState::Flushed(_, fence) => match self.behavior {
                FenceSignalFutureBehavior::Block { timeout } => {
                    unsafe { fence.wait_unchecked(timeout) }?;
                }
                FenceSignalFutureBehavior::Continue => (),
            },
            FenceSignalFutureState::Cleaned | FenceSignalFutureState::Poisoned => (),
            FenceSignalFutureState::Pending(_, _) => unreachable!(),
            FenceSignalFutureState::PartiallyFlushed(_, _) => unreachable!(),
        }

        Ok(SubmitAnyBuilder::Empty)
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        let mut state = self.state.lock();
        self.flush_impl(&mut state)
    }

    unsafe fn signal_finished(&self) {
        let state = self.state.lock();
        match *state {
            FenceSignalFutureState::Flushed(ref prev, _) => {
                unsafe { prev.signal_finished() };
            }
            FenceSignalFutureState::Cleaned | FenceSignalFutureState::Poisoned => (),
            _ => unreachable!(),
        }
    }

    fn queue_change_allowed(&self) -> bool {
        match self.behavior {
            FenceSignalFutureBehavior::Continue => {
                let state = self.state.lock();
                state.get_prev().is_none()
            }
            FenceSignalFutureBehavior::Block { .. } => true,
        }
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        let state = self.state.lock();
        if let Some(prev) = state.get_prev() {
            prev.queue()
        } else {
            None
        }
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        let state = self.state.lock();
        if let Some(previous) = state.get_prev() {
            previous.check_buffer_access(buffer, range, exclusive, queue)
        } else {
            Err(AccessCheckError::Unknown)
        }
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        let state = self.state.lock();
        if let Some(previous) = state.get_prev() {
            previous.check_image_access(image, range, exclusive, expected_layout, queue)
        } else {
            Err(AccessCheckError::Unknown)
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        _before: bool,
    ) -> Result<(), AccessCheckError> {
        if let Some(previous) = self.state.lock().get_prev() {
            previous.check_swapchain_image_acquired(swapchain, image_index, false)
        } else {
            Err(AccessCheckError::Unknown)
        }
    }
}

unsafe impl<F> DeviceOwned for FenceSignalFuture<F>
where
    F: GpuFuture,
{
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<F> Drop for FenceSignalFuture<F>
where
    F: GpuFuture,
{
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }

        let mut state = self.state.lock();

        // We ignore any possible error while submitting for now. Problems are handled below.
        let _ = self.flush_impl(&mut state);

        match replace(&mut *state, FenceSignalFutureState::Cleaned) {
            FenceSignalFutureState::Flushed(previous, fence) => {
                // This is a normal situation. Submitting worked.
                // TODO: handle errors?
                fence.wait(None).unwrap();
                unsafe { previous.signal_finished() };
            }
            FenceSignalFutureState::Cleaned => {
                // Also a normal situation. The user called `cleanup_finished()` before dropping.
            }
            FenceSignalFutureState::Poisoned => {
                // The previous future was already dropped and blocked the current queue.
            }
            FenceSignalFutureState::Pending(_, _)
            | FenceSignalFutureState::PartiallyFlushed(_, _) => {
                // Flushing produced an error. There's nothing more we can do except drop the
                // previous future and let it block the current queue.
            }
        }
    }
}

unsafe impl<F> GpuFuture for Arc<FenceSignalFuture<F>>
where
    F: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.cleanup_finished_impl()
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        // Note that this is sound because we always return `SubmitAnyBuilder::Empty`. See the
        // documentation of `build_submission`.
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
