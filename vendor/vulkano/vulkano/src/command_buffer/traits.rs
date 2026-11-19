use super::{
    CommandBuffer, CommandBufferInheritanceInfo, CommandBufferResourcesUsage, CommandBufferState,
    CommandBufferUsage, OldSubmitInfo, SecondaryCommandBufferResourcesUsage,
};
use crate::{
    buffer::Buffer,
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageLayout},
    swapchain::Swapchain,
    sync::future::{
        now, queue_submit, AccessCheckError, AccessError, GpuFuture, NowFuture, SubmitAnyBuilder,
    },
    DeviceSize, SafeDeref, Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use parking_lot::{Mutex, MutexGuard};
use std::{
    borrow::Cow,
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

pub unsafe trait PrimaryCommandBufferAbstract:
    VulkanObject<Handle = vk::CommandBuffer> + DeviceOwned + Send + Sync
{
    /// Returns the underlying raw command buffer.
    fn as_raw(&self) -> &CommandBuffer;

    /// Returns the queue family index of this command buffer.
    fn queue_family_index(&self) -> u32;

    /// Returns the usage of this command buffer.
    fn usage(&self) -> CommandBufferUsage;

    /// Executes this command buffer on a queue.
    ///
    /// This function returns an object that implements the [`GpuFuture`] trait. See the
    /// documentation of the [`future`][crate::sync::future] module for more information.
    ///
    /// The command buffer is not actually executed until you call [`flush()`][GpuFuture::flush] on
    /// the future.  You are encouraged to chain together as many futures as possible prior to
    /// calling [`flush()`][GpuFuture::flush]. In order to know when the future has completed, call
    /// one of [`then_signal_fence()`][GpuFuture::then_signal_fence] or
    /// [`then_signal_semaphore()`][GpuFuture::then_signal_semaphore]. You can do both together
    /// with [`then_signal_fence_and_flush()`][GpuFuture::then_signal_fence_and_flush] or
    /// [`then_signal_semaphore_and_flush()`][GpuFuture::then_signal_semaphore_and_flush],
    /// respectively.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > In the future this function may return `-> impl GpuFuture` instead of a concrete type.
    ///
    /// </div>
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > This is just a shortcut for `execute_after(vulkano::sync::now(), queue)`.
    ///
    /// </div>
    ///
    /// # Panics
    ///
    /// - Panics if the device of the command buffer is not the same as the device of the future.
    #[inline]
    fn execute(
        self: Arc<Self>,
        queue: Arc<Queue>,
    ) -> Result<CommandBufferExecFuture<NowFuture>, CommandBufferExecError>
    where
        Self: Sized + 'static,
    {
        let device = queue.device().clone();
        self.execute_after(now(device), queue)
    }

    /// Executes the command buffer after an existing future.
    ///
    /// This function returns an object that implements the [`GpuFuture`] trait. See the
    /// documentation of the [`future`][crate::sync::future] module for more information.
    ///
    /// The command buffer is not actually executed until you call [`flush()`][GpuFuture::flush] on
    /// the future.  You are encouraged to chain together as many futures as possible prior to
    /// calling [`flush()`][GpuFuture::flush]. In order to know when the future has completed, call
    /// one of [`then_signal_fence()`][GpuFuture::then_signal_fence] or
    /// [`then_signal_semaphore()`][GpuFuture::then_signal_semaphore]. You can do both together
    /// with [`then_signal_fence_and_flush()`][GpuFuture::then_signal_fence_and_flush] or
    /// [`then_signal_semaphore_and_flush()`][GpuFuture::then_signal_semaphore_and_flush],
    /// respectively.
    ///
    /// <div class="vulkano-alert-note">
    ///
    /// > Note
    /// >
    /// > In the future this function may return `-> impl GpuFuture` instead of a concrete type.
    ///
    /// </div>
    ///
    /// This function requires the `'static` lifetime to be on the command buffer. This is because
    /// this function returns a `CommandBufferExecFuture` whose job is to lock resources and keep
    /// them alive while they are in use by the GPU. If `'static` wasn't required, you could call
    /// `std::mem::forget` on that object and "unlock" these resources. For more information about
    /// this problem, search the web for "rust thread scoped leakpocalypse".
    ///
    /// # Panics
    ///
    /// - Panics if the device of the command buffer is not the same as the device of the future.
    fn execute_after<F>(
        self: Arc<Self>,
        future: F,
        queue: Arc<Queue>,
    ) -> Result<CommandBufferExecFuture<F>, CommandBufferExecError>
    where
        Self: Sized + 'static,
        F: GpuFuture,
    {
        assert_eq!(self.device().handle(), future.device().handle());

        if !future.queue_change_allowed() {
            assert_eq!(future.queue().unwrap(), queue);
        }

        Ok(CommandBufferExecFuture {
            previous: future,
            command_buffer: self,
            queue,
            submitted: Mutex::new(false),
            finished: AtomicBool::new(false),
        })
    }

    #[doc(hidden)]
    fn state(&self) -> MutexGuard<'_, CommandBufferState>;

    #[doc(hidden)]
    fn resources_usage(&self) -> &CommandBufferResourcesUsage;
}

impl Debug for dyn PrimaryCommandBufferAbstract {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        Debug::fmt(&self.handle(), f)
    }
}

unsafe impl<T> PrimaryCommandBufferAbstract for T
where
    T: VulkanObject<Handle = vk::CommandBuffer> + SafeDeref + Send + Sync,
    T::Target: PrimaryCommandBufferAbstract,
{
    fn as_raw(&self) -> &CommandBuffer {
        (**self).as_raw()
    }

    fn queue_family_index(&self) -> u32 {
        (**self).queue_family_index()
    }

    fn usage(&self) -> CommandBufferUsage {
        (**self).usage()
    }

    fn state(&self) -> MutexGuard<'_, CommandBufferState> {
        (**self).state()
    }

    fn resources_usage(&self) -> &CommandBufferResourcesUsage {
        (**self).resources_usage()
    }
}

pub unsafe trait SecondaryCommandBufferAbstract:
    VulkanObject<Handle = vk::CommandBuffer> + DeviceOwned + Send + Sync
{
    /// Returns the underlying raw command buffer.
    fn as_raw(&self) -> &CommandBuffer;

    /// Returns the usage of this command buffer.
    fn usage(&self) -> CommandBufferUsage;

    /// Returns a `CommandBufferInheritance` value describing the properties that the command
    /// buffer inherits from its parent primary command buffer.
    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo;

    /// Checks whether this command buffer is allowed to be recorded to a command buffer,
    /// and if so locks it.
    ///
    /// If you call this function, then you should call `unlock` afterwards.
    fn lock_record(&self) -> Result<(), Box<ValidationError>>;

    /// Unlocks the command buffer. Should be called once for each call to `lock_record`.
    ///
    /// # Safety
    ///
    /// Must not be called if you haven't called `lock_record` before.
    unsafe fn unlock(&self);

    #[doc(hidden)]
    fn resources_usage(&self) -> &SecondaryCommandBufferResourcesUsage;
}

unsafe impl<T> SecondaryCommandBufferAbstract for T
where
    T: VulkanObject<Handle = vk::CommandBuffer> + SafeDeref + Send + Sync,
    T::Target: SecondaryCommandBufferAbstract,
{
    fn as_raw(&self) -> &CommandBuffer {
        (**self).as_raw()
    }

    fn usage(&self) -> CommandBufferUsage {
        (**self).usage()
    }

    fn inheritance_info(&self) -> &CommandBufferInheritanceInfo {
        (**self).inheritance_info()
    }

    fn lock_record(&self) -> Result<(), Box<ValidationError>> {
        (**self).lock_record()
    }

    unsafe fn unlock(&self) {
        unsafe { (**self).unlock() };
    }

    fn resources_usage(&self) -> &SecondaryCommandBufferResourcesUsage {
        (**self).resources_usage()
    }
}

/// Represents a command buffer being executed by the GPU and the moment when the execution
/// finishes.
#[derive(Debug)]
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    previous: F,
    command_buffer: Arc<dyn PrimaryCommandBufferAbstract>,
    queue: Arc<Queue>,
    // True if the command buffer has already been submitted.
    // If flush is called multiple times, we want to block so that only one flushing is executed.
    // Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
    submitted: Mutex<bool>,
    finished: AtomicBool,
}

impl<F> CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    // Implementation of `build_submission`. Doesn't check whenever the future was already flushed.
    // You must make sure to not submit same command buffer multiple times.
    unsafe fn build_submission_impl(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        Ok(match unsafe { self.previous.build_submission() }? {
            SubmitAnyBuilder::Empty => SubmitAnyBuilder::CommandBuffer(
                OldSubmitInfo {
                    command_buffers: vec![self.command_buffer.clone()],
                    protected_submit: self.command_buffer.as_raw().is_protected(),
                    ..Default::default()
                },
                None,
            ),
            SubmitAnyBuilder::SemaphoresWait(semaphores) => SubmitAnyBuilder::CommandBuffer(
                OldSubmitInfo {
                    wait_semaphores: semaphores.into_iter().collect(),
                    command_buffers: vec![self.command_buffer.clone()],
                    protected_submit: self.command_buffer.as_raw().is_protected(),
                    ..Default::default()
                },
                None,
            ),
            SubmitAnyBuilder::CommandBuffer(mut submit_info, fence) => {
                // FIXME: add pipeline barrier
                submit_info
                    .command_buffers
                    .push(self.command_buffer.clone());
                SubmitAnyBuilder::CommandBuffer(submit_info, fence)
            }
            SubmitAnyBuilder::QueuePresent(_) => {
                unimplemented!() // TODO:
                                 /*present.submit();     // TODO: wrong
                                 let mut builder = SubmitCommandBufferBuilder::new();
                                 builder.add_command_buffer(self.command_buffer.inner());
                                 SubmitAnyBuilder::CommandBuffer(builder)*/
            }
        })
    }
}

unsafe impl<F> GpuFuture for CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        if *self.submitted.lock() {
            return Ok(SubmitAnyBuilder::Empty);
        }

        unsafe { self.build_submission_impl() }
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        let mut submitted = self.submitted.lock();

        if *submitted {
            return Ok(());
        }

        match unsafe { self.build_submission_impl() }? {
            SubmitAnyBuilder::Empty => {}
            SubmitAnyBuilder::CommandBuffer(submit_info, fence) => {
                unsafe { queue_submit(&self.queue, submit_info, fence, &self.previous) }.unwrap();
            }
            _ => unreachable!(),
        };

        // Only write `true` here in order to try again next time if we failed to submit.
        *submitted = true;
        Ok(())
    }

    unsafe fn signal_finished(&self) {
        if !self.finished.swap(true, Ordering::SeqCst) {
            let resource_usage = self.command_buffer.resources_usage();

            for usage in &resource_usage.buffers {
                let mut state = usage.buffer.state();

                for (range, range_usage) in usage.ranges.iter() {
                    if range_usage.mutable {
                        unsafe { state.gpu_write_unlock(range.clone()) };
                    } else {
                        unsafe { state.gpu_read_unlock(range.clone()) };
                    }
                }
            }

            for usage in &resource_usage.images {
                let mut state = usage.image.state();

                for (range, range_usage) in usage.ranges.iter() {
                    if range_usage.mutable {
                        unsafe { state.gpu_write_unlock(range.clone()) };
                    } else {
                        unsafe { state.gpu_read_unlock(range.clone()) };
                    }
                }
            }

            unsafe { self.command_buffer.state().set_submit_finished() };
        }

        unsafe { self.previous.signal_finished() };
    }

    fn queue_change_allowed(&self) -> bool {
        false
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        Some(self.queue.clone())
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        let resources_usage = self.command_buffer.resources_usage();
        let usage = match resources_usage.buffer_indices.get(buffer) {
            Some(&index) => &resources_usage.buffers[index],
            None => return Err(AccessCheckError::Unknown),
        };

        // TODO: check the queue family

        let result = usage
            .ranges
            .range(&range)
            .try_fold((), |_, (_range, range_usage)| {
                if !range_usage.mutable && exclusive {
                    Err(AccessCheckError::Unknown)
                } else {
                    Ok(())
                }
            });

        match result {
            Ok(()) => Ok(()),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => self
                .previous
                .check_buffer_access(buffer, range, exclusive, queue),
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
        let resources_usage = self.command_buffer.resources_usage();
        let usage = match resources_usage.image_indices.get(image) {
            Some(&index) => &resources_usage.images[index],
            None => return Err(AccessCheckError::Unknown),
        };

        // TODO: check the queue family

        let result = usage
            .ranges
            .range(&range)
            .try_fold((), |_, (_range, range_usage)| {
                if expected_layout != ImageLayout::Undefined
                    && range_usage.final_layout != expected_layout
                {
                    return Err(AccessCheckError::Denied(
                        AccessError::UnexpectedImageLayout {
                            allowed: range_usage.final_layout,
                            requested: expected_layout,
                        },
                    ));
                }

                if !range_usage.mutable && exclusive {
                    Err(AccessCheckError::Unknown)
                } else {
                    Ok(())
                }
            });

        match result {
            Ok(()) => Ok(()),
            Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
            Err(AccessCheckError::Unknown) => {
                self.previous
                    .check_image_access(image, range, exclusive, expected_layout, queue)
            }
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        _before: bool,
    ) -> Result<(), AccessCheckError> {
        self.previous
            .check_swapchain_image_acquired(swapchain, image_index, false)
    }
}

unsafe impl<F> DeviceOwned for CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    fn device(&self) -> &Arc<Device> {
        self.command_buffer.device()
    }
}

impl<F> Drop for CommandBufferExecFuture<F>
where
    F: GpuFuture,
{
    fn drop(&mut self) {
        if !*self.finished.get_mut() && !thread::panicking() {
            // TODO: handle errors?
            self.flush().unwrap();
            // Block until the queue finished.
            self.queue.with(|mut q| q.wait_idle()).unwrap();

            unsafe {
                self.signal_finished();
            }
        }
    }
}

/// Error that can happen when attempting to execute a command buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandBufferExecError {
    /// Access to a resource has been denied.
    AccessError {
        error: AccessError,
        command_name: Cow<'static, str>,
        command_param: Cow<'static, str>,
        command_offset: usize,
    },

    /// The command buffer or one of the secondary command buffers it executes was created with the
    /// "one time submit" flag, but has already been submitted it the past.
    OneTimeSubmitAlreadySubmitted,

    /// The command buffer or one of the secondary command buffers it executes is already in use by
    /// the GPU and was not created with the "concurrent" flag.
    ExclusiveAlreadyInUse,
    // TODO: missing entries (eg. wrong queue family, secondary command buffer)
}

impl Error for CommandBufferExecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CommandBufferExecError::AccessError { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl Display for CommandBufferExecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            CommandBufferExecError::AccessError {
                error: _,
                command_name,
                command_offset,
                command_param,
            } => write!(
                f,
                "access to a resource has been denied on command {} (offset: {}, param: {})",
                command_name, command_offset, command_param,
            ),
            CommandBufferExecError::OneTimeSubmitAlreadySubmitted => write!(
                f,
                "the command buffer or one of the secondary command buffers it executes was \
                created with the \"one time submit\" flag, but has already been submitted in \
                the past",
            ),
            CommandBufferExecError::ExclusiveAlreadyInUse => write!(
                f,
                "the command buffer or one of the secondary command buffers it executes is \
                already in use was not created with the \"concurrent\" flag"
            ),
        }
    }
}
