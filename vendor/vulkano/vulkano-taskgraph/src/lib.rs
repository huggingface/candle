//! Vulkano's **EXPERIMENTAL** task graph implementation. Expect many bugs and incomplete features.
//! There is also currently no validation except the most bare-bones sanity checks. You may also
//! get panics in random places.

use command_buffer::RecordingCommandBuffer;
use concurrent_slotmap::{hyaline, Key, SlotId};
use graph::{CompileInfo, ExecuteError, ResourceMap, TaskGraph};
use linear_map::LinearMap;
use resource::{
    AccessTypes, BufferState, Flight, HostAccessType, ImageLayoutType, ImageState, Resources,
    SwapchainState,
};
use std::{
    any::{Any, TypeId},
    cell::Cell,
    cmp,
    error::Error,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem,
    ops::{Deref, RangeBounds},
    ptr::NonNull,
    sync::Arc,
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferMemory, Subbuffer},
    command_buffer as raw,
    device::Queue,
    format::ClearValue,
    image::Image,
    render_pass::Framebuffer,
    swapchain::Swapchain,
    DeviceSize, Validated, ValidationError,
};

pub mod collector;
pub mod command_buffer;
pub mod descriptor_set;
pub mod graph;
mod linear_map;
pub mod resource;
mod slotmap;

/// Creates a [`TaskGraph`] with one task node, compiles it, and executes it.
pub unsafe fn execute(
    queue: &Arc<Queue>,
    resources: &Arc<Resources>,
    flight_id: Id<Flight>,
    task: impl FnOnce(&mut RecordingCommandBuffer<'_>, &mut TaskContext<'_>) -> TaskResult,
    host_buffer_accesses: impl IntoIterator<Item = (Id<Buffer>, HostAccessType)>,
    buffer_accesses: impl IntoIterator<Item = (Id<Buffer>, AccessTypes)>,
    image_accesses: impl IntoIterator<Item = (Id<Image>, AccessTypes, ImageLayoutType)>,
) -> Result<(), ExecuteError> {
    #[repr(transparent)]
    struct OnceTask<'a>(
        &'a dyn Fn(&mut RecordingCommandBuffer<'_>, &mut TaskContext<'_>) -> TaskResult,
    );

    // SAFETY: The task is constructed inside this function and never leaves its scope, so there is
    // no way it could be sent to another thread.
    unsafe impl Send for OnceTask<'_> {}

    // SAFETY: The task is constructed inside this function and never leaves its scope, so there is
    // no way it could be shared with another thread.
    unsafe impl Sync for OnceTask<'_> {}

    impl Task for OnceTask<'static> {
        type World = ();

        unsafe fn execute(
            &self,
            cbf: &mut RecordingCommandBuffer<'_>,
            tcx: &mut TaskContext<'_>,
            _: &Self::World,
        ) -> TaskResult {
            (self.0)(cbf, tcx)
        }
    }

    let task = Cell::new(Some(task));
    let trampoline = move |cbf: &mut RecordingCommandBuffer<'_>, tcx: &mut TaskContext<'_>| {
        // `ExecutableTaskGraph::execute` calls each task exactly once, and we only execute the
        // task graph once.
        (Cell::take(&task).unwrap())(cbf, tcx)
    };

    let mut task_graph = TaskGraph::new(resources);

    for (id, access_type) in host_buffer_accesses {
        task_graph.add_host_buffer_access(id, access_type);
    }

    let mut node = task_graph.create_task_node(
        "",
        QueueFamilyType::Specific {
            index: queue.queue_family_index(),
        },
        // SAFETY: The task never leaves this function scope, so it is safe to pretend that the
        // local `trampoline` and its captures from the outer scope live forever.
        unsafe { mem::transmute::<OnceTask<'_>, OnceTask<'static>>(OnceTask(&trampoline)) },
    );

    for (id, access_types) in buffer_accesses {
        node.buffer_access(id, access_types);
    }

    for (id, access_types, layout_type) in image_accesses {
        node.image_access(id, access_types, layout_type);
    }

    // SAFETY:
    // * The user must ensure that there are no accesses that are incompatible with the queue.
    // * The user must ensure that there are no accesses incompatible with the device.
    let task_graph = unsafe {
        task_graph.compile(&CompileInfo {
            queues: &[queue],
            present_queue: None,
            flight_id,
            _ne: crate::NE,
        })
    }
    .unwrap();

    let resource_map = ResourceMap::new(&task_graph).unwrap();

    // SAFETY: The user must ensure that there are no other task graphs executing that access any
    // of the same subresources.
    unsafe { task_graph.execute(resource_map, &(), || {}) }
}

/// A task represents a unit of work to be recorded to a command buffer.
pub trait Task: Any + Send + Sync {
    type World: ?Sized;

    // Potentially TODO:
    // fn update(&mut self, ...) {}

    /// If the task node was created with any attachments which were [set to be cleared], this
    /// method is invoked to allow the task to set clear values for such attachments.
    ///
    /// This method is invoked at least once for every attachment to be cleared before every
    /// execution of the task. It's possible that it is invoked multiple times before the task is
    /// executed, and it may not be invoked right before [`execute`], just at some point between
    /// when the task graph has begun execution and when the task is executed.
    ///
    /// [set to be cleared]: graph::AttachmentInfo::clear
    /// [`execute`]: Self::execute
    #[allow(unused)]
    fn clear_values(&self, clear_values: &mut ClearValues<'_>, world: &Self::World) {}

    /// Executes the task, which should record its commands using the provided command buffer and
    /// context.
    ///
    /// # Safety
    ///
    /// - Every resource in the [task's access set] must not be written to concurrently in any
    ///   other tasks during execution on the device.
    /// - Every resource in the task's access set, if it's an [image access], must have had its
    ///   layout transitioned to the layout specified in the access.
    /// - Every resource in the task's access set, if the resource's [sharing mode] is exclusive,
    ///   must be currently owned by the queue family the task is executing on.
    ///
    /// [sharing mode]: vulkano::sync::Sharing
    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult;
}

impl<W: ?Sized + 'static> dyn Task<World = W> {
    /// Returns `true` if `self` is of type `T`.
    #[inline]
    pub fn is<T: Task<World = W>>(&self) -> bool {
        self.type_id() == TypeId::of::<T>()
    }

    /// Returns a reference to the inner value if it is of type `T`, or returns `None` otherwise.
    #[inline]
    pub fn downcast_ref<T: Task<World = W>>(&self) -> Option<&T> {
        if self.is::<T>() {
            // SAFETY: We just checked that the type is correct.
            Some(unsafe { self.downcast_unchecked_ref() })
        } else {
            None
        }
    }

    /// Returns a reference to the inner value if it is of type `T`, or returns `None` otherwise.
    #[inline]
    pub fn downcast_mut<T: Task<World = W>>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            // SAFETY: We just checked that the type is correct.
            Some(unsafe { self.downcast_unchecked_mut() })
        } else {
            None
        }
    }

    /// Returns a reference to the inner value without checking if it is of type `T`.
    ///
    /// # Safety
    ///
    /// `self` must be of type `T`.
    #[inline]
    pub unsafe fn downcast_unchecked_ref<T: Task<World = W>>(&self) -> &T {
        // SAFETY: The caller must guarantee that the type is correct.
        unsafe { &*<*const dyn Task<World = W>>::cast::<T>(self) }
    }

    /// Returns a reference to the inner value without checking if it is of type `T`.
    ///
    /// # Safety
    ///
    /// `self` must be of type `T`.
    #[inline]
    pub unsafe fn downcast_unchecked_mut<T: Task<World = W>>(&mut self) -> &mut T {
        // SAFETY: The caller must guarantee that the type is correct.
        unsafe { &mut *<*mut dyn Task<World = W>>::cast::<T>(self) }
    }
}

impl<W: ?Sized> fmt::Debug for dyn Task<World = W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Task").finish_non_exhaustive()
    }
}

/// An implementation of a phantom task, which is zero-sized and doesn't do anything.
///
/// You may want to use this if all you're interested in is the automatic synchronization and don't
/// have any other commands to execute. A common example would be doing a queue family ownership
/// transfer after doing an upload.
impl<W: ?Sized + 'static> Task for PhantomData<fn() -> W> {
    type World = W;

    unsafe fn execute(
        &self,
        _cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        _world: &Self::World,
    ) -> TaskResult {
        Ok(())
    }
}

/// The context of a task.
///
/// This gives you access to the resources.
pub struct TaskContext<'a> {
    resource_map: &'a ResourceMap<'a>,
    current_frame_index: u32,
    command_buffers: &'a mut Vec<Arc<raw::CommandBuffer>>,
}

impl<'a> TaskContext<'a> {
    /// Returns the buffer corresponding to `id`, panicking on an invalid slot error.
    ///
    /// This is a shortcut for `try_buffer().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_buffer`] returns an [`InvalidSlotError`].
    ///
    /// [`try_buffer`]: Self::try_buffer
    #[inline]
    #[track_caller]
    pub fn buffer(&self, id: Id<Buffer>) -> &'a BufferState {
        self.try_buffer(id).unwrap()
    }

    /// Returns the buffer corresponding to `id`.
    #[inline]
    pub fn try_buffer(&self, id: Id<Buffer>) -> Result<&'a BufferState, InvalidSlotError> {
        if id.is_virtual() {
            // SAFETY: The caller of `Task::execute` must ensure that `self.resource_map` maps the
            // virtual IDs of the graph exhaustively.
            Ok(unsafe { self.resource_map.buffer(id) }?)
        } else {
            let resources = self.resource_map.resources();
            let guard = self.resource_map.guard();

            Ok(resources.try_buffer_protected(id, guard)?)
        }
    }

    /// Returns the image corresponding to `id`, panicking on an invalid slot error.
    ///
    /// This is a shortcut for `try_image().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_image`] returns an [`InvalidSlotError`].
    /// - Panics if `id` refers to a swapchain image.
    ///
    /// [`try_image`]: Self::try_image
    #[inline]
    #[track_caller]
    pub fn image(&self, id: Id<Image>) -> &'a ImageState {
        self.try_image(id).unwrap()
    }

    /// Returns the image corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if `id` refers to a swapchain image.
    #[inline]
    pub fn try_image(&self, id: Id<Image>) -> Result<&'a ImageState, InvalidSlotError> {
        assert_ne!(id.object_type(), ObjectType::Swapchain);

        if id.is_virtual() {
            // SAFETY: The caller of `Task::execute` must ensure that `self.resource_map` maps the
            // virtual IDs of the graph exhaustively.
            Ok(unsafe { self.resource_map.image(id) }?)
        } else {
            let resources = self.resource_map.resources();
            let guard = self.resource_map.guard();

            Ok(resources.try_image_protected(id, guard)?)
        }
    }

    /// Returns the swapchain corresponding to `id`, panicking on an invalid slot error.
    ///
    /// This is a shortcut for `try_swapchain().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_swapchain`] returns an [`InvalidSlotError`].
    ///
    /// [`try_swapchain`]: Self::try_swapchain
    #[inline]
    #[track_caller]
    pub fn swapchain(&self, id: Id<Swapchain>) -> &'a SwapchainState {
        self.try_swapchain(id).unwrap()
    }

    /// Returns the swapchain corresponding to `id`, or returns an error if it isn't present.
    #[inline]
    pub fn try_swapchain(&self, id: Id<Swapchain>) -> Result<&'a SwapchainState, InvalidSlotError> {
        if id.is_virtual() {
            // SAFETY: The caller of `Task::execute` must ensure that `self.resource_map` maps the
            // virtual IDs of the graph exhaustively.
            Ok(unsafe { self.resource_map.swapchain(id) }?)
        } else {
            let resources = self.resource_map.resources();
            let guard = self.resource_map.guard();

            Ok(resources.try_swapchain_protected(id, guard)?)
        }
    }

    /// Returns the `ResourceMap`.
    #[inline]
    pub fn resource_map(&self) -> &'a ResourceMap<'a> {
        self.resource_map
    }

    /// Returns the index of the current [frame] in [flight].
    #[inline]
    #[must_use]
    pub fn current_frame_index(&self) -> u32 {
        self.current_frame_index
    }

    /// Tries to get read access to a portion of the buffer corresponding to `id`, panicking on a
    /// validation error or task error.
    ///
    /// If host read access for the buffer is not accounted for in the [task graph's host access
    /// set], this method will return an error.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be read using this method and an error will
    /// be returned.
    ///
    /// This is a shortcut for `try_read_buffer().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_read_buffer`] returns a [`ValidationError`] or [`TaskError`].
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    /// [`try_read_buffer`]: Self::try_read_buffer
    #[track_caller]
    pub fn read_buffer<T: BufferContents + ?Sized>(
        &self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> &T {
        self.try_read_buffer(id, range).unwrap()
    }

    /// Tries to get read access to a portion of the buffer corresponding to `id`.
    ///
    /// If host read access for the buffer is not accounted for in the [task graph's host access
    /// set], this method will return an error.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be read using this method and an error will
    /// be returned.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub fn try_read_buffer<T: BufferContents + ?Sized>(
        &self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> Result<&T, Validated<TaskError>> {
        self.validate_read_buffer(id)?;

        // SAFETY: We checked that the task has read access to the buffer above, which also
        // includes the guarantee that no other tasks can be writing the subbuffer on neither the
        // host nor the device. The same task cannot obtain another mutable reference to the buffer
        // because `TaskContext::write_buffer` requires a mutable reference.
        Ok(unsafe { self.read_buffer_unchecked(id, range) }?)
    }

    fn validate_read_buffer(&self, id: Id<Buffer>) -> Result<(), Box<ValidationError>> {
        if !self
            .resource_map
            .virtual_resources()
            .contains_host_buffer_access(id, HostAccessType::Read)
        {
            return Err(Box::new(ValidationError {
                context: "TaskContext::read_buffer".into(),
                problem: "the task graph does not have an access of type `HostAccessType::Read` \
                    for the buffer"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    /// Gets read access to a portion of the buffer corresponding to `id` without checking if this
    /// access is accounted for in the [task graph's host access set].
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be read using this method and an error will
    /// be returned.
    ///
    /// # Safety
    ///
    /// This access must be accounted for in the task graph's host access set.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub unsafe fn read_buffer_unchecked<T: BufferContents + ?Sized>(
        &self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> Result<&T, TaskError> {
        assert!(T::LAYOUT.alignment().as_devicesize() <= 64);

        let buffer = self
            .try_buffer(id)
            .map_err(TaskError::InvalidSlot)?
            .buffer();
        let subbuffer = Subbuffer::from(buffer.clone())
            .slice(range)
            .reinterpret::<T>();

        let allocation = match buffer.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => {
                todo!("`TaskContext::read_buffer` doesn't support sparse binding yet");
            }
            BufferMemory::External => {
                return Err(TaskError::HostAccess(HostAccessError::Unmanaged));
            }
            _ => unreachable!(),
        };

        match unsafe { allocation.mapped_slice_unchecked(..) } {
            Ok(_) => {}
            Err(vulkano::sync::HostAccessError::NotHostMapped) => {
                return Err(TaskError::HostAccess(HostAccessError::NotHostMapped));
            }
            Err(vulkano::sync::HostAccessError::OutOfMappedRange) => {
                return Err(TaskError::HostAccess(HostAccessError::OutOfMappedRange));
            }
            Err(_) => unreachable!(),
        }

        let mapped_slice = subbuffer.mapped_slice().unwrap();

        // SAFETY: The caller must ensure that access to the data is synchronized.
        let data_ptr = unsafe { T::ptr_from_slice(mapped_slice) };
        let data = unsafe { &*data_ptr };

        Ok(data)
    }

    /// Tries to get write access to a portion of the buffer corresponding to `id`, panicking on a
    /// validation error or task error.
    ///
    /// If host write access for the buffer is not accounted for in the [task graph's host access
    /// set], this method will return an error.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be written using this method and an error
    /// will be returned.
    ///
    /// This is a shortcut for `try_write_buffer().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_write_buffer`] returns a [`ValidationError`] or [`TaskError`].
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    /// [`try_write_buffer`]: Self::try_write_buffer
    #[track_caller]
    pub fn write_buffer<T: BufferContents + ?Sized>(
        &mut self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> &mut T {
        self.try_write_buffer(id, range).unwrap()
    }

    /// Tries to get write access to a portion of the buffer corresponding to `id`.
    ///
    /// If host write access for the buffer is not accounted for in the [task graph's host access
    /// set], this method will return an error.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be written using this method and an error
    /// will be returned.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub fn try_write_buffer<T: BufferContents + ?Sized>(
        &mut self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> Result<&mut T, Validated<TaskError>> {
        self.validate_write_buffer(id)?;

        // SAFETY: We checked that the task has write access to the buffer above, which also
        // includes the guarantee that no other tasks can be accessing the buffer on neither the
        // host nor the device. The same task cannot obtain another mutable reference to the buffer
        // because `TaskContext::write_buffer` requires a mutable reference.
        Ok(unsafe { self.write_buffer_unchecked(id, range) }?)
    }

    fn validate_write_buffer(&self, id: Id<Buffer>) -> Result<(), Box<ValidationError>> {
        if !self
            .resource_map
            .virtual_resources()
            .contains_host_buffer_access(id, HostAccessType::Write)
        {
            return Err(Box::new(ValidationError {
                context: "TaskContext::write_buffer".into(),
                problem: "the task graph does not have an access of type `HostAccessType::Write` \
                    for the buffer"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    /// Gets write access to a portion of the buffer corresponding to `id` without checking if this
    /// access is accounted for in the [task graph's host access set].
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be written using this method and an error
    /// will be returned.
    ///
    /// # Safety
    ///
    /// This access must be accounted for in the task graph's host access set.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub unsafe fn write_buffer_unchecked<T: BufferContents + ?Sized>(
        &mut self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> Result<&mut T, TaskError> {
        assert!(T::LAYOUT.alignment().as_devicesize() <= 64);

        let buffer = self
            .try_buffer(id)
            .map_err(TaskError::InvalidSlot)?
            .buffer();
        let subbuffer = Subbuffer::from(buffer.clone())
            .slice(range)
            .reinterpret::<T>();

        let allocation = match buffer.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => {
                todo!("`TaskContext::write_buffer` doesn't support sparse binding yet");
            }
            BufferMemory::External => {
                return Err(TaskError::HostAccess(HostAccessError::Unmanaged));
            }
            _ => unreachable!(),
        };

        match unsafe { allocation.mapped_slice_unchecked(..) } {
            Ok(_) => {}
            Err(vulkano::sync::HostAccessError::NotHostMapped) => {
                return Err(TaskError::HostAccess(HostAccessError::NotHostMapped));
            }
            Err(vulkano::sync::HostAccessError::OutOfMappedRange) => {
                return Err(TaskError::HostAccess(HostAccessError::OutOfMappedRange));
            }
            Err(_) => unreachable!(),
        }

        let mapped_slice = subbuffer.mapped_slice().unwrap();

        // SAFETY: The caller must ensure that access to the data is synchronized.
        let data_ptr = unsafe { T::ptr_from_slice(mapped_slice) };
        let data = unsafe { &mut *data_ptr };

        Ok(data)
    }

    /// Pushes a command buffer into the list of command buffers to be executed on the queue.
    ///
    /// All command buffers will be executed in the order in which they are pushed after the task
    /// has finished execution. That means in particular, that commands recorded by the task will
    /// start execution before execution of any pushed command buffers starts.
    ///
    /// # Safety
    ///
    /// Since the command buffer will be executed on the same queue right after the current command
    /// buffer, without any added synchronization, it must be safe to do so. The given command
    /// buffer must not do any accesses not accounted for in the [task's access set], or ensure
    /// that such accesses are appropriately synchronized.
    #[inline]
    pub unsafe fn push_command_buffer(&mut self, command_buffer: Arc<raw::CommandBuffer>) {
        self.command_buffers.push(command_buffer);
    }

    /// Extends the list of command buffers to be executed on the queue.
    ///
    /// This function behaves identically to the [`push_command_buffer`] method, except that it
    /// pushes all command buffers from the given iterator in order.
    ///
    /// # Safety
    ///
    /// See the [`push_command_buffer`] method for the safety preconditions.
    ///
    /// [`push_command_buffer`]: Self::push_command_buffer
    #[inline]
    pub unsafe fn extend_command_buffers(
        &mut self,
        command_buffers: impl IntoIterator<Item = Arc<raw::CommandBuffer>>,
    ) {
        self.command_buffers.extend(command_buffers);
    }
}

/// Stores the clear value for each attachment that was [set to be cleared] when creating the task
/// node.
///
/// This is used to set the clear values in [`Task::clear_values`].
///
/// [set to be cleared]: graph::AttachmentInfo::clear
pub struct ClearValues<'a> {
    inner: &'a mut LinearMap<Id, Option<ClearValue>>,
    resource_map: &'a ResourceMap<'a>,
}

impl ClearValues<'_> {
    /// Sets the clear value for the image corresponding to `id`.
    #[inline]
    pub fn set(&mut self, id: Id<Image>, clear_value: impl Into<ClearValue>) {
        self.set_inner(id, clear_value.into());
    }

    fn set_inner(&mut self, id: Id<Image>, clear_value: ClearValue) {
        let mut id = id.erase();

        if !id.is_virtual() {
            let virtual_resources = self.resource_map.virtual_resources();

            if let Some(&virtual_id) = virtual_resources.physical_map().get(&id.erase()) {
                id = virtual_id;
            } else {
                return;
            }
        }

        if let Some(value) = self.inner.get_mut(&id) {
            if value.is_none() {
                *value = Some(clear_value);
            }
        }
    }
}

/// The type of result returned by a task.
pub type TaskResult<T = (), E = Validated<TaskError>> = ::std::result::Result<T, E>;

/// Error that can happen inside a task.
#[derive(Debug)]
pub enum TaskError {
    InvalidSlot(InvalidSlotError),
    HostAccess(HostAccessError),
}

impl From<InvalidSlotError> for Validated<TaskError> {
    fn from(err: InvalidSlotError) -> Self {
        Self::Error(TaskError::InvalidSlot(err))
    }
}

impl From<TaskError> for Validated<TaskError> {
    fn from(err: TaskError) -> Self {
        Self::Error(err)
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::InvalidSlot(_) => "invalid slot",
            Self::HostAccess(_) => "a host access error occurred",
        };

        f.write_str(msg)
    }
}

impl Error for TaskError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidSlot(err) => Some(err),
            Self::HostAccess(err) => Some(err),
        }
    }
}

/// Error that can happen when trying to retrieve a Vulkan object or state by [`Id`].
#[derive(Debug)]
pub struct InvalidSlotError {
    id: Id,
}

impl InvalidSlotError {
    fn new<O>(id: Id<O>) -> Self {
        InvalidSlotError { id: id.erase() }
    }
}

impl fmt::Display for InvalidSlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self { id } = self;
        let object_type = id.object_type();

        write!(f, "invalid slot for object type `{object_type:?}`: {id:?}")
    }
}

impl Error for InvalidSlotError {}

/// Error that can happen when attempting to read or write a resource from the host.
#[derive(Debug)]
pub enum HostAccessError {
    Unmanaged,
    NotHostMapped,
    OutOfMappedRange,
}

impl fmt::Display for HostAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Unmanaged => "the resource is not managed by vulkano",
            Self::NotHostMapped => "the device memory is not current host-mapped",
            Self::OutOfMappedRange => {
                "the requested range is not within the currently mapped range of device memory"
            }
        };

        f.write_str(msg)
    }
}

impl Error for HostAccessError {}

/// Specifies the type of queue family that a task can be executed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueueFamilyType {
    /// Picks a queue family that supports graphics and transfer operations.
    Graphics,

    /// Picks a queue family that supports compute and transfer operations.
    Compute,

    /// Picks a queue family that supports transfer operations.
    Transfer,

    // TODO:
    // VideoDecode,

    // TODO:
    // VideoEncode,
    /// Picks the queue family of the given index. You should generally avoid this and use one of
    /// the other variants, so that the task graph compiler can pick the most optimal queue family
    /// indices that still satisfy the supported operations that the tasks require (and also, it's
    /// more convenient that way, as there's less to think about). Nevertheless, you may want to
    /// use this if you're looking for some very specific outcome.
    Specific { index: u32 },
}

/// This ID type is used throughout the crate to refer to Vulkan objects such as resource objects
/// and their synchronization state, synchronization object state, and other state.
///
/// The type parameter denotes the type of object or state being referred to.
///
/// Note that this ID **is not** globally unique. It is unique in the scope of a logical device.
#[repr(transparent)]
pub struct Id<T = ()> {
    slot: SlotId,
    marker: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    /// An ID that's guaranteed to be invalid.
    pub const INVALID: Self = Id {
        slot: SlotId::INVALID,
        marker: PhantomData,
    };

    const unsafe fn new(slot: SlotId) -> Self {
        Id {
            slot,
            marker: PhantomData,
        }
    }

    fn index(self) -> u32 {
        self.slot.index()
    }

    /// Returns `true` if this ID represents a [virtual resource].
    #[inline]
    pub const fn is_virtual(self) -> bool {
        self.slot.tag() & Id::VIRTUAL_BIT != 0
    }

    /// Returns `true` if this ID represents a resource with the exclusive sharing mode.
    fn is_exclusive(self) -> bool {
        self.slot.tag() & Id::EXCLUSIVE_BIT != 0
    }

    fn erase(self) -> Id {
        unsafe { Id::new(self.slot) }
    }

    fn is<O: Object>(self) -> bool {
        self.object_type() == O::TYPE
    }

    fn object_type(self) -> ObjectType {
        match self.slot.tag() & Id::OBJECT_TYPE_MASK {
            Buffer::TAG => ObjectType::Buffer,
            Image::TAG => ObjectType::Image,
            Swapchain::TAG => ObjectType::Swapchain,
            Flight::TAG => ObjectType::Flight,
            _ => unreachable!(),
        }
    }

    unsafe fn parametrize<O: Object>(self) -> Id<O> {
        unsafe { Id::new(self.slot) }
    }
}

impl Id<Swapchain> {
    /// Returns the ID that always refers to the swapchain image that's currently acquired from the
    /// swapchain.
    #[inline]
    pub const fn current_image_id(self) -> Id<Image> {
        unsafe { Id::new(self.slot) }
    }
}

impl Id {
    const OBJECT_TYPE_MASK: u32 = 0b111;

    const VIRTUAL_BIT: u32 = 1 << 7;
    const EXCLUSIVE_BIT: u32 = 1 << 6;
}

impl<T> Clone for Id<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.slot, f)
    }
}

impl<T> PartialEq for Id<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}

impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.slot.hash(state);
    }
}

#[doc(hidden)]
impl Key for Id {
    #[inline]
    fn from_id(id: SlotId) -> Self {
        unsafe { Id::new(id) }
    }

    #[inline]
    fn as_id(self) -> SlotId {
        self.slot
    }
}

#[doc(hidden)]
impl Key for Id<Framebuffer> {
    #[inline]
    fn from_id(id: SlotId) -> Self {
        unsafe { Id::new(id) }
    }

    #[inline]
    fn as_id(self) -> SlotId {
        self.slot
    }
}

impl<T> PartialOrd for Id<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.slot.cmp(&other.slot)
    }
}

/// A reference to some Vulkan object or state.
///
/// When you use [`Id`] to retrieve something, you can get back a `Ref` with the same type
/// parameter, which you can then dereference to get at the underlying data denoted by the type
/// parameter.
pub struct Ref<'a, T: 'a> {
    // We cannot use a reference here because references must remain valid for the duration of a
    // function call they are passed to. Our reference is only valid until the guard is dropped,
    // therefore passing `Ref` to a function would be unsound as the reference could get
    // invalidated in the middle (or even at the end would be unsound).
    inner: NonNull<T>,
    #[allow(unused)]
    guard: hyaline::Guard<'a>,
}

impl<'a, T> Ref<'a, T> {
    fn new(inner: &'a T, guard: hyaline::Guard<'a>) -> Self {
        Ref {
            inner: inner.into(),
            guard,
        }
    }
}

impl<T> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: The pointer was constructured from an existing reference in `Ref::new`.
        unsafe { self.inner.as_ref() }
    }
}

impl<T: fmt::Debug> fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

trait Object {
    const TYPE: ObjectType;

    const TAG: u32 = Self::TYPE as u32;
}

impl Object for Buffer {
    const TYPE: ObjectType = ObjectType::Buffer;
}

impl Object for Image {
    const TYPE: ObjectType = ObjectType::Image;
}

impl Object for Swapchain {
    const TYPE: ObjectType = ObjectType::Swapchain;
}

impl Object for Flight {
    const TYPE: ObjectType = ObjectType::Flight;
}

impl Object for Framebuffer {
    const TYPE: ObjectType = ObjectType::Framebuffer;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjectType {
    Buffer = 0,
    Image = 1,
    Swapchain = 2,
    Flight = 3,
    Framebuffer = 4,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NonExhaustive<'a>(PhantomData<&'a ()>);

impl fmt::Debug for NonExhaustive<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("NonExhaustive")
    }
}

const NE: NonExhaustive<'static> = NonExhaustive(PhantomData);

macro_rules! assert_unsafe_precondition {
    ($condition:expr, $message:expr $(,)?) => {
        // The nesting is intentional. There is a special path in the compiler for `if false`
        // facilitating conditional compilation without `#[cfg]` and the problems that come with it.
        if cfg!(debug_assertions) {
            if !$condition {
                crate::panic_nounwind(concat!("unsafe precondition(s) validated: ", $message));
            }
        }
    };
}
use assert_unsafe_precondition;

/// Polyfill for `core::panicking::panic_nounwind`.
#[cold]
#[inline(never)]
fn panic_nounwind(message: &'static str) -> ! {
    struct UnwindGuard;

    impl Drop for UnwindGuard {
        fn drop(&mut self) {
            panic!();
        }
    }

    let _guard = UnwindGuard;
    std::panic::panic_any(message);
}

#[cfg(test)]
mod tests {
    macro_rules! test_queues {
        () => {
            test_queues!(;;)
        };
        (
            $($device_feature:ident),*;
            $($device_extension:ident),*;
            $($instance_extension:ident),* $(;)?
        ) => {{
            let Ok(library) = (unsafe { vulkano::VulkanLibrary::new() }) else {
                return;
            };
            let instance_extensions = vulkano::instance::InstanceExtensions {
                $($instance_extension: true,)*
                ..Default::default()
            };
            if !library.supported_extensions().contains(&instance_extensions) {
                return;
            }
            let Ok(instance) = vulkano::instance::Instance::new(
                &library,
                &vulkano::instance::InstanceCreateInfo {
                    enabled_extensions: &instance_extensions,
                    ..Default::default()
                },
            ) else {
                return;
            };
            let Ok(physical_devices) = instance.enumerate_physical_devices() else {
                return;
            };
            let device_extensions = vulkano::device::DeviceExtensions {
                $($device_extension: true,)*
                ..Default::default()
            };
            let device_features = vulkano::device::DeviceFeatures {
                $($device_feature: true,)*
                ..Default::default()
            };
            let Some(physical_device) = physical_devices
                .into_iter()
                .filter(|p|
                    p.supported_extensions().contains(&device_extensions)
                        && p.supported_features().contains(&device_features)
                )
                .find(|p| {
                    p.queue_family_properties().iter().any(|q| {
                        q.queue_flags
                            .contains(vulkano::device::QueueFlags::GRAPHICS)
                    })
                })
            else {
                return;
            };
            let queue_create_infos = physical_device
                .queue_family_properties()
                .iter()
                .enumerate()
                .map(|(i, _)| vulkano::device::QueueCreateInfo {
                    queue_family_index: i as u32,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            let Ok((device, queues)) = vulkano::device::Device::new(
                &physical_device,
                &vulkano::device::DeviceCreateInfo {
                    queue_create_infos: &queue_create_infos,
                    enabled_extensions: &device_extensions,
                    enabled_features: &device_features,
                    ..Default::default()
                },
            ) else {
                return;
            };

            (
                $crate::resource::Resources::new(&device, &Default::default()).unwrap(),
                queues,
            )
        }};
    }
    pub(crate) use test_queues;
}
