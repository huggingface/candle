//! An event provides fine-grained synchronization within a single queue, or from the host to a
//! queue.
//!
//! When an event is signaled from a queue using the [`set_event`] command buffer command,
//! an event acts similar to a [pipeline barrier], but the synchronization scopes are split:
//! the source synchronization scope includes only commands before the `set_event` command,
//! while the destination synchronization scope includes only commands after the
//! [`wait_events`] command. Commands in between the two are not included.
//!
//! An event can also be signaled from the host, by calling the [`set`] method directly on the
//! [`Event`].
//!
//! [`set_event`]: crate::command_buffer::RecordingCommandBuffer::set_event
//! [pipeline barrier]: crate::command_buffer::RecordingCommandBuffer::pipeline_barrier
//! [`wait_events`]: crate::command_buffer::RecordingCommandBuffer::wait_events
//! [`set`]: Event::set

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use std::{mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// Used to block the GPU execution until an event on the CPU occurs.
///
/// Note that Vulkan implementations may have limits on how long a command buffer will wait for an
/// event to be signaled, in order to avoid interfering with progress of other clients of the GPU.
/// If the event isn't signaled within these limits, results are undefined and may include
/// device loss.
#[derive(Debug)]
pub struct Event {
    handle: vk::Event,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,
    must_put_in_pool: bool,

    flags: EventCreateFlags,
}

impl Event {
    /// Creates a new `Event`, panicking on a validation error.
    ///
    /// On [portability subset] devices, the [`events`] feature must be enabled on the device.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [portability subset]: crate::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`events`]: crate::device::DeviceFeatures::events
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        create_info: &EventCreateInfo<'_>,
    ) -> Result<Event, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `Event`.
    ///
    /// On [portability subset] devices, the [`events`] feature must be enabled on the device.
    ///
    /// [portability subset]: crate::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`events`]: crate::device::DeviceFeatures::events
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &EventCreateInfo<'_>,
    ) -> Result<Event, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &EventCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if device.enabled_extensions().khr_portability_subset && !device.enabled_features().events {
            return Err(Box::new(ValidationError {
                problem: "this device is a portability subset device".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "events",
                )])]),
                vuids: &["VUID-vkCreateEvent-events-04468"],
                ..Default::default()
            }));
        }

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &EventCreateInfo<'_>,
    ) -> Result<Event, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let mut output = MaybeUninit::uninit();
            let fns = device.fns();
            unsafe {
                (fns.v1_0.create_event)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(device, handle, create_info) })
    }

    /// Takes an event from the vulkano-provided event pool.
    /// If the pool is empty, a new event will be allocated.
    /// Upon `drop`, the event is put back into the pool.
    ///
    /// For most applications, using the event pool should be preferred,
    /// in order to avoid creating new events every frame.
    #[inline]
    pub fn from_pool(device: &Arc<Device>) -> Result<Event, VulkanError> {
        let handle = device.event_pool().lock().pop();
        let event = match handle {
            Some(handle) => {
                // Make sure the event isn't signaled
                let fns = device.fns();
                unsafe {
                    (fns.v1_0.reset_event)(device.handle(), handle)
                        .result()
                        .map_err(VulkanError::from)
                }?;

                Event {
                    handle,
                    device: InstanceOwnedDebugWrapper(device.clone()),
                    id: Self::next_id(),
                    must_put_in_pool: true,

                    flags: EventCreateFlags::empty(),
                }
            }
            None => {
                // Pool is empty, alloc new event
                let mut event = unsafe { Event::new_unchecked(device, &Default::default()) }?;
                event.must_put_in_pool = true;
                event
            }
        };

        Ok(event)
    }

    /// Creates a new `Event` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Event,
        create_info: &EventCreateInfo<'_>,
    ) -> Event {
        let &EventCreateInfo { flags, _ne: _ } = create_info;

        Event {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),
            must_put_in_pool: false,
            flags,
        }
    }

    /// Returns the flags that the event was created with.
    #[inline]
    pub fn flags(&self) -> EventCreateFlags {
        self.flags
    }

    /// Returns true if the event is signaled, panicking on a validation error.
    ///
    /// This is a shortcut for `try_is_signaled().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_is_signaled`] returns a [`ValidationError`].
    ///
    /// [`try_is_signaled`]: Self::try_is_signaled
    #[inline]
    #[track_caller]
    pub fn is_signaled(&self) -> Result<bool, VulkanError> {
        match self.try_is_signaled() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Returns true if the event is signaled.
    #[inline]
    pub fn try_is_signaled(&self) -> Result<bool, Validated<VulkanError>> {
        self.validate_is_signaled()?;

        Ok(unsafe { self.is_signaled_unchecked() }?)
    }

    fn validate_is_signaled(&self) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn is_signaled_unchecked(&self) -> Result<bool, VulkanError> {
        let fns = self.device.fns();
        let result = unsafe { (fns.v1_0.get_event_status)(self.device.handle(), self.handle) };
        match result {
            vk::Result::EVENT_SET => Ok(true),
            vk::Result::EVENT_RESET => Ok(false),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Changes the `Event` to the signaled state, panicking on a validation error.
    ///
    /// If a command buffer is waiting on this event, it is then unblocked.
    ///
    /// This is a shortcut for `try_set().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_set`] returns a [`ValidationError`].
    ///
    /// [`try_set`]: Self::try_set
    #[track_caller]
    pub fn set(&mut self) -> Result<(), VulkanError> {
        match self.try_set() {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Changes the `Event` to the signaled state.
    ///
    /// If a command buffer is waiting on this event, it is then unblocked.
    pub fn try_set(&mut self) -> Result<(), Validated<VulkanError>> {
        self.validate_set()?;

        Ok(unsafe { self.set_unchecked() }?)
    }

    fn validate_set(&mut self) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn set_unchecked(&mut self) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.set_event)(self.device.handle(), self.handle) }
            .result()
            .map_err(VulkanError::from)?;
        Ok(())
    }

    /// Changes the `Event` to the unsignaled state, panicking on a validation error.
    ///
    /// This is a shortcut for `try_reset().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - There must be an execution dependency between `reset` and the execution of any
    ///   [`wait_events`] command that includes this event in its `events` parameter.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_reset`] returns a [`ValidationError`].
    ///
    /// [`try_reset`]: Self::try_reset
    /// [`wait_events`]: crate::command_buffer::RecordingCommandBuffer::wait_events
    #[inline]
    #[track_caller]
    pub unsafe fn reset(&mut self) -> Result<(), VulkanError> {
        match unsafe { self.try_reset() } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Changes the `Event` to the unsignaled state.
    ///
    /// # Safety
    ///
    /// - There must be an execution dependency between `reset` and the execution of any
    ///   [`wait_events`] command that includes this event in its `events` parameter.
    ///
    /// [`wait_events`]: crate::command_buffer::RecordingCommandBuffer::wait_events
    #[inline]
    pub unsafe fn try_reset(&mut self) -> Result<(), Validated<VulkanError>> {
        self.validate_reset()?;

        Ok(unsafe { self.reset_unchecked() }?)
    }

    fn validate_reset(&mut self) -> Result<(), Box<ValidationError>> {
        // VUID-vkResetEvent-event-03821
        // VUID-vkResetEvent-event-03822
        // Unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn reset_unchecked(&mut self) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.reset_event)(self.device.handle(), self.handle) }
            .result()
            .map_err(VulkanError::from)?;
        Ok(())
    }
}

impl Drop for Event {
    #[inline]
    fn drop(&mut self) {
        if self.must_put_in_pool {
            let raw_event = self.handle;
            self.device.event_pool().lock().push(raw_event);
        } else {
            let fns = self.device.fns();
            unsafe { (fns.v1_0.destroy_event)(self.device.handle(), self.handle, ptr::null()) };
        }
    }
}

unsafe impl VulkanObject for Event {
    type Handle = vk::Event;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Event {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Event);

/// Parameters to create a new `Event`.
#[derive(Clone, Debug)]
pub struct EventCreateInfo<'a> {
    /// Additional properties of the event.
    ///
    /// The default value is empty.
    pub flags: EventCreateFlags,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for EventCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl EventCreateInfo<'_> {
    /// Returns a default `EventCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: EventCreateFlags::empty(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self { flags, _ne: _ } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkEventCreateInfo-flags-parameter"])
        })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::EventCreateInfo<'static> {
        let &Self { flags, _ne: _ } = self;

        vk::EventCreateInfo::default().flags(flags.into())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of an event.
    EventCreateFlags = EventCreateFlags(u32);

    DEVICE_ONLY = DEVICE_ONLY
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),
}

#[cfg(test)]
mod tests {
    use crate::{sync::event::Event, VulkanObject};

    #[test]
    fn event_create() {
        let (device, _) = gfx_dev_and_queue!();
        let event = Event::new(&device, &Default::default()).unwrap();
        assert!(!event.is_signaled().unwrap());
    }

    #[test]
    fn event_set() {
        let (device, _) = gfx_dev_and_queue!();
        let mut event = Event::new(&device, &Default::default()).unwrap();
        assert!(!event.is_signaled().unwrap());

        event.set().unwrap();
        assert!(event.is_signaled().unwrap());
    }

    #[test]
    fn event_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let mut event = Event::new(&device, &Default::default()).unwrap();
        event.set().unwrap();
        assert!(event.is_signaled().unwrap());

        unsafe { event.reset().unwrap() };
        assert!(!event.is_signaled().unwrap());
    }

    #[test]
    fn event_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.event_pool().lock().len(), 0);
        let event1_internal_obj = {
            let event = Event::from_pool(&device).unwrap();
            assert_eq!(device.event_pool().lock().len(), 0);
            event.handle()
        };

        assert_eq!(device.event_pool().lock().len(), 1);
        let event2 = Event::from_pool(&device).unwrap();
        assert_eq!(device.event_pool().lock().len(), 0);
        assert_eq!(event2.handle(), event1_internal_obj);
    }
}
