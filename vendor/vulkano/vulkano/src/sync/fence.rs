//! A fence provides synchronization between the device and the host, or between an external source
//! and the host.
//!
//! A fence has two states: **signaled** and **unsignaled**.
//!
//! The device can only perform one operation on a fence:
//! - A **fence signal operation** will put the fence into the signaled state.
//!
//! The host can poll a fence's status, wait for it to become signaled, or reset the fence back
//! to the unsignaled state.
//!
//! # Queue-to-host synchronization
//!
//! The primary use of a fence is to know when a queue operation has completed executing.
//! When adding a command to a queue, a fence can be provided with the command, to be signaled
//! when the operation finishes. You can check for a fence's current status by calling
//! `is_signaled`, `wait` or `await` on it. If the fence is found to be signaled, that means that
//! the queue has completed the operation that is associated with the fence, and all operations
//! that happened-before it have been completed as well.
//!
//! # Safety
//!
//! - There must never be more than one fence signal operation queued at any given time.
//! - The fence must be unsignaled at the time the function (for example [`submit`]) is called.
//!
//! [`submit`]: crate::device::QueueGuard::submit

use crate::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_bitflags_enum},
    self_referential::borrow_wrapper_impls,
    RawFd, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError, VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use smallvec::SmallVec;
use std::{
    future::Future,
    mem::MaybeUninit,
    num::NonZero,
    pin::Pin,
    ptr,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

/// A two-state synchronization primitive that is signaled by the device and waited on by the
/// host.
#[derive(Debug)]
pub struct Fence {
    handle: vk::Fence,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    flags: FenceCreateFlags,
    export_handle_types: ExternalFenceHandleTypes,

    must_put_in_pool: bool,
}

impl Fence {
    /// Creates a new `Fence`, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        create_info: &FenceCreateInfo<'_>,
    ) -> Result<Fence, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new `Fence`.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &FenceCreateInfo<'_>,
    ) -> Result<Fence, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &FenceCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &FenceCreateInfo<'_>,
    ) -> Result<Fence, VulkanError> {
        let mut create_info_extensions_vk = create_info.to_vk_extensions();
        let create_info_vk = create_info.to_vk(&mut create_info_extensions_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_fence)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
                .result()
            }
            .map_err(VulkanError::from)?;

            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle(device, handle, create_info) })
    }

    /// Takes a fence from the vulkano-provided fence pool.
    /// If the pool is empty, a new fence will be created.
    /// Upon `drop`, the fence is put back into the pool.
    ///
    /// For most applications, using the fence pool should be preferred,
    /// in order to avoid creating new fences every frame.
    #[inline]
    pub fn from_pool(device: &Arc<Device>) -> Result<Fence, VulkanError> {
        let handle = device.fence_pool().lock().pop();
        let fence = match handle {
            Some(handle) => {
                // Make sure the fence isn't signaled
                let fns = device.fns();
                unsafe { (fns.v1_0.reset_fences)(device.handle(), 1, &handle) }
                    .result()
                    .map_err(VulkanError::from)?;

                Fence {
                    handle,
                    device: InstanceOwnedDebugWrapper(device.clone()),
                    id: Self::next_id(),

                    flags: FenceCreateFlags::empty(),
                    export_handle_types: ExternalFenceHandleTypes::empty(),

                    must_put_in_pool: true,
                }
            }
            None => {
                // Pool is empty, alloc new fence
                let mut fence =
                    unsafe { Fence::new_unchecked(device, &FenceCreateInfo::default()) }?;
                fence.must_put_in_pool = true;
                fence
            }
        };

        Ok(fence)
    }

    /// Creates a new `Fence` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::Fence,
        create_info: &FenceCreateInfo<'_>,
    ) -> Fence {
        let &FenceCreateInfo {
            flags,
            export_handle_types,
            _ne: _,
        } = create_info;

        Fence {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),

            flags,
            export_handle_types,

            must_put_in_pool: false,
        }
    }

    /// Returns the flags that the fence was created with.
    #[inline]
    pub fn flags(&self) -> FenceCreateFlags {
        self.flags
    }

    /// Returns the handle types that can be exported from the fence.
    #[inline]
    pub fn export_handle_types(&self) -> ExternalFenceHandleTypes {
        self.export_handle_types
    }

    /// Returns true if the fence is signaled, panicking on a validation error.
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

    /// Returns true if the fence is signaled.
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
        let result = unsafe { (fns.v1_0.get_fence_status)(self.device.handle(), self.handle) };
        match result {
            vk::Result::SUCCESS => Ok(true),
            vk::Result::NOT_READY => Ok(false),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Waits until the fence is signaled, or at least until the timeout duration has elapsed,
    /// panicking on a validation error.
    ///
    /// If you pass a duration of 0, then the function will return without blocking.
    ///
    /// This is a shortcut for `try_wait().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_wait`] returns a [`ValidationError`].
    ///
    /// [`try_wait`]: Self::try_wait
    #[track_caller]
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), VulkanError> {
        match self.try_wait(timeout) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Waits until the fence is signaled, or at least until the timeout duration has elapsed,
    /// returning an error on failure.
    ///
    /// If you pass a duration of 0, then the function will return without blocking.
    pub fn try_wait(&self, timeout: Option<Duration>) -> Result<(), Validated<VulkanError>> {
        self.validate_wait(timeout)?;

        Ok(unsafe { self.wait_unchecked(timeout) }?)
    }

    fn validate_wait(&self, _timeout: Option<Duration>) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_unchecked(&self, timeout: Option<Duration>) -> Result<(), VulkanError> {
        let timeout_ns = timeout.map_or(u64::MAX, |timeout| {
            timeout
                .as_secs()
                .saturating_mul(1_000_000_000)
                .saturating_add(timeout.subsec_nanos() as u64)
        });

        let fns = self.device.fns();
        let result = unsafe {
            (fns.v1_0.wait_for_fences)(self.device.handle(), 1, &self.handle, vk::TRUE, timeout_ns)
        };

        match result {
            vk::Result::SUCCESS => Ok(()),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Waits for many fences at once, panicking on a validation error.
    ///
    /// This is a shortcut for `try_wait_many().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_wait_many`] returns a [`ValidationError`].
    /// - Panics if not all fences belong to the same device.
    ///
    /// [`try_wait_many`]: Self::try_wait_many
    #[track_caller]
    pub fn wait_many(fences: &[&Fence], timeout: Option<Duration>) -> Result<(), VulkanError> {
        match Self::try_wait_many(fences, timeout) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Waits for many fences at once.
    ///
    /// # Panics
    ///
    /// - Panics if not all fences belong to the same device.
    pub fn try_wait_many(
        fences: &[&Fence],
        timeout: Option<Duration>,
    ) -> Result<(), Validated<VulkanError>> {
        Self::validate_wait_many(fences, timeout)?;

        Ok(unsafe { Self::wait_many_unchecked(fences, timeout) }?)
    }

    fn validate_wait_many(
        fences: &[&Fence],
        _timeout: Option<Duration>,
    ) -> Result<(), Box<ValidationError>> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;

        for fence in fences {
            // VUID-vkWaitForFences-pFences-parent
            assert_eq!(device, &fence.device);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_many_unchecked(
        fences: &[&Fence],
        timeout: Option<Duration>,
    ) -> Result<(), VulkanError> {
        let iter = fences.iter();
        let mut fences_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut fences: SmallVec<[_; 8]> = SmallVec::new();

        for fence in iter {
            fences_vk.push(fence.handle);
            fences.push(fence);
        }

        // VUID-vkWaitForFences-fenceCount-arraylength
        // If there are no fences, we don't need to wait.
        if fences_vk.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;
        let timeout_ns = timeout.map_or(u64::MAX, |timeout| {
            timeout
                .as_secs()
                .saturating_mul(1_000_000_000)
                .saturating_add(timeout.subsec_nanos() as u64)
        });

        let result = {
            let fns = device.fns();
            unsafe {
                (fns.v1_0.wait_for_fences)(
                    device.handle(),
                    fences_vk.len() as u32,
                    fences_vk.as_ptr(),
                    vk::TRUE, // TODO: let the user choose false here?
                    timeout_ns,
                )
            }
        };

        match result {
            vk::Result::SUCCESS => Ok(()),
            err => Err(VulkanError::from(err)),
        }
    }

    /// Resets the fence, panicking on a validation error.
    ///
    /// This is a shortcut for `try_reset().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_reset`] returns a [`ValidationError`].
    ///
    /// [`try_reset`]: Self::try_reset
    #[inline]
    #[track_caller]
    pub unsafe fn reset(&self) -> Result<(), VulkanError> {
        match unsafe { self.try_reset() } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Resets the fence.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    #[inline]
    pub unsafe fn try_reset(&self) -> Result<(), Validated<VulkanError>> {
        self.validate_reset()?;

        Ok(unsafe { self.reset_unchecked() }?)
    }

    fn validate_reset(&self) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_unchecked(&self) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.reset_fences)(self.device.handle(), 1, &self.handle) }
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Resets many fences at once, panicking on a validation error.
    ///
    /// This is a shortcut for `try_reset_many().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - The elements of `fences` must not be in use by the device.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_reset_many`] returns a [`ValidationError`].
    ///
    /// [`try_reset_many`]: Self::try_reset_many
    #[track_caller]
    pub unsafe fn reset_many(fences: &[&Fence]) -> Result<(), VulkanError> {
        match unsafe { Self::try_reset_many(fences) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Resets many fences at once.
    ///
    /// # Safety
    ///
    /// - The elements of `fences` must not be in use by the device.
    pub unsafe fn try_reset_many(fences: &[&Fence]) -> Result<(), Validated<VulkanError>> {
        Self::validate_reset_many(fences)?;

        Ok(unsafe { Self::reset_many_unchecked(fences) }?)
    }

    fn validate_reset_many(fences: &[&Fence]) -> Result<(), Box<ValidationError>> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;

        for fence in fences {
            // VUID-vkResetFences-pFences-parent
            assert_eq!(device, &fence.device);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_many_unchecked(fences: &[&Fence]) -> Result<(), VulkanError> {
        if fences.is_empty() {
            return Ok(());
        }

        let device = &fences[0].device;
        let fences_vk: SmallVec<[_; 8]> = fences.iter().map(|fence| fence.handle).collect();

        let fns = device.fns();
        unsafe {
            (fns.v1_0.reset_fences)(device.handle(), fences_vk.len() as u32, fences_vk.as_ptr())
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Exports the fence into a POSIX file descriptor, panicking on a validation error. The caller
    /// owns the returned file descriptor.
    ///
    /// The [`khr_external_fence_fd`] extension must be enabled on the device.
    ///
    /// This is a shortcut for `try_export_fd().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the fence must be signaled, or a signal
    ///   operation on the fence must be pending.
    /// - The fence must not currently have an imported payload from a swapchain acquire operation.
    /// - If the fence has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_fence_properties`].
    ///
    /// # Panics
    ///
    /// - Panics if [`try_export_fd`] returns an error.
    ///
    /// [`khr_external_fence_fd`]: crate::device::DeviceExtensions::khr_external_fence_fd
    /// [`try_export_fd`]: Self::try_export_fd
    #[inline]
    #[track_caller]
    pub unsafe fn export_fd(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<RawFd, VulkanError> {
        match unsafe { self.try_export_fd(handle_type) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Exports the fence into a POSIX file descriptor. The caller owns the returned file
    /// descriptor.
    ///
    /// The [`khr_external_fence_fd`] extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the fence must be signaled, or a signal
    ///   operation on the fence must be pending.
    /// - The fence must not currently have an imported payload from a swapchain acquire operation.
    /// - If the fence has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_fence_properties`].
    ///
    /// [`khr_external_fence_fd`]: crate::device::DeviceExtensions::khr_external_fence_fd
    #[inline]
    pub unsafe fn try_export_fd(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<RawFd, Validated<VulkanError>> {
        self.validate_export_fd(handle_type)?;

        Ok(unsafe { self.export_fd_unchecked(handle_type) }?)
    }

    fn validate_export_fd(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_fd {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_fd",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkFenceGetFdInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueFd | ExternalFenceHandleType::SyncFd
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueFd` or \
                    `ExternalFenceHandleType::SyncFd`"
                    .into(),
                vuids: &["VUID-VkFenceGetFdInfoKHR-handleType-01456"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(Box::new(ValidationError {
                problem: "`self.export_handle_types()` does not contain `handle_type`".into(),
                vuids: &["VUID-VkFenceGetFdInfoKHR-handleType-01453"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_fd_unchecked(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<RawFd, VulkanError> {
        let info_vk = vk::FenceGetFdInfoKHR::default()
            .fence(self.handle)
            .handle_type(handle_type.into());

        let fd = {
            let mut output = MaybeUninit::uninit();
            let fns = self.device.fns();
            unsafe {
                (fns.khr_external_fence_fd.get_fence_fd_khr)(
                    self.device.handle(),
                    &info_vk,
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            unsafe { output.assume_init() }
        };

        Ok(fd)
    }

    /// Exports the fence into a Win32 handle, panicking on a validation error.
    ///
    /// The [`khr_external_fence_win32`] extension must be enabled on the device.
    ///
    /// This is a shortcut for `try_export_win32_handle().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the fence must be signaled, or a signal
    ///   operation on the fence must be pending.
    /// - The fence must not currently have an imported payload from a swapchain acquire operation.
    /// - If the fence has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_fence_properties`].
    /// - If `handle_type` is `ExternalFenceHandleType::OpaqueWin32`, then a handle of this type
    ///   must not have been already exported from this fence.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_export_win32_handle`] returns a [`ValidationError`].
    ///
    /// [`khr_external_fence_win32`]: crate::device::DeviceExtensions::khr_external_fence_win32
    /// [`try_export_win32_handle`]: Self::try_export_win32_handle
    #[inline]
    #[track_caller]
    pub fn export_win32_handle(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<vk::HANDLE, VulkanError> {
        match self.try_export_win32_handle(handle_type) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Exports the fence into a Win32 handle.
    ///
    /// The [`khr_external_fence_win32`] extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - If `handle_type` has copy transference, then the fence must be signaled, or a signal
    ///   operation on the fence must be pending.
    /// - The fence must not currently have an imported payload from a swapchain acquire operation.
    /// - If the fence has an imported payload, its handle type must allow re-exporting as
    ///   `handle_type`, as returned by [`PhysicalDevice::external_fence_properties`].
    /// - If `handle_type` is `ExternalFenceHandleType::OpaqueWin32`, then a handle of this type
    ///   must not have been already exported from this fence.
    ///
    /// [`khr_external_fence_win32`]: crate::device::DeviceExtensions::khr_external_fence_win32
    #[inline]
    pub fn try_export_win32_handle(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<vk::HANDLE, Validated<VulkanError>> {
        self.validate_export_win32_handle(handle_type)?;

        Ok(unsafe { self.export_win32_handle_unchecked(handle_type) }?)
    }

    fn validate_export_win32_handle(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_win32 {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_win32",
                )])]),
                ..Default::default()
            }));
        }

        handle_type.validate_device(&self.device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkFenceGetWin32HandleInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueWin32 | ExternalFenceHandleType::OpaqueWin32Kmt
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueWin32` or \
                    `ExternalFenceHandleType::OpaqueWin32Kmt`"
                    .into(),
                vuids: &["VUID-VkFenceGetWin32HandleInfoKHR-handleType-01452"],
                ..Default::default()
            }));
        }

        if !self.export_handle_types.intersects(handle_type.into()) {
            return Err(Box::new(ValidationError {
                problem: "`self.export_handle_types()` does not contain `handle_type`".into(),
                vuids: &["VUID-VkFenceGetWin32HandleInfoKHR-handleType-01448"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn export_win32_handle_unchecked(
        &self,
        handle_type: ExternalFenceHandleType,
    ) -> Result<vk::HANDLE, VulkanError> {
        let info_vk = vk::FenceGetWin32HandleInfoKHR::default()
            .fence(self.handle)
            .handle_type(handle_type.into());

        let handle = {
            let mut output = MaybeUninit::uninit();
            let fns = self.device.fns();
            unsafe {
                (fns.khr_external_fence_win32.get_fence_win32_handle_khr)(
                    self.device.handle(),
                    &info_vk,
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(handle)
    }

    /// Imports a fence from a POSIX file descriptor, panicking on a validation error.
    ///
    /// The [`khr_external_fence_fd`] extension must be enabled on the device.
    ///
    /// This is a shortcut for `try_import_fd().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    /// - If in `import_fence_fd_info`, `handle_type` is `ExternalHandleType::OpaqueFd`, then `fd`
    ///   must represent a fence that was exported from Vulkan or a compatible API, with a driver
    ///   and device UUID equal to those of the device that owns `self`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_import_fd`] returns a [`ValidationError`].
    ///
    /// [`khr_external_fence_fd`]: crate::device::DeviceExtensions::khr_external_fence_fd
    /// [`try_import_fd`]: Self::try_import_fd
    #[inline]
    #[track_caller]
    pub unsafe fn import_fd(
        &self,
        import_fence_fd_info: &ImportFenceFdInfo<'_>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_import_fd(import_fence_fd_info) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Imports a fence from a POSIX file descriptor.
    ///
    /// The [`khr_external_fence_fd`] extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    /// - If in `import_fence_fd_info`, `handle_type` is `ExternalHandleType::OpaqueFd`, then `fd`
    ///   must represent a fence that was exported from Vulkan or a compatible API, with a driver
    ///   and device UUID equal to those of the device that owns `self`.
    ///
    /// [`khr_external_fence_fd`]: crate::device::DeviceExtensions::khr_external_fence_fd
    #[inline]
    pub unsafe fn try_import_fd(
        &self,
        import_fence_fd_info: &ImportFenceFdInfo<'_>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_import_fd(import_fence_fd_info)?;

        Ok(unsafe { self.import_fd_unchecked(import_fence_fd_info) }?)
    }

    fn validate_import_fd(
        &self,
        import_fence_fd_info: &ImportFenceFdInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_fd {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_fd",
                )])]),
                ..Default::default()
            }));
        }

        import_fence_fd_info
            .validate(&self.device)
            .map_err(|err| err.add_context("import_fence_fd_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn import_fd_unchecked(
        &self,
        import_fence_fd_info: &ImportFenceFdInfo<'_>,
    ) -> Result<(), VulkanError> {
        let info_vk = import_fence_fd_info.to_vk(self.handle());

        let fns = self.device.fns();
        unsafe { (fns.khr_external_fence_fd.import_fence_fd_khr)(self.device.handle(), &info_vk) }
            .result()
            .map_err(VulkanError::from)?;

        Ok(())
    }

    /// Imports a fence from a Win32 handle, panicking on a validation error.
    ///
    /// The [`khr_external_fence_win32`] extension must be enabled on the device.
    ///
    /// This is a shortcut for `try_import_win32_handle().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    /// - In `import_fence_win32_handle_info`, `handle` must represent a fence that was exported
    ///   from Vulkan or a compatible API, with a driver and device UUID equal to those of the
    ///   device that owns `self`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_import_win32_handle`] returns a [`ValidationError`].
    ///
    /// [`khr_external_fence_win32`]: crate::device::DeviceExtensions::khr_external_fence_win32
    /// [`try_import_win32_handle`]: Self::try_import_win32_handle
    #[inline]
    #[track_caller]
    pub unsafe fn import_win32_handle(
        &self,
        import_fence_win32_handle_info: &ImportFenceWin32HandleInfo<'_>,
    ) -> Result<(), VulkanError> {
        match unsafe { self.try_import_win32_handle(import_fence_win32_handle_info) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Imports a fence from a Win32 handle.
    ///
    /// The [`khr_external_fence_win32`] extension must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - The fence must not be in use by the device.
    /// - In `import_fence_win32_handle_info`, `handle` must represent a fence that was exported
    ///   from Vulkan or a compatible API, with a driver and device UUID equal to those of the
    ///   device that owns `self`.
    ///
    /// [`khr_external_fence_win32`]: crate::device::DeviceExtensions::khr_external_fence_win32
    #[inline]
    pub unsafe fn try_import_win32_handle(
        &self,
        import_fence_win32_handle_info: &ImportFenceWin32HandleInfo<'_>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_import_win32_handle(import_fence_win32_handle_info)?;

        Ok(unsafe { self.import_win32_handle_unchecked(import_fence_win32_handle_info) }?)
    }

    fn validate_import_win32_handle(
        &self,
        import_fence_win32_handle_info: &ImportFenceWin32HandleInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !self.device.enabled_extensions().khr_external_fence_win32 {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_external_fence_win32",
                )])]),
                ..Default::default()
            }));
        }

        import_fence_win32_handle_info
            .validate(&self.device)
            .map_err(|err| err.add_context("import_fence_win32_handle_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn import_win32_handle_unchecked(
        &self,
        import_fence_win32_handle_info: &ImportFenceWin32HandleInfo<'_>,
    ) -> Result<(), VulkanError> {
        let info_vk = import_fence_win32_handle_info.to_vk(self.handle());

        let fns = self.device.fns();
        unsafe {
            (fns.khr_external_fence_win32.import_fence_win32_handle_khr)(
                self.device.handle(),
                &info_vk,
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    // Shared by Fence and FenceSignalFuture
    pub(crate) fn poll_impl(&self, cx: &mut Context<'_>) -> Poll<Result<(), VulkanError>> {
        // Vulkan only allows polling of the fence status, so we have to use a spin future.
        // This is still better than blocking in async applications, since a smart-enough async
        // engine can choose to run some other tasks between probing this one.

        // Check if we are done without blocking
        match unsafe { self.is_signaled_unchecked() } {
            Err(e) => return Poll::Ready(Err(e)),
            Ok(signaled) => {
                if signaled {
                    return Poll::Ready(Ok(()));
                }
            }
        }

        // Otherwise spin
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

impl Drop for Fence {
    #[inline]
    fn drop(&mut self) {
        if self.must_put_in_pool {
            let raw_fence = self.handle;
            self.device.fence_pool().lock().push(raw_fence);
        } else {
            let fns = self.device.fns();
            unsafe { (fns.v1_0.destroy_fence)(self.device.handle(), self.handle, ptr::null()) };
        }
    }
}

impl Future for Fence {
    type Output = Result<(), VulkanError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.poll_impl(cx)
    }
}

unsafe impl VulkanObject for Fence {
    type Handle = vk::Fence;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Fence {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(Fence);

/// Parameters to create a new `Fence`.
#[derive(Clone, Debug)]
pub struct FenceCreateInfo<'a> {
    /// Additional properties of the fence.
    ///
    /// The default value is empty.
    pub flags: FenceCreateFlags,

    /// The handle types that can be exported from the fence.
    pub export_handle_types: ExternalFenceHandleTypes,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for FenceCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> FenceCreateInfo<'a> {
    /// Returns a default `FenceCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: FenceCreateFlags::empty(),
            export_handle_types: ExternalFenceHandleTypes::empty(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            export_handle_types,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkFenceCreateInfo-flags-parameter"])
        })?;

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_fence)
            {
                return Err(Box::new(ValidationError {
                    context: "export_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_fence")]),
                    ]),
                    ..Default::default()
                }));
            }

            export_handle_types.validate_device(device).map_err(|err| {
                err.add_context("export_handle_types")
                    .set_vuids(&["VUID-VkExportFenceCreateInfo-handleTypes-parameter"])
            })?;

            for handle_type in export_handle_types.into_iter() {
                let external_fence_properties = unsafe {
                    device
                        .physical_device()
                        .external_fence_properties_unchecked(&ExternalFenceInfo::new(handle_type))
                };

                if !external_fence_properties.exportable {
                    return Err(Box::new(ValidationError {
                        context: "export_handle_types".into(),
                        problem: format!(
                            "the handle type `ExternalFenceHandleTypes::{:?}` is not exportable, \
                            as returned by `PhysicalDevice::external_fence_properties`",
                            ExternalFenceHandleTypes::from(handle_type)
                        )
                        .into(),
                        vuids: &["VUID-VkExportFenceCreateInfo-handleTypes-01446"],
                        ..Default::default()
                    }));
                }

                if !external_fence_properties
                    .compatible_handle_types
                    .contains(export_handle_types)
                {
                    return Err(Box::new(ValidationError {
                        context: "export_handle_types".into(),
                        problem: format!(
                            "the handle type `ExternalFenceHandleTypes::{:?}` is not compatible \
                            with the other specified handle types, as returned by \
                            `PhysicalDevice::external_fence_properties`",
                            ExternalFenceHandleTypes::from(handle_type)
                        )
                        .into(),
                        vuids: &["VUID-VkExportFenceCreateInfo-handleTypes-01446"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        extensions_vk: &'a mut FenceCreateInfoExtensionsVk,
    ) -> vk::FenceCreateInfo<'a> {
        let &Self {
            flags,
            export_handle_types: _,
            _ne: _,
        } = self;

        let mut val_vk = vk::FenceCreateInfo::default().flags(flags.into());

        let FenceCreateInfoExtensionsVk { export_vk } = extensions_vk;

        if let Some(next) = export_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_extensions(&self) -> FenceCreateInfoExtensionsVk {
        let &Self {
            flags: _,
            export_handle_types,
            _ne: _,
        } = self;

        let export_vk = (!export_handle_types.is_empty())
            .then(|| vk::ExportFenceCreateInfo::default().handle_types(export_handle_types.into()));

        FenceCreateInfoExtensionsVk { export_vk }
    }
}

pub(crate) struct FenceCreateInfoExtensionsVk {
    pub(crate) export_vk: Option<vk::ExportFenceCreateInfo<'static>>,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a fence.
    FenceCreateFlags = FenceCreateFlags(u32);

    /// Creates the fence in the signaled state.
    SIGNALED = SIGNALED,
}

vulkan_bitflags_enum! {
    #[non_exhaustive]
    /// A set of [`ExternalFenceHandleType`] values.
    ExternalFenceHandleTypes,

    /// The handle type used to export or import fences to/from an external source.
    ExternalFenceHandleType impl {
        /// Returns whether the given handle type has *copy transference* rather than *reference
        /// transference*.
        ///
        /// Imports of handles with copy transference must always be temporary. Exports of such
        /// handles must only occur if the fence is already signaled, or if there is a fence signal
        /// operation pending in a queue.
        #[inline]
        pub fn has_copy_transference(self) -> bool {
            // As defined by
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-fence-handletypes-win32
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-fence-handletypes-fd
            matches!(self, Self::SyncFd)
        }
    },

    = ExternalFenceHandleTypeFlags(u32);

    /// A POSIX file descriptor handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OPAQUE_FD, OpaqueFd = OPAQUE_FD,

    /// A Windows NT handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OPAQUE_WIN32, OpaqueWin32 = OPAQUE_WIN32,

    /// A Windows global share handle that is only usable with Vulkan and compatible APIs.
    ///
    /// This handle type has *reference transference*.
    OPAQUE_WIN32_KMT, OpaqueWin32Kmt = OPAQUE_WIN32_KMT,

    /// A POSIX file descriptor handle to a Linux Sync File or Android Fence object.
    ///
    /// This handle type has *copy transference*.
    SYNC_FD, SyncFd = SYNC_FD,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Additional parameters for a fence payload import.
    FenceImportFlags = FenceImportFlags(u32);

    /// The fence payload will be imported only temporarily, regardless of the permanence of the
    /// imported handle type.
    TEMPORARY = TEMPORARY,
}

#[derive(Debug)]
pub struct ImportFenceFdInfo<'a> {
    /// Additional parameters for the import operation.
    ///
    /// If `handle_type` has *copy transference*, this must include the `temporary` flag.
    ///
    /// The default value is [`FenceImportFlags::empty()`].
    pub flags: FenceImportFlags,

    /// The handle type of `fd`.
    ///
    /// There is no default value.
    pub handle_type: ExternalFenceHandleType,

    /// The file descriptor to import the fence from.
    ///
    /// If `handle_type` is `ExternalFenceHandleType::SyncFd`, then `fd` can be `None`. Instead of
    /// an imported file descriptor, a dummy file descriptor `-1` is used, which represents a fence
    /// that is always signaled.
    ///
    /// The default value is `None`, which must be overridden if `handle_type` is not
    /// `ExternalFenceHandleType::SyncFd`.
    pub fd: Option<RawFd>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl ImportFenceFdInfo<'_> {
    /// Returns a default `ImportFenceFdInfo` with the provided `handle_type`.
    #[inline]
    pub const fn new(handle_type: ExternalFenceHandleType) -> Self {
        Self {
            flags: FenceImportFlags::empty(),
            handle_type,
            fd: None,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn handle_type(handle_type: ExternalFenceHandleType) -> Self {
        Self::new(handle_type)
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            handle_type,
            fd: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImportFenceFdInfoKHR-flags-parameter"])
        })?;

        handle_type.validate_device(device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkImportFenceFdInfoKHR-handleType-parameter"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueFd | ExternalFenceHandleType::SyncFd
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueFd` or \
                    `ExternalFenceHandleType::SyncFd`"
                    .into(),
                vuids: &["VUID-VkImportFenceFdInfoKHR-handleType-01464"],
                ..Default::default()
            }));
        }

        // VUID-VkImportFenceFdInfoKHR-fd-01541
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.intersects(FenceImportFlags::TEMPORARY) {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `flags` does not contain `FenceImportFlags::TEMPORARY`"
                    .into(),
                vuids: &["VUID-VkImportFenceFdInfoKHR-handleType-07306"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self, fence_vk: vk::Fence) -> vk::ImportFenceFdInfoKHR<'static> {
        let &Self {
            flags,
            handle_type,
            fd,
            _ne: _,
        } = self;

        vk::ImportFenceFdInfoKHR::default()
            .fence(fence_vk)
            .flags(flags.into())
            .handle_type(handle_type.into())
            .fd(fd.unwrap_or(-1))
    }
}

#[derive(Debug)]
pub struct ImportFenceWin32HandleInfo<'a> {
    /// Additional parameters for the import operation.
    ///
    /// If `handle_type` has *copy transference*, this must include the `temporary` flag.
    ///
    /// The default value is [`FenceImportFlags::empty()`].
    pub flags: FenceImportFlags,

    /// The handle type of `handle`.
    ///
    /// There is no default value.
    pub handle_type: ExternalFenceHandleType,

    /// The file to import the fence from.
    ///
    /// The default value is `0`, which must be overridden.
    pub handle: vk::HANDLE,

    pub _ne: crate::NonExhaustive<'a>,
}

impl ImportFenceWin32HandleInfo<'_> {
    /// Returns a default `ImportFenceWin32HandleInfo` with the provided `handle_type`.
    #[inline]
    pub const fn new(handle_type: ExternalFenceHandleType) -> Self {
        Self {
            flags: FenceImportFlags::empty(),
            handle_type,
            handle: 0,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn handle_type(handle_type: ExternalFenceHandleType) -> Self {
        Self::new(handle_type)
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            handle_type,
            handle: _,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkImportFenceWin32HandleInfoKHR-flags-parameter"])
        })?;

        handle_type.validate_device(device).map_err(|err| {
            err.add_context("handle_type")
                .set_vuids(&["VUID-VkImportFenceWin32HandleInfoKHR-handleType-01457"])
        })?;

        if !matches!(
            handle_type,
            ExternalFenceHandleType::OpaqueWin32 | ExternalFenceHandleType::OpaqueWin32Kmt
        ) {
            return Err(Box::new(ValidationError {
                context: "handle_type".into(),
                problem: "is not `ExternalFenceHandleType::OpaqueWin32` or \
                    `ExternalFenceHandleType::OpaqueWin32Kmt`"
                    .into(),
                vuids: &["VUID-VkImportFenceWin32HandleInfoKHR-handleType-01457"],
                ..Default::default()
            }));
        }

        // VUID-VkImportFenceWin32HandleInfoKHR-handle-01539
        // Can't validate, therefore unsafe

        if handle_type.has_copy_transference() && !flags.intersects(FenceImportFlags::TEMPORARY) {
            return Err(Box::new(ValidationError {
                problem: "`handle_type` has copy transference, but \
                    `flags` does not contain `FenceImportFlags::TEMPORARY`"
                    .into(),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self, fence_vk: vk::Fence) -> vk::ImportFenceWin32HandleInfoKHR<'static> {
        let &Self {
            flags,
            handle_type,
            handle,
            _ne: _,
        } = self;

        vk::ImportFenceWin32HandleInfoKHR::default()
            .fence(fence_vk)
            .flags(flags.into())
            .handle_type(handle_type.into())
            .handle(handle)
        // .name() // TODO: support?
    }
}

/// The fence configuration to query in
/// [`PhysicalDevice::external_fence_properties`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalFenceInfo<'a> {
    /// The external handle type that will be used with the fence.
    pub handle_type: ExternalFenceHandleType,

    pub _ne: crate::NonExhaustive<'a>,
}

impl ExternalFenceInfo<'_> {
    /// Returns a default `ExternalFenceInfo` with the provided `handle_type`.
    #[inline]
    pub const fn new(handle_type: ExternalFenceHandleType) -> Self {
        Self {
            handle_type,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn handle_type(handle_type: ExternalFenceHandleType) -> Self {
        Self::new(handle_type)
    }

    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            handle_type,
            _ne: _,
        } = self;

        handle_type
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("handle_type")
                    .set_vuids(&["VUID-VkPhysicalDeviceExternalFenceInfo-handleType-parameter"])
            })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PhysicalDeviceExternalFenceInfo<'static> {
        let &Self {
            handle_type,
            _ne: _,
        } = self;

        vk::PhysicalDeviceExternalFenceInfo::default().handle_type(handle_type.into())
    }

    pub(crate) fn to_owned(&self) -> ExternalFenceInfo<'static> {
        ExternalFenceInfo {
            _ne: crate::NE,
            ..*self
        }
    }
}

borrow_wrapper_impls!(ExternalFenceInfo<'_>, PartialEq, Eq, Hash);

/// The properties for exporting or importing external handles, when a fence is created
/// with a specific configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalFenceProperties {
    /// Whether a handle can be exported to an external source with the queried
    /// external handle type.
    pub exportable: bool,

    /// Whether a handle can be imported from an external source with the queried
    /// external handle type.
    pub importable: bool,

    /// Which external handle types can be re-exported after the queried external handle type has
    /// been imported.
    pub export_from_imported_handle_types: ExternalFenceHandleTypes,

    /// Which external handle types can be enabled along with the queried external handle type
    /// when creating the fence.
    pub compatible_handle_types: ExternalFenceHandleTypes,
}

impl ExternalFenceProperties {
    pub(crate) fn to_mut_vk() -> vk::ExternalFenceProperties<'static> {
        vk::ExternalFenceProperties::default()
    }

    pub(crate) fn from_vk(val_vk: &vk::ExternalFenceProperties<'_>) -> Self {
        let &vk::ExternalFenceProperties {
            export_from_imported_handle_types,
            compatible_handle_types,
            external_fence_features,
            ..
        } = val_vk;

        ExternalFenceProperties {
            exportable: external_fence_features
                .intersects(vk::ExternalFenceFeatureFlags::EXPORTABLE),
            importable: external_fence_features
                .intersects(vk::ExternalFenceFeatureFlags::IMPORTABLE),
            export_from_imported_handle_types: export_from_imported_handle_types.into(),
            compatible_handle_types: compatible_handle_types.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        sync::fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        VulkanObject,
    };
    use std::time::Duration;

    #[test]
    fn fence_create() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(&device, &Default::default()).unwrap();
        assert!(!fence.is_signaled().unwrap());
    }

    #[test]
    fn fence_create_signaled() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            &device,
            &FenceCreateInfo {
                flags: FenceCreateFlags::SIGNALED,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fence.is_signaled().unwrap());
    }

    #[test]
    fn fence_signaled_wait() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            &device,
            &FenceCreateInfo {
                flags: FenceCreateFlags::SIGNALED,
                ..Default::default()
            },
        )
        .unwrap();
        fence.wait(Some(Duration::new(0, 10))).unwrap();
    }

    #[test]
    fn fence_reset() {
        let (device, _) = gfx_dev_and_queue!();

        let fence = Fence::new(
            &device,
            &FenceCreateInfo {
                flags: FenceCreateFlags::SIGNALED,
                ..Default::default()
            },
        )
        .unwrap();

        unsafe { fence.reset() }.unwrap();

        assert!(!fence.is_signaled().unwrap());
    }

    #[test]
    fn multiwait_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let fence1 = Fence::new(
                &device1,
                &FenceCreateInfo {
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
            .unwrap();
            let fence2 = Fence::new(
                &device2,
                &FenceCreateInfo {
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
            .unwrap();

            let _ = Fence::wait_many(&[&fence1, &fence2], Some(Duration::new(0, 10)));
        });
    }

    #[test]
    fn multireset_different_devices() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        assert_should_panic!({
            let fence1 = Fence::new(
                &device1,
                &FenceCreateInfo {
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
            .unwrap();
            let fence2 = Fence::new(
                &device2,
                &FenceCreateInfo {
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
            .unwrap();

            let _ = unsafe { Fence::reset_many(&[&fence1, &fence2]) };
        });
    }

    #[test]
    fn fence_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.fence_pool().lock().len(), 0);
        let fence1_internal_obj = {
            let fence = Fence::from_pool(&device).unwrap();
            assert_eq!(device.fence_pool().lock().len(), 0);
            fence.handle()
        };

        assert_eq!(device.fence_pool().lock().len(), 1);
        let fence2 = Fence::from_pool(&device).unwrap();
        assert_eq!(device.fence_pool().lock().len(), 0);
        assert_eq!(fence2.handle(), fence1_internal_obj);
    }
}
