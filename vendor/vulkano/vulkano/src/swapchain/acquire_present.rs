use super::{PresentMode, Swapchain};
use crate::{
    buffer::Buffer,
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageLayout},
    sync::{
        fence::Fence,
        future::{queue_present, AccessCheckError, AccessError, GpuFuture, SubmitAnyBuilder},
        semaphore::{Semaphore, SemaphoreType},
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError,
    VulkanObject,
};
use ash::vk;
use ash::vk::TaggedStructure;

use smallvec::{smallvec, SmallVec};
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    num::NonZero,
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

/// Parameters to acquire the next image from a swapchain.
#[derive(Clone, Debug)]
pub struct AcquireNextImageInfo<'a> {
    /// If no image is immediately available, how long to wait for one to become available.
    ///
    /// Specify `None` to wait forever. This is only allowed if at least one image in the swapchain
    /// has not been acquired yet.
    ///
    /// The default value is `None`.
    pub timeout: Option<Duration>,

    /// The semaphore to signal when an image has become available.
    ///
    /// `semaphore` and `fence` must not both be `None`.
    ///
    /// The default value is `None`.
    pub semaphore: Option<&'a Semaphore>,

    /// The fence to signal when an image has become available.
    ///
    /// `semaphore` and `fence` must not both be `None`.
    ///
    /// The default value is `None`.
    pub fence: Option<&'a Fence>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for AcquireNextImageInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AcquireNextImageInfo<'_> {
    /// Returns a default `AcquireNextImageInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            timeout: None,
            semaphore: None,
            fence: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            timeout,
            semaphore,
            fence,
            _ne: _,
        } = self;

        if let Some(timeout) = timeout {
            if timeout.as_nanos() >= u64::MAX as u128 {
                return Err(Box::new(ValidationError {
                    context: "timeout".into(),
                    problem: "is not less than `u64::MAX` nanoseconds".into(),
                    ..Default::default()
                }));
            }
        }

        if semaphore.is_none() && fence.is_none() {
            return Err(Box::new(ValidationError {
                problem: "`semaphore` and `fence` are both `None`".into(),
                vuids: &["VUID-VkAcquireNextImageInfoKHR-semaphore-01782"],
                ..Default::default()
            }));
        }

        if let Some(semaphore) = semaphore {
            // VUID-VkAcquireNextImageInfoKHR-commonparent
            assert_eq!(device, semaphore.device().as_ref());

            if semaphore.semaphore_type() != SemaphoreType::Binary {
                return Err(Box::new(ValidationError {
                    context: "semaphore.semaphore_type()".into(),
                    problem: "is not `SemaphoreType::Binary`".into(),
                    vuids: &["VUID-VkAcquireNextImageInfoKHR-semaphore-03266"],
                    ..Default::default()
                }));
            }
        }

        if let Some(fence) = fence {
            // VUID-VkAcquireNextImageInfoKHR-commonparent
            assert_eq!(device, fence.device().as_ref());
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        swapchain_vk: vk::SwapchainKHR,
        device_mask_vk: u32,
    ) -> vk::AcquireNextImageInfoKHR<'static> {
        let &Self {
            timeout,
            semaphore,
            fence,
            _ne: _,
        } = self;

        vk::AcquireNextImageInfoKHR::default()
            .swapchain(swapchain_vk)
            .timeout(timeout.map_or(u64::MAX, |duration| {
                u64::try_from(duration.as_nanos()).unwrap()
            }))
            .semaphore(
                semaphore
                    .as_ref()
                    .map(VulkanObject::handle)
                    .unwrap_or_default(),
            )
            .fence(fence.as_ref().map(VulkanObject::handle).unwrap_or_default())
            .device_mask(device_mask_vk)
    }
}

/// Tries to take ownership of an image in order to draw on it.
///
/// The function returns the index of the image in the array of images that was returned
/// when creating the swapchain, plus a future that represents the moment when the image will
/// become available from the GPU (which may not be *immediately*).
///
/// If you try to draw on an image without acquiring it first, the execution will block. (TODO
/// behavior may change).
///
/// The second field in the tuple in the Ok result is a bool represent if the acquisition was
/// suboptimal. In this case the acquired image is still usable, but the swapchain should be
/// recreated as the Surface's properties no longer match the swapchain.
pub fn acquire_next_image(
    swapchain: Arc<Swapchain>,
    timeout: Option<Duration>,
) -> Result<(u32, bool, SwapchainAcquireFuture), Validated<VulkanError>> {
    let semaphore = Arc::new(Semaphore::from_pool(&swapchain.device)?);
    let fence = Arc::new(Fence::from_pool(&swapchain.device)?);

    let AcquiredImage {
        image_index,
        is_suboptimal,
    } = unsafe {
        swapchain.try_acquire_next_image(&AcquireNextImageInfo {
            timeout,
            semaphore: Some(&semaphore),
            fence: Some(&fence),
            ..Default::default()
        })
    }?;

    Ok((
        image_index,
        is_suboptimal,
        SwapchainAcquireFuture {
            swapchain,
            semaphore: Some(semaphore),
            fence: Some(fence),
            image_index,
            finished: AtomicBool::new(false),
        },
    ))
}

/// Unsafe variant of `acquire_next_image`.
///
/// # Safety
///
/// - The semaphore and/or the fence must be kept alive until it is signaled.
/// - The swapchain must not have been replaced by being passed as the old swapchain when creating
///   a new one.
#[deprecated(
    since = "0.35.0",
    note = "use `Swapchain::acquire_next_image_unchecked` instead"
)]
pub unsafe fn acquire_next_image_raw(
    swapchain: &Swapchain,
    timeout: Option<Duration>,
    semaphore: Option<&Semaphore>,
    fence: Option<&Fence>,
) -> Result<AcquiredImage, Validated<VulkanError>> {
    let fns = swapchain.device.fns();

    let timeout_ns = if let Some(timeout) = timeout {
        timeout
            .as_secs()
            .saturating_mul(1_000_000_000)
            .saturating_add(timeout.subsec_nanos() as u64)
    } else {
        u64::MAX
    };

    let (image_index, is_suboptimal) = {
        let mut output = MaybeUninit::uninit();
        let result = unsafe {
            (fns.khr_swapchain.acquire_next_image_khr)(
                swapchain.device.handle(),
                swapchain.handle,
                timeout_ns,
                semaphore
                    .map(|s| s.handle())
                    .unwrap_or(vk::Semaphore::null()),
                fence.map(|f| f.handle()).unwrap_or(vk::Fence::null()),
                output.as_mut_ptr(),
            )
        };

        match result {
            vk::Result::SUCCESS => (unsafe { output.assume_init() }, false),
            vk::Result::SUBOPTIMAL_KHR => (unsafe { output.assume_init() }, true),
            vk::Result::NOT_READY => return Err(VulkanError::NotReady.into()),
            vk::Result::TIMEOUT => return Err(VulkanError::Timeout.into()),
            err => return Err(VulkanError::from(err).into()),
        }
    };

    Ok(AcquiredImage {
        image_index,
        is_suboptimal,
    })
}

/// An image that will be acquired from a swapchain.
#[derive(Clone, Copy, Debug)]
pub struct AcquiredImage {
    /// The index of the swapchain image that will be acquired.
    pub image_index: u32,

    /// Whether the swapchain is no longer optimally configured to present to its surface.
    ///
    /// If this is `true`, then the acquired image will still be usable, but the swapchain should
    /// be recreated soon to ensure an optimal configuration.
    pub is_suboptimal: bool,
}

/// Represents the moment when the GPU will have access to a swapchain image.
#[must_use]
pub struct SwapchainAcquireFuture {
    swapchain: Arc<Swapchain>,
    image_index: u32,
    // Semaphore that is signaled when the acquire is complete. Empty if the acquire has already
    // happened.
    semaphore: Option<Arc<Semaphore>>,
    // Fence that is signaled when the acquire is complete. Empty if the acquire has already
    // happened.
    fence: Option<Arc<Fence>>,
    finished: AtomicBool,
}

impl SwapchainAcquireFuture {
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    pub fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Returns the corresponding swapchain.
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }

    /// Blocks the current thread until the swapchain image has been acquired, or timeout
    ///
    /// If timeout is `None`, will potentially block forever
    ///
    /// You still need to join with this future for present to work
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), VulkanError> {
        match &self.fence {
            Some(fence) => unsafe { fence.wait_unchecked(timeout) },
            None => Ok(()),
        }
    }
}

unsafe impl GpuFuture for SwapchainAcquireFuture {
    fn cleanup_finished(&mut self) {}

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        if let Some(ref semaphore) = self.semaphore {
            let sem = smallvec![semaphore.clone()];
            Ok(SubmitAnyBuilder::SemaphoresWait(sem))
        } else {
            Ok(SubmitAnyBuilder::Empty)
        }
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        Ok(())
    }

    unsafe fn signal_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
    }

    fn queue_change_allowed(&self) -> bool {
        true
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        None
    }

    fn check_buffer_access(
        &self,
        _buffer: &Buffer,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        _queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    fn check_image_access(
        &self,
        image: &Image,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        expected_layout: ImageLayout,
        _queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        if self.swapchain.index_of_image(image) != Some(self.image_index) {
            return Err(AccessCheckError::Unknown);
        }

        if !self.swapchain.images[self.image_index as usize]
            .layout_initialized
            .load(Ordering::Relaxed)
            && expected_layout != ImageLayout::Undefined
        {
            return Err(AccessCheckError::Denied(AccessError::ImageNotInitialized {
                requested: expected_layout,
            }));
        }

        if expected_layout != ImageLayout::Undefined && expected_layout != ImageLayout::PresentSrc {
            return Err(AccessCheckError::Denied(
                AccessError::UnexpectedImageLayout {
                    allowed: ImageLayout::PresentSrc,
                    requested: expected_layout,
                },
            ));
        }

        Ok(())
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        if before {
            Ok(())
        } else {
            if swapchain == self.swapchain.as_ref() && image_index == self.image_index {
                Ok(())
            } else {
                Err(AccessCheckError::Unknown)
            }
        }
    }
}

impl Drop for SwapchainAcquireFuture {
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }

        if let Some(fence) = &self.fence {
            fence.wait(None).unwrap(); // TODO: handle error?
            self.semaphore = None;
        }

        // TODO: if this future is destroyed without being presented, then eventually acquiring
        // a new image will block forever ; difficulty: hard
    }
}

unsafe impl DeviceOwned for SwapchainAcquireFuture {
    fn device(&self) -> &Arc<Device> {
        &self.swapchain.device
    }
}

/// Presents an image on the screen.
///
/// The actual behavior depends on the present mode that you passed when creating the swapchain.
pub fn present<F>(
    before: F,
    queue: Arc<Queue>,
    swapchain_info: SwapchainPresentInfo,
) -> PresentFuture<F>
where
    F: GpuFuture,
{
    assert!(swapchain_info.image_index < swapchain_info.swapchain.image_count());

    // TODO: restore this check with a dummy ImageAccess implementation
    /*let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
    // Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
    // function on the image instead. But since we know that this method on `SwapchainImage`
    // always returns false anyway (by design), we don't need to do it.
    assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead*/

    PresentFuture {
        previous: before,
        queue,
        swapchain_info,
        flushed: AtomicBool::new(false),
        finished: AtomicBool::new(false),
    }
}

/// Parameters to execute present operations on a queue.
#[derive(Clone, Debug)]
pub struct PresentInfo {
    /// The semaphores to wait for before beginning the execution of the present operations.
    ///
    /// The default value is empty.
    pub wait_semaphores: Vec<SemaphorePresentInfo>,

    /// The present operations to perform.
    ///
    /// The default value is empty.
    pub swapchain_infos: Vec<SwapchainPresentInfo>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for PresentInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PresentInfo {
    /// Returns a default `PresentInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            wait_semaphores: Vec::new(),
            swapchain_infos: Vec::new(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self {
            wait_semaphores,
            swapchain_infos,
            _ne: _,
        } = self;

        for (index, semaphore_present_info) in wait_semaphores.iter().enumerate() {
            semaphore_present_info
                .validate(device)
                .map_err(|err| err.add_context(format!("wait_semaphores[{}]", index)))?;
        }

        assert!(!swapchain_infos.is_empty());

        let has_present_mode = swapchain_infos
            .first()
            .is_some_and(|first| first.present_mode.is_some());

        for (index, swapchain_info) in swapchain_infos.iter().enumerate() {
            swapchain_info
                .validate(device)
                .map_err(|err| err.add_context(format!("swapchain_infos[{}]", index)))?;

            let &SwapchainPresentInfo {
                swapchain: _,
                image_index: _,
                present_region: _,
                present_id: _,
                fence: _,
                present_mode,
                _ne: _,
            } = swapchain_info;

            if has_present_mode {
                if present_mode.is_none() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`swapchain_infos[0].present_mode` is `Some`, but \
                            `swapchain_infos[{}].present_mode` is not also `Some`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkPresentInfoKHR-pSwapchains-09199"],
                        ..Default::default()
                    }));
                }
            } else {
                if present_mode.is_some() {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`swapchain_infos[0].present_mode` is `None`, but \
                            `swapchain_infos[{}].present_mode` is not also `None`",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkPresentInfoKHR-pSwapchains-09199"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a PresentInfoFields1Vk<'_>,
        results_vk: &'a mut [vk::Result],
        extensions_vk: &'a mut PresentInfoExtensionsVk<'_>,
    ) -> vk::PresentInfoKHR<'a> {
        let &Self {
            wait_semaphores: _,
            swapchain_infos: _,
            _ne: _,
        } = self;
        let PresentInfoFields1Vk {
            wait_semaphores_vk,
            swapchains_vk,
            image_indices_vk,
            present_regions_vk: _,
            present_ids_vk: _,
            fences_vk: _,
            present_modes_vk: _,
        } = fields1_vk;
        let PresentInfoExtensionsVk {
            present_regions_vk,
            present_id_vk,
            present_fence_vk,
            present_mode_vk,
        } = extensions_vk;

        let mut val_vk = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores_vk)
            .swapchains(swapchains_vk)
            .image_indices(image_indices_vk)
            .results(results_vk);

        if let Some(next) = present_regions_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_id_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_fence_vk {
            val_vk = val_vk.push(next);
        }

        if let Some(next) = present_mode_vk {
            val_vk = val_vk.push(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_results(&self) -> Vec<vk::Result> {
        vec![vk::Result::SUCCESS; self.swapchain_infos.len()]
    }

    pub(crate) fn to_vk_extensions<'a>(
        &self,
        fields1_vk: &'a PresentInfoFields1Vk<'_>,
    ) -> PresentInfoExtensionsVk<'a> {
        let PresentInfoFields1Vk {
            wait_semaphores_vk: _,
            swapchains_vk: _,
            image_indices_vk: _,
            present_regions_vk,
            present_ids_vk,
            fences_vk,
            present_modes_vk,
        } = fields1_vk;

        let mut has_fences = false;
        let mut has_present_modes = false;
        let mut has_present_regions = false;
        let mut has_present_ids = false;

        for swapchain_info in &self.swapchain_infos {
            let &SwapchainPresentInfo {
                swapchain: _,
                image_index: _,
                ref present_region,
                present_id,
                ref fence,
                present_mode,
                _ne: _,
            } = swapchain_info;

            has_present_regions |= !present_region.is_empty();
            has_present_ids |= present_id.is_some();
            has_fences |= fence.is_some();
            has_present_modes |= present_mode.is_some();
        }

        let present_regions_vk = has_present_regions
            .then(|| vk::PresentRegionsKHR::default().regions(present_regions_vk));
        let present_id_vk =
            has_present_ids.then(|| vk::PresentIdKHR::default().present_ids(present_ids_vk));
        let present_fence_vk =
            has_fences.then(|| vk::SwapchainPresentFenceInfoEXT::default().fences(fences_vk));
        let present_mode_vk = has_present_modes
            .then(|| vk::SwapchainPresentModeInfoEXT::default().present_modes(present_modes_vk));

        PresentInfoExtensionsVk {
            present_regions_vk,
            present_id_vk,
            present_fence_vk,
            present_mode_vk,
        }
    }

    pub(crate) fn to_vk_fields1<'a>(
        &self,
        fields2_vk: &'a PresentInfoFields2Vk,
    ) -> PresentInfoFields1Vk<'a> {
        let Self {
            wait_semaphores,
            swapchain_infos,
            _ne: _,
        } = self;
        let PresentInfoFields2Vk {
            swapchain_infos_fields1_vk,
        } = fields2_vk;

        let wait_semaphores_vk = wait_semaphores
            .iter()
            .map(SemaphorePresentInfo::to_vk)
            .collect();

        let mut swapchains_vk = SmallVec::with_capacity(swapchain_infos.len());
        let mut image_indices_vk = SmallVec::with_capacity(swapchain_infos.len());
        let mut present_regions_vk = SmallVec::with_capacity(swapchain_infos.len());
        let mut present_ids_vk = SmallVec::with_capacity(swapchain_infos.len());
        let mut fences_vk = SmallVec::with_capacity(swapchain_infos.len());
        let mut present_modes_vk = SmallVec::with_capacity(swapchain_infos.len());

        for (swapchain_info, fields1_vk) in swapchain_infos.iter().zip(swapchain_infos_fields1_vk) {
            let &SwapchainPresentInfo {
                ref swapchain,
                image_index,
                present_region: _,
                present_id,
                ref fence,
                present_mode,
                _ne: _,
            } = swapchain_info;

            let SwapchainPresentInfoFields1Vk {
                present_region_rectangles_vk,
            } = fields1_vk;

            swapchains_vk.push(swapchain.handle());
            image_indices_vk.push(image_index);
            present_regions_vk
                .push(vk::PresentRegionKHR::default().rectangles(present_region_rectangles_vk));
            present_ids_vk.push(present_id.map_or(0, u64::from));
            fences_vk.push(fence.as_ref().map_or_else(Default::default, |f| f.handle()));
            present_modes_vk.push(present_mode.map_or_else(Default::default, Into::into));
        }

        PresentInfoFields1Vk {
            wait_semaphores_vk,
            swapchains_vk,
            image_indices_vk,
            present_regions_vk,
            present_ids_vk,
            fences_vk,
            present_modes_vk,
        }
    }

    pub(crate) fn to_vk_fields2(&self) -> PresentInfoFields2Vk {
        let swapchain_infos_fields1_vk = self
            .swapchain_infos
            .iter()
            .map(SwapchainPresentInfo::to_vk_fields1)
            .collect();

        PresentInfoFields2Vk {
            swapchain_infos_fields1_vk,
        }
    }
}

pub(crate) struct PresentInfoExtensionsVk<'a> {
    pub(crate) present_regions_vk: Option<vk::PresentRegionsKHR<'a>>,
    pub(crate) present_id_vk: Option<vk::PresentIdKHR<'a>>,
    pub(crate) present_fence_vk: Option<vk::SwapchainPresentFenceInfoEXT<'a>>,
    pub(crate) present_mode_vk: Option<vk::SwapchainPresentModeInfoEXT<'a>>,
}

pub(crate) struct PresentInfoFields1Vk<'a> {
    pub(crate) wait_semaphores_vk: SmallVec<[vk::Semaphore; 4]>,
    pub(crate) swapchains_vk: SmallVec<[vk::SwapchainKHR; 4]>,
    pub(crate) image_indices_vk: SmallVec<[u32; 4]>,
    pub(crate) present_regions_vk: SmallVec<[vk::PresentRegionKHR<'a>; 4]>,
    pub(crate) present_ids_vk: SmallVec<[u64; 4]>,
    pub(crate) fences_vk: SmallVec<[vk::Fence; 4]>,
    pub(crate) present_modes_vk: SmallVec<[vk::PresentModeKHR; 4]>,
}

pub(crate) struct PresentInfoFields2Vk {
    pub(crate) swapchain_infos_fields1_vk: SmallVec<[SwapchainPresentInfoFields1Vk; 4]>,
}

/// Parameters for a single present operation on a swapchain.
#[derive(Clone, Debug)]
pub struct SwapchainPresentInfo {
    /// The swapchain to present to.
    ///
    /// There is no default value.
    pub swapchain: Arc<Swapchain>,

    /// The index of the swapchain image to present to.
    ///
    /// The image must have been acquired first; this is the index that `acquire_next_image`
    /// returns.
    ///
    /// There is no default value.
    pub image_index: u32,

    /// An optimization hint to the implementation, that only some parts of the swapchain image are
    /// going to be updated by the present operation.
    ///
    /// If `present_regions` is not empty, then the
    /// [`khr_incremental_present`](crate::device::DeviceExtensions::khr_incremental_present)
    /// extension must be enabled on the device. The implementation will update the provided
    /// regions of the swapchain image, and _may_ ignore the other areas. However, as this is just
    /// a hint, the Vulkan implementation is free to ignore the regions altogether and update
    /// everything.
    ///
    /// If `present_regions` is empty, that means that all of the swapchain image must be updated.
    ///
    /// The default value is empty.
    pub present_region: Vec<RectangleLayer>,

    /// An id used to identify this present operation.
    ///
    /// If `present_id` is `Some`, the [`present_id`](crate::device::DeviceFeatures::present_id)
    /// feature must be enabled on the device. The id must be greater than any id previously
    /// used for `swapchain`. If a swapchain is recreated, this resets.
    ///
    /// The default value is `None`.
    pub present_id: Option<NonZero<u64>>,

    /// The fence to signal when the presentation has completed.
    ///
    /// If this is not `None`, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub fence: Option<Arc<Fence>>,

    /// The new present mode to use for presenting. This mode will be used for the current
    /// present, and any future presents where this value is `None`.
    ///
    /// If this is not `None`, then the provided present mode must be one of the present modes
    /// specified with [`present_modes`] when creating the
    /// swapchain.
    ///
    /// The default value is `None`.
    ///
    /// [`present_modes`]: crate::swapchain::SwapchainCreateInfo::present_modes
    pub present_mode: Option<PresentMode>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl SwapchainPresentInfo {
    /// Returns a default `SwapchainPresentInfo` with the provided `swapchain` and `image_index`.
    #[inline]
    pub const fn new(swapchain: Arc<Swapchain>, image_index: u32) -> Self {
        Self {
            swapchain,
            image_index,
            present_region: Vec::new(),
            present_id: None,
            fence: None,
            present_mode: None,
            _ne: crate::NE,
        }
    }

    #[deprecated(since = "0.36.0", note = "use `new` instead")]
    #[inline]
    pub fn swapchain_image_index(swapchain: Arc<Swapchain>, image_index: u32) -> Self {
        Self::new(swapchain, image_index)
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref swapchain,
            image_index,
            ref present_region,
            present_id,
            ref fence,
            present_mode,
            _ne: _,
        } = self;

        // VUID-VkPresentInfoKHR-commonparent
        assert_eq!(device, swapchain.device().as_ref());

        if image_index >= swapchain.image_count() {
            return Err(Box::new(ValidationError {
                problem: "`image_index` is not less than `swapchain.image_count()`".into(),
                vuids: &["VUID-VkPresentInfoKHR-pImageIndices-01430"],
                ..Default::default()
            }));
        }

        if present_id.is_some() && !device.enabled_features().present_id {
            return Err(Box::new(ValidationError {
                context: "present_id".into(),
                problem: "is `Some`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "present_id",
                )])]),
                vuids: &["VUID-VkPresentInfoKHR-pNext-06235"],
            }));
        }

        if !present_region.is_empty() && !device.enabled_extensions().khr_incremental_present {
            return Err(Box::new(ValidationError {
                context: "present_regions".into(),
                problem: "is not empty".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_incremental_present",
                )])]),
                ..Default::default()
            }));
        }

        for (index, rectangle_layer) in present_region.iter().enumerate() {
            let &RectangleLayer {
                offset,
                extent,
                layer,
            } = rectangle_layer;

            if offset[0] + extent[0] > swapchain.image_extent()[0] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`present_region[{0}].offset[0]` + `present_regions[{0}].extent[0]` is \
                        greater than `swapchain.image_extent()[0]`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkRectLayerKHR-offset-04864"],
                    ..Default::default()
                }));
            }

            if offset[1] + extent[1] > swapchain.image_extent()[1] {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`present_region[{0}].offset[1]` + `present_regions[{0}].extent[1]` is \
                        greater than `swapchain.image_extent()[1]`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkRectLayerKHR-offset-04864"],
                    ..Default::default()
                }));
            }

            if layer >= swapchain.image_array_layers() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`present_region[{0}].layer` is greater than \
                        `swapchain.image_array_layers()`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkRectLayerKHR-layer-01262"],
                    ..Default::default()
                }));
            }
        }

        if fence.is_some() && !device.enabled_features().swapchain_maintenance1 {
            return Err(Box::new(ValidationError {
                context: "fence".into(),
                problem: "is `Some`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "swapchain_maintenance1",
                )])]),
                vuids: &["VUID-VkPresentInfoKHR-swapchainMaintenance1-10158"],
            }));
        }

        if let Some(present_mode) = present_mode {
            if !device.enabled_features().swapchain_maintenance1 {
                return Err(Box::new(ValidationError {
                    context: "present_mode".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "swapchain_maintenance1",
                    )])]),
                    vuids: &["VUID-VkPresentInfoKHR-swapchainMaintenance1-10158"],
                }));
            }

            if !swapchain.present_modes().contains(&present_mode) {
                return Err(Box::new(ValidationError {
                    problem: "`swapchain.present_modes()` does not contain `present_mode`".into(),
                    vuids: &["VUID-VkSwapchainPresentModeInfoEXT-pPresentModes-07761"],
                    ..Default::default()
                }));
            }
        }

        // unsafe
        // VUID-VkPresentInfoKHR-pImageIndices-01430
        // VUID-VkPresentIdKHR-presentIds-04999

        Ok(())
    }

    pub(crate) fn to_vk_fields1(&self) -> SwapchainPresentInfoFields1Vk {
        let present_region_rectangles_vk = self
            .present_region
            .iter()
            .map(RectangleLayer::to_vk)
            .collect();

        SwapchainPresentInfoFields1Vk {
            present_region_rectangles_vk,
        }
    }
}

pub(crate) struct SwapchainPresentInfoFields1Vk {
    pub(crate) present_region_rectangles_vk: SmallVec<[vk::RectLayerKHR; 4]>,
}

/// Represents a rectangular region on an image layer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RectangleLayer {
    /// Coordinates in pixels of the top-left hand corner of the rectangle.
    pub offset: [u32; 2],

    /// Dimensions in pixels of the rectangle.
    pub extent: [u32; 2],

    /// The layer of the image. For images with only one layer, the value of layer must be 0.
    pub layer: u32,
}

impl RectangleLayer {
    /// Returns true if this rectangle layer is compatible with swapchain.
    #[inline]
    pub fn is_compatible_with(&self, swapchain: &Swapchain) -> bool {
        self.offset[0] + self.extent[0] <= swapchain.image_extent()[0]
            && self.offset[1] + self.extent[1] <= swapchain.image_extent()[1]
            && self.layer < swapchain.image_array_layers()
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_vk(&self) -> vk::RectLayerKHR {
        let &Self {
            offset,
            extent,
            layer,
        } = self;

        vk::RectLayerKHR {
            offset: vk::Offset2D {
                x: offset[0] as i32,
                y: offset[1] as i32,
            },
            extent: vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
            layer,
        }
    }
}

/// Parameters for a semaphore wait operation in a queue present operation.
#[derive(Clone, Debug)]
pub struct SemaphorePresentInfo {
    /// The semaphore to wait for.
    ///
    /// There is no default value.
    pub semaphore: Arc<Semaphore>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl SemaphorePresentInfo {
    /// Returns a default `SemaphorePresentInfo` with the provided `semaphore`.
    #[inline]
    pub const fn new(semaphore: Arc<Semaphore>) -> Self {
        Self {
            semaphore,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self { semaphore, _ne: _ } = self;

        // VUID-VkPresentInfoKHR-commonparent
        assert_eq!(device, semaphore.device().as_ref());

        if semaphore.semaphore_type() != SemaphoreType::Binary {
            return Err(Box::new(ValidationError {
                context: "semaphore.semaphore_type()".into(),
                problem: "is not `SemaphoreType::Binary`".into(),
                vuids: &["VUID-vkQueuePresentKHR-pWaitSemaphores-03267"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::Semaphore {
        let Self { semaphore, _ne: _ } = self;

        semaphore.handle()
    }
}

/// Represents a swapchain image being presented on the screen.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct PresentFuture<P>
where
    P: GpuFuture,
{
    previous: P,
    queue: Arc<Queue>,
    swapchain_info: SwapchainPresentInfo,
    // True if `flush()` has been called on the future, which means that the present command has
    // been submitted.
    flushed: AtomicBool,
    // True if `signal_finished()` has been called on the future, which means that the future has
    // been submitted and has already been processed by the GPU.
    finished: AtomicBool,
}

impl<P> PresentFuture<P>
where
    P: GpuFuture,
{
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    pub fn image_id(&self) -> u32 {
        self.swapchain_info.image_index
    }

    /// Returns the corresponding swapchain.
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain_info.swapchain
    }
}

unsafe impl<P> GpuFuture for PresentFuture<P>
where
    P: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        if self.flushed.load(Ordering::SeqCst) {
            return Ok(SubmitAnyBuilder::Empty);
        }

        let mut swapchain_info = self.swapchain_info.clone();
        debug_assert!(swapchain_info.image_index < swapchain_info.swapchain.image_count());
        let device = swapchain_info.swapchain.device();

        if !device.enabled_features().present_id {
            swapchain_info.present_id = None;
        }

        if device.enabled_extensions().khr_incremental_present {
            for rectangle in &swapchain_info.present_region {
                assert!(rectangle.is_compatible_with(swapchain_info.swapchain.as_ref()));
            }
        } else {
            swapchain_info.present_region = Default::default();
        }

        let _queue = self.previous.queue();

        // TODO: if the swapchain image layout is not PRESENT, should add a transition command
        // buffer

        Ok(match unsafe { self.previous.build_submission() }? {
            SubmitAnyBuilder::Empty => SubmitAnyBuilder::QueuePresent(PresentInfo {
                swapchain_infos: vec![self.swapchain_info.clone()],
                ..Default::default()
            }),
            SubmitAnyBuilder::SemaphoresWait(semaphores) => {
                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    wait_semaphores: semaphores
                        .into_iter()
                        .map(SemaphorePresentInfo::new)
                        .collect(),
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::CommandBuffer(_, _) => {
                self.previous.flush()?;

                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::QueuePresent(mut present_info) => {
                if present_info.swapchain_infos.first().is_some_and(|prev| {
                    prev.present_mode.is_some() != self.swapchain_info.present_mode.is_some()
                }) {
                    // If the present mode Option variants don't match, create a new command.
                    self.previous.flush()?;

                    SubmitAnyBuilder::QueuePresent(PresentInfo {
                        swapchain_infos: vec![self.swapchain_info.clone()],
                        ..Default::default()
                    })
                } else {
                    // Otherwise, add our swapchain to the previous.
                    present_info
                        .swapchain_infos
                        .push(self.swapchain_info.clone());

                    SubmitAnyBuilder::QueuePresent(present_info)
                }
            }
        })
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        // SAFETY: If `flushed` already contains `true`, then `build_submission` will return
        // `Empty`.
        let build_submission_result = unsafe { self.build_submission() };
        self.flushed.store(true, Ordering::SeqCst);

        match build_submission_result? {
            SubmitAnyBuilder::Empty => Ok(()),
            SubmitAnyBuilder::QueuePresent(present_info) => {
                let PresentInfo {
                    wait_semaphores: _,
                    swapchain_infos: swapchains,
                    _ne: _,
                } = &present_info;

                for swapchain_info in swapchains {
                    let &SwapchainPresentInfo {
                        ref swapchain,
                        image_index: _,
                        present_region: _,
                        present_id,
                        fence: _,
                        present_mode: _,
                        _ne: _,
                    } = swapchain_info;

                    if present_id.is_some_and(|present_id| !unsafe {
                        swapchain.try_claim_present_id(present_id)
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
                }

                match self.previous.check_swapchain_image_acquired(
                    &self.swapchain_info.swapchain,
                    self.swapchain_info.image_index,
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

                Ok(unsafe { queue_present(&self.queue, present_info) }?
                    .map(|r| r.map(|_| ()))
                    .fold(Ok(()), Result::and)?)
            }
            _ => unreachable!(),
        }
    }

    unsafe fn signal_finished(&self) {
        self.flushed.store(true, Ordering::SeqCst);
        self.finished.store(true, Ordering::SeqCst);
        unsafe { self.previous.signal_finished() };
    }

    fn queue_change_allowed(&self) -> bool {
        false
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        debug_assert!(match self.previous.queue() {
            None => true,
            Some(q) => q == self.queue,
        });

        Some(self.queue.clone())
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        self.previous
            .check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        if self.swapchain_info.swapchain.index_of_image(image)
            == Some(self.swapchain_info.image_index)
        {
            // This future presents the swapchain image, which "unlocks" it. Therefore any attempt
            // to use this swapchain image afterwards shouldn't get granted automatic access.
            // Instead any attempt to access the image afterwards should get an authorization from
            // a later swapchain acquire future. Hence why we return `Unknown` here.
            Err(AccessCheckError::Unknown)
        } else {
            self.previous
                .check_image_access(image, range, exclusive, expected_layout, queue)
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        if before {
            self.previous
                .check_swapchain_image_acquired(swapchain, image_index, false)
        } else if swapchain == self.swapchain_info.swapchain.as_ref()
            && image_index == self.swapchain_info.image_index
        {
            Err(AccessError::SwapchainImageNotAcquired.into())
        } else {
            self.previous
                .check_swapchain_image_acquired(swapchain, image_index, false)
        }
    }
}

unsafe impl<P> DeviceOwned for PresentFuture<P>
where
    P: GpuFuture,
{
    fn device(&self) -> &Arc<Device> {
        self.queue.device()
    }
}

impl<P> Drop for PresentFuture<P>
where
    P: GpuFuture,
{
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }

        if !*self.flushed.get_mut() {
            // Flushing may fail, that's okay. We will still wait for the queue later, so any
            // previous futures that were flushed correctly will still be waited upon.
            self.flush().ok();
        }

        if !*self.finished.get_mut() {
            // Block until the queue finished.
            self.queue().unwrap().with(|mut q| q.wait_idle()).unwrap();
            unsafe { self.previous.signal_finished() };
        }
    }
}

/// Wait for an image to be presented to the user. Must be used with a `present_id` given to
/// `present_with_id`.
///
/// Returns a bool to represent if the presentation was suboptimal. In this case the swapchain is
/// still usable, but the swapchain should be recreated as the Surface's properties no longer match
/// the swapchain.
#[deprecated(since = "0.35.0", note = "use `Swapchain::wait_for_present` instead")]
pub fn wait_for_present(
    swapchain: Arc<Swapchain>,
    present_id: u64,
    timeout: Option<Duration>,
) -> Result<bool, Validated<VulkanError>> {
    swapchain.try_wait_for_present(present_id.try_into().unwrap(), timeout)
}
