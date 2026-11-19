//! Cache the pipeline objects to disk for faster reloads.
//!
//! A pipeline cache is an opaque type that allow you to cache your graphics and compute
//! pipelines on the disk.
//!
//! You can create either an empty cache or a cache from some initial data. Whenever you create a
//! graphics or compute pipeline, you have the possibility to pass a reference to that cache.
//! The Vulkan implementation will then look in the cache for an existing entry, or add one if it
//! doesn't exist.
//!
//! Once that is done, you can extract the data from the cache and store it. See the documentation
//! of [`get_data`](PipelineCache::get_data) for example of how to store
//! the data on the disk, and [`new`](PipelineCache::new) for how to reload
//! it.

use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk;
use smallvec::SmallVec;
use std::{mem::MaybeUninit, num::NonZero, ptr, sync::Arc};

/// Opaque cache that contains pipeline objects.
///
/// See [the documentation of the module](crate::pipeline::cache) for more info.
#[derive(Debug)]
pub struct PipelineCache {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    handle: vk::PipelineCache,
    id: NonZero<u64>,

    flags: PipelineCacheCreateFlags,
}

impl PipelineCache {
    /// Builds a new pipeline cache, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// # Examples
    ///
    /// This example loads a cache from a file, if it exists. See [`get_data`] for how to store the
    /// data in a file.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use vulkano::device::Device;
    /// use std::{fs::File, io::Read};
    /// use vulkano::pipeline::cache::{PipelineCache, PipelineCacheCreateInfo, PipelineCacheData};
    ///
    /// # let device: Arc<Device> = return;
    /// #
    /// let initial_data = {
    ///     let file = File::open("pipeline_cache.bin");
    ///     if let Ok(mut file) = file {
    ///         let mut data = Vec::new();
    ///         if file.read_to_end(&mut data).is_ok() {
    ///             Some(data)
    ///         } else {
    ///             None
    ///         }
    ///     } else {
    ///         None
    ///     }
    /// };
    ///
    /// let cache = PipelineCache::new(
    ///     &device,
    ///     &PipelineCacheCreateInfo {
    ///         // This is unsafe because there is no way to be sure that the file contains valid
    ///         // data.
    ///         initial_data: initial_data
    ///             .as_deref()
    ///             .map(|data| unsafe { PipelineCacheData::new(data) }),
    ///         ..Default::default()
    ///     },
    /// )
    /// .unwrap();
    /// ```
    ///
    /// [`get_data`]: Self::get_data
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        device: &Arc<Device>,
        create_info: &PipelineCacheCreateInfo<'_>,
    ) -> Result<Arc<PipelineCache>, VulkanError> {
        match Self::try_new(device, create_info) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Builds a new pipeline cache.
    #[inline]
    pub fn try_new(
        device: &Arc<Device>,
        create_info: &PipelineCacheCreateInfo<'_>,
    ) -> Result<Arc<PipelineCache>, Validated<VulkanError>> {
        Self::validate_new(device, create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &PipelineCacheCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &PipelineCacheCreateInfo<'_>,
    ) -> Result<Arc<PipelineCache>, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_pipeline_cache)(
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

    /// Creates a new `PipelineCache` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::PipelineCache,
        create_info: &PipelineCacheCreateInfo<'_>,
    ) -> Arc<PipelineCache> {
        let &PipelineCacheCreateInfo {
            flags,
            initial_data: _,
            _ne: _,
        } = create_info;

        Arc::new(PipelineCache {
            device: InstanceOwnedDebugWrapper(device.clone()),
            handle,
            id: Self::next_id(),

            flags,
        })
    }

    /// Returns the flags that the pipeline cache was created with.
    pub fn flags(&self) -> PipelineCacheCreateFlags {
        self.flags
    }

    /// Obtains the data from the cache.
    ///
    /// This data can be stored and then reloaded and passed to `PipelineCache::new`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::{fs, fs::File, io::Write};
    /// # use std::sync::Arc;
    /// # use vulkano::pipeline::cache::PipelineCache;
    ///
    /// # let cache: Arc<PipelineCache> = return;
    /// // If an error happens (eg. no permission for the file) we simply skip storing the cache.
    /// if let Ok(data) = cache.get_data() {
    ///     if let Ok(mut file) = File::create("pipeline_cache.bin.tmp") {
    ///         if file.write_all(&data).is_ok() {
    ///             let _ = fs::rename("pipeline_cache.bin.tmp", "pipeline_cache.bin");
    ///         } else {
    ///             let _ = fs::remove_file("pipeline_cache.bin.tmp");
    ///         }
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn get_data(&self) -> Result<Vec<u8>, VulkanError> {
        let fns = self.device.fns();

        let data = loop {
            let mut count = 0;
            unsafe {
                (fns.v1_0.get_pipeline_cache_data)(
                    self.device.handle(),
                    self.handle,
                    &mut count,
                    ptr::null_mut(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            let mut data: Vec<u8> = Vec::with_capacity(count);
            let result = unsafe {
                (fns.v1_0.get_pipeline_cache_data)(
                    self.device.handle(),
                    self.handle,
                    &mut count,
                    data.as_mut_ptr().cast(),
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    unsafe { data.set_len(count) };
                    break data;
                }
                vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        Ok(data)
    }

    /// Merges other pipeline caches into this one, panicking on a validation error.
    ///
    /// It is `self` that is modified here. The pipeline caches passed as parameter are untouched.
    ///
    /// This is a shortcut for `try_merge().map_err(Validated::unwrap)`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_merge`] returns a [`ValidationError`].
    ///
    /// # Examples
    ///
    /// This example loads a pipeline cache from the disk and merges it with the current pipeline
    /// cache.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use vulkano::device::Device;
    /// use std::{fs::File, io::Read};
    /// use vulkano::pipeline::cache::{PipelineCache, PipelineCacheCreateInfo, PipelineCacheData};
    ///
    /// # let device: Arc<Device> = return;
    /// #
    /// // Imagine this is an existing cache that got modified at runtime.
    /// let current_cache = PipelineCache::new(&device, &Default::default()).unwrap();
    ///
    /// // Load a new pipeline cache from the disk
    /// let new_data = {
    ///     let file = File::open("pipeline_cache.bin");
    ///     if let Ok(mut file) = file {
    ///         let mut data = Vec::new();
    ///         if file.read_to_end(&mut data).is_ok() {
    ///             Some(data)
    ///         } else {
    ///             None
    ///         }
    ///     } else {
    ///         None
    ///     }
    /// };
    ///
    /// let new_cache = PipelineCache::new(
    ///     &device,
    ///     &PipelineCacheCreateInfo {
    ///         // This is unsafe because there is no way to be sure that the file contains valid
    ///         // data.
    ///         initial_data: new_data
    ///             .as_deref()
    ///             .map(|data| unsafe { PipelineCacheData::new(data) }),
    ///         ..Default::default()
    ///     },
    /// )
    /// .unwrap();
    ///
    /// // Now merge the new pipeline cache into the existing one
    /// current_cache.merge([new_cache.as_ref()]).unwrap();
    /// ```
    ///
    /// [`try_merge`]: Self::try_merge
    #[track_caller]
    pub fn merge<'a>(
        &self,
        src_caches: impl IntoIterator<Item = &'a PipelineCache>,
    ) -> Result<(), VulkanError> {
        match self.try_merge(src_caches) {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Merges other pipeline caches into this one.
    ///
    /// It is `self` that is modified here. The pipeline caches passed as parameter are untouched.
    //
    // FIXME: vkMergePipelineCaches is not thread safe for the destination cache
    pub fn try_merge<'a>(
        &self,
        src_caches: impl IntoIterator<Item = &'a PipelineCache>,
    ) -> Result<(), Validated<VulkanError>> {
        let src_caches: SmallVec<[_; 8]> = src_caches.into_iter().collect();
        self.validate_merge(&src_caches)?;

        Ok(unsafe { self.merge_unchecked(src_caches) }?)
    }

    fn validate_merge(&self, src_caches: &[&PipelineCache]) -> Result<(), Box<ValidationError>> {
        for (index, &src_cache) in src_caches.iter().enumerate() {
            if src_cache == self {
                return Err(Box::new(ValidationError {
                    context: format!("src_caches[{}]", index).into(),
                    problem: "equals `self`".into(),
                    vuids: &["VUID-vkMergePipelineCaches-dstCache-00770"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn merge_unchecked<'a>(
        &self,
        src_caches: impl IntoIterator<Item = &'a PipelineCache>,
    ) -> Result<(), VulkanError> {
        let src_caches_vk: SmallVec<[_; 8]> =
            src_caches.into_iter().map(VulkanObject::handle).collect();

        let fns = self.device.fns();
        unsafe {
            (fns.v1_0.merge_pipeline_caches)(
                self.device.handle(),
                self.handle,
                src_caches_vk.len() as u32,
                src_caches_vk.as_ptr(),
            )
        }
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }
}

impl Drop for PipelineCache {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe {
            (fns.v1_0.destroy_pipeline_cache)(self.device.handle(), self.handle, ptr::null())
        };
    }
}

unsafe impl VulkanObject for PipelineCache {
    type Handle = vk::PipelineCache;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for PipelineCache {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(PipelineCache);

/// Parameters to create a new `PipelineCache`.
#[derive(Clone, Debug)]
pub struct PipelineCacheCreateInfo<'a> {
    /// Additional properties of the pipeline cache.
    ///
    /// The default value is empty.
    pub flags: PipelineCacheCreateFlags,

    /// The initial data to provide to the cache.
    ///
    /// If this is not `None`, then the data must have been previously retrieved by calling
    /// [`PipelineCache::get_data`].
    ///
    /// The default value is `None`.
    pub initial_data: Option<PipelineCacheData<'a>>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for PipelineCacheCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCacheCreateInfo<'_> {
    /// Returns a default `PipelineCacheCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: PipelineCacheCreateFlags::empty(),
            initial_data: None,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            initial_data: _,
            _ne,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkPipelineCacheCreateInfo-flags-parameter"])
        })?;

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::PipelineCacheCreateInfo<'_> {
        let &Self {
            flags,
            initial_data,
            _ne: _,
        } = self;

        let mut val_vk = vk::PipelineCacheCreateInfo::default().flags(flags.into());

        if let Some(initial_data) = initial_data {
            val_vk = val_vk.initial_data(initial_data.data());
        }

        val_vk
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a pipeline cache.
    PipelineCacheCreateFlags = PipelineCacheCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    EXTERNALLY_SYNCHRONIZED = EXTERNALLY_SYNCHRONIZED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_pipeline_creation_cache_control)]),
    ]), */
}

/// Represents the data of a **valid** Vulkan pipeline cache binary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PipelineCacheData<'a>(&'a [u8]);

impl<'a> PipelineCacheData<'a> {
    /// Creates new pipeline cache data from the given bytes.
    ///
    /// # Safety
    ///
    /// - The data passed to this function will most likely be blindly trusted by the Vulkan
    ///   implementation. Therefore you can easily crash your application or the system by passing
    ///   wrong data and using it for creating a pipeline cache.
    #[inline]
    pub unsafe fn new(data: &'a [u8]) -> Self {
        Self(data)
    }

    /// Returns a slice of bytes of the pipeline cache data.
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.0
    }
}

impl AsRef<[u8]> for PipelineCacheData<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.data()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        pipeline::{
            cache::PipelineCache, compute::ComputePipelineCreateInfo, ComputePipeline,
            PipelineLayout, PipelineShaderStageCreateInfo,
        },
        shader::{ShaderModule, ShaderModuleCreateInfo},
    };
    use std::slice;

    #[test]
    fn merge_self_forbidden() {
        let (device, _queue) = gfx_dev_and_queue!();
        let pipeline = PipelineCache::new(&device, &Default::default()).unwrap();
        assert!(pipeline.try_merge([pipeline.as_ref()]).is_err());
    }

    #[test]
    fn cache_returns_same_data() {
        let (device, _queue) = gfx_dev_and_queue!();

        let cache = PipelineCache::new(&device, &Default::default()).unwrap();

        let cs = {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u32; 48] = [
                119734787, 65536, 524298, 6, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1,
                196611, 2, 450, 262149, 4, 1852399981, 0, 131091, 2, 196641, 3, 2, 327734, 2, 4, 0,
                3, 131320, 5, 65789, 65592,
            ];
            let module =
                unsafe { ShaderModule::new(&device, &ShaderModuleCreateInfo::new(&MODULE)) }
                    .unwrap();
            module.entry_point("main").unwrap()
        };

        let _pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                Some(&cache),
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let cache_data = cache.get_data().unwrap();
        let second_data = cache.get_data().unwrap();

        assert_eq!(cache_data, second_data);
    }

    #[test]
    fn cache_returns_different_data() {
        let (device, _queue) = gfx_dev_and_queue!();

        let cache = PipelineCache::new(&device, &Default::default()).unwrap();

        let _first_pipeline = {
            let cs = {
                /*
                 * #version 450
                 * void main() {
                 * }
                 */
                const MODULE: [u32; 48] = [
                    119734787, 65536, 524298, 6, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                    808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1,
                    1, 196611, 2, 450, 262149, 4, 1852399981, 0, 131091, 2, 196641, 3, 2, 327734,
                    2, 4, 0, 3, 131320, 5, 65789, 65592,
                ];
                let module =
                    unsafe { ShaderModule::new(&device, &ShaderModuleCreateInfo::new(&MODULE)) }
                        .unwrap();
                module.entry_point("main").unwrap()
            };

            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                Some(&cache),
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let cache_data = cache.get_data().unwrap();

        let _second_pipeline = {
            let cs = {
                /*
                 * #version 450
                 *
                 * void main() {
                 *     uint idx = gl_GlobalInvocationID.x;
                 * }
                 */
                const MODULE: [u32; 108] = [
                    119734787, 65536, 524298, 16, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                    808793134, 0, 196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 1,
                    1, 1, 196611, 2, 450, 262149, 4, 1852399981, 0, 196613, 8, 7890025, 524293, 11,
                    1197436007, 1633841004, 1986939244, 1952539503, 1231974249, 68, 262215, 11, 11,
                    28, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176, 7, 7, 6, 262167, 9, 6,
                    3, 262176, 10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0, 262176, 13, 1, 6,
                    327734, 2, 4, 0, 3, 131320, 5, 262203, 7, 8, 7, 327745, 13, 14, 11, 12, 262205,
                    6, 15, 14, 196670, 8, 15, 65789, 65592,
                ];
                let module =
                    unsafe { ShaderModule::new(&device, &ShaderModuleCreateInfo::new(&MODULE)) }
                        .unwrap();
                module.entry_point("main").unwrap()
            };

            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                Some(&cache),
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let second_data = cache.get_data().unwrap();

        if cache_data.is_empty() {
            assert_eq!(cache_data, second_data);
        } else {
            assert_ne!(cache_data, second_data);
        }
    }

    #[test]
    fn cache_data_does_not_change() {
        let (device, _queue) = gfx_dev_and_queue!();

        let cache = PipelineCache::new(&device, &Default::default()).unwrap();

        let cs = {
            /*
             * #version 450
             * void main() {
             * }
             */
            const MODULE: [u32; 48] = [
                119734787, 65536, 524298, 6, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1,
                196611, 2, 450, 262149, 4, 1852399981, 0, 131091, 2, 196641, 3, 2, 327734, 2, 4, 0,
                3, 131320, 5, 65789, 65592,
            ];
            let module =
                unsafe { ShaderModule::new(&device, &ShaderModuleCreateInfo::new(&MODULE)) }
                    .unwrap();
            module.entry_point("main").unwrap()
        };

        let _first_pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                Some(&cache),
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let cache_data = cache.get_data().unwrap();

        let _second_pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(&device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                &device,
                Some(&cache),
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let second_data = cache.get_data().unwrap();

        assert_eq!(cache_data, second_data);
    }
}
