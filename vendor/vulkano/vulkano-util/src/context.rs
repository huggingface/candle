use std::sync::Arc;
#[cfg(target_os = "macos")]
use vulkano::instance::InstanceCreateFlags;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
    },
    instance::{
        debug::{DebugUtilsMessenger, DebugUtilsMessengerCreateInfo},
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    memory::allocator::StandardMemoryAllocator,
    Version, VulkanLibrary,
};

/// A configuration struct to pass various creation options to create [`VulkanoContext`].
///
/// Instance extensions that are required for surface creation will be appended to the config when
/// creating [`VulkanoContext`].
pub struct VulkanoConfig<'a> {
    pub instance_create_info: InstanceCreateInfo<'a>,

    /// Pass the `DebugUtilsMessengerCreateInfo` to create the debug callback
    /// for printing debug information at runtime.
    pub debug_create_info: Option<DebugUtilsMessengerCreateInfo<'a>>,

    /// Pass filter function for your physical device selection. See default for example.
    pub device_filter_fn: Arc<dyn Fn(&PhysicalDevice) -> bool>,

    /// Pass priority order function for your physical device selection. See default for example.
    pub device_priority_fn: Arc<dyn Fn(&PhysicalDevice) -> u32>,

    pub device_extensions: DeviceExtensions,

    pub device_features: DeviceFeatures,

    /// Print your selected device name at start.
    pub print_device_name: bool,
}

impl Default for VulkanoConfig<'_> {
    #[inline]
    fn default() -> Self {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        VulkanoConfig {
            instance_create_info: InstanceCreateInfo {
                #[cfg(target_vendor = "apple")]
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                application_version: Version::V1_3,
                ..Default::default()
            },
            debug_create_info: None,
            device_filter_fn: Arc::new(move |p| {
                p.supported_extensions().contains(&device_extensions)
            }),
            device_priority_fn: Arc::new(|p| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 1,
                PhysicalDeviceType::IntegratedGpu => 2,
                PhysicalDeviceType::VirtualGpu => 3,
                PhysicalDeviceType::Cpu => 4,
                PhysicalDeviceType::Other => 5,
                _ => 6,
            }),
            print_device_name: false,
            device_extensions,
            device_features: DeviceFeatures::empty(),
        }
    }
}

/// A utility struct to create, access and hold alive Vulkano device, instance and queues.
///
/// Vulkano context is used in the creation of your graphics or compute pipelines, images and
/// in the creation of [`VulkanoWindowRenderer`](crate::renderer::VulkanoWindowRenderer) through
/// [`VulkanoWindows`](crate::window::VulkanoWindows).
///
/// ## Examples
///
/// ```no_run
/// use vulkano_util::context::{VulkanoConfig, VulkanoContext};
///
/// fn test() {
///     let context = VulkanoContext::new(VulkanoConfig::default());
///     // Then create event loop, windows, pipelines, etc.
/// }
/// ```
pub struct VulkanoContext {
    instance: Arc<Instance>,
    _debug_utils_messenger: Option<DebugUtilsMessenger>,
    device: Arc<Device>,
    queues: Queues,
    memory_allocator: Arc<StandardMemoryAllocator>,
}

impl Default for VulkanoContext {
    #[inline]
    fn default() -> Self {
        VulkanoContext::new(VulkanoConfig::default())
    }
}

impl VulkanoContext {
    /// Creates a new [`VulkanoContext`].
    ///
    /// # Panics
    ///
    /// - Panics where the underlying Vulkano struct creations fail
    pub fn new(config: VulkanoConfig<'_>) -> Self {
        let library = match unsafe { VulkanLibrary::new() } {
            Ok(x) => x,
            #[cfg(target_vendor = "apple")]
            Err(vulkano::library::LoadingError::LibraryLoadFailure(err)) => panic!(
                "failed to load Vulkan library: {err}; did you install VulkanSDK from \
                https://vulkan.lunarg.com/sdk/home?",
            ),
            Err(err) => panic!("failed to load Vulkan library: {err}"),
        };

        // Append required extensions
        // HACK: This should be replaced with `Surface::required_extensions`, but will need to
        // happen in the next minor version bump. It should have been done before releasing 0.34.
        let instance_extensions = library
            .supported_extensions()
            .intersection(&InstanceExtensions {
                khr_surface: true,
                khr_xlib_surface: true,
                khr_xcb_surface: true,
                khr_wayland_surface: true,
                khr_android_surface: true,
                khr_win32_surface: true,
                ext_metal_surface: true,
                ..InstanceExtensions::empty()
            })
            .union(config.instance_create_info.enabled_extensions);

        // Create instance
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                enabled_extensions: &instance_extensions,
                ..config.instance_create_info
            },
        )
        .expect("failed to create instance");

        // Create debug callback
        let _debug_utils_messenger = config.debug_create_info.as_ref().map(|dbg_create_info| {
            DebugUtilsMessenger::new(&instance, dbg_create_info)
                .expect("failed to create debug callback")
        });

        // Get prioritized device
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .into_iter()
            .filter(|p| (config.device_filter_fn)(p))
            .min_by_key(|p| (config.device_priority_fn)(p))
            .expect("failed to create physical device");
        // Print used device
        if config.print_device_name {
            println!(
                "Using device {}, type: {:?}",
                physical_device.properties().device_name,
                physical_device.properties().device_type,
            );
        }

        // Create device
        let (device, queues) = Self::create_device(
            &physical_device,
            &config.device_extensions,
            &config.device_features,
        );
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        Self {
            instance,
            _debug_utils_messenger,
            device,
            queues,
            memory_allocator,
        }
    }

    /// Creates vulkano device with required queue families and required extensions. Creates a
    /// separate queue for compute if possible. If not, same queue as graphics is used.
    fn create_device(
        physical_device: &Arc<PhysicalDevice>,
        device_extensions: &DeviceExtensions,
        device_features: &DeviceFeatures,
    ) -> (Arc<Device>, Queues) {
        let queue_family_graphics = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .map(|(i, q)| (i as u32, q))
            .find(|(_i, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
            .map(|(i, _)| i)
            .expect("could not find a queue that supports graphics");
        // Try finding a separate queue for compute
        let queue_family_compute = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .map(|(i, q)| (i as u32, q))
            .find(|(i, q)| {
                q.queue_flags.intersects(QueueFlags::COMPUTE) && *i != queue_family_graphics
            })
            .map(|(i, _)| i);
        // Try finding a separate queue for transfer
        let queue_family_transfer = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .map(|(i, q)| (i as u32, q))
            .find(|(_i, q)| {
                q.queue_flags.intersects(QueueFlags::TRANSFER)
                    && !q
                        .queue_flags
                        .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            })
            .map(|(i, _)| i);

        let queue_create_infos: Vec<_> = [
            Some(QueueCreateInfo {
                queue_family_index: queue_family_graphics,
                ..Default::default()
            }),
            queue_family_compute.map(|queue_family_index| QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }),
            queue_family_transfer.map(|queue_family_index| QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }),
        ]
        .into_iter()
        .flatten()
        .collect();

        let (device, queues) = {
            Device::new(
                physical_device,
                &DeviceCreateInfo {
                    queue_create_infos: &queue_create_infos,
                    enabled_extensions: device_extensions,
                    enabled_features: device_features,
                    ..Default::default()
                },
            )
            .expect("failed to create device")
        };

        let mut queues = queues.into_iter();

        let graphics_queue = queues.next().unwrap();
        let compute_queue = if queue_family_compute.is_some() {
            queues.next().unwrap()
        } else {
            graphics_queue.clone()
        };
        let transfer_queue = queue_family_transfer
            .is_some()
            .then(|| queues.next().unwrap());

        (
            device,
            Queues {
                graphics_queue,
                compute_queue,
                transfer_queue,
            },
        )
    }

    /// Returns the name of the device.
    #[inline]
    pub fn device_name(&self) -> &str {
        &self.device.physical_device().properties().device_name
    }

    /// Returns the type of the device.
    #[inline]
    pub fn device_type(&self) -> PhysicalDeviceType {
        self.device.physical_device().properties().device_type
    }

    /// Returns the maximum memory allocation of the device.
    #[inline]
    pub fn max_memory(&self) -> u32 {
        self.device
            .physical_device()
            .properties()
            .max_memory_allocation_count
    }

    /// Returns the instance.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the device.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the graphics queue.
    #[inline]
    pub fn graphics_queue(&self) -> &Arc<Queue> {
        &self.queues.graphics_queue
    }

    /// Returns the compute queue.
    ///
    /// Depending on your device, this might be the same as the graphics queue.
    #[inline]
    pub fn compute_queue(&self) -> &Arc<Queue> {
        &self.queues.compute_queue
    }

    /// Returns the transfer queue, if the device has a queue family that is dedicated for
    /// transfers (does not support graphics or compute).
    #[inline]
    pub fn transfer_queue(&self) -> Option<&Arc<Queue>> {
        self.queues.transfer_queue.as_ref()
    }

    /// Returns the memory allocator.
    #[inline]
    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }
}

struct Queues {
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    transfer_queue: Option<Arc<Queue>>,
}
