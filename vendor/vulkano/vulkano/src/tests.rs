#![cfg(test)]

/// Creates an instance or returns if initialization fails.
macro_rules! instance {
    () => {{
        use crate::{instance::Instance, VulkanLibrary};

        let library = match unsafe { VulkanLibrary::new() } {
            Ok(x) => x,
            Err(_) => return,
        };

        match Instance::new(&library, &Default::default()) {
            Ok(x) => x,
            Err(_) => return,
        }
    }};
}

/// Creates a device and a queue for graphics operations.
macro_rules! gfx_dev_and_queue {
    () => ({
        gfx_dev_and_queue!(;)
    });
    ($($feature:ident),*) => ({
        gfx_dev_and_queue!($($feature),*;)
    });
    ($($feature:ident),*; $($extension:ident),*) => ({
        use crate::device::physical::PhysicalDeviceType;
        use crate::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
        use crate::device::DeviceFeatures;

        let instance = instance!();
        let enabled_extensions = DeviceExtensions {
            $(
                $extension: true,
            )*
            ..DeviceExtensions::empty()
        };
        let enabled_features = DeviceFeatures {
            $(
                $feature: true,
            )*
            ..DeviceFeatures::empty()
        };

        let select = match instance.enumerate_physical_devices() {
            Ok(x) => x.into_iter(),
            Err(_) => return,
        }
        .filter(|p| {
            p.supported_extensions().contains(&enabled_extensions) &&
            p.supported_features().contains(&enabled_features)
        })
        .filter_map(|p| {
            p.queue_family_properties().iter()
                .position(|q| q.queue_flags.intersects(crate::device::QueueFlags::GRAPHICS))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            }
        });

        let (physical_device, queue_family_index) = match select {
            Some(x) => x,
            None => return,
        };

        let (device, queues) = match Device::new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: &enabled_extensions,
                enabled_features: &enabled_features,
                ..Default::default()
            },
        ) {
            Ok(r) => r,
            Err(_) => return,
        };

        let [queue] = queues.try_into().unwrap();

        (device, queue)
    });
}

macro_rules! assert_should_panic {
    ($msg:expr, $code:block) => {{
        let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $code));

        match res {
            Ok(_) => panic!("Test expected to panic but didn't"),
            Err(err) => {
                if let Some(msg) = err.downcast_ref::<String>() {
                    assert!(msg.contains($msg));
                } else if let Some(&msg) = err.downcast_ref::<&str>() {
                    assert!(msg.contains($msg));
                } else {
                    panic!("Couldn't decipher the panic message of the test")
                }
            }
        }
    }};

    ($code:block) => {{
        let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $code));

        match res {
            Ok(_) => panic!("Test expected to panic but didn't"),
            Err(_) => {}
        }
    }};
}
