use core::ffi::c_void;
use metal::Device;

use crate::ffi::*;

const GPU_CORE_COUNT_KEY: &str = "gpu-core-count\0";
const AGXACCELERATOR_KEY: &str = "AGXAccelerator\0";

struct IOIterator(io_iterator_t);

impl IOIterator {
    fn new(it: io_iterator_t) -> Self {
        IOIterator(it)
    }

    fn next(&self) -> Option<io_object_t> {
        let result = unsafe { IOIteratorNext(self.0) };
        if result == MACH_PORT_NULL as u32 {
            return None;
        }
        Some(result)
    }
}

impl Drop for IOIterator {
    fn drop(&mut self) {
        unsafe { IOObjectRelease(self.0 as _) };
    }
}

unsafe fn get_io_service_matching(val: &str) -> Result<CFMutableDictionaryRef, String> {
    let matching = IOServiceMatching(val.as_ptr().cast());
    if matching.is_null() {
        return Err(format!("IOServiceMatching call failed, `{val}` not found"));
    }
    Ok(matching)
}

unsafe fn get_matching_services(
    main_port: mach_port_t,
    matching: CFMutableDictionaryRef,
) -> Result<IOIterator, String> {
    let mut iterator: io_iterator_t = 0;
    let result = IOServiceGetMatchingServices(main_port, matching, &mut iterator);
    if result != 0 {
        return Err("Error getting matching services".to_string());
    }
    Ok(IOIterator::new(iterator))
}

unsafe fn get_gpu_io_service() -> Result<io_object_t, String> {
    let matching = get_io_service_matching(AGXACCELERATOR_KEY)?;
    let iterator = get_matching_services(kIOMainPortDefault, matching)?;
    iterator
        .next()
        .ok_or("Error getting GPU IO Service".to_string())
}

unsafe fn get_property_by_key(
    entry: io_registry_entry_t,
    plane: &str,
    key: &str,
    allocator: CFAllocatorRef,
    options: IOOptionBits,
) -> Result<CFTypeRef, String> {
    let result = IORegistryEntrySearchCFProperty(
        entry,
        plane.as_ptr().cast(),
        cfstr(key),
        allocator,
        options,
    );
    if result.is_null() {
        return Err(format!("Error getting {key} property"));
    }
    Ok(result)
}

unsafe fn get_int_value(number: CFNumberRef) -> Result<i64, String> {
    let mut value: i64 = 0;
    let result = CFNumberGetValue(
        number,
        kCFNumberSInt64Type,
        &mut value as *mut i64 as *mut c_void,
    );
    if !result {
        return Err("Error getting int value".to_string());
    }
    Ok(value)
}

unsafe fn find_core_count() -> Result<usize, String> {
    let gpu_io_service = get_gpu_io_service()?;
    let gpu_core_count = get_property_by_key(
        gpu_io_service,
        kIOServicePlane,
        GPU_CORE_COUNT_KEY,
        core::ptr::null(),
        0,
    )?;
    let value = get_int_value(gpu_core_count as CFNumberRef)?;
    Ok(value as usize)
}

pub(crate) fn get_device_core_count(device: &Device) -> usize {
    #[cfg(target_os = "macos")]
    {
        unsafe { find_core_count().expect("Retrieving gpu core count failed") }
    }
    #[cfg(target_os = "ios")]
    {
        if device.name().starts_with("A") {
            if device.supports_family(MTLGPUFamily::Apple9) {
                6
            } else {
                5
            }
        } else {
            10
        }
    }
}
