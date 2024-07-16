use core::ffi::c_void;
use metal::Device;

use crate::ffi::*;

#[cfg(target_os = "macos")]
fn find_core_count() -> Result<usize, String> {
    let matching_dict = unsafe { IOServiceMatching(b"AGXAccelerator\0".as_ptr().cast()) };
    if matching_dict.is_null() {
        return Err("IOServiceMatching call failed, `AGXAccelerator` not found".to_string());
    }

    let mut iterator: io_iterator_t = 0;
    let result =
        unsafe { IOServiceGetMatchingServices(kIOMainPortDefault, matching_dict, &mut iterator) };
    if result != 0 {
        return Err("Error getting matching services".to_string());
    }

    let gpu_entry = unsafe { IOIteratorNext(iterator) };
    unsafe {
        IOObjectRelease(iterator);
    }

    if gpu_entry == MACH_PORT_NULL as u32 {
        return Err("Error getting GPU entry".to_string());
    }
    let key = "gpu-core-count\0";
    let options: IOOptionBits = 0;

    let gpu_core_count = unsafe {
        IORegistryEntrySearchCFProperty(
            gpu_entry,
            kIOServicePlane as *mut i8,
            cfstr(key.as_ptr().cast()),
            core::ptr::null(),
            options,
        )
    };

    if gpu_core_count.is_null() {
        return Err("Error getting gpu-core-count property".to_string());
    }
    let gpu_core_count_number = gpu_core_count as CFNumberRef;

    let mut value: i64 = 0;
    let result = unsafe {
        CFNumberGetValue(
            gpu_core_count_number,
            kCFNumberSInt64Type,
            &mut value as *mut i64 as *mut c_void,
        )
    };
    if !result {
        return Err("Error getting value of gpu-core-count".to_string());
    }
    Ok(value as usize)
}

pub(crate) fn get_device_core_count(device: &Device) -> usize {
    #[cfg(target_os = "macos")]
    {
        find_core_count().unwrap()
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
