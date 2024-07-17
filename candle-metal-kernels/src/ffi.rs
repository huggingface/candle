#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_uint, c_void};

pub type CFTypeRef = *const c_void;
pub type CFAllocatorRef = *const c_void;
pub type CFMutableDictionaryRef = *mut c_void;
pub type CFStringRef = *const c_void;
pub type CFNumberRef = *const c_void;

pub type mach_port_t = c_uint;
pub type kern_return_t = c_int;
pub type io_object_t = mach_port_t;
pub type io_iterator_t = io_object_t;
pub type io_registry_entry_t = io_object_t;

pub type IOOptionBits = u32;
pub type CFNumberType = u32;

pub const kIOMainPortDefault: mach_port_t = 0;
pub const kIOServicePlane: &str = "IOService\0";
pub const kCFNumberSInt64Type: CFNumberType = 4;

pub const MACH_PORT_NULL: i32 = 0;

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    pub fn IOServiceGetMatchingServices(
        mainPort: mach_port_t,
        matching: CFMutableDictionaryRef,
        existing: *mut io_iterator_t,
    ) -> kern_return_t;

    pub fn IOServiceMatching(a: *const c_char) -> CFMutableDictionaryRef;

    pub fn IOIteratorNext(iterator: io_iterator_t) -> io_object_t;

    pub fn IOObjectRelease(obj: io_object_t) -> kern_return_t;

    pub fn IORegistryEntrySearchCFProperty(
        entry: io_registry_entry_t,
        plane: *const c_char,
        key: CFStringRef,
        allocator: CFAllocatorRef,
        options: IOOptionBits,
    ) -> CFTypeRef;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub fn CFNumberGetValue(
        number: CFNumberRef,
        theType: CFNumberType,
        valuePtr: *mut c_void,
    ) -> bool;
}

extern "C" {
    fn __CFStringMakeConstantString(c_str: *const c_char) -> CFStringRef;
}

pub fn cfstr(val: &str) -> CFStringRef {
    unsafe { __CFStringMakeConstantString(val.as_ptr().cast()) }
}
