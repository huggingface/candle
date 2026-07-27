use core::ffi::c_void;

extern "C" {
    pub(crate) fn lora_bgmv_shrink(
        x: *const c_void,
        a: *const c_void,
        slots: *const u32,
        tmp: *mut c_void,
        batch: u32,
        in_dim: u32,
        r: u32,
        dtype: u32,
        stream: *mut c_void,
    );

    pub(crate) fn lora_bgmv_expand(
        tmp: *const c_void,
        b: *const c_void,
        slots: *const u32,
        delta: *mut c_void,
        batch: u32,
        r: u32,
        out_dim: u32,
        dtype: u32,
        stream: *mut c_void,
    );
}
