use libc::{c_char, c_double, c_float, c_int};

mod ffi {
    use super::*;
    extern "C" {
        pub fn sgemm_(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const c_float,
            a: *const c_float,
            lda: *const c_int,
            b: *const c_float,
            ldb: *const c_int,
            beta: *const c_float,
            c: *mut c_float,
            ldc: *const c_int,
        );
        pub fn dgemm_(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const c_double,
            a: *const c_double,
            lda: *const c_int,
            b: *const c_double,
            ldb: *const c_int,
            beta: *const c_double,
            c: *mut c_double,
            ldc: *const c_int,
        );
        pub fn hgemm_(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const half::f16,
            a: *const half::f16,
            lda: *const c_int,
            b: *const half::f16,
            ldb: *const c_int,
            beta: *const half::f16,
            c: *mut half::f16,
            ldc: *const c_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    ffi::sgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn dgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    ffi::dgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn hgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: half::f16,
    a: &[half::f16],
    lda: i32,
    b: &[half::f16],
    ldb: i32,
    beta: half::f16,
    c: &mut [half::f16],
    ldc: i32,
) {
    ffi::hgemm_(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}
