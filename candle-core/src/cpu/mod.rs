//! Traits and methods for CPU-backed Tensors
#![allow(clippy::needless_range_loop)]
pub mod erf;
pub mod kernels;

#[allow(unused)]
trait Cpu {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f32) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit);
}

#[allow(unused)]
trait CpuF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;
    // How many outer loop steps to accumulate in Self::Unit before flushing to f32.
    // Set to a finite value to bound rounding error (16 for example).
    const FLUSH_INTERVAL: usize = usize::MAX;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit);
}

#[allow(unused)]
trait CpuBF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const bf16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit);
}

use half::{bf16, f16};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
pub mod avx;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
pub use avx::{CurrentCpu, CurrentCpuBF16, CurrentCpuF16};

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod simd128;
#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub use simd128::CurrentCpu;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub use neon::{CurrentCpu, CurrentCpuBF16, CurrentCpuF16};

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    let mut sum = CurrentCpu::zero_array();

    let mut i = 0;
    while i + CurrentCpu::STEP <= k {
        for j in 0..CurrentCpu::ARR {
            sum[j] = CurrentCpu::vec_fma(
                sum[j],
                CurrentCpu::load(a_row.add(i + j * CurrentCpu::EPR)),
                CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR)),
            );
        }
        i += CurrentCpu::STEP;
    }

    CurrentCpu::vec_reduce(sum, c);

    // leftovers
    while i < k {
        *c += *a_row.add(i) * (*b_row.add(i));
        i += 1;
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    for i in 0..k {
        *c += *a_row.add(i) * (*b_row.add(i));
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
#[inline(always)]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    let np = k & !(CurrentCpu::STEP - 1);

    let mut sum = CurrentCpu::zero_array();
    let mut x = CurrentCpu::zero_array();

    for i in (0..np).step_by(CurrentCpu::STEP) {
        for j in 0..CurrentCpu::ARR {
            x[j] = CurrentCpu::load(row.add(i + j * CurrentCpu::EPR));
            sum[j] = CurrentCpu::vec_add(sum[j], x[j]);
        }
    }

    CurrentCpu::vec_reduce(sum, b);

    // leftovers
    for i in np..k {
        *b += *row.add(i)
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    *b = 0f32;
    for i in 0..k {
        *b += *row.add(i)
    }
}

#[cfg(any(target_feature = "avx2", target_feature = "neon"))]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    let mut sumf = 0.0f32;
    let mut sum = CurrentCpuF16::zero_array();
    let mut steps_since_flush = 0usize;
    let mut i = 0;
    while i + CurrentCpuF16::STEP <= k {
        for j in 0..CurrentCpuF16::ARR {
            sum[j] = CurrentCpuF16::vec_fma(
                sum[j],
                CurrentCpuF16::load(a_row.add(i + j * CurrentCpuF16::EPR)),
                CurrentCpuF16::load(b_row.add(i + j * CurrentCpuF16::EPR)),
            );
        }
        steps_since_flush += 1;
        if steps_since_flush == CurrentCpuF16::FLUSH_INTERVAL {
            let mut partial = 0.0f32;
            CurrentCpuF16::vec_reduce(sum, &mut partial);
            sumf += partial;
            sum = CurrentCpuF16::zero_array();
            steps_since_flush = 0;
        }

        i += CurrentCpuF16::STEP;
    }

    let mut partial = 0.0f32;
    CurrentCpuF16::vec_reduce(sum, &mut partial);
    sumf += partial;

    // leftovers
    while i < k {
        sumf += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        i += 1;
    }
    *c = sumf;
}

#[cfg(any(target_feature = "avx2", target_feature = "neon"))]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    let mut sum = CurrentCpuBF16::zero_array();

    let mut i = 0;
    while i + CurrentCpuBF16::STEP <= k {
        for j in 0..CurrentCpuBF16::ARR {
            sum[j] = CurrentCpuBF16::vec_fma(
                sum[j],
                CurrentCpuBF16::load(a_row.add(i + j * CurrentCpuBF16::EPR)),
                CurrentCpuBF16::load(b_row.add(i + j * CurrentCpuBF16::EPR)),
            );
        }
        i += CurrentCpuBF16::STEP;
    }

    CurrentCpuBF16::vec_reduce(sum, c);

    // leftovers
    while i < k {
        *c += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        i += 1;
    }
}

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    // leftovers
    let mut sum = 0.0;
    for i in 0..k {
        sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
    }
    *c = sum;
}

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    // leftovers
    let mut sum = 0.0;
    for i in 0..k {
        sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
    }
    *c = sum;
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
pub(crate) unsafe fn vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    let mut i = 0;
    while i + CurrentCpuF16::STEP <= k {
        for j in 0..CurrentCpuF16::ARR {
            CurrentCpuF16::vec_store(
                c.add(i + j * CurrentCpuF16::EPR),
                CurrentCpuF16::vec_add(
                    CurrentCpuF16::load(a_row.add(i + j * CurrentCpuF16::EPR)),
                    CurrentCpuF16::load(b_row.add(i + j * CurrentCpuF16::EPR)),
                ),
            );
        }
        i += CurrentCpuF16::STEP;
    }

    // leftovers
    for j in i..k {
        *c.add(j) = *a_row.add(j) + *b_row.add(j);
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    for i in 0..k {
        *c.add(i) = *a_row.add(i) + *b_row.add(i);
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
pub(crate) unsafe fn vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    let mut i = 0;
    while i + CurrentCpuBF16::STEP <= k {
        for j in 0..CurrentCpuBF16::ARR {
            CurrentCpuBF16::vec_store(
                c.add(i + j * CurrentCpuBF16::EPR),
                CurrentCpuBF16::vec_add(
                    CurrentCpuBF16::load(a_row.add(i + j * CurrentCpuBF16::EPR)),
                    CurrentCpuBF16::load(b_row.add(i + j * CurrentCpuBF16::EPR)),
                ),
            );
        }
        i += CurrentCpuBF16::STEP;
    }

    // leftovers
    for j in i..k {
        *c.add(j) = *a_row.add(j) + *b_row.add(j);
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    for i in 0..k {
        *c.add(i) = *a_row.add(i) + *b_row.add(i);
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    let sv = CurrentCpuF16::from_f32(scalar.to_f32());
    let mut i = 0;
    while i + CurrentCpuF16::STEP <= k {
        for j in 0..CurrentCpuF16::ARR {
            CurrentCpuF16::vec_store(
                ys.add(i + j * CurrentCpuF16::EPR),
                CurrentCpuF16::vec_add(CurrentCpuF16::load(xs.add(i + j * CurrentCpuF16::EPR)), sv),
            );
        }
        i += CurrentCpuF16::STEP;
    }
    for j in i..k {
        *ys.add(j) = *xs.add(j) + scalar;
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    for i in 0..k {
        *ys.add(i) = *xs.add(i) + scalar;
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    let sv = CurrentCpuBF16::from_f32(scalar.to_f32());
    let mut i = 0;
    while i + CurrentCpuBF16::STEP <= k {
        for j in 0..CurrentCpuBF16::ARR {
            CurrentCpuBF16::vec_store(
                ys.add(i + j * CurrentCpuBF16::EPR),
                CurrentCpuBF16::vec_add(
                    CurrentCpuBF16::load(xs.add(i + j * CurrentCpuBF16::EPR)),
                    sv,
                ),
            );
        }
        i += CurrentCpuBF16::STEP;
    }
    for j in i..k {
        *ys.add(j) = *xs.add(j) + scalar;
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    for i in 0..k {
        *ys.add(i) = *xs.add(i) + scalar;
    }
}
