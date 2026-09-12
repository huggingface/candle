use super::{Cpu, CpuBF16, CpuF16};
use core::arch::wasm32::*;
use half::{bf16, f16};

pub struct CurrentCpu {}

const STEP: usize = 16;
const EPR: usize = 4;
const ARR: usize = STEP / EPR;

impl Cpu for CurrentCpu {
    type Unit = v128;
    type Array = [v128; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        f32x4_splat(0.0)
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); ARR]
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        f32x4_splat(v)
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        v128_load(mem_addr as *mut v128)
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        f32x4_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        f32x4_add(f32x4_mul(b, c), a)
    }

    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit) {
        v128_store(mem_addr as *mut v128, a);
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        for i in 0..ARR / 2 {
            x[2 * i] = f32x4_add(x[2 * i], x[2 * i + 1]);
        }
        for i in 0..ARR / 4 {
            x[4 * i] = f32x4_add(x[4 * i], x[4 * i + 2]);
        }
        for i in 0..ARR / 8 {
            x[8 * i] = f32x4_add(x[8 * i], x[8 * i + 4]);
        }
        *y = f32x4_extract_lane::<0>(x[0])
            + f32x4_extract_lane::<1>(x[0])
            + f32x4_extract_lane::<2>(x[0])
            + f32x4_extract_lane::<3>(x[0]);
    }
}

// WebAssembly SIMD has no half-precision arithmetic, so widen f16 lanes to
// f32x4 as the AVX software fallback and the NEON fallback do.
pub struct CurrentCpuF16 {}

impl CpuF16 for CurrentCpuF16 {
    type Unit = v128;
    type Array = [v128; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        <CurrentCpu as Cpu>::zero()
    }

    unsafe fn zero_array() -> Self::Array {
        <CurrentCpu as Cpu>::zero_array()
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        <CurrentCpu as Cpu>::from_f32(v)
    }

    unsafe fn load(mem_addr: *const f16) -> Self::Unit {
        let values = [
            (*mem_addr).to_f32(),
            (*mem_addr.add(1)).to_f32(),
            (*mem_addr.add(2)).to_f32(),
            (*mem_addr.add(3)).to_f32(),
        ];
        v128_load(values.as_ptr().cast())
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        <CurrentCpu as Cpu>::vec_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        <CurrentCpu as Cpu>::vec_fma(a, b, c)
    }

    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
        let mut values = [0f32; EPR];
        v128_store(values.as_mut_ptr().cast(), a);
        for (i, value) in values.into_iter().enumerate() {
            *mem_addr.add(i) = f16::from_f32(value);
        }
    }

    unsafe fn vec_reduce(x: Self::Array, y: *mut f32) {
        <CurrentCpu as Cpu>::vec_reduce(x, y)
    }
}

// bf16 is the upper half of an f32. Widen its bits before doing arithmetic in
// f32x4, then use half's scalar conversion to preserve round-to-even on store.
pub struct CurrentCpuBF16 {}

impl CpuBF16 for CurrentCpuBF16 {
    type Unit = v128;
    type Array = [v128; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        <CurrentCpu as Cpu>::zero()
    }

    unsafe fn zero_array() -> Self::Array {
        <CurrentCpu as Cpu>::zero_array()
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        <CurrentCpu as Cpu>::from_f32(v)
    }

    unsafe fn load(mem_addr: *const bf16) -> Self::Unit {
        let values = v128_load64_zero(mem_addr.cast());
        u32x4_shl(u32x4_extend_low_u16x8(values), 16)
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        <CurrentCpu as Cpu>::vec_add(a, b)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        <CurrentCpu as Cpu>::vec_fma(a, b, c)
    }

    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit) {
        let mut values = [0f32; EPR];
        v128_store(values.as_mut_ptr().cast(), a);
        for (i, value) in values.into_iter().enumerate() {
            *mem_addr.add(i) = bf16::from_f32(value);
        }
    }

    unsafe fn vec_reduce(x: Self::Array, y: *mut f32) {
        <CurrentCpu as Cpu>::vec_reduce(x, y)
    }
}
