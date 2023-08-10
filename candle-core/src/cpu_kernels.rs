pub trait VecDot: num_traits::NumAssign + Copy {
    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *lhs.add(i) * *rhs.add(i)
        }
    }
}

impl VecDot for f32 {
    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        ggblas::ggml::vec_dot_f32(lhs, rhs, res, len)
    }
}

impl VecDot for f64 {}
impl VecDot for half::bf16 {}
impl VecDot for half::f16 {}
impl VecDot for u8 {}
impl VecDot for u32 {}
