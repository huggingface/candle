pub trait VecOps: num_traits::NumAssign + Copy {
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;

    /// Wide accumulator used when summing many `Self` values one at a time (e.g. a
    /// reduction over a non-contiguous or non-trailing axis). Narrow floats override
    /// this with `f32` so long running sums don't saturate the way a plain `Self +=`
    /// would; every other type just accumulates in itself.
    type Accum: Copy;

    /// Seed the accumulator from a starting `Self` value (usually `Self::zero()`).
    fn to_accum(self) -> Self::Accum;

    /// Fold `val` into `acc`.
    fn accum_add(acc: &mut Self::Accum, val: Self);

    /// Narrow the accumulator back down to `Self` once the reduction is complete.
    fn from_accum(acc: Self::Accum) -> Self;

    /// Element-wise addition of two slices into a third.
    #[inline(always)]
    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        lhs.iter()
            .zip(rhs)
            .zip(res)
            .for_each(|((&a, &b), y)| *y = a + b)
    }

    /// Add a broadcast scalar to every element of a slice: `ys[i] = xs[i] + scalar`.
    #[inline(always)]
    fn scalar_add(scalar: Self, xs: &[Self], ys: &mut [Self]) {
        xs.iter().zip(ys).for_each(|(&x, y)| *y = x + scalar)
    }

    /// Dot-product of two vectors.
    ///
    /// # Safety
    ///
    /// The length of `lhs` and `rhs` have to be at least `len`. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *lhs.add(i) * *rhs.add(i)
        }
    }

    /// Sum of all elements in a vector.
    ///
    /// # Safety
    ///
    /// The length of `xs` must be at least `len`. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len: usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *xs.add(i)
        }
    }

    /// Maximum element in a non-empty vector.
    ///
    /// # Safety
    ///
    /// The length of `xs` must be at least `len` and positive. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_reduce_max(xs: *const Self, res: *mut Self, len: usize) {
        *res = *xs;
        for i in 1..len {
            *res = (*res).max(*xs.add(i))
        }
    }

    /// Minimum element in a non-empty vector.
    ///
    /// # Safety
    ///
    /// The length of `xs` must be at least `len` and positive. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_reduce_min(xs: *const Self, res: *mut Self, len: usize) {
        *res = *xs;
        for i in 1..len {
            *res = (*res).min(*xs.add(i))
        }
    }
}

impl VecOps for f32 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }

    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        #[cfg(feature = "mkl")]
        crate::mkl::vs_add(lhs, rhs, res);
        #[cfg(all(feature = "accelerate", not(feature = "mkl")))]
        crate::accelerate::vs_add(lhs, rhs, res);
        #[cfg(not(any(feature = "mkl", feature = "accelerate")))]
        lhs.iter()
            .zip(rhs)
            .zip(res)
            .for_each(|((&a, &b), y)| *y = a + b)
    }

    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        super::vec_dot_f32(lhs, rhs, res, len)
    }

    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len: usize) {
        super::vec_sum(xs, res, len)
    }
}

impl VecOps for half::f16 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        unsafe { super::vec_add_f16(lhs.as_ptr(), rhs.as_ptr(), res.as_mut_ptr(), lhs.len()) }
    }

    #[inline(always)]
    fn scalar_add(scalar: Self, xs: &[Self], ys: &mut [Self]) {
        unsafe { super::vec_scalar_add_f16(scalar, xs.as_ptr(), ys.as_mut_ptr(), xs.len()) }
    }

    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        let mut res_f32 = 0f32;
        super::vec_dot_f16(lhs, rhs, &mut res_f32, len);
        *res = half::f16::from_f32(res_f32);
    }

    // Accumulate in f32: a sequential sum in f16 saturates on long axes (e.g. an
    // f16 sum of 4096 ones stalls at 2048 once the running ULP exceeds 1),
    // giving wrong `sum`/`mean` results that also disagree with the Metal
    // backend. This mirrors the f32 accumulation already used by `vec_dot`.
    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len: usize) {
        let mut sum = 0f32;
        for i in 0..len {
            sum += (*xs.add(i)).to_f32();
        }
        *res = half::f16::from_f32(sum);
    }

    type Accum = f32;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self.to_f32()
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val.to_f32();
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        half::f16::from_f32(acc)
    }
}

impl VecOps for f64 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    #[inline(always)]
    fn vec_add(lhs: &[f64], rhs: &[f64], res: &mut [f64]) {
        #[cfg(feature = "mkl")]
        crate::mkl::vd_add(lhs, rhs, res);
        #[cfg(all(feature = "accelerate", not(feature = "mkl")))]
        crate::accelerate::vd_add(lhs, rhs, res);
        #[cfg(not(any(feature = "mkl", feature = "accelerate")))]
        lhs.iter()
            .zip(rhs)
            .zip(res)
            .for_each(|((&a, &b), y)| *y = a + b)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}
impl VecOps for half::bf16 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    #[inline(always)]
    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        unsafe { super::vec_add_bf16(lhs.as_ptr(), rhs.as_ptr(), res.as_mut_ptr(), lhs.len()) }
    }

    #[inline(always)]
    fn scalar_add(scalar: Self, xs: &[Self], ys: &mut [Self]) {
        unsafe { super::vec_scalar_add_bf16(scalar, xs.as_ptr(), ys.as_mut_ptr(), xs.len()) }
    }

    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        let mut res_f32 = 0f32;
        super::vec_dot_bf16(lhs, rhs, &mut res_f32, len);
        *res = half::bf16::from_f32(res_f32);
    }

    // Accumulate in f32: a sequential sum in bf16 saturates on long axes (bf16
    // has only 8 mantissa bits, so the running sum stalls even sooner than
    // f16), giving wrong `sum`/`mean` results that also disagree with the Metal
    // backend. This mirrors the f32 accumulation already used by `vec_dot`.
    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len: usize) {
        let mut sum = 0f32;
        for i in 0..len {
            sum += (*xs.add(i)).to_f32();
        }
        *res = half::bf16::from_f32(sum);
    }

    type Accum = f32;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self.to_f32()
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val.to_f32();
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        half::bf16::from_f32(acc)
    }
}
impl VecOps for u8 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        <Self as Ord>::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        <Self as Ord>::max(self, other)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}
impl VecOps for u32 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        <Self as Ord>::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        <Self as Ord>::max(self, other)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}
impl VecOps for i16 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        <Self as Ord>::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        <Self as Ord>::max(self, other)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}
impl VecOps for i32 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        <Self as Ord>::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        <Self as Ord>::max(self, other)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}
impl VecOps for i64 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        <Self as Ord>::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        <Self as Ord>::max(self, other)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}

impl VecOps for float8::F8E4M3 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    type Accum = Self;
    #[inline(always)]
    fn to_accum(self) -> Self::Accum {
        self
    }
    #[inline(always)]
    fn accum_add(acc: &mut Self::Accum, val: Self) {
        *acc += val;
    }
    #[inline(always)]
    fn from_accum(acc: Self::Accum) -> Self {
        acc
    }
}

#[inline(always)]
pub fn par_for_each(n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        func(0)
    } else {
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                s.spawn(move |_| func(thread_idx));
            }
        })
    }
}

#[inline(always)]
pub fn par_range(lo: usize, up: usize, n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        for i in lo..up {
            func(i)
        }
    } else {
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                s.spawn(move |_| {
                    for i in (thread_idx..up).step_by(n_threads) {
                        func(i)
                    }
                });
            }
        })
    }
}
