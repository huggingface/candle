use crate::DeviceSize;
use std::{
    cmp::Ordering,
    error::Error,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    mem,
    num::NonZero,
};

/// Vulkan analog of std's [`Alignment`], stored as a [`DeviceSize`] that is guaranteed to be a
/// valid Vulkan alignment.
///
/// [`Alignment`]: std::ptr::Alignment
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DeviceAlignment(AlignmentEnum);

const _: () = assert!(size_of::<DeviceAlignment>() == size_of::<DeviceSize>());
const _: () = assert!(align_of::<DeviceAlignment>() == align_of::<DeviceSize>());

const _: () = assert!(size_of::<DeviceSize>() >= size_of::<usize>());

impl DeviceAlignment {
    /// The smallest possible alignment, 1.
    pub const MIN: Self = Self(AlignmentEnum::_Align1Shl0);

    /// The largest possible alignment, 2<sup>63</sup>.
    pub const MAX: Self = Self(AlignmentEnum::_Align1Shl63);

    /// Returns the alignment for a type.
    #[inline]
    pub const fn of<T>() -> Self {
        // SAFETY: rustc guarantees that the alignment of types is a power of two.
        unsafe { DeviceAlignment::new_unchecked(align_of::<T>() as DeviceSize) }
    }

    /// Returns the alignment for a value.
    #[inline]
    pub fn of_val<T: ?Sized>(value: &T) -> Self {
        // SAFETY: rustc guarantees that the alignment of types is a power of two.
        unsafe { DeviceAlignment::new_unchecked(align_of_val(value) as DeviceSize) }
    }

    /// Tries to create a `DeviceAlignment` from a [`DeviceSize`], returning [`None`] if it's not a
    /// power of two.
    #[inline]
    pub const fn new(alignment: DeviceSize) -> Option<Self> {
        if alignment.is_power_of_two() {
            Some(unsafe { DeviceAlignment::new_unchecked(alignment) })
        } else {
            None
        }
    }

    /// Creates a `DeviceAlignment` from a [`DeviceSize`] without checking if it's a power of two.
    ///
    /// # Safety
    ///
    /// - `alignment` must be a power of two, which also means it must be non-zero.
    #[inline]
    pub const unsafe fn new_unchecked(alignment: DeviceSize) -> Self {
        debug_assert!(alignment.is_power_of_two());

        unsafe { mem::transmute::<DeviceSize, DeviceAlignment>(alignment) }
    }

    /// Returns the alignment as a [`DeviceSize`].
    #[inline]
    pub const fn as_devicesize(self) -> DeviceSize {
        self.0 as DeviceSize
    }

    /// Returns the alignment as a `NonZero<DeviceSize>`.
    #[inline]
    pub const fn as_nonzero(self) -> NonZero<DeviceSize> {
        // SAFETY: All the discriminants are non-zero.
        unsafe { NonZero::new_unchecked(self.as_devicesize()) }
    }

    /// Returns the base-2 logarithm of the alignment.
    #[inline]
    pub const fn log2(self) -> u32 {
        self.as_nonzero().trailing_zeros()
    }

    // TODO: Replace with `Ord::max` once its constness is stabilized.
    #[inline(always)]
    pub(crate) const fn max(self, other: Self) -> Self {
        if self.as_devicesize() >= other.as_devicesize() {
            self
        } else {
            other
        }
    }
}

impl Debug for DeviceAlignment {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?} (1 << {:?})", self.as_nonzero(), self.log2())
    }
}

impl Default for DeviceAlignment {
    #[inline]
    fn default() -> Self {
        DeviceAlignment::MIN
    }
}

impl TryFrom<NonZero<DeviceSize>> for DeviceAlignment {
    type Error = TryFromIntError;

    #[inline]
    fn try_from(alignment: NonZero<DeviceSize>) -> Result<Self, Self::Error> {
        if alignment.is_power_of_two() {
            Ok(unsafe { DeviceAlignment::new_unchecked(alignment.get()) })
        } else {
            Err(TryFromIntError)
        }
    }
}

impl TryFrom<DeviceSize> for DeviceAlignment {
    type Error = TryFromIntError;

    #[inline]
    fn try_from(alignment: DeviceSize) -> Result<Self, Self::Error> {
        DeviceAlignment::new(alignment).ok_or(TryFromIntError)
    }
}

impl From<DeviceAlignment> for NonZero<DeviceSize> {
    #[inline]
    fn from(alignment: DeviceAlignment) -> Self {
        alignment.as_nonzero()
    }
}

impl From<DeviceAlignment> for DeviceSize {
    #[inline]
    fn from(alignment: DeviceAlignment) -> Self {
        alignment.as_devicesize()
    }
}

// This is a false-positive, the underlying values that this impl and the derived `PartialEq` work
// with are the same.
#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for DeviceAlignment {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_nonzero().hash(state);
    }
}

impl PartialOrd for DeviceAlignment {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeviceAlignment {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_nonzero().cmp(&other.as_nonzero())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
enum AlignmentEnum {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
    _Align1Shl16 = 1 << 16,
    _Align1Shl17 = 1 << 17,
    _Align1Shl18 = 1 << 18,
    _Align1Shl19 = 1 << 19,
    _Align1Shl20 = 1 << 20,
    _Align1Shl21 = 1 << 21,
    _Align1Shl22 = 1 << 22,
    _Align1Shl23 = 1 << 23,
    _Align1Shl24 = 1 << 24,
    _Align1Shl25 = 1 << 25,
    _Align1Shl26 = 1 << 26,
    _Align1Shl27 = 1 << 27,
    _Align1Shl28 = 1 << 28,
    _Align1Shl29 = 1 << 29,
    _Align1Shl30 = 1 << 30,
    _Align1Shl31 = 1 << 31,
    _Align1Shl32 = 1 << 32,
    _Align1Shl33 = 1 << 33,
    _Align1Shl34 = 1 << 34,
    _Align1Shl35 = 1 << 35,
    _Align1Shl36 = 1 << 36,
    _Align1Shl37 = 1 << 37,
    _Align1Shl38 = 1 << 38,
    _Align1Shl39 = 1 << 39,
    _Align1Shl40 = 1 << 40,
    _Align1Shl41 = 1 << 41,
    _Align1Shl42 = 1 << 42,
    _Align1Shl43 = 1 << 43,
    _Align1Shl44 = 1 << 44,
    _Align1Shl45 = 1 << 45,
    _Align1Shl46 = 1 << 46,
    _Align1Shl47 = 1 << 47,
    _Align1Shl48 = 1 << 48,
    _Align1Shl49 = 1 << 49,
    _Align1Shl50 = 1 << 50,
    _Align1Shl51 = 1 << 51,
    _Align1Shl52 = 1 << 52,
    _Align1Shl53 = 1 << 53,
    _Align1Shl54 = 1 << 54,
    _Align1Shl55 = 1 << 55,
    _Align1Shl56 = 1 << 56,
    _Align1Shl57 = 1 << 57,
    _Align1Shl58 = 1 << 58,
    _Align1Shl59 = 1 << 59,
    _Align1Shl60 = 1 << 60,
    _Align1Shl61 = 1 << 61,
    _Align1Shl62 = 1 << 62,
    _Align1Shl63 = 1 << 63,
}

/// Error that can happen when trying to convert an integer to a `DeviceAlignment`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryFromIntError;

impl Error for TryFromIntError {}

impl Display for TryFromIntError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("attempted to convert a non-power-of-two integer to a `DeviceAlignment`")
    }
}
