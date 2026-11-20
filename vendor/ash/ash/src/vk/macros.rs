#[macro_export]
macro_rules! vk_bitflags_wrapped {
    ($ name : ident , $ flag_type : ty) => {
        impl Default for $name {
            fn default() -> Self {
                Self(0)
            }
        }
        impl $name {
            #[inline]
            pub const fn empty() -> Self {
                Self(0)
            }
            #[inline]
            pub const fn from_raw(x: $flag_type) -> Self {
                Self(x)
            }
            #[inline]
            pub const fn as_raw(self) -> $flag_type {
                self.0
            }
            #[inline]
            pub const fn is_empty(self) -> bool {
                self.0 == Self::empty().0
            }
            #[inline]
            pub const fn intersects(self, other: Self) -> bool {
                !Self(self.0 & other.0).is_empty()
            }
            #[doc = r" Returns whether `other` is a subset of `self`"]
            #[inline]
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }
        }
        impl ::core::ops::BitOr for $name {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
        impl ::core::ops::BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs
            }
        }
        impl ::core::ops::BitAnd for $name {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }
        impl ::core::ops::BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs
            }
        }
        impl ::core::ops::BitXor for $name {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }
        impl ::core::ops::BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs
            }
        }
        impl ::core::ops::Not for $name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                Self(!self.0)
            }
        }
    };
}
#[macro_export]
macro_rules! handle_nondispatchable {
    ($ name : ident , $ ty : ident , $ doc_link : meta $(, $ cfgs : meta )?) => {
        #[$doc_link]
        $(#[$cfgs])?
        #[repr(transparent)]
        #[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash, Default)]
        pub struct $name(u64);
        $(#[$cfgs])?
        impl Handle for $name {
            const TYPE: ObjectType = ObjectType::$ty;
            fn as_raw(self) -> u64 {
                self.0
            }
            fn from_raw(x: u64) -> Self {
                Self(x)
            }
        }
        $(#[$cfgs])?
        impl $name {
            pub const fn null() -> Self {
                Self(0)
            }
        }
        $(#[$cfgs])?
        impl fmt::Pointer for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0x{:x}", self.0)
            }
        }
        $(#[$cfgs])?
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "0x{:x}", self.0)
            }
        }
    };
}
#[macro_export]
macro_rules! define_handle {
    ($ name : ident , $ ty : ident , $ doc_link : meta $(, $ cfgs : meta )?) => {
        #[$doc_link]
        $(#[$cfgs])?
        #[repr(transparent)]
        #[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Hash)]
        pub struct $name(*mut u8);
        $(#[$cfgs])?
        impl Default for $name {
            fn default() -> Self {
                Self::null()
            }
        }
        $(#[$cfgs])?
        impl Handle for $name {
            const TYPE: ObjectType = ObjectType::$ty;
            fn as_raw(self) -> u64 {
                self.0 as u64
            }
            fn from_raw(x: u64) -> Self {
                Self(x as _)
            }
        }
        $(#[$cfgs])?
        unsafe impl Send for $name {}
        $(#[$cfgs])?
        unsafe impl Sync for $name {}
        $(#[$cfgs])?
        impl $name {
            pub const fn null() -> Self {
                Self(::core::ptr::null_mut())
            }
        }
        $(#[$cfgs])?
        impl fmt::Pointer for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Pointer::fmt(&self.0, f)
            }
        }
        $(#[$cfgs])?
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }
    };
}
