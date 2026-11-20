use super::align_up;
use crate::{buffer::BufferContents, macros::try_opt, memory::DeviceAlignment, DeviceSize};
use std::{
    alloc::Layout,
    error::Error,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::Hash,
    num::NonZero,
};

/// Vulkan analog of std's [`Layout`], represented using [`DeviceSize`]s.
///
/// Unlike `Layout`s, `DeviceLayout`s are required to have non-zero size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceLayout {
    size: NonZero<DeviceSize>,
    alignment: DeviceAlignment,
}

const _: () = assert!(size_of::<DeviceSize>() >= size_of::<usize>());
const _: () = assert!(DeviceLayout::MAX_SIZE >= isize::MAX as DeviceSize);

impl DeviceLayout {
    /// The maximum size of a memory block after its layout's size has been rounded up to the
    /// nearest multiple of its layout's alignment.
    ///
    /// This invariant is enforced to avoid arithmetic overflows when constructing layouts and when
    /// allocating memory. Any layout that doesn't uphold this invariant will therefore *lead to
    /// undefined behavior*.
    pub const MAX_SIZE: DeviceSize = DeviceAlignment::MAX.as_devicesize() - 1;

    /// Creates a new `DeviceLayout` from a [`Layout`], or returns [`None`] if the `Layout` has
    /// zero size.
    #[inline]
    pub const fn from_layout(layout: Layout) -> Option<DeviceLayout> {
        let (size, alignment) = Self::size_alignment_from_layout(&layout);

        if let Some(size) = NonZero::new(size) {
            // SAFETY: Under the precondition that `usize` can't overflow `DeviceSize`, which we
            // checked above, `Layout`'s overflow-invariant is the same if not stricter than that
            // of `DeviceLayout`.
            Some(unsafe { DeviceLayout::new_unchecked(size, alignment) })
        } else {
            None
        }
    }

    /// Converts the `DeviceLayout` into a [`Layout`], or returns [`None`] if the `DeviceLayout`
    /// doesn't meet the invariants of `Layout`.
    #[inline]
    pub const fn into_layout(self) -> Option<Layout> {
        let (size, alignment) = (self.size(), self.alignment().as_devicesize());

        #[cfg(target_pointer_width = "64")]
        {
            // SAFETY: Under the precondition that `DeviceSize` can't overflow `usize`, which we
            // checked above, `DeviceLayout`'s overflow-invariant is the same if not stricter that
            // of `Layout`.
            Some(unsafe { Layout::from_size_align_unchecked(size as usize, alignment as usize) })
        }
        #[cfg(any(target_pointer_width = "32", target_pointer_width = "16"))]
        {
            if size > usize::MAX as DeviceSize || alignment > usize::MAX as DeviceSize {
                None
            } else if let Ok(layout) = Layout::from_size_align(size as usize, alignment as usize) {
                Some(layout)
            } else {
                None
            }
        }
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment`.
    ///
    /// Returns [`None`] if `size` is zero, `alignment` is not a power of two, or if `size` would
    /// exceed [`DeviceLayout::MAX_SIZE`] when rounded up to the nearest multiple of `alignment`.
    #[inline]
    pub const fn from_size_alignment(size: DeviceSize, alignment: DeviceSize) -> Option<Self> {
        let size = try_opt!(NonZero::new(size));
        let alignment = try_opt!(DeviceAlignment::new(alignment));

        DeviceLayout::new(size, alignment)
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment` without doing any
    /// checks.
    ///
    /// # Safety
    ///
    /// - `size` must be non-zero.
    /// - `alignment` must be a power of two, which also means it must be non-zero.
    /// - `size`, when rounded up to the nearest multiple of `alignment`, must not exceed
    ///   [`DeviceLayout::MAX_SIZE`].
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub const unsafe fn from_size_alignment_unchecked(
        size: DeviceSize,
        alignment: DeviceSize,
    ) -> Self {
        let size = unsafe { NonZero::new_unchecked(size) };
        let alignment = unsafe { DeviceAlignment::new_unchecked(alignment) };
        unsafe { DeviceLayout::new_unchecked(size, alignment) }
    }

    /// Creates a new `DeviceLayout` for a sized `T`.
    #[inline]
    pub const fn new_sized<T: BufferContents>() -> DeviceLayout {
        T::LAYOUT.unwrap_sized()
    }

    /// Creates a new `DeviceLayout` for an unsized `T` with an unsized tail of `len` elements.
    ///
    /// Returns [`None`] if `len` is zero or if the total size would exceed
    /// [`DeviceLayout::MAX_SIZE`], unless the `T` is actually sized, in which case this behaves
    /// identically to [`new_sized`] and `len` is ignored.
    ///
    /// [`new_sized`]: Self::new_sized
    #[inline]
    pub const fn new_unsized<T: BufferContents + ?Sized>(len: DeviceSize) -> Option<DeviceLayout> {
        T::LAYOUT.layout_for_len(len)
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment`.
    ///
    /// Returns [`None`] if `size` would exceed [`DeviceLayout::MAX_SIZE`] when rounded up to the
    /// nearest multiple of `alignment`.
    #[inline]
    pub const fn new(size: NonZero<DeviceSize>, alignment: DeviceAlignment) -> Option<Self> {
        if size.get() > Self::max_size_for_alignment(alignment) {
            None
        } else {
            // SAFETY: We checked that the rounded-up size won't exceed `DeviceLayout::MAX_SIZE`.
            Some(unsafe { DeviceLayout::new_unchecked(size, alignment) })
        }
    }

    #[inline(always)]
    const fn max_size_for_alignment(alignment: DeviceAlignment) -> DeviceSize {
        // `DeviceLayout::MAX_SIZE` is `DeviceAlignment::MAX - 1`, so this can't overflow.
        DeviceLayout::MAX_SIZE - (alignment.as_devicesize() - 1)
    }

    /// Creates a new `DeviceLayout` from the given `size` and `alignment` without checking for
    /// potential overflow.
    ///
    /// # Safety
    ///
    /// - `size`, when rounded up to the nearest multiple of `alignment`, must not exceed
    ///   [`DeviceLayout::MAX_SIZE`].
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub const unsafe fn new_unchecked(
        size: NonZero<DeviceSize>,
        alignment: DeviceAlignment,
    ) -> Self {
        debug_assert!(size.get() <= Self::max_size_for_alignment(alignment));

        DeviceLayout { size, alignment }
    }

    /// Creates a new `DeviceLayout` for the given `value`.
    ///
    /// Returns [`None`] if the value is zero-sized.
    #[inline]
    pub fn for_value<T: BufferContents + ?Sized>(value: &T) -> Option<DeviceLayout> {
        DeviceLayout::from_layout(Layout::for_value(value))
    }

    /// Returns the minimum size in bytes for a memory block of this layout.
    #[inline]
    pub const fn size(&self) -> DeviceSize {
        self.size.get()
    }

    /// Returns the minimum alignment for a memory block of this layout.
    #[inline]
    pub const fn alignment(&self) -> DeviceAlignment {
        self.alignment
    }

    /// Creates a new `DeviceLayout` from `self` that is also aligned to `alignment` at minimum.
    ///
    /// Returns [`None`] if `self.size()` would overflow [`DeviceLayout::MAX_SIZE`] when rounded up
    /// to the nearest multiple of `alignment`.
    #[inline]
    pub const fn align_to(&self, alignment: DeviceAlignment) -> Option<Self> {
        DeviceLayout::new(self.size, DeviceAlignment::max(self.alignment, alignment))
    }

    /// Returns the amount of padding that needs to be added to `self.size()` such that the result
    /// is a multiple of `alignment`.
    #[inline]
    pub const fn padding_needed_for(&self, alignment: DeviceAlignment) -> DeviceSize {
        let size = self.size();

        align_up(size, alignment).wrapping_sub(size)
    }

    /// Creates a new `DeviceLayout` by rounding up `self.size()` to the nearest multiple of
    /// `self.alignment()`.
    #[inline]
    pub const fn pad_to_alignment(&self) -> Self {
        // SAFETY: `DeviceLayout`'s invariant guarantees that the rounded up size won't exceed
        // `DeviceLayout::MAX_SIZE`.
        unsafe { DeviceLayout::new_unchecked(self.padded_size(), self.alignment) }
    }

    #[inline(always)]
    const fn padded_size(&self) -> NonZero<DeviceSize> {
        let size = align_up(self.size(), self.alignment);

        // SAFETY: `DeviceLayout`'s invariant guarantees that the rounded up size won't overflow.
        unsafe { NonZero::new_unchecked(size) }
    }

    /// Creates a new `DeviceLayout` describing the record for `n` instances of `self`, possibly
    /// with padding at the end of each to ensure correct alignment of all instances.
    ///
    /// Returns a tuple consisting of the new layout and the stride, in bytes, of `self`, or
    /// returns [`None`] if `n` is zero or when the total size would exceed
    /// [`DeviceLayout::MAX_SIZE`].
    #[inline]
    pub const fn repeat(&self, n: DeviceSize) -> Option<(Self, DeviceSize)> {
        let n = try_opt!(NonZero::new(n));
        let stride = self.padded_size();
        let size = try_opt!(stride.checked_mul(n));
        let layout = try_opt!(DeviceLayout::new(size, self.alignment));

        Some((layout, stride.get()))
    }

    /// Creates a new `DeviceLayout` describing the record for `self` followed by `next`, including
    /// potential padding between them to ensure `next` will be properly aligned, but without any
    /// trailing padding. You should use [`pad_to_alignment`] after you are done extending the
    /// layout with all fields to get a valid `#[repr(C)]` layout.
    ///
    /// The alignments of the two layouts get combined by picking the maximum between them.
    ///
    /// Returns a tuple consisting of the resulting layout as well as the offset, in bytes, of
    /// `next`.
    ///
    /// Returns [`None`] on arithmetic overflow or when the total size rounded up to the nearest
    /// multiple of the combined alignment would exceed [`DeviceLayout::MAX_SIZE`].
    ///
    /// [`pad_to_alignment`]: Self::pad_to_alignment
    #[inline]
    pub const fn extend(&self, next: Self) -> Option<(Self, DeviceSize)> {
        self.extend_inner(next.size(), next.alignment)
    }

    /// Same as [`extend`], except it extends with a [`Layout`].
    ///
    /// [`extend`]: Self::extend
    #[inline]
    pub const fn extend_with_layout(&self, next: Layout) -> Option<(Self, DeviceSize)> {
        let (next_size, next_alignment) = Self::size_alignment_from_layout(&next);

        self.extend_inner(next_size, next_alignment)
    }

    const fn extend_inner(
        &self,
        next_size: DeviceSize,
        next_alignment: DeviceAlignment,
    ) -> Option<(Self, DeviceSize)> {
        let padding = self.padding_needed_for(next_alignment);
        let offset = try_opt!(self.size.checked_add(padding));
        let size = try_opt!(offset.checked_add(next_size));
        let alignment = DeviceAlignment::max(self.alignment, next_alignment);
        let layout = try_opt!(DeviceLayout::new(size, alignment));

        Some((layout, offset.get()))
    }

    /// Creates a new `DeviceLayout` describing the record for the `previous` [`Layout`] followed
    /// by `self`. This function is otherwise the same as [`extend`].
    ///
    /// [`extend`]: Self::extend
    #[inline]
    pub const fn extend_from_layout(self, previous: &Layout) -> Option<(Self, DeviceSize)> {
        let (size, alignment) = Self::size_alignment_from_layout(previous);

        let padding = align_up(size, self.alignment).wrapping_sub(size);
        let offset = try_opt!(size.checked_add(padding));
        let size = try_opt!(self.size.checked_add(offset));
        let alignment = DeviceAlignment::max(alignment, self.alignment);
        let layout = try_opt!(DeviceLayout::new(size, alignment));

        Some((layout, offset))
    }

    #[inline(always)]
    const fn size_alignment_from_layout(layout: &Layout) -> (DeviceSize, DeviceAlignment) {
        // We checked that `usize` can't overflow `DeviceSize` above, so this can't truncate.
        let (size, alignment) = (layout.size() as DeviceSize, layout.align() as DeviceSize);

        // SAFETY: `Layout`'s alignment-invariant guarantees that it is a power of two.
        let alignment = unsafe { DeviceAlignment::new_unchecked(alignment) };

        (size, alignment)
    }
}

impl TryFrom<Layout> for DeviceLayout {
    type Error = TryFromLayoutError;

    #[inline]
    fn try_from(layout: Layout) -> Result<Self, Self::Error> {
        DeviceLayout::from_layout(layout).ok_or(TryFromLayoutError)
    }
}

impl TryFrom<DeviceLayout> for Layout {
    type Error = TryFromDeviceLayoutError;

    #[inline]
    fn try_from(device_layout: DeviceLayout) -> Result<Self, Self::Error> {
        DeviceLayout::into_layout(device_layout).ok_or(TryFromDeviceLayoutError)
    }
}

/// Error that can happen when converting a [`Layout`] to a [`DeviceLayout`].
///
/// It occurs when the `Layout` has zero size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryFromLayoutError;

impl Error for TryFromLayoutError {}

impl Display for TryFromLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("attempted to convert a zero-size `Layout` to a `DeviceLayout`")
    }
}

/// Error that can happen when converting a [`DeviceLayout`] to a [`Layout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryFromDeviceLayoutError;

impl Error for TryFromDeviceLayoutError {}

impl Display for TryFromDeviceLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(
            "attempted to convert a `DeviceLayout` to a `Layout` which would result in a \
            violation of `Layout`'s overflow-invariant",
        )
    }
}
