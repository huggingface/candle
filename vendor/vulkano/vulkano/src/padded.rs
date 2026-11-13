//! A newtype wrapper for enforcing correct alignment for external types.

use crate::buffer::{BufferContents, BufferContentsLayout};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    alloc::Layout,
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// A newtype wrapper around `T`, with `N` bytes of trailing padding.
///
/// In Vulkan, the layout of buffer contents is not necessarily as one would expect from the type
/// signature in the shader code. For example, the *extended layout* or *std140 layout* in GLSL,
/// which is used for uniform buffers by default, requires that array elements are aligned to 16
/// bytes at minimum. That means that even if the array contains a scalar type like `u32` for
/// example, it must be aligned to 16 bytes. We can not enforce that with primitive Rust types
/// alone. In such cases, we can use `Padded` to enforce correct alignment on the Rust side.
///
/// See also [the `shader` module documentation] for more information about layout in shaders.
///
/// # Examples
///
/// ## Aligning struct members
///
/// Consider this GLSL code:
///
/// ```glsl
/// layout(binding = 0) uniform MyData {
///     int x;
///     vec3 y;
///     vec4 z;
/// };
/// ```
///
/// By default, the alignment rules require that `y` and `z` are placed at an offset that is an
/// integer multiple of 16. However, `x` is only 4 bytes, which means that there must be 12 bytes
/// of padding between `x` and `y`. Furthermore, `y` is only 12 bytes, which means that there must
/// be 4 bytes of padding between `y` and `z`.
///
/// We can model this in Rust using `Padded`:
///
/// ```
/// # use vulkano::{buffer::BufferContents, padded::Padded};
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     x: Padded<i32, 12>,
///     y: Padded<[f32; 3], 4>,
///     z: [f32; 4],
/// }
///
/// let data = MyData {
///     x: Padded(42),
///     y: Padded([1.0, 2.0, 3.0]),
///     z: [10.0; 4],
/// };
/// ```
///
/// **But note that this layout is extremely suboptimal.** What you should do instead is reorder
/// your fields such that you don't need any padding:
///
/// ```glsl
/// layout(binding = 0) uniform MyData {
///     vec3 y;
///     int x;
///     vec4 z;
/// };
/// ```
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     y: [f32; 3],
///     x: i32,
///     z: [f32; 4],
/// }
/// ```
///
/// This way, the fields are aligned naturally. But reordering fields is not always an option: the
/// notable case being when your structure only contains `vec3`s and `vec4`s, or `vec3`s and
/// `vec2`s, so that there are no scalar fields to fill the gaps with.
///
/// ## Aligning array elements
///
/// If you need an array of `vec3`s, then that necessitates that each array element has 4 bytes of
/// trailing padding. The same goes for a matrix with 3 rows, each column will have to have 4 bytes
/// of trailing padding (assuming its column-major).
///
/// We can model those using `Padded` too:
///
/// ```glsl
/// layout(binding = 0) uniform MyData {
///     vec3 x[10];
///     mat3 y;
/// };
/// ```
///
/// ```
/// # use vulkano::{buffer::BufferContents, padded::Padded};
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     x: [Padded<[f32; 3], 4>; 10],
///     y: [Padded<[f32; 3], 4>; 3],
/// }
/// ```
///
/// Another example would be if you have an array of scalars or `vec2`s inside a uniform block:
///
/// ```glsl
/// layout(binding = 0) uniform MyData {
///     int x[10];
///     vec2 y[10];
/// };
/// ```
///
/// By default, arrays inside uniform blocks must have their elements aligned to 16 bytes at
/// minimum, which would look like this in Rust:
///
/// ```
/// # use vulkano::{buffer::BufferContents, padded::Padded};
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     x: [Padded<i32, 12>; 10],
///     y: [Padded<[f32; 2], 8>; 10],
/// }
/// ```
///
/// **But note again, that this layout is suboptimal.** You can instead use a buffer block instead
/// of the uniform block, if memory usage could become an issue:
///
/// ```glsl
/// layout(binding = 0) buffer MyData {
///     int x[10];
///     vec2 y[10];
/// };
/// ```
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents)]
/// #[repr(C)]
/// struct MyData {
///     x: [i32; 10],
///     y: [[f32; 2]; 10],
/// }
/// ```
///
/// You may also want to consider using [the `uniform_buffer_standard_layout` feature].
///
/// [the `shader` module documentation]: crate::shader
/// [the `uniform_buffer_standard_layout` feature]: crate::device::DeviceFeatures::uniform_buffer_standard_layout
#[repr(C)]
pub struct Padded<T, const N: usize> {
    value: T,
    _padding: [MaybeUninit<u8>; N],
}

#[allow(non_snake_case)]
#[doc(hidden)]
#[inline(always)]
pub const fn Padded<T, const N: usize>(value: T) -> Padded<T, N> {
    Padded {
        value,
        _padding: [MaybeUninit::uninit(); N],
    }
}

impl<T, const N: usize> AsRef<T> for Padded<T, N> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T, const N: usize> AsMut<T> for Padded<T, N> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T, const N: usize> Clone for Padded<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Padded(self.value.clone())
    }
}

impl<T, const N: usize> Copy for Padded<T, N> where T: Copy {}

impl<T, const N: usize> Debug for Padded<T, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.value.fmt(f)
    }
}

impl<T, const N: usize> Default for Padded<T, N>
where
    T: Default,
{
    fn default() -> Self {
        Padded(T::default())
    }
}

impl<T, const N: usize> Deref for Padded<T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T, const N: usize> DerefMut for Padded<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T, const N: usize> Display for Padded<T, N>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.value.fmt(f)
    }
}

impl<T, const N: usize> From<T> for Padded<T, N> {
    fn from(value: T) -> Self {
        Padded(value)
    }
}

impl<T, const N: usize> PartialEq for Padded<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T, const N: usize> Eq for Padded<T, N> where T: Eq {}

impl<T, const N: usize> Hash for Padded<T, N>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T, const N: usize> PartialOrd for Padded<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T, const N: usize> Ord for Padded<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

unsafe impl<T, const N: usize> BufferContents for Padded<T, N>
where
    T: BufferContents,
{
    const LAYOUT: BufferContentsLayout = BufferContentsLayout::from_sized(Layout::new::<Self>());

    unsafe fn ptr_from_slice(slice: NonNull<[u8]>) -> *mut Self {
        <*mut [u8]>::cast::<Padded<T, N>>(slice.as_ptr())
    }
}

#[cfg(feature = "serde")]
impl<T, const N: usize> Serialize for Padded<T, N>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize> Deserialize<'de> for Padded<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Padded)
    }
}
