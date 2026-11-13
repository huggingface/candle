use crate::format::Format;

/// Implements the `Vertex` trait on a struct.
///
/// # Examples
///
/// ```
/// # use vulkano::buffer::BufferContents;
/// #[derive(BufferContents, Default)]
/// #[repr(C)]
/// struct Vertex {
///     position: [f32; 3],
///     color: [f32; 4],
/// }
///
/// vulkano::impl_vertex!(Vertex, position, color);
/// ```
#[deprecated(
    since = "0.33.0",
    note = "derive `Vertex` instead and use field-level attributes to specify format"
)]
#[macro_export]
macro_rules! impl_vertex {
    ($out:ty $(, $member:ident)*) => (
        #[allow(unsafe_code)]
        unsafe impl $crate::pipeline::graphics::vertex_input::Vertex for $out {
            #[inline(always)]
            #[allow(deprecated)]
            fn per_vertex() -> $crate::pipeline::graphics::vertex_input::VertexBufferDescription {
                #[allow(unused_imports)]
                use std::collections::HashMap;
                use $crate::format::Format;
                use $crate::pipeline::graphics::vertex_input::VertexMember;
                use $crate::pipeline::graphics::vertex_input::{VertexInputRate, VertexMemberInfo};

                let mut members = HashMap::default();
                $(
                    {
                        let dummy = <$out>::default();
                        #[inline] fn f<T: VertexMember>(_: &T) -> Format { T::format() }
                        let format = f(&dummy.$member);
                        let field_size = {
                            let dummy_ptr: *const $out = <*const _>::cast(&dummy);
                            let member_ptr = unsafe {
                                &raw const (*dummy_ptr).$member
                            };
                            const fn size_of_raw<T>(_: *const T) -> usize {
                                core::mem::size_of::<T>()
                            }
                            size_of_raw(member_ptr)
                        } as u32;
                        let format_size = format.block_size() as u32;
                        let num_elements = field_size / format_size;
                        let remainder = field_size % format_size;
                        assert!(remainder == 0, "struct field `{}` size does not fit multiple of format size", stringify!($member));

                        let dummy_ptr = &raw const dummy;
                        let member_ptr = &raw const dummy.$member;

                        members.insert(stringify!($member).to_string(), VertexMemberInfo {
                            offset: u32::try_from(member_ptr as usize - dummy_ptr as usize).unwrap(),
                            format,
                            num_elements,
                            stride: format_size,
                        });
                    }
                )*

                $crate::pipeline::graphics::vertex_input::VertexBufferDescription {
                    members,
                    stride: std::mem::size_of::<$out>() as u32,
                    input_rate: VertexInputRate::Vertex,
                }
            }
            #[inline(always)]
            #[allow(deprecated)]
            fn per_instance() -> $crate::pipeline::graphics::vertex_input::VertexBufferDescription {
                <$out as $crate::pipeline::graphics::vertex_input::Vertex>::per_vertex().per_instance()
            }
            #[inline(always)]
            #[allow(deprecated)]
            fn per_instance_with_divisor(divisor: u32) -> $crate::pipeline::graphics::vertex_input::VertexBufferDescription {
                <$out as $crate::pipeline::graphics::vertex_input::Vertex>::per_vertex().per_instance_with_divisor(divisor)
            }

        }
    )
}

/// Trait for data types that can be used as vertex members. Used by the `impl_vertex!` macro.
#[deprecated(
    since = "0.33.0",
    note = "derive `Vertex` instead and use field-level attributes to specify format"
)]
pub unsafe trait VertexMember {
    /// Returns the format and array size of the member.
    fn format() -> Format;
}

#[macro_export]
macro_rules! impl_vertex_member {
    ($out:ty, $format:ident) => {
        #[allow(deprecated)]
        unsafe impl VertexMember for $out {
            #[inline]
            fn format() -> Format {
                Format::$format
            }
        }
    };
}

impl_vertex_member!(i8, R8_SINT);
impl_vertex_member!(u8, R8_UINT);
impl_vertex_member!(i16, R16_SINT);
impl_vertex_member!(u16, R16_UINT);
impl_vertex_member!(i32, R32_SINT);
impl_vertex_member!(u32, R32_UINT);
impl_vertex_member!(f32, R32_SFLOAT);
impl_vertex_member!(f64, R64_SFLOAT);
impl_vertex_member!([i8; 2], R8G8_SINT);
impl_vertex_member!([u8; 2], R8G8_UINT);
impl_vertex_member!([i16; 2], R16G16_SINT);
impl_vertex_member!([u16; 2], R16G16_UINT);
impl_vertex_member!([i32; 2], R32G32_SINT);
impl_vertex_member!([u32; 2], R32G32_UINT);
impl_vertex_member!([f32; 2], R32G32_SFLOAT);
impl_vertex_member!([f64; 2], R64G64_SFLOAT);
impl_vertex_member!([i8; 3], R8G8B8_SINT);
impl_vertex_member!([u8; 3], R8G8B8_UINT);
impl_vertex_member!([i16; 3], R16G16B16_SINT);
impl_vertex_member!([u16; 3], R16G16B16_UINT);
impl_vertex_member!([i32; 3], R32G32B32_SINT);
impl_vertex_member!([u32; 3], R32G32B32_UINT);
impl_vertex_member!([f32; 3], R32G32B32_SFLOAT);
impl_vertex_member!([f64; 3], R64G64B64_SFLOAT);
impl_vertex_member!([i8; 4], R8G8B8A8_SINT);
impl_vertex_member!([u8; 4], R8G8B8A8_UINT);
impl_vertex_member!([i16; 4], R16G16B16A16_SINT);
impl_vertex_member!([u16; 4], R16G16B16A16_UINT);
impl_vertex_member!([i32; 4], R32G32B32A32_SINT);
impl_vertex_member!([u32; 4], R32G32B32A32_UINT);
impl_vertex_member!([f32; 4], R32G32B32A32_SFLOAT);
impl_vertex_member!([f64; 4], R64G64B64A64_SFLOAT);
impl_vertex_member!([f32; 9], R32G32B32_SFLOAT);
impl_vertex_member!([f32; 16], R32G32B32A32_SFLOAT);

#[allow(deprecated)]
unsafe impl<T> VertexMember for [T; 1]
where
    T: VertexMember,
{
    fn format() -> Format {
        <T as VertexMember>::format()
    }
}

#[cfg(test)]
mod tests {
    use crate::format::Format;
    #[allow(deprecated)]
    use crate::pipeline::graphics::vertex_input::Vertex;
    use bytemuck::{Pod, Zeroable};

    #[test]
    #[allow(deprecated)]
    fn impl_vertex() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default)]
        struct TestVertex {
            matrix: [f32; 16],
            vector: [f32; 4],
            scalar: u16,
            _padding: u16,
        }

        unsafe impl Pod for TestVertex {}
        unsafe impl Zeroable for TestVertex {}

        impl_vertex!(TestVertex, scalar, vector, matrix);

        let info = TestVertex::per_vertex();
        let matrix = info.members.get("matrix").unwrap();
        let vector = info.members.get("vector").unwrap();
        let scalar = info.members.get("scalar").unwrap();
        assert_eq!(matrix.format, Format::R32G32B32A32_SFLOAT);
        assert_eq!(matrix.offset, 0);
        assert_eq!(matrix.num_elements, 4);
        assert_eq!(vector.format, Format::R32G32B32A32_SFLOAT);
        assert_eq!(vector.offset, 16 * 4);
        assert_eq!(vector.num_elements, 1);
        assert_eq!(scalar.format, Format::R16_UINT);
        assert_eq!(scalar.offset, 16 * 5);
        assert_eq!(scalar.num_elements, 1);
    }
}
