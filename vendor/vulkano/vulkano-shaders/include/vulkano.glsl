// IMPORTANT NOTE: Names starting with an underscore are private and NOT PART OF THE PUBLIC API.
// They can be changed at any moment without warning.

#ifndef _VULKANO_HEADER
#define _VULKANO_HEADER

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_samplerless_texture_functions : enable

#if VKO_STORAGE_IMAGE_WITHOUT_FORMAT_ENABLED
#extension GL_EXT_shader_image_load_formatted : enable
#endif // VKO_STORAGE_IMAGE_WITHOUT_FORMAT_ENABLED

#if VKO_IMAGE_INT64_ATOMICS_ENABLED
#extension GL_EXT_shader_image_int64 : enable
#endif // VKO_IMAGE_INT64_ATOMICS_ENABLED

// NOTE(Marc): The index getter macros below may look strange, because they are crafted such that
// they enforce type safety while also preserving `nonuniformEXT` qualifiers that the IDs may have.
// That means in particular that we can't use a function which returns the index, as the qualifier
// doesn't get propagated through function calls, however it does get propagated through struct
// decomposition. The hope here is that this leads to the least surprising behavior for the user.

struct SamplerId {
    uint _private_index;
    uint _private_generation;
};

void _vko_assert_is_sampler_id(SamplerId id) {}

#define _vko_sampler_index(id)                                                                     \
    (_vko_assert_is_sampler_id(id), id._private_index)

struct SampledImageId {
    uint _private_index;
    uint _private_generation;
};

void _vko_assert_is_sampled_image_id(SampledImageId id) {}

#define _vko_sampled_image_index(id)                                                               \
    (_vko_assert_is_sampled_image_id(id), id._private_index)

struct StorageImageId {
    uint _private_index;
    uint _private_generation;
};

void _vko_assert_is_storage_image_id(StorageImageId id) {}

#define _vko_storage_image_index(id)                                                               \
    (_vko_assert_is_storage_image_id(id), id._private_index)

struct StorageBufferId {
    uint _private_index;
    uint _private_generation;
};

void _vko_assert_is_storage_buffer_id(StorageBufferId id) {}

#define _vko_storage_buffer_index(id)                                                              \
    (_vko_assert_is_storage_buffer_id(id), id._private_index)

struct AccelerationStructureId {
    uint _private_index;
    uint _private_generation;
};

void _vko_assert_is_acceleration_structure_id(AccelerationStructureId id) {}

#define _vko_acceleration_structure_index(id)                                                      \
    (_vko_assert_is_acceleration_structure_id(id), id._private_index)

// NOTE(Marc): The following constants must match the definitions in
// vulkano_taskgraph/src/descriptor_set.rs!

#define VKO_GLOBAL_SET 0
#define VKO_SAMPLER_BINDING 0
#define VKO_SAMPLED_IMAGE_BINDING 1
#define VKO_STORAGE_IMAGE_BINDING 2
#define VKO_STORAGE_BUFFER_BINDING 3
#define VKO_ACCELERATION_STRUCTURE_BINDING 4

#define VKO_LOCAL_SET 1
#define VKO_INPUT_ATTACHMENT_BINDING 0

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#define _VKO_DECLARE_SAMPLER(TYPE)                                                                 \
    layout(set = VKO_GLOBAL_SET, binding = VKO_SAMPLER_BINDING)                                    \
        uniform TYPE _vko_samplers_##TYPE[];

_VKO_DECLARE_SAMPLER(sampler)
#define vko_sampler(id)                                                                            \
    _vko_samplers_sampler[_vko_sampler_index(id)]

_VKO_DECLARE_SAMPLER(samplerShadow)
#define vko_samplerShadow(id)                                                                      \
    _vko_samplers_samplerShadow[_vko_sampler_index(id)]

#undef _VKO_DECLARE_SAMPLER

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#define _VKO_DECLARE_SAMPLED_IMAGE(TYPE)                                                           \
    layout(set = VKO_GLOBAL_SET, binding = VKO_SAMPLED_IMAGE_BINDING)                              \
        uniform TYPE _vko_sampled_images_##TYPE[];

#define _VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(DIMENSION)                                            \
    _VKO_DECLARE_SAMPLED_IMAGE(texture##DIMENSION)                                                 \
    _VKO_DECLARE_SAMPLED_IMAGE(itexture##DIMENSION)                                                \
    _VKO_DECLARE_SAMPLED_IMAGE(utexture##DIMENSION)

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(1D)
#define vko_texture1D(id)                                                                          \
    _vko_sampled_images_texture1D[_vko_sampled_image_index(id)]
#define vko_itexture1D(id)                                                                         \
    _vko_sampled_images_itexture1D[_vko_sampled_image_index(id)]
#define vko_utexture1D(id)                                                                         \
    _vko_sampled_images_utexture1D[_vko_sampled_image_index(id)]

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(1DArray)
#define vko_texture1DArray(id)                                                                     \
    _vko_sampled_images_texture1DArray[_vko_sampled_image_index(id)]
#define vko_itexture1DArray(id)                                                                    \
    _vko_sampled_images_itexture1DArray[_vko_sampled_image_index(id)]
#define vko_utexture1DArray(id)                                                                    \
    _vko_sampled_images_utexture1DArray[_vko_sampled_image_index(id)]

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(2D)
#define vko_texture2D(id)                                                                          \
    _vko_sampled_images_texture2D[_vko_sampled_image_index(id)]
#define vko_itexture2D(id)                                                                         \
    _vko_sampled_images_itexture2D[_vko_sampled_image_index(id)]
#define vko_utexture2D(id)                                                                         \
    _vko_sampled_images_utexture2D[_vko_sampled_image_index(id)]

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(2DArray)
#define vko_texture2DArray(id)                                                                     \
    _vko_sampled_images_texture2DArray[_vko_sampled_image_index(id)]
#define vko_itexture2DArray(id)                                                                    \
    _vko_sampled_images_itexture2DArray[_vko_sampled_image_index(id)]
#define vko_utexture2DArray(id)                                                                    \
    _vko_sampled_images_utexture2DArray[_vko_sampled_image_index(id)]

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(2DMS)
#define vko_texture2DMS(id)                                                                        \
    _vko_sampled_images_texture2DMS[_vko_sampled_image_index(id)]
#define vko_itexture2DMS(id)                                                                       \
    _vko_sampled_images_itexture2DMS[_vko_sampled_image_index(id)]
#define vko_utexture2DMS(id)                                                                       \
    _vko_sampled_images_utexture2DMS[_vko_sampled_image_index(id)]

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(2DMSArray)
#define vko_texture2DMSArray(id)                                                                   \
    _vko_sampled_images_texture2DMSArray[_vko_sampled_image_index(id)]
#define vko_itexture2DMSArray(id)                                                                  \
    _vko_sampled_images_itexture2DMSArray[_vko_sampled_image_index(id)]
#define vko_utexture2DMSArray(id)                                                                  \
    _vko_sampled_images_utexture2DMSArray[_vko_sampled_image_index(id)]

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(3D)
#define vko_texture3D(id)                                                                          \
    _vko_sampled_images_texture3D[_vko_sampled_image_index(id)]
#define vko_itexture3D(id)                                                                         \
    _vko_sampled_images_itexture3D[_vko_sampled_image_index(id)]
#define vko_utexture3D(id)                                                                         \
    _vko_sampled_images_utexture3D[_vko_sampled_image_index(id)]

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(Cube)
#define vko_textureCube(id)                                                                        \
    _vko_sampled_images_textureCube[_vko_sampled_image_index(id)]
#define vko_itextureCube(id)                                                                       \
    _vko_sampled_images_itextureCube[_vko_sampled_image_index(id)]
#define vko_utextureCube(id)                                                                       \
    _vko_sampled_images_utextureCube[_vko_sampled_image_index(id)]

#if VKO_IMAGE_CUBE_ARRAY_ENABLED

_VKO_DECLARE_SAMPLED_IMAGE_DIMENSION(CubeArray)
#define vko_textureCubeArray(id)                                                                   \
    _vko_sampled_images_textureCubeArray[_vko_sampled_image_index(id)]
#define vko_itextureCubeArray(id)                                                                  \
    _vko_sampled_images_itextureCubeArray[_vko_sampled_image_index(id)]
#define vko_utextureCubeArray(id)                                                                  \
    _vko_sampled_images_utextureCubeArray[_vko_sampled_image_index(id)]

#endif // VKO_IMAGE_CUBE_ARRAY_ENABLED

#undef _VKO_DECLARE_SAMPLED_IMAGE_DIMENSION
#undef _VKO_DECLARE_SAMPLED_IMAGE_TYPE

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#define vko_sampler1D(sampled_image_id, sampler_id)                                                \
    sampler1D(                                                                                     \
        vko_texture1D(sampled_image_id),                                                           \
        vko_sampler(sampler_id))
#define vko_isampler1D(sampled_image_id, sampler_id)                                               \
    isampler1D(                                                                                    \
        vko_itexture1D(sampled_image_id),                                                          \
        vko_sampler(sampler_id))
#define vko_usampler1D(sampled_image_id, sampler_id)                                               \
    usampler1D(                                                                                    \
        vko_utexture1D(sampled_image_id),                                                          \
        vko_sampler(sampler_id))
#define vko_sampler1DShadow(sampled_image_id, sampler_id)                                          \
    sampler1DShadow(                                                                               \
        vko_texture1D(sampled_image_id),                                                           \
        vko_samplerShadow(sampler_id))
#define vko_isampler1DShadow(sampled_image_id, sampler_id)                                         \
    isampler1DShadow(                                                                              \
        vko_itexture1D(sampled_image_id),                                                          \
        vko_samplerShadow(sampler_id))
#define vko_usampler1DShadow(sampled_image_id, sampler_id)                                         \
    usampler1DShadow(                                                                              \
        vko_utexture1D(sampled_image_id),                                                          \
        vko_samplerShadow(sampler_id))

#define vko_sampler1DArray(sampled_image_id, sampler_id)                                           \
    sampler1DArray(                                                                                \
        vko_texture1DArray(sampled_image_id),                                                      \
        vko_sampler(sampler_id))
#define vko_isampler1DArray(sampled_image_id, sampler_id)                                          \
    isampler1DArray(                                                                               \
        vko_itexture1DArray(sampled_image_id),                                                     \
        vko_sampler(sampler_id))
#define vko_usampler1DArray(sampled_image_id, sampler_id)                                          \
    usampler1DArray(                                                                               \
        vko_utexture1DArray(sampled_image_id),                                                     \
        vko_sampler(sampler_id))
#define vko_sampler1DArrayShadow(sampled_image_id, sampler_id)                                     \
    sampler1DArrayShadow(                                                                          \
        vko_texture1DArray(sampled_image_id),                                                      \
        vko_samplerShadow(sampler_id))
#define vko_isampler1DArrayShadow(sampled_image_id, sampler_id)                                    \
    isampler1DArrayShadow(                                                                         \
        vko_itexture1DArray(sampled_image_id),                                                     \
        vko_samplerShadow(sampler_id))
#define vko_usampler1DArrayShadow(sampled_image_id, sampler_id)                                    \
    usampler1DArrayShadow(                                                                         \
        vko_utexture1DArray(sampled_image_id),                                                     \
        vko_samplerShadow(sampler_id))

#define vko_sampler2D(sampled_image_id, sampler_id)                                                \
    sampler2D(                                                                                     \
        vko_texture2D(sampled_image_id),                                                           \
        vko_sampler(sampler_id))
#define vko_isampler2D(sampled_image_id, sampler_id)                                               \
    isampler2D(                                                                                    \
        vko_itexture2D(sampled_image_id),                                                          \
        vko_sampler(sampler_id))
#define vko_usampler2D(sampled_image_id, sampler_id)                                               \
    usampler2D(                                                                                    \
        vko_utexture2D(sampled_image_id),                                                          \
        vko_sampler(sampler_id))
#define vko_sampler2DShadow(sampled_image_id, sampler_id)                                          \
    sampler2DShadow(                                                                               \
        vko_texture2D(sampled_image_id),                                                           \
        vko_samplerShadow(sampler_id))
#define vko_isampler2DShadow(sampled_image_id, sampler_id)                                         \
    isampler2DShadow(                                                                              \
        vko_itexture2D(sampled_image_id),                                                          \
        vko_samplerShadow(sampler_id))
#define vko_usampler2DShadow(sampled_image_id, sampler_id)                                         \
    usampler2DShadow(                                                                              \
        vko_utexture2D(sampled_image_id),                                                          \
        vko_samplerShadow(sampler_id))

#define vko_sampler2DArray(sampled_image_id, sampler_id)                                           \
    sampler2DArray(                                                                                \
        vko_texture2DArray(sampled_image_id),                                                      \
        vko_sampler(sampler_id))
#define vko_isampler2DArray(sampled_image_id, sampler_id)                                          \
    isampler2DArray(                                                                               \
        vko_itexture2DArray(sampled_image_id),                                                     \
        vko_sampler(sampler_id))
#define vko_usampler2DArray(sampled_image_id, sampler_id)                                          \
    usampler2DArray(                                                                               \
        vko_utexture2DArray(sampled_image_id),                                                     \
        vko_sampler(sampler_id))
#define vko_sampler2DArrayShadow(sampled_image_id, sampler_id)                                     \
    sampler2DArrayShadow(                                                                          \
        vko_texture2DArray(sampled_image_id),                                                      \
        vko_samplerShadow(sampler_id))
#define vko_isampler2DArrayShadow(sampled_image_id, sampler_id)                                    \
    isampler2DArrayShadow(                                                                         \
        vko_itexture2DArray(sampled_image_id),                                                     \
        vko_samplerShadow(sampler_id))
#define vko_usampler2DArrayShadow(sampled_image_id, sampler_id)                                    \
    usampler2DArrayShadow(                                                                         \
        vko_utexture2DArray(sampled_image_id),                                                     \
        vko_samplerShadow(sampler_id))

#define vko_sampler2DMS(sampled_image_id, sampler_id)                                              \
    sampler2DMS(                                                                                   \
        vko_texture2DMS(sampled_image_id),                                                         \
        vko_sampler(sampler_id))
#define vko_isampler2DMS(sampled_image_id, sampler_id)                                             \
    isampler2DMS(                                                                                  \
        vko_itexture2DMS(sampled_image_id),                                                        \
        vko_sampler(sampler_id))
#define vko_usampler2DMS(sampled_image_id, sampler_id)                                             \
    usampler2DMS(                                                                                  \
        vko_utexture2DMS(sampled_image_id),                                                        \
        vko_sampler(sampler_id))
#define vko_sampler2DMSShadow(sampled_image_id, sampler_id)                                        \
    sampler2DMSShadow(                                                                             \
        vko_texture2DMS(sampled_image_id),                                                         \
        vko_samplerShadow(sampler_id))
#define vko_isampler2DMSShadow(sampled_image_id, sampler_id)                                       \
    isampler2DMSShadow(                                                                            \
        vko_itexture2DMS(sampled_image_id),                                                        \
        vko_samplerShadow(sampler_id))
#define vko_usampler2DMSShadow(sampled_image_id, sampler_id)                                       \
    usampler2DMSShadow(                                                                            \
        vko_utexture2DMS(sampled_image_id),                                                        \
        vko_samplerShadow(sampler_id))

#define vko_sampler2DMSArray(sampled_image_id, sampler_id)                                         \
    sampler2DMSArray(                                                                              \
        vko_texture2DMSArray(sampled_image_id),                                                    \
        vko_sampler(sampler_id))
#define vko_isampler2DMSArray(sampled_image_id, sampler_id)                                        \
    isampler2DMSArray(                                                                             \
        vko_itexture2DMSArray(sampled_image_id),                                                   \
        vko_sampler(sampler_id))
#define vko_usampler2DMSArray(sampled_image_id, sampler_id)                                        \
    usampler2DMSArray(                                                                             \
        vko_utexture2DMSArray(sampled_image_id),                                                   \
        vko_sampler(sampler_id))
#define vko_sampler2DMSArrayShadow(sampled_image_id, sampler_id)                                   \
    sampler2DMSArrayShadow(                                                                        \
        vko_texture2DMSArray(sampled_image_id),                                                    \
        vko_samplerShadow(sampler_id))
#define vko_isampler2DMSArrayShadow(sampled_image_id, sampler_id)                                  \
    isampler2DMSArrayShadow(                                                                       \
        vko_itexture2DMSArray(sampled_image_id),                                                   \
        vko_samplerShadow(sampler_id))
#define vko_usampler2DMSArrayShadow(sampled_image_id, sampler_id)                                  \
    usampler2DMSArrayShadow(                                                                       \
        vko_utexture2DMSArray(sampled_image_id),                                                   \
        vko_samplerShadow(sampler_id))

#define vko_sampler3D(sampled_image_id, sampler_id)                                                \
    sampler3D(                                                                                     \
        vko_texture3D(sampled_image_id),                                                           \
        vko_sampler(sampler_id))
#define vko_isampler3D(sampled_image_id, sampler_id)                                               \
    isampler3D(                                                                                    \
        vko_itexture3D(sampled_image_id),                                                          \
        vko_sampler(sampler_id))
#define vko_usampler3D(sampled_image_id, sampler_id)                                               \
    usampler3D(                                                                                    \
        vko_utexture3D(sampled_image_id),                                                          \
        vko_sampler(sampler_id))
#define vko_sampler3DShadow(sampled_image_id, sampler_id)                                          \
    sampler3DShadow(                                                                               \
        vko_texture3D(sampled_image_id),                                                           \
        vko_samplerShadow(sampler_id))
#define vko_isampler3DShadow(sampled_image_id, sampler_id)                                         \
    isampler3DShadow(                                                                              \
        vko_itexture3D(sampled_image_id),                                                          \
        vko_samplerShadow(sampler_id))
#define vko_usampler3DShadow(sampled_image_id, sampler_id)                                         \
    usampler3DShadow(                                                                              \
        vko_utexture3D(sampled_image_id),                                                          \
        vko_samplerShadow(sampler_id))

#define vko_samplerCube(sampled_image_id, sampler_id)                                              \
    samplerCube(                                                                                   \
        vko_textureCube(sampled_image_id),                                                         \
        vko_sampler(sampler_id))
#define vko_isamplerCube(sampled_image_id, sampler_id)                                             \
    isamplerCube(                                                                                  \
        vko_itextureCube(sampled_image_id),                                                        \
        vko_sampler(sampler_id))
#define vko_usamplerCube(sampled_image_id, sampler_id)                                             \
    usamplerCube(                                                                                  \
        vko_utextureCube(sampled_image_id),                                                        \
        vko_sampler(sampler_id))
#define vko_samplerCubeShadow(sampled_image_id, sampler_id)                                        \
    samplerCubeShadow(                                                                             \
        vko_textureCube(sampled_image_id),                                                         \
        vko_samplerShadow(sampler_id))
#define vko_isamplerCubeShadow(sampled_image_id, sampler_id)                                       \
    isamplerCubeShadow(                                                                            \
        vko_itextureCube(sampled_image_id),                                                        \
        vko_samplerShadow(sampler_id))
#define vko_usamplerCubeShadow(sampled_image_id, sampler_id)                                       \
    usamplerCubeShadow(                                                                            \
        vko_utextureCube(sampled_image_id),                                                        \
        vko_samplerShadow(sampler_id))

#if VKO_IMAGE_CUBE_ARRAY_ENABLED

#define vko_samplerCubeArray(sampled_image_id, sampler_id)                                         \
    samplerCubeArray(                                                                              \
        vko_textureCubeArray(sampled_image_id),                                                    \
        vko_sampler(sampler_id))
#define vko_isamplerCubeArray(sampled_image_id, sampler_id)                                        \
    isamplerCubeArray(                                                                             \
        vko_itextureCubeArray(sampled_image_id),                                                   \
        vko_sampler(sampler_id))
#define vko_usamplerCubeArray(sampled_image_id, sampler_id)                                        \
    usamplerCubeArray(                                                                             \
        vko_utextureCubeArray(sampled_image_id),                                                   \
        vko_sampler(sampler_id))
#define vko_samplerCubeArrayShadow(sampled_image_id, sampler_id)                                   \
    samplerCubeArrayShadow(                                                                        \
        vko_textureCubeArray(sampled_image_id),                                                    \
        vko_samplerShadow(sampler_id))
#define vko_isamplerCubeArrayShadow(sampled_image_id, sampler_id)                                  \
    isamplerCubeArrayShadow(                                                                       \
        vko_itextureCubeArray(sampled_image_id),                                                   \
        vko_samplerShadow(sampler_id))
#define vko_usamplerCubeArrayShadow(sampled_image_id, sampler_id)                                  \
    usamplerCubeArrayShadow(                                                                       \
        vko_utextureCubeArray(sampled_image_id),                                                   \
        vko_samplerShadow(sampler_id))

#endif // VKO_IMAGE_CUBE_ARRAY_ENABLED

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#define VKO_DECLARE_STORAGE_IMAGE(NAME, TYPE, FORMAT)                                              \
    layout(set = VKO_GLOBAL_SET, binding = VKO_STORAGE_IMAGE_BINDING, FORMAT)                      \
        uniform TYPE _vko_##NAME##_storage_images[];

#define VKO_DECLARE_STORAGE_IMAGE_WITHOUT_FORMAT(NAME, TYPE)                                       \
    layout(set = VKO_GLOBAL_SET, binding = VKO_STORAGE_IMAGE_BINDING)                              \
        uniform TYPE _vko_##NAME##_storage_images[];

#define vko_image(NAME, id)                                                                        \
    _vko_##NAME##_storage_images[_vko_storage_image_index(id)]

#define _VKO_DECLARE_STORAGE_IMAGE(TYPE, FORMAT)                                                   \
    layout(set = VKO_GLOBAL_SET, binding = VKO_STORAGE_IMAGE_BINDING, FORMAT)                      \
        uniform TYPE _vko_storage_images_##TYPE##_##FORMAT[];

#define _VKO_DECLARE_STORAGE_IMAGE_DIMENSION(DIMENSION)                                            \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rgba32f)                                          \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rgba16f)                                          \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rg32f)                                            \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rg16f)                                            \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, r11f_g11f_b10f)                                   \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, r32f)                                             \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, r16f)                                             \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rgba16)                                           \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rgb10_a2)                                         \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rgba8)                                            \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rg16)                                             \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rg8)                                              \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, r16)                                              \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, r8)                                               \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rgba16_snorm)                                     \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rgba8_snorm)                                      \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rg16_snorm)                                       \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, rg8_snorm)                                        \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, r16_snorm)                                        \
    _VKO_DECLARE_STORAGE_IMAGE(image##DIMENSION, r8_snorm)                                         \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, rgba32i)                                         \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, rgba16i)                                         \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, rgba8i)                                          \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, rg32i)                                           \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, rg16i)                                           \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, rg8i)                                            \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, r32i)                                            \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, r16i)                                            \
    _VKO_DECLARE_STORAGE_IMAGE(iimage##DIMENSION, r8i)                                             \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, rgba32ui)                                        \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, rgba16ui)                                        \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, rgb10_a2ui)                                      \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, rgba8ui)                                         \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, rg32ui)                                          \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, rg16ui)                                          \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, rg8ui)                                           \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, r32ui)                                           \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, r16ui)                                           \
    _VKO_DECLARE_STORAGE_IMAGE(uimage##DIMENSION, r8ui)

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(1D)
#define vko_image1D_rgba32f(id)                                                                    \
    _vko_storage_images_image1D_rgba32f[_vko_storage_image_index(id)]
#define vko_image1D_rgba16f(id)                                                                    \
    _vko_storage_images_image1D_rgba16f[_vko_storage_image_index(id)]
#define vko_image1D_rg32f(id)                                                                      \
    _vko_storage_images_image1D_rg32f[_vko_storage_image_index(id)]
#define vko_image1D_rg16f(id)                                                                      \
    _vko_storage_images_image1D_rg16f[_vko_storage_image_index(id)]
#define vko_image1D_r11f_g11f_b10f(id)                                                             \
    _vko_storage_images_image1D_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_image1D_r32f(id)                                                                       \
    _vko_storage_images_image1D_r32f[_vko_storage_image_index(id)]
#define vko_image1D_r16f(id)                                                                       \
    _vko_storage_images_image1D_r16f[_vko_storage_image_index(id)]
#define vko_image1D_rgba16(id)                                                                     \
    _vko_storage_images_image1D_rgba16[_vko_storage_image_index(id)]
#define vko_image1D_rgb10_a2(id)                                                                   \
    _vko_storage_images_image1D_rgb10_a2[_vko_storage_image_index(id)]
#define vko_image1D_rgba8(id)                                                                      \
    _vko_storage_images_image1D_rgba8[_vko_storage_image_index(id)]
#define vko_image1D_rg16(id)                                                                       \
    _vko_storage_images_image1D_rg16[_vko_storage_image_index(id)]
#define vko_image1D_rg8(id)                                                                        \
    _vko_storage_images_image1D_rg8[_vko_storage_image_index(id)]
#define vko_image1D_r16(id)                                                                        \
    _vko_storage_images_image1D_r16[_vko_storage_image_index(id)]
#define vko_image1D_r8(id)                                                                         \
    _vko_storage_images_image1D_r8[_vko_storage_image_index(id)]
#define vko_image1D_rgba16_snorm(id)                                                               \
    _vko_storage_images_image1D_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_image1D_rgba8_snorm(id)                                                                \
    _vko_storage_images_image1D_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_image1D_rg16_snorm(id)                                                                 \
    _vko_storage_images_image1D_rg16_snorm[_vko_storage_image_index(id)]
#define vko_image1D_rg8_snorm(id)                                                                  \
    _vko_storage_images_image1D_rg8_snorm[_vko_storage_image_index(id)]
#define vko_image1D_r16_snorm(id)                                                                  \
    _vko_storage_images_image1D_r16_snorm[_vko_storage_image_index(id)]
#define vko_image1D_r8_snorm(id)                                                                   \
    _vko_storage_images_image1D_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimage1D_rgba32i(id)                                                                   \
    _vko_storage_images_iimage1D_rgba32i[_vko_storage_image_index(id)]
#define vko_iimage1D_rgba16i(id)                                                                   \
    _vko_storage_images_iimage1D_rgba16i[_vko_storage_image_index(id)]
#define vko_iimage1D_rgba8i(id)                                                                    \
    _vko_storage_images_iimage1D_rgba8i[_vko_storage_image_index(id)]
#define vko_iimage1D_rg32i(id)                                                                     \
    _vko_storage_images_iimage1D_rg32i[_vko_storage_image_index(id)]
#define vko_iimage1D_rg16i(id)                                                                     \
    _vko_storage_images_iimage1D_rg16i[_vko_storage_image_index(id)]
#define vko_iimage1D_rg8i(id)                                                                      \
    _vko_storage_images_iimage1D_rg8i[_vko_storage_image_index(id)]
#define vko_iimage1D_r32i(id)                                                                      \
    _vko_storage_images_iimage1D_r32i[_vko_storage_image_index(id)]
#define vko_iimage1D_r16i(id)                                                                      \
    _vko_storage_images_iimage1D_r16i[_vko_storage_image_index(id)]
#define vko_iimage1D_r8i(id)                                                                       \
    _vko_storage_images_iimage1D_r8i[_vko_storage_image_index(id)]
#define vko_uimage1D_rgba32ui(id)                                                                  \
    _vko_storage_images_uimage1D_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimage1D_rgba16ui(id)                                                                  \
    _vko_storage_images_uimage1D_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimage1D_rgb10_a2ui(id)                                                                \
    _vko_storage_images_uimage1D_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimage1D_rgba8ui(id)                                                                   \
    _vko_storage_images_uimage1D_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimage1D_rg32ui(id)                                                                    \
    _vko_storage_images_uimage1D_rg32ui[_vko_storage_image_index(id)]
#define vko_uimage1D_rg16ui(id)                                                                    \
    _vko_storage_images_uimage1D_rg16ui[_vko_storage_image_index(id)]
#define vko_uimage1D_rg8ui(id)                                                                     \
    _vko_storage_images_uimage1D_rg8ui[_vko_storage_image_index(id)]
#define vko_uimage1D_r32ui(id)                                                                     \
    _vko_storage_images_uimage1D_r32ui[_vko_storage_image_index(id)]
#define vko_uimage1D_r16ui(id)                                                                     \
    _vko_storage_images_uimage1D_r16ui[_vko_storage_image_index(id)]
#define vko_uimage1D_r8ui(id)                                                                      \
    _vko_storage_images_uimage1D_r8ui[_vko_storage_image_index(id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(1DArray)
#define vko_image1DArray_rgba32f(id)                                                               \
    _vko_storage_images_image1DArray_rgba32f[_vko_storage_image_index(id)]
#define vko_image1DArray_rgba16f(id)                                                               \
    _vko_storage_images_image1DArray_rgba16f[_vko_storage_image_index(id)]
#define vko_image1DArray_rg32f(id)                                                                 \
    _vko_storage_images_image1DArray_rg32f[_vko_storage_image_index(id)]
#define vko_image1DArray_rg16f(id)                                                                 \
    _vko_storage_images_image1DArray_rg16f[_vko_storage_image_index(id)]
#define vko_image1DArray_r11f_g11f_b10f(id)                                                        \
    _vko_storage_images_image1DArray_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_image1DArray_r32f(id)                                                                  \
    _vko_storage_images_image1DArray_r32f[_vko_storage_image_index(id)]
#define vko_image1DArray_r16f(id)                                                                  \
    _vko_storage_images_image1DArray_r16f[_vko_storage_image_index(id)]
#define vko_image1DArray_rgba16(id)                                                                \
    _vko_storage_images_image1DArray_rgba16[_vko_storage_image_index(id)]
#define vko_image1DArray_rgb10_a2(id)                                                              \
    _vko_storage_images_image1DArray_rgb10_a2[_vko_storage_image_index(id)]
#define vko_image1DArray_rgba8(id)                                                                 \
    _vko_storage_images_image1DArray_rgba8[_vko_storage_image_index(id)]
#define vko_image1DArray_rg16(id)                                                                  \
    _vko_storage_images_image1DArray_rg16[_vko_storage_image_index(id)]
#define vko_image1DArray_rg8(id)                                                                   \
    _vko_storage_images_image1DArray_rg8[_vko_storage_image_index(id)]
#define vko_image1DArray_r16(id)                                                                   \
    _vko_storage_images_image1DArray_r16[_vko_storage_image_index(id)]
#define vko_image1DArray_r8(id)                                                                    \
    _vko_storage_images_image1DArray_r8[_vko_storage_image_index(id)]
#define vko_image1DArray_rgba16_snorm(id)                                                          \
    _vko_storage_images_image1DArray_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_image1DArray_rgba8_snorm(id)                                                           \
    _vko_storage_images_image1DArray_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_image1DArray_rg16_snorm(id)                                                            \
    _vko_storage_images_image1DArray_rg16_snorm[_vko_storage_image_index(id)]
#define vko_image1DArray_rg8_snorm(id)                                                             \
    _vko_storage_images_image1DArray_rg8_snorm[_vko_storage_image_index(id)]
#define vko_image1DArray_r16_snorm(id)                                                             \
    _vko_storage_images_image1DArray_r16_snorm[_vko_storage_image_index(id)]
#define vko_image1DArray_r8_snorm(id)                                                              \
    _vko_storage_images_image1DArray_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimage1DArray_rgba32i(id)                                                              \
    _vko_storage_images_iimage1DArray_rgba32i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_rgba16i(id)                                                              \
    _vko_storage_images_iimage1DArray_rgba16i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_rgba8i(id)                                                               \
    _vko_storage_images_iimage1DArray_rgba8i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_rg32i(id)                                                                \
    _vko_storage_images_iimage1DArray_rg32i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_rg16i(id)                                                                \
    _vko_storage_images_iimage1DArray_rg16i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_rg8i(id)                                                                 \
    _vko_storage_images_iimage1DArray_rg8i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_r32i(id)                                                                 \
    _vko_storage_images_iimage1DArray_r32i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_r16i(id)                                                                 \
    _vko_storage_images_iimage1DArray_r16i[_vko_storage_image_index(id)]
#define vko_iimage1DArray_r8i(id)                                                                  \
    _vko_storage_images_iimage1DArray_r8i[_vko_storage_image_index(id)]
#define vko_uimage1DArray_rgba32ui(id)                                                             \
    _vko_storage_images_uimage1DArray_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_rgba16ui(id)                                                             \
    _vko_storage_images_uimage1DArray_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_rgb10_a2ui(id)                                                           \
    _vko_storage_images_uimage1DArray_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_rgba8ui(id)                                                              \
    _vko_storage_images_uimage1DArray_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_rg32ui(id)                                                               \
    _vko_storage_images_uimage1DArray_rg32ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_rg16ui(id)                                                               \
    _vko_storage_images_uimage1DArray_rg16ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_rg8ui(id)                                                                \
    _vko_storage_images_uimage1DArray_rg8ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_r32ui(id)                                                                \
    _vko_storage_images_uimage1DArray_r32ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_r16ui(id)                                                                \
    _vko_storage_images_uimage1DArray_r16ui[_vko_storage_image_index(id)]
#define vko_uimage1DArray_r8ui(id)                                                                 \
    _vko_storage_images_uimage1DArray_r8ui[_vko_storage_image_index(id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(2D)
#define vko_image2D_rgba32f(id)                                                                    \
    _vko_storage_images_image2D_rgba32f[_vko_storage_image_index(id)]
#define vko_image2D_rgba16f(id)                                                                    \
    _vko_storage_images_image2D_rgba16f[_vko_storage_image_index(id)]
#define vko_image2D_rg32f(id)                                                                      \
    _vko_storage_images_image2D_rg32f[_vko_storage_image_index(id)]
#define vko_image2D_rg16f(id)                                                                      \
    _vko_storage_images_image2D_rg16f[_vko_storage_image_index(id)]
#define vko_image2D_r11f_g11f_b10f(id)                                                             \
    _vko_storage_images_image2D_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_image2D_r32f(id)                                                                       \
    _vko_storage_images_image2D_r32f[_vko_storage_image_index(id)]
#define vko_image2D_r16f(id)                                                                       \
    _vko_storage_images_image2D_r16f[_vko_storage_image_index(id)]
#define vko_image2D_rgba16(id)                                                                     \
    _vko_storage_images_image2D_rgba16[_vko_storage_image_index(id)]
#define vko_image2D_rgb10_a2(id)                                                                   \
    _vko_storage_images_image2D_rgb10_a2[_vko_storage_image_index(id)]
#define vko_image2D_rgba8(id)                                                                      \
    _vko_storage_images_image2D_rgba8[_vko_storage_image_index(id)]
#define vko_image2D_rg16(id)                                                                       \
    _vko_storage_images_image2D_rg16[_vko_storage_image_index(id)]
#define vko_image2D_rg8(id)                                                                        \
    _vko_storage_images_image2D_rg8[_vko_storage_image_index(id)]
#define vko_image2D_r16(id)                                                                        \
    _vko_storage_images_image2D_r16[_vko_storage_image_index(id)]
#define vko_image2D_r8(id)                                                                         \
    _vko_storage_images_image2D_r8[_vko_storage_image_index(id)]
#define vko_image2D_rgba16_snorm(id)                                                               \
    _vko_storage_images_image2D_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_image2D_rgba8_snorm(id)                                                                \
    _vko_storage_images_image2D_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_image2D_rg16_snorm(id)                                                                 \
    _vko_storage_images_image2D_rg16_snorm[_vko_storage_image_index(id)]
#define vko_image2D_rg8_snorm(id)                                                                  \
    _vko_storage_images_image2D_rg8_snorm[_vko_storage_image_index(id)]
#define vko_image2D_r16_snorm(id)                                                                  \
    _vko_storage_images_image2D_r16_snorm[_vko_storage_image_index(id)]
#define vko_image2D_r8_snorm(id)                                                                   \
    _vko_storage_images_image2D_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimage2D_rgba32i(id)                                                                   \
    _vko_storage_images_iimage2D_rgba32i[_vko_storage_image_index(id)]
#define vko_iimage2D_rgba16i(id)                                                                   \
    _vko_storage_images_iimage2D_rgba16i[_vko_storage_image_index(id)]
#define vko_iimage2D_rgba8i(id)                                                                    \
    _vko_storage_images_iimage2D_rgba8i[_vko_storage_image_index(id)]
#define vko_iimage2D_rg32i(id)                                                                     \
    _vko_storage_images_iimage2D_rg32i[_vko_storage_image_index(id)]
#define vko_iimage2D_rg16i(id)                                                                     \
    _vko_storage_images_iimage2D_rg16i[_vko_storage_image_index(id)]
#define vko_iimage2D_rg8i(id)                                                                      \
    _vko_storage_images_iimage2D_rg8i[_vko_storage_image_index(id)]
#define vko_iimage2D_r32i(id)                                                                      \
    _vko_storage_images_iimage2D_r32i[_vko_storage_image_index(id)]
#define vko_iimage2D_r16i(id)                                                                      \
    _vko_storage_images_iimage2D_r16i[_vko_storage_image_index(id)]
#define vko_iimage2D_r8i(id)                                                                       \
    _vko_storage_images_iimage2D_r8i[_vko_storage_image_index(id)]
#define vko_uimage2D_rgba32ui(id)                                                                  \
    _vko_storage_images_uimage2D_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimage2D_rgba16ui(id)                                                                  \
    _vko_storage_images_uimage2D_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimage2D_rgb10_a2ui(id)                                                                \
    _vko_storage_images_uimage2D_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimage2D_rgba8ui(id)                                                                   \
    _vko_storage_images_uimage2D_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimage2D_rg32ui(id)                                                                    \
    _vko_storage_images_uimage2D_rg32ui[_vko_storage_image_index(id)]
#define vko_uimage2D_rg16ui(id)                                                                    \
    _vko_storage_images_uimage2D_rg16ui[_vko_storage_image_index(id)]
#define vko_uimage2D_rg8ui(id)                                                                     \
    _vko_storage_images_uimage2D_rg8ui[_vko_storage_image_index(id)]
#define vko_uimage2D_r32ui(id)                                                                     \
    _vko_storage_images_uimage2D_r32ui[_vko_storage_image_index(id)]
#define vko_uimage2D_r16ui(id)                                                                     \
    _vko_storage_images_uimage2D_r16ui[_vko_storage_image_index(id)]
#define vko_uimage2D_r8ui(id)                                                                      \
    _vko_storage_images_uimage2D_r8ui[_vko_storage_image_index(id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(2DArray)
#define vko_image2DArray_rgba32f(id)                                                               \
    _vko_storage_images_image2DArray_rgba32f[_vko_storage_image_index(id)]
#define vko_image2DArray_rgba16f(id)                                                               \
    _vko_storage_images_image2DArray_rgba16f[_vko_storage_image_index(id)]
#define vko_image2DArray_rg32f(id)                                                                 \
    _vko_storage_images_image2DArray_rg32f[_vko_storage_image_index(id)]
#define vko_image2DArray_rg16f(id)                                                                 \
    _vko_storage_images_image2DArray_rg16f[_vko_storage_image_index(id)]
#define vko_image2DArray_r11f_g11f_b10f(id)                                                        \
    _vko_storage_images_image2DArray_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_image2DArray_r32f(id)                                                                  \
    _vko_storage_images_image2DArray_r32f[_vko_storage_image_index(id)]
#define vko_image2DArray_r16f(id)                                                                  \
    _vko_storage_images_image2DArray_r16f[_vko_storage_image_index(id)]
#define vko_image2DArray_rgba16(id)                                                                \
    _vko_storage_images_image2DArray_rgba16[_vko_storage_image_index(id)]
#define vko_image2DArray_rgb10_a2(id)                                                              \
    _vko_storage_images_image2DArray_rgb10_a2[_vko_storage_image_index(id)]
#define vko_image2DArray_rgba8(id)                                                                 \
    _vko_storage_images_image2DArray_rgba8[_vko_storage_image_index(id)]
#define vko_image2DArray_rg16(id)                                                                  \
    _vko_storage_images_image2DArray_rg16[_vko_storage_image_index(id)]
#define vko_image2DArray_rg8(id)                                                                   \
    _vko_storage_images_image2DArray_rg8[_vko_storage_image_index(id)]
#define vko_image2DArray_r16(id)                                                                   \
    _vko_storage_images_image2DArray_r16[_vko_storage_image_index(id)]
#define vko_image2DArray_r8(id)                                                                    \
    _vko_storage_images_image2DArray_r8[_vko_storage_image_index(id)]
#define vko_image2DArray_rgba16_snorm(id)                                                          \
    _vko_storage_images_image2DArray_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_image2DArray_rgba8_snorm(id)                                                           \
    _vko_storage_images_image2DArray_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_image2DArray_rg16_snorm(id)                                                            \
    _vko_storage_images_image2DArray_rg16_snorm[_vko_storage_image_index(id)]
#define vko_image2DArray_rg8_snorm(id)                                                             \
    _vko_storage_images_image2DArray_rg8_snorm[_vko_storage_image_index(id)]
#define vko_image2DArray_r16_snorm(id)                                                             \
    _vko_storage_images_image2DArray_r16_snorm[_vko_storage_image_index(id)]
#define vko_image2DArray_r8_snorm(id)                                                              \
    _vko_storage_images_image2DArray_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimage2DArray_rgba32i(id)                                                              \
    _vko_storage_images_iimage2DArray_rgba32i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_rgba16i(id)                                                              \
    _vko_storage_images_iimage2DArray_rgba16i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_rgba8i(id)                                                               \
    _vko_storage_images_iimage2DArray_rgba8i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_rg32i(id)                                                                \
    _vko_storage_images_iimage2DArray_rg32i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_rg16i(id)                                                                \
    _vko_storage_images_iimage2DArray_rg16i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_rg8i(id)                                                                 \
    _vko_storage_images_iimage2DArray_rg8i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_r32i(id)                                                                 \
    _vko_storage_images_iimage2DArray_r32i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_r16i(id)                                                                 \
    _vko_storage_images_iimage2DArray_r16i[_vko_storage_image_index(id)]
#define vko_iimage2DArray_r8i(id)                                                                  \
    _vko_storage_images_iimage2DArray_r8i[_vko_storage_image_index(id)]
#define vko_uimage2DArray_rgba32ui(id)                                                             \
    _vko_storage_images_uimage2DArray_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_rgba16ui(id)                                                             \
    _vko_storage_images_uimage2DArray_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_rgb10_a2ui(id)                                                           \
    _vko_storage_images_uimage2DArray_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_rgba8ui(id)                                                              \
    _vko_storage_images_uimage2DArray_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_rg32ui(id)                                                               \
    _vko_storage_images_uimage2DArray_rg32ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_rg16ui(id)                                                               \
    _vko_storage_images_uimage2DArray_rg16ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_rg8ui(id)                                                                \
    _vko_storage_images_uimage2DArray_rg8ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_r32ui(id)                                                                \
    _vko_storage_images_uimage2DArray_r32ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_r16ui(id)                                                                \
    _vko_storage_images_uimage2DArray_r16ui[_vko_storage_image_index(id)]
#define vko_uimage2DArray_r8ui(id)                                                                 \
    _vko_storage_images_uimage2DArray_r8ui[_vko_storage_image_index(id)]

#if VKO_STORAGE_IMAGE_MULTISAMPLE_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(2DMS)
#define vko_image2DMS_rgba32f(id)                                                                  \
    _vko_storage_images_image2DMS_rgba32f[_vko_storage_image_index(id)]
#define vko_image2DMS_rgba16f(id)                                                                  \
    _vko_storage_images_image2DMS_rgba16f[_vko_storage_image_index(id)]
#define vko_image2DMS_rg32f(id)                                                                    \
    _vko_storage_images_image2DMS_rg32f[_vko_storage_image_index(id)]
#define vko_image2DMS_rg16f(id)                                                                    \
    _vko_storage_images_image2DMS_rg16f[_vko_storage_image_index(id)]
#define vko_image2DMS_r11f_g11f_b10f(id)                                                           \
    _vko_storage_images_image2DMS_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_image2DMS_r32f(id)                                                                     \
    _vko_storage_images_image2DMS_r32f[_vko_storage_image_index(id)]
#define vko_image2DMS_r16f(id)                                                                     \
    _vko_storage_images_image2DMS_r16f[_vko_storage_image_index(id)]
#define vko_image2DMS_rgba16(id)                                                                   \
    _vko_storage_images_image2DMS_rgba16[_vko_storage_image_index(id)]
#define vko_image2DMS_rgb10_a2(id)                                                                 \
    _vko_storage_images_image2DMS_rgb10_a2[_vko_storage_image_index(id)]
#define vko_image2DMS_rgba8(id)                                                                    \
    _vko_storage_images_image2DMS_rgba8[_vko_storage_image_index(id)]
#define vko_image2DMS_rg16(id)                                                                     \
    _vko_storage_images_image2DMS_rg16[_vko_storage_image_index(id)]
#define vko_image2DMS_rg8(id)                                                                      \
    _vko_storage_images_image2DMS_rg8[_vko_storage_image_index(id)]
#define vko_image2DMS_r16(id)                                                                      \
    _vko_storage_images_image2DMS_r16[_vko_storage_image_index(id)]
#define vko_image2DMS_r8(id)                                                                       \
    _vko_storage_images_image2DMS_r8[_vko_storage_image_index(id)]
#define vko_image2DMS_rgba16_snorm(id)                                                             \
    _vko_storage_images_image2DMS_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_image2DMS_rgba8_snorm(id)                                                              \
    _vko_storage_images_image2DMS_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_image2DMS_rg16_snorm(id)                                                               \
    _vko_storage_images_image2DMS_rg16_snorm[_vko_storage_image_index(id)]
#define vko_image2DMS_rg8_snorm(id)                                                                \
    _vko_storage_images_image2DMS_rg8_snorm[_vko_storage_image_index(id)]
#define vko_image2DMS_r16_snorm(id)                                                                \
    _vko_storage_images_image2DMS_r16_snorm[_vko_storage_image_index(id)]
#define vko_image2DMS_r8_snorm(id)                                                                 \
    _vko_storage_images_image2DMS_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimage2DMS_rgba32i(id)                                                                 \
    _vko_storage_images_iimage2DMS_rgba32i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_rgba16i(id)                                                                 \
    _vko_storage_images_iimage2DMS_rgba16i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_rgba8i(id)                                                                  \
    _vko_storage_images_iimage2DMS_rgba8i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_rg32i(id)                                                                   \
    _vko_storage_images_iimage2DMS_rg32i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_rg16i(id)                                                                   \
    _vko_storage_images_iimage2DMS_rg16i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_rg8i(id)                                                                    \
    _vko_storage_images_iimage2DMS_rg8i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_r32i(id)                                                                    \
    _vko_storage_images_iimage2DMS_r32i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_r16i(id)                                                                    \
    _vko_storage_images_iimage2DMS_r16i[_vko_storage_image_index(id)]
#define vko_iimage2DMS_r8i(id)                                                                     \
    _vko_storage_images_iimage2DMS_r8i[_vko_storage_image_index(id)]
#define vko_uimage2DMS_rgba32ui(id)                                                                \
    _vko_storage_images_uimage2DMS_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_rgba16ui(id)                                                                \
    _vko_storage_images_uimage2DMS_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_rgb10_a2ui(id)                                                              \
    _vko_storage_images_uimage2DMS_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_rgba8ui(id)                                                                 \
    _vko_storage_images_uimage2DMS_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_rg32ui(id)                                                                  \
    _vko_storage_images_uimage2DMS_rg32ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_rg16ui(id)                                                                  \
    _vko_storage_images_uimage2DMS_rg16ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_rg8ui(id)                                                                   \
    _vko_storage_images_uimage2DMS_rg8ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_r32ui(id)                                                                   \
    _vko_storage_images_uimage2DMS_r32ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_r16ui(id)                                                                   \
    _vko_storage_images_uimage2DMS_r16ui[_vko_storage_image_index(id)]
#define vko_uimage2DMS_r8ui(id)                                                                    \
    _vko_storage_images_uimage2DMS_r8ui[_vko_storage_image_index(id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(2DMSArray)
#define vko_image2DMSArray_rgba32f(id)                                                             \
    _vko_storage_images_image2DMSArray_rgba32f[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rgba16f(id)                                                             \
    _vko_storage_images_image2DMSArray_rgba16f[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rg32f(id)                                                               \
    _vko_storage_images_image2DMSArray_rg32f[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rg16f(id)                                                               \
    _vko_storage_images_image2DMSArray_rg16f[_vko_storage_image_index(id)]
#define vko_image2DMSArray_r11f_g11f_b10f(id)                                                      \
    _vko_storage_images_image2DMSArray_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_image2DMSArray_r32f(id)                                                                \
    _vko_storage_images_image2DMSArray_r32f[_vko_storage_image_index(id)]
#define vko_image2DMSArray_r16f(id)                                                                \
    _vko_storage_images_image2DMSArray_r16f[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rgba16(id)                                                              \
    _vko_storage_images_image2DMSArray_rgba16[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rgb10_a2(id)                                                            \
    _vko_storage_images_image2DMSArray_rgb10_a2[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rgba8(id)                                                               \
    _vko_storage_images_image2DMSArray_rgba8[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rg16(id)                                                                \
    _vko_storage_images_image2DMSArray_rg16[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rg8(id)                                                                 \
    _vko_storage_images_image2DMSArray_rg8[_vko_storage_image_index(id)]
#define vko_image2DMSArray_r16(id)                                                                 \
    _vko_storage_images_image2DMSArray_r16[_vko_storage_image_index(id)]
#define vko_image2DMSArray_r8(id)                                                                  \
    _vko_storage_images_image2DMSArray_r8[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rgba16_snorm(id)                                                        \
    _vko_storage_images_image2DMSArray_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rgba8_snorm(id)                                                         \
    _vko_storage_images_image2DMSArray_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rg16_snorm(id)                                                          \
    _vko_storage_images_image2DMSArray_rg16_snorm[_vko_storage_image_index(id)]
#define vko_image2DMSArray_rg8_snorm(id)                                                           \
    _vko_storage_images_image2DMSArray_rg8_snorm[_vko_storage_image_index(id)]
#define vko_image2DMSArray_r16_snorm(id)                                                           \
    _vko_storage_images_image2DMSArray_r16_snorm[_vko_storage_image_index(id)]
#define vko_image2DMSArray_r8_snorm(id)                                                            \
    _vko_storage_images_image2DMSArray_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_rgba32i(id)                                                            \
    _vko_storage_images_iimage2DMSArray_rgba32i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_rgba16i(id)                                                            \
    _vko_storage_images_iimage2DMSArray_rgba16i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_rgba8i(id)                                                             \
    _vko_storage_images_iimage2DMSArray_rgba8i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_rg32i(id)                                                              \
    _vko_storage_images_iimage2DMSArray_rg32i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_rg16i(id)                                                              \
    _vko_storage_images_iimage2DMSArray_rg16i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_rg8i(id)                                                               \
    _vko_storage_images_iimage2DMSArray_rg8i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_r32i(id)                                                               \
    _vko_storage_images_iimage2DMSArray_r32i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_r16i(id)                                                               \
    _vko_storage_images_iimage2DMSArray_r16i[_vko_storage_image_index(id)]
#define vko_iimage2DMSArray_r8i(id)                                                                \
    _vko_storage_images_iimage2DMSArray_r8i[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_rgba32ui(id)                                                           \
    _vko_storage_images_uimage2DMSArray_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_rgba16ui(id)                                                           \
    _vko_storage_images_uimage2DMSArray_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_rgb10_a2ui(id)                                                         \
    _vko_storage_images_uimage2DMSArray_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_rgba8ui(id)                                                            \
    _vko_storage_images_uimage2DMSArray_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_rg32ui(id)                                                             \
    _vko_storage_images_uimage2DMSArray_rg32ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_rg16ui(id)                                                             \
    _vko_storage_images_uimage2DMSArray_rg16ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_rg8ui(id)                                                              \
    _vko_storage_images_uimage2DMSArray_rg8ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_r32ui(id)                                                              \
    _vko_storage_images_uimage2DMSArray_r32ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_r16ui(id)                                                              \
    _vko_storage_images_uimage2DMSArray_r16ui[_vko_storage_image_index(id)]
#define vko_uimage2DMSArray_r8ui(id)                                                               \
    _vko_storage_images_uimage2DMSArray_r8ui[_vko_storage_image_index(id)]

#endif // VKO_STORAGE_IMAGE_MULTISAMPLE_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(3D)
#define vko_image3D_rgba32f(id)                                                                    \
    _vko_storage_images_image3D_rgba32f[_vko_storage_image_index(id)]
#define vko_image3D_rgba16f(id)                                                                    \
    _vko_storage_images_image3D_rgba16f[_vko_storage_image_index(id)]
#define vko_image3D_rg32f(id)                                                                      \
    _vko_storage_images_image3D_rg32f[_vko_storage_image_index(id)]
#define vko_image3D_rg16f(id)                                                                      \
    _vko_storage_images_image3D_rg16f[_vko_storage_image_index(id)]
#define vko_image3D_r11f_g11f_b10f(id)                                                             \
    _vko_storage_images_image3D_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_image3D_r32f(id)                                                                       \
    _vko_storage_images_image3D_r32f[_vko_storage_image_index(id)]
#define vko_image3D_r16f(id)                                                                       \
    _vko_storage_images_image3D_r16f[_vko_storage_image_index(id)]
#define vko_image3D_rgba16(id)                                                                     \
    _vko_storage_images_image3D_rgba16[_vko_storage_image_index(id)]
#define vko_image3D_rgb10_a2(id)                                                                   \
    _vko_storage_images_image3D_rgb10_a2[_vko_storage_image_index(id)]
#define vko_image3D_rgba8(id)                                                                      \
    _vko_storage_images_image3D_rgba8[_vko_storage_image_index(id)]
#define vko_image3D_rg16(id)                                                                       \
    _vko_storage_images_image3D_rg16[_vko_storage_image_index(id)]
#define vko_image3D_rg8(id)                                                                        \
    _vko_storage_images_image3D_rg8[_vko_storage_image_index(id)]
#define vko_image3D_r16(id)                                                                        \
    _vko_storage_images_image3D_r16[_vko_storage_image_index(id)]
#define vko_image3D_r8(id)                                                                         \
    _vko_storage_images_image3D_r8[_vko_storage_image_index(id)]
#define vko_image3D_rgba16_snorm(id)                                                               \
    _vko_storage_images_image3D_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_image3D_rgba8_snorm(id)                                                                \
    _vko_storage_images_image3D_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_image3D_rg16_snorm(id)                                                                 \
    _vko_storage_images_image3D_rg16_snorm[_vko_storage_image_index(id)]
#define vko_image3D_rg8_snorm(id)                                                                  \
    _vko_storage_images_image3D_rg8_snorm[_vko_storage_image_index(id)]
#define vko_image3D_r16_snorm(id)                                                                  \
    _vko_storage_images_image3D_r16_snorm[_vko_storage_image_index(id)]
#define vko_image3D_r8_snorm(id)                                                                   \
    _vko_storage_images_image3D_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimage3D_rgba32i(id)                                                                   \
    _vko_storage_images_iimage3D_rgba32i[_vko_storage_image_index(id)]
#define vko_iimage3D_rgba16i(id)                                                                   \
    _vko_storage_images_iimage3D_rgba16i[_vko_storage_image_index(id)]
#define vko_iimage3D_rgba8i(id)                                                                    \
    _vko_storage_images_iimage3D_rgba8i[_vko_storage_image_index(id)]
#define vko_iimage3D_rg32i(id)                                                                     \
    _vko_storage_images_iimage3D_rg32i[_vko_storage_image_index(id)]
#define vko_iimage3D_rg16i(id)                                                                     \
    _vko_storage_images_iimage3D_rg16i[_vko_storage_image_index(id)]
#define vko_iimage3D_rg8i(id)                                                                      \
    _vko_storage_images_iimage3D_rg8i[_vko_storage_image_index(id)]
#define vko_iimage3D_r32i(id)                                                                      \
    _vko_storage_images_iimage3D_r32i[_vko_storage_image_index(id)]
#define vko_iimage3D_r16i(id)                                                                      \
    _vko_storage_images_iimage3D_r16i[_vko_storage_image_index(id)]
#define vko_iimage3D_r8i(id)                                                                       \
    _vko_storage_images_iimage3D_r8i[_vko_storage_image_index(id)]
#define vko_uimage3D_rgba32ui(id)                                                                  \
    _vko_storage_images_uimage3D_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimage3D_rgba16ui(id)                                                                  \
    _vko_storage_images_uimage3D_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimage3D_rgb10_a2ui(id)                                                                \
    _vko_storage_images_uimage3D_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimage3D_rgba8ui(id)                                                                   \
    _vko_storage_images_uimage3D_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimage3D_rg32ui(id)                                                                    \
    _vko_storage_images_uimage3D_rg32ui[_vko_storage_image_index(id)]
#define vko_uimage3D_rg16ui(id)                                                                    \
    _vko_storage_images_uimage3D_rg16ui[_vko_storage_image_index(id)]
#define vko_uimage3D_rg8ui(id)                                                                     \
    _vko_storage_images_uimage3D_rg8ui[_vko_storage_image_index(id)]
#define vko_uimage3D_r32ui(id)                                                                     \
    _vko_storage_images_uimage3D_r32ui[_vko_storage_image_index(id)]
#define vko_uimage3D_r16ui(id)                                                                     \
    _vko_storage_images_uimage3D_r16ui[_vko_storage_image_index(id)]
#define vko_uimage3D_r8ui(id)                                                                      \
    _vko_storage_images_uimage3D_r8ui[_vko_storage_image_index(id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(Cube)
#define vko_imageCube_rgba32f(id)                                                                  \
    _vko_storage_images_imageCube_rgba32f[_vko_storage_image_index(id)]
#define vko_imageCube_rgba16f(id)                                                                  \
    _vko_storage_images_imageCube_rgba16f[_vko_storage_image_index(id)]
#define vko_imageCube_rg32f(id)                                                                    \
    _vko_storage_images_imageCube_rg32f[_vko_storage_image_index(id)]
#define vko_imageCube_rg16f(id)                                                                    \
    _vko_storage_images_imageCube_rg16f[_vko_storage_image_index(id)]
#define vko_imageCube_r11f_g11f_b10f(id)                                                           \
    _vko_storage_images_imageCube_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_imageCube_r32f(id)                                                                     \
    _vko_storage_images_imageCube_r32f[_vko_storage_image_index(id)]
#define vko_imageCube_r16f(id)                                                                     \
    _vko_storage_images_imageCube_r16f[_vko_storage_image_index(id)]
#define vko_imageCube_rgba16(id)                                                                   \
    _vko_storage_images_imageCube_rgba16[_vko_storage_image_index(id)]
#define vko_imageCube_rgb10_a2(id)                                                                 \
    _vko_storage_images_imageCube_rgb10_a2[_vko_storage_image_index(id)]
#define vko_imageCube_rgba8(id)                                                                    \
    _vko_storage_images_imageCube_rgba8[_vko_storage_image_index(id)]
#define vko_imageCube_rg16(id)                                                                     \
    _vko_storage_images_imageCube_rg16[_vko_storage_image_index(id)]
#define vko_imageCube_rg8(id)                                                                      \
    _vko_storage_images_imageCube_rg8[_vko_storage_image_index(id)]
#define vko_imageCube_r16(id)                                                                      \
    _vko_storage_images_imageCube_r16[_vko_storage_image_index(id)]
#define vko_imageCube_r8(id)                                                                       \
    _vko_storage_images_imageCube_r8[_vko_storage_image_index(id)]
#define vko_imageCube_rgba16_snorm(id)                                                             \
    _vko_storage_images_imageCube_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_imageCube_rgba8_snorm(id)                                                              \
    _vko_storage_images_imageCube_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_imageCube_rg16_snorm(id)                                                               \
    _vko_storage_images_imageCube_rg16_snorm[_vko_storage_image_index(id)]
#define vko_imageCube_rg8_snorm(id)                                                                \
    _vko_storage_images_imageCube_rg8_snorm[_vko_storage_image_index(id)]
#define vko_imageCube_r16_snorm(id)                                                                \
    _vko_storage_images_imageCube_r16_snorm[_vko_storage_image_index(id)]
#define vko_imageCube_r8_snorm(id)                                                                 \
    _vko_storage_images_imageCube_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimageCube_rgba32i(id)                                                                 \
    _vko_storage_images_iimageCube_rgba32i[_vko_storage_image_index(id)]
#define vko_iimageCube_rgba16i(id)                                                                 \
    _vko_storage_images_iimageCube_rgba16i[_vko_storage_image_index(id)]
#define vko_iimageCube_rgba8i(id)                                                                  \
    _vko_storage_images_iimageCube_rgba8i[_vko_storage_image_index(id)]
#define vko_iimageCube_rg32i(id)                                                                   \
    _vko_storage_images_iimageCube_rg32i[_vko_storage_image_index(id)]
#define vko_iimageCube_rg16i(id)                                                                   \
    _vko_storage_images_iimageCube_rg16i[_vko_storage_image_index(id)]
#define vko_iimageCube_rg8i(id)                                                                    \
    _vko_storage_images_iimageCube_rg8i[_vko_storage_image_index(id)]
#define vko_iimageCube_r32i(id)                                                                    \
    _vko_storage_images_iimageCube_r32i[_vko_storage_image_index(id)]
#define vko_iimageCube_r16i(id)                                                                    \
    _vko_storage_images_iimageCube_r16i[_vko_storage_image_index(id)]
#define vko_iimageCube_r8i(id)                                                                     \
    _vko_storage_images_iimageCube_r8i[_vko_storage_image_index(id)]
#define vko_uimageCube_rgba32ui(id)                                                                \
    _vko_storage_images_uimageCube_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimageCube_rgba16ui(id)                                                                \
    _vko_storage_images_uimageCube_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimageCube_rgb10_a2ui(id)                                                              \
    _vko_storage_images_uimageCube_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimageCube_rgba8ui(id)                                                                 \
    _vko_storage_images_uimageCube_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimageCube_rg32ui(id)                                                                  \
    _vko_storage_images_uimageCube_rg32ui[_vko_storage_image_index(id)]
#define vko_uimageCube_rg16ui(id)                                                                  \
    _vko_storage_images_uimageCube_rg16ui[_vko_storage_image_index(id)]
#define vko_uimageCube_rg8ui(id)                                                                   \
    _vko_storage_images_uimageCube_rg8ui[_vko_storage_image_index(id)]
#define vko_uimageCube_r32ui(id)                                                                   \
    _vko_storage_images_uimageCube_r32ui[_vko_storage_image_index(id)]
#define vko_uimageCube_r16ui(id)                                                                   \
    _vko_storage_images_uimageCube_r16ui[_vko_storage_image_index(id)]
#define vko_uimageCube_r8ui(id)                                                                    \
    _vko_storage_images_uimageCube_r8ui[_vko_storage_image_index(id)]

#if VKO_IMAGE_CUBE_ARRAY_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION(CubeArray)
#define vko_imageCubeArray_rgba32f(id)                                                             \
    _vko_storage_images_imageCubeArray_rgba32f[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rgba16f(id)                                                             \
    _vko_storage_images_imageCubeArray_rgba16f[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rg32f(id)                                                               \
    _vko_storage_images_imageCubeArray_rg32f[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rg16f(id)                                                               \
    _vko_storage_images_imageCubeArray_rg16f[_vko_storage_image_index(id)]
#define vko_imageCubeArray_r11f_g11f_b10f(id)                                                      \
    _vko_storage_images_imageCubeArray_r11f_g11f_b10f[_vko_storage_image_index(id)]
#define vko_imageCubeArray_r32f(id)                                                                \
    _vko_storage_images_imageCubeArray_r32f[_vko_storage_image_index(id)]
#define vko_imageCubeArray_r16f(id)                                                                \
    _vko_storage_images_imageCubeArray_r16f[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rgba16(id)                                                              \
    _vko_storage_images_imageCubeArray_rgba16[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rgb10_a2(id)                                                            \
    _vko_storage_images_imageCubeArray_rgb10_a2[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rgba8(id)                                                               \
    _vko_storage_images_imageCubeArray_rgba8[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rg16(id)                                                                \
    _vko_storage_images_imageCubeArray_rg16[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rg8(id)                                                                 \
    _vko_storage_images_imageCubeArray_rg8[_vko_storage_image_index(id)]
#define vko_imageCubeArray_r16(id)                                                                 \
    _vko_storage_images_imageCubeArray_r16[_vko_storage_image_index(id)]
#define vko_imageCubeArray_r8(id)                                                                  \
    _vko_storage_images_imageCubeArray_r8[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rgba16_snorm(id)                                                        \
    _vko_storage_images_imageCubeArray_rgba16_snorm[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rgba8_snorm(id)                                                         \
    _vko_storage_images_imageCubeArray_rgba8_snorm[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rg16_snorm(id)                                                          \
    _vko_storage_images_imageCubeArray_rg16_snorm[_vko_storage_image_index(id)]
#define vko_imageCubeArray_rg8_snorm(id)                                                           \
    _vko_storage_images_imageCubeArray_rg8_snorm[_vko_storage_image_index(id)]
#define vko_imageCubeArray_r16_snorm(id)                                                           \
    _vko_storage_images_imageCubeArray_r16_snorm[_vko_storage_image_index(id)]
#define vko_imageCubeArray_r8_snorm(id)                                                            \
    _vko_storage_images_imageCubeArray_r8_snorm[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_rgba32i(id)                                                            \
    _vko_storage_images_iimageCubeArray_rgba32i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_rgba16i(id)                                                            \
    _vko_storage_images_iimageCubeArray_rgba16i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_rgba8i(id)                                                             \
    _vko_storage_images_iimageCubeArray_rgba8i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_rg32i(id)                                                              \
    _vko_storage_images_iimageCubeArray_rg32i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_rg16i(id)                                                              \
    _vko_storage_images_iimageCubeArray_rg16i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_rg8i(id)                                                               \
    _vko_storage_images_iimageCubeArray_rg8i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_r32i(id)                                                               \
    _vko_storage_images_iimageCubeArray_r32i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_r16i(id)                                                               \
    _vko_storage_images_iimageCubeArray_r16i[_vko_storage_image_index(id)]
#define vko_iimageCubeArray_r8i(id)                                                                \
    _vko_storage_images_iimageCubeArray_r8i[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_rgba32ui(id)                                                           \
    _vko_storage_images_uimageCubeArray_rgba32ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_rgba16ui(id)                                                           \
    _vko_storage_images_uimageCubeArray_rgba16ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_rgb10_a2ui(id)                                                         \
    _vko_storage_images_uimageCubeArray_rgb10_a2ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_rgba8ui(id)                                                            \
    _vko_storage_images_uimageCubeArray_rgba8ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_rg32ui(id)                                                             \
    _vko_storage_images_uimageCubeArray_rg32ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_rg16ui(id)                                                             \
    _vko_storage_images_uimageCubeArray_rg16ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_rg8ui(id)                                                              \
    _vko_storage_images_uimageCubeArray_rg8ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_r32ui(id)                                                              \
    _vko_storage_images_uimageCubeArray_r32ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_r16ui(id)                                                              \
    _vko_storage_images_uimageCubeArray_r16ui[_vko_storage_image_index(id)]
#define vko_uimageCubeArray_r8ui(id)                                                               \
    _vko_storage_images_uimageCubeArray_r8ui[_vko_storage_image_index(id)]

#endif // VKO_IMAGE_CUBE_ARRAY_ENABLED

#undef _VKO_DECLARE_STORAGE_IMAGE
#undef _VKO_DECLARE_STORAGE_IMAGE_DIMENSION

///////////////////////////////////////////////////////////////////////////////////////////////////

#if VKO_IMAGE_INT64_ATOMICS_ENABLED

#define _VKO_DECLARE_STORAGE_IMAGE_INT64(TYPE, FORMAT)                                             \
    layout(set = VKO_GLOBAL_SET, binding = VKO_STORAGE_IMAGE_BINDING, FORMAT)                      \
        uniform TYPE _vko_storage_images_##TYPE##_##FORMAT[];

#define _VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(DIMENSION)                                      \
    _VKO_DECLARE_STORAGE_IMAGE_INT64(i64image##DIMENSION, r64i)                                    \
    _VKO_DECLARE_STORAGE_IMAGE_INT64(u64image##DIMENSION, r64ui)

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(1D)
#define vko_i64image1D_r64i(storage_image_id)                                                      \
    _vko_images_i64image1D_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64image1D_r64ui(storage_image_id)                                                     \
    _vko_images_u64image1D_r64ui[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(1DArray)
#define vko_i64image1DArray_r64i(storage_image_id)                                                 \
    _vko_images_i64image1DArray_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64image1DArray_r64ui(storage_image_id)                                                \
    _vko_images_u64image1DArray_r64ui[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(2D)
#define vko_i64image2D_r64i(storage_image_id)                                                      \
    _vko_images_i64image2D_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64image2D_r64ui(storage_image_id)                                                     \
    _vko_images_u64image2D_r64ui[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(2DArray)
#define vko_i64image2DArray_r64i(storage_image_id)                                                 \
    _vko_images_i64image2DArray_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64image2DArray_r64ui(storage_image_id)                                                \
    _vko_images_u64image2DArray_r64ui[_vko_storage_image_index(storage_image_id)]

#if VKO_STORAGE_IMAGE_MULTISAMPLE_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(2DMS)
#define vko_i64image2DMS_r64i(storage_image_id)                                                    \
    _vko_images_i64image2DMS_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64image2DMS_r64ui(storage_image_id)                                                   \
    _vko_images_u64image2DMS_r64ui[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(2DMSArray)
#define vko_i64image2DMSArray_r64i(storage_image_id)                                               \
    _vko_images_i64image2DMSArray_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64image2DMSArray_r64ui(storage_image_id)                                              \
    _vko_images_u64image2DMSArray_r64ui[_vko_storage_image_index(storage_image_id)]

#endif // VKO_STORAGE_IMAGE_MULTISAMPLE_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(3D)
#define vko_i64image3D_r64i(storage_image_id)                                                      \
    _vko_images_i64image3D_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64image3D_r64ui(storage_image_id)                                                     \
    _vko_images_u64image3D_r64ui[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(Cube)
#define vko_i64imageCube_r64i(storage_image_id)                                                    \
    _vko_images_i64imageCube_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64imageCube_r64ui(storage_image_id)                                                   \
    _vko_images_u64imageCube_r64ui[_vko_storage_image_index(storage_image_id)]

#if VKO_IMAGE_CUBE_ARRAY_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_INT64_DIMENSION(CubeArray)
#define vko_i64imageCubeArray_r64i(storage_image_id)                                               \
    _vko_images_i64imageCubeArray_r64i[_vko_storage_image_index(storage_image_id)]
#define vko_u64imageCubeArray_r64ui(storage_image_id)                                              \
    _vko_images_u64imageCubeArray_r64ui[_vko_storage_image_index(storage_image_id)]

#endif // VKO_IMAGE_CUBE_ARRAY_ENABLED

#undef _VKO_DECLARE_IMAGE_INT64
#undef _VKO_DECLARE_IMAGE_INT64_DIMENSION

#endif // VKO_IMAGE_INT64_ATOMICS_ENABLED

///////////////////////////////////////////////////////////////////////////////////////////////////

#if VKO_STORAGE_IMAGE_WITHOUT_FORMAT_ENABLED

#define _VKO_DECLARE_STORAGE_IMAGE_WITHOUT_FORMAT(TYPE)                                            \
    layout(set = VKO_GLOBAL_SET, binding = VKO_STORAGE_IMAGE_BINDING)                              \
        uniform TYPE _vko_storage_images_##TYPE[];

#define _VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(DIMENSION)                             \
    _VKO_DECLARE_STORAGE_IMAGE_WITHOUT_FORMAT(image##DIMENSION)                                    \
    _VKO_DECLARE_STORAGE_IMAGE_WITHOUT_FORMAT(iimage##DIMENSION)                                   \
    _VKO_DECLARE_STORAGE_IMAGE_WITHOUT_FORMAT(uimage##DIMENSION)

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(1D)
#define vko_image1D(storage_image_id)                                                              \
    _vko_storage_images_image1D[_vko_storage_image_index(storage_image_id)]
#define vko_iimage1D(storage_image_id)                                                             \
    _vko_storage_images_iimage1D[_vko_storage_image_index(storage_image_id)]
#define vko_uimage1D(storage_image_id)                                                             \
    _vko_storage_images_uimage1D[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(1DArray)
#define vko_image1DArray(storage_image_id)                                                         \
    _vko_storage_images_image1DArray[_vko_storage_image_index(storage_image_id)]
#define vko_iimage1DArray(storage_image_id)                                                        \
    _vko_storage_images_iimage1DArray[_vko_storage_image_index(storage_image_id)]
#define vko_uimage1DArray(storage_image_id)                                                        \
    _vko_storage_images_uimage1DArray[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(2D)
#define vko_image2D(storage_image_id)                                                              \
    _vko_storage_images_image2D[_vko_storage_image_index(storage_image_id)]
#define vko_iimage2D(storage_image_id)                                                             \
    _vko_storage_images_iimage2D[_vko_storage_image_index(storage_image_id)]
#define vko_uimage2D(storage_image_id)                                                             \
    _vko_storage_images_uimage2D[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(2DArray)
#define vko_image2DArray(storage_image_id)                                                         \
    _vko_storage_images_image2DArray[_vko_storage_image_index(storage_image_id)]
#define vko_iimage2DArray(storage_image_id)                                                        \
    _vko_storage_images_iimage2DArray[_vko_storage_image_index(storage_image_id)]
#define vko_uimage2DArray(storage_image_id)                                                        \
    _vko_storage_images_uimage2DArray[_vko_storage_image_index(storage_image_id)]

#if VKO_STORAGE_IMAGE_MULTISAMPLE_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(2DMS)
#define vko_image2DMS(storage_image_id)                                                            \
    _vko_storage_images_image2DMS[_vko_storage_image_index(storage_image_id)]
#define vko_iimage2DMS(storage_image_id)                                                           \
    _vko_storage_images_iimage2DMS[_vko_storage_image_index(storage_image_id)]
#define vko_uimage2DMS(storage_image_id)                                                           \
    _vko_storage_images_uimage2DMS[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(2DMSArray)
#define vko_image2DMSArray(storage_image_id)                                                       \
    _vko_storage_images_image2DMSArray[_vko_storage_image_index(storage_image_id)]
#define vko_iimage2DMSArray(storage_image_id)                                                      \
    _vko_storage_images_iimage2DMSArray[_vko_storage_image_index(storage_image_id)]
#define vko_uimage2DMSArray(storage_image_id)                                                      \
    _vko_storage_images_uimage2DMSArray[_vko_storage_image_index(storage_image_id)]

#endif // VKO_STORAGE_IMAGE_MULTISAMPLE_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(3D)
#define vko_image3D(storage_image_id)                                                              \
    _vko_storage_images_image3D[_vko_storage_image_index(storage_image_id)]
#define vko_iimage3D(storage_image_id)                                                             \
    _vko_storage_images_iimage3D[_vko_storage_image_index(storage_image_id)]
#define vko_uimage3D(storage_image_id)                                                             \
    _vko_storage_images_uimage3D[_vko_storage_image_index(storage_image_id)]

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(Cube)
#define vko_imageCube(storage_image_id)                                                            \
    _vko_storage_images_imageCube[_vko_storage_image_index(storage_image_id)]
#define vko_iimageCube(storage_image_id)                                                           \
    _vko_storage_images_iimageCube[_vko_storage_image_index(storage_image_id)]
#define vko_uimageCube(storage_image_id)                                                           \
    _vko_storage_images_uimageCube[_vko_storage_image_index(storage_image_id)]

#if VKO_IMAGE_CUBE_ARRAY_ENABLED

_VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT(CubeArray)
#define vko_imageCubeArray(storage_image_id)                                                       \
    _vko_storage_images_imageCubeArray[_vko_storage_image_index(storage_image_id)]
#define vko_iimageCubeArray(storage_image_id)                                                      \
    _vko_storage_images_iimageCubeArray[_vko_storage_image_index(storage_image_id)]
#define vko_uimageCubeArray(storage_image_id)                                                      \
    _vko_storage_images_uimageCubeArray[_vko_storage_image_index(storage_image_id)]

#endif // VKO_IMAGE_CUBE_ARRAY_ENABLED

#undef _VKO_DECLARE_STORAGE_IMAGE_WITHOUT_FORMAT
#undef _VKO_DECLARE_STORAGE_IMAGE_DIMENSION_WITHOUT_FORMAT

#endif // VKO_STORAGE_IMAGE_WITHOUT_FORMAT_ENABLED

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#define VKO_DECLARE_STORAGE_BUFFER(NAME, BLOCK)                                                    \
    layout(set = VKO_GLOBAL_SET, binding = VKO_STORAGE_BUFFER_BINDING)                             \
        buffer BLOCK _vko_##NAME##_storage_buffers[];

#define VKO_DECLARE_STORAGE_BUFFER_WITH_LAYOUT(NAME, BLOCK, LAYOUT)                                \
    layout(set = VKO_GLOBAL_SET, binding = VKO_STORAGE_BUFFER_BINDING, LAYOUT)                     \
        buffer BLOCK _vko_##NAME##_storage_buffers[];

#define vko_buffer(NAME, id)                                                                       \
    _vko_##NAME##_storage_buffers[_vko_storage_buffer_index(id)]

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#if VKO_ACCELERATION_STRUCTURE_ENABLED

layout(set = VKO_GLOBAL_SET, binding = VKO_ACCELERATION_STRUCTURE_BINDING)
    uniform accelerationStructureEXT _vko_acceleration_structures_accelerationStructureEXT[];

#define vko_accelerationStructureEXT(id)                                                           \
    _vko_acceleration_structures_accelerationStructureEXT[_vko_acceleration_structure_index(id)]

#endif // VKO_ACCELERATION_STRUCTURE_ENABLED

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#if VKO_INPUT_ATTACHMENT_ENABLED

#define _VKO_DECLARE_INPUT_ATTACHMENT(TYPE)                                                        \
    layout(set = VKO_LOCAL_SET, binding = VKO_INPUT_ATTACHMENT_BINDING, input_attachment_index = 0)\
        uniform TYPE _vko_input_attachments_##TYPE[];

_VKO_DECLARE_INPUT_ATTACHMENT(subpassInput)
#define vko_subpassInput(INDEX)                                                                    \
    _vko_input_attachments_subpassInput[INDEX]

_VKO_DECLARE_INPUT_ATTACHMENT(subpassInputMS)
#define vko_subpassInputMS(INDEX)                                                                  \
    _vko_input_attachments_subpassInputMS[INDEX]

_VKO_DECLARE_INPUT_ATTACHMENT(isubpassInput)
#define vko_isubpassInput(INDEX)                                                                   \
    _vko_input_attachments_isubpassInput[INDEX]

_VKO_DECLARE_INPUT_ATTACHMENT(isubpassInputMS)
#define vko_isubpassInputMS(INDEX)                                                                 \
    _vko_input_attachments_isubpassInputMS[INDEX]

_VKO_DECLARE_INPUT_ATTACHMENT(usubpassInput)
#define vko_usubpassInput(INDEX)                                                                   \
    _vko_input_attachments_usubpassInput[INDEX]

_VKO_DECLARE_INPUT_ATTACHMENT(usubpassInputMS)
#define vko_usubpassInputMS(INDEX)                                                                 \
    _vko_input_attachments_usubpassInputMS[INDEX]

#undef _VKO_DECLARE_INPUT_ATTACHMENT

#endif // VKO_INPUT_ATTACHMENT_ENABLED

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // _VULKANO_HEADER
