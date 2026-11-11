// IMPORTANT NOTE: Names starting with an underscore are private and NOT PART OF THE PUBLIC API.
// They can be changed at any moment without warning.

#pragma once

struct SamplerId {
    let _private_index: uint;
    let _private_generation: uint;
}

struct SampledImageId {
    let _private_index: uint;
    let _private_generation: uint;
}

struct StorageImageId {
    let _private_index: uint;
    let _private_generation: uint;
}

struct StorageBufferId {
    let _private_index: uint;
    let _private_generation: uint;
}

struct AccelerationStructureId {
    let _private_index: uint;
    let _private_generation: uint;
}

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

namespace vko {

[UnscopedEnum]
enum Format: int {
    formatless,
    rgba32f,
    rgba16f,
    rg32f,
    rg16f,
    r11f_g11f_b10f,
    r32f,
    r16f,
    rgba16,
    rgb10_a2,
    rgba8,
    rg16,
    rg8,
    r16,
    r8,
    rgba16_snorm,
    rgba8_snorm,
    rg16_snorm,
    rg8_snorm,
    r16_snorm,
    r8_snorm,
    rgba32i,
    rgba16i,
    rgba8i,
    rg32i,
    rg16i,
    rg8i,
    r32i,
    r16i,
    r8i,
    rgba32ui,
    rgba16ui,
    rgb10_a2ui,
    rgba8ui,
    rg32ui,
    rg16ui,
    rg8ui,
    r32ui,
    r16ui,
    r8ui,
    r64ui,
    r64i,
}

// This cringe interface is unfortunately needed because Slang can't comprehend that a generic type
// can be cast from `__DynamicResource` when it conforms to the `__IDynamicResourceCastable`
// interface. What is the interface even for???
#define _VKO_DEFINE_RESOURCE(NAME, ID, INTERFACE)                                                  \
let _##NAME##s: __DynamicResource[];                                                               \
interface INTERFACE {                                                                              \
    static func _vko_from_dyn(res: __DynamicResource) -> This;                                     \
}                                                                                                  \
func _uniform_##NAME<T: INTERFACE>(id: ID) -> T {                                                  \
    return T._vko_from_dyn(_##NAME##s[id._private_index]);                                         \
}                                                                                                  \
func _nonuniform_##NAME<T: INTERFACE>(id: ID) -> T {                                               \
    return T._vko_from_dyn(_##NAME##s[NonUniformResourceIndex(id._private_index)]);                \
}

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

[vk::binding(VKO_SAMPLER_BINDING, VKO_GLOBAL_SET)]
_VKO_DEFINE_RESOURCE(sampler, SamplerId, _ISamplerState)

#define _VKO_DEFINE_SAMPLER(TYPE)                                                                  \
extension TYPE: _ISamplerState {                                                                   \
    static func _vko_from_dyn(res: __DynamicResource) -> This {                                    \
        return res;                                                                                \
    }                                                                                              \
}                                                                                                  \
func uniform##TYPE(id: SamplerId) -> TYPE {                                                        \
    return _uniform_sampler<TYPE>(id);                                                             \
}                                                                                                  \
func nonuniform##TYPE(id: SamplerId) -> TYPE {                                                     \
    return _nonuniform_sampler<TYPE>(id);                                                          \
}

_VKO_DEFINE_SAMPLER(SamplerState)
_VKO_DEFINE_SAMPLER(SamplerComparisonState)

#undef _VKO_DEFINE_SAMPLER

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

[vk::binding(VKO_SAMPLED_IMAGE_BINDING, VKO_GLOBAL_SET)]
_VKO_DEFINE_RESOURCE(sampled_image, SampledImageId, _ISampledImage)

#define _VKO_DEFINE_SAMPLED_IMAGE(TYPE)                                                            \
extension<                                                                                         \
    T: ITexelElement,                                                                              \
    let sampleCount: int,                                                                          \
    let format: int,                                                                               \
> TYPE<T, sampleCount, format>: _ISampledImage {                                                   \
    static func _vko_from_dyn(res: __DynamicResource) -> This {                                    \
        return res;                                                                                \
    }                                                                                              \
}                                                                                                  \
func uniform##TYPE<T: ITexelElement = float4, let format: Format = formatless>(                    \
    id: SampledImageId,                                                                            \
) -> TYPE<T, 0, format> {                                                                          \
    return _uniform_sampled_image<TYPE<T, 0, format>>(id);                                         \
}                                                                                                  \
func nonuniform##TYPE<T: ITexelElement = float4, let format: Format = formatless>(                 \
    id: SampledImageId,                                                                            \
) -> TYPE<T, 0, format> {                                                                          \
    return _nonuniform_sampled_image<TYPE<T, 0, format>>(id);                                      \
}

_VKO_DEFINE_SAMPLED_IMAGE(Texture1D)
_VKO_DEFINE_SAMPLED_IMAGE(Texture1DArray)
_VKO_DEFINE_SAMPLED_IMAGE(Texture2D)
_VKO_DEFINE_SAMPLED_IMAGE(Texture2DArray)
_VKO_DEFINE_SAMPLED_IMAGE(Texture2DMS)
_VKO_DEFINE_SAMPLED_IMAGE(Texture2DMSArray)
_VKO_DEFINE_SAMPLED_IMAGE(Texture3D)
_VKO_DEFINE_SAMPLED_IMAGE(TextureCube)

#ifdef VKO_IMAGE_CUBE_ARRAY_ENABLED
#if VKO_IMAGE_CUBE_ARRAY_ENABLED
_VKO_DEFINE_SAMPLED_IMAGE(TextureCubeArray, textureCubeArray)
#endif // VKO_IMAGE_CUBE_ARRAY_ENABLED
#endif // VKO_IMAGE_CUBE_ARRAY_ENABLED

#undef _VKO_DEFINE_SAMPLED_IMAGE

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

[vk::binding(VKO_STORAGE_IMAGE_BINDING, VKO_GLOBAL_SET)]
_VKO_DEFINE_RESOURCE(storage_image, StorageImageId, _IStorageImage)

#define _VKO_DEFINE_RW_STORAGE_IMAGE(TYPE)                                                         \
extension<                                                                                         \
    T: ITexelElement,                                                                              \
    let sampleCount: int,                                                                          \
    let format: int,                                                                               \
> TYPE<T, sampleCount, format>: _IStorageImage {                                                   \
    static func _vko_from_dyn(res: __DynamicResource) -> This {                                    \
        return res;                                                                                \
    }                                                                                              \
}                                                                                                  \
func uniform##TYPE<T: ITexelElement, let format: Format>(                                          \
    id: StorageImageId,                                                                            \
) -> TYPE<T, 0, format> {                                                                          \
    return _uniform_storage_image<TYPE<T, 0, format>>(id);                                         \
}                                                                                                  \
func nonuniform##TYPE<T: ITexelElement, let format: Format>(                                       \
    id: StorageImageId,                                                                            \
) -> TYPE<T, 0, format> {                                                                          \
    return _nonuniform_storage_image<TYPE<T, 0, format>>(id);                                      \
}

_VKO_DEFINE_RW_STORAGE_IMAGE(RWTexture1D)
_VKO_DEFINE_RW_STORAGE_IMAGE(RWTexture1DArray)
_VKO_DEFINE_RW_STORAGE_IMAGE(RWTexture2D)
_VKO_DEFINE_RW_STORAGE_IMAGE(RWTexture2DArray)
_VKO_DEFINE_RW_STORAGE_IMAGE(RWTexture3D)

#undef _VKO_DEFINE_RW_STORAGE_IMAGE

#define _VKO_DEFINE_WO_STORAGE_IMAGE(TYPE)                                                         \
extension<                                                                                         \
    T: ITexelElement,                                                                              \
    let sampleCount: int,                                                                          \
    let format: int,                                                                               \
> TYPE<T, sampleCount, format>: _IStorageImage {                                                   \
    static func _vko_from_dyn(res: __DynamicResource) -> This {                                    \
        return res;                                                                                \
    }                                                                                              \
}                                                                                                  \
func uniform##TYPE<T: ITexelElement = float4, let format: Format = formatless>(                    \
    id: StorageImageId,                                                                            \
) -> TYPE<T, 0, format> {                                                                          \
    return _uniform_storage_image<TYPE<T, 0, format>>(id);                                         \
}                                                                                                  \
func nonuniform##TYPE<T: ITexelElement = float4, let format: Format = formatless>(                 \
    id: StorageImageId,                                                                            \
) -> TYPE<T, 0, format> {                                                                          \
    return _nonuniform_storage_image<TYPE<T, 0, format>>(id);                                      \
}

_VKO_DEFINE_WO_STORAGE_IMAGE(WTexture1D)
_VKO_DEFINE_WO_STORAGE_IMAGE(WTexture1DArray)
_VKO_DEFINE_WO_STORAGE_IMAGE(WTexture2D)
_VKO_DEFINE_WO_STORAGE_IMAGE(WTexture2DArray)
_VKO_DEFINE_WO_STORAGE_IMAGE(WTexture3D)

#undef _VKO_DEFINE_WO_STORAGE_IMAGE

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

[vk::binding(VKO_STORAGE_BUFFER_BINDING, VKO_GLOBAL_SET)]
_VKO_DEFINE_RESOURCE(storage_buffer, StorageBufferId, _IStorageBuffer)

#define _VKO_DEFINE_BUFFER(TYPE)                                                                   \
extension<T, I: IBufferDataLayout> TYPE<T, I>: _IStorageBuffer {                                   \
    static func _vko_from_dyn(res: __DynamicResource) -> This {                                    \
        return res;                                                                                \
    }                                                                                              \
}                                                                                                  \
func uniform##TYPE<T, I: IBufferDataLayout = DefaultDataLayout>(                                   \
    id: StorageBufferId,                                                                           \
) -> TYPE<T, I> {                                                                                  \
    return _uniform_storage_buffer<TYPE<T, I>>(id);                                                \
}                                                                                                  \
func nonuniform##TYPE<T, I: IBufferDataLayout = DefaultDataLayout>(                                \
    id: StorageBufferId,                                                                           \
) -> TYPE<T, I> {                                                                                  \
    return _nonuniform_storage_buffer<TYPE<T, I>>(id);                                             \
}

_VKO_DEFINE_BUFFER(StructuredBuffer)
_VKO_DEFINE_BUFFER(RWStructuredBuffer)

#undef _VKO_DEFINE_BUFFER

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef VKO_ACCELERATION_STRUCTURE_ENABLED
#if VKO_ACCELERATION_STRUCTURE_ENABLED

[vk::binding(VKO_ACCELERATION_STRUCTURE_BINDING, VKO_GLOBAL_SET)]
let _acceleration_structures: RaytracingAccelerationStructure[];

func uniformRaytracingAccelerationStructure(
    id: AccelerationStructureId,
) -> RaytracingAccelerationStructure {
    return _acceleration_structures[id._private_index];
}

func nonuniformRaytracingAccelerationStructure(
    id: AccelerationStructureId,
) -> RaytracingAccelerationStructure {
    return _acceleration_structures[NonUniformResourceIndex(id._private_index)];
}

#endif // VKO_ACCELERATION_STRUCTURE_ENABLED
#endif // VKO_ACCELERATION_STRUCTURE_ENABLED

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef VKO_INPUT_ATTACHMENT_ENABLED
#if VKO_INPUT_ATTACHMENT_ENABLED

[vk::binding(VKO_INPUT_ATTACHMENT_BINDING, VKO_LOCAL_SET)]
[vk::input_attachment_index(0)]
let _input_attachments: __DynamicResource[];

interface _IInputAttachment {
    static func _vko_from_dyn(res: __DynamicResource) -> This;
}

func _uniform_input_attachment<T: _IInputAttachment>(index: uint) -> T {
    return T._vko_from_dyn(_input_attachments[index]);
}

func _nonuniform_input_attachment<T: _IInputAttachment>(index: uint) -> T {
    return T._vko_from_dyn(_input_attachments[NonUniformResourceIndex(index)]);
}

#define _VKO_DEFINE_INPUT_ATTACHMENT(TYPE, GETTER, IS_MS)                                          \
extension<T, let isMS: int> TYPE<T, isMS>: _IInputAttachment {                                     \
    static func _vko_from_dyn(res: __DynamicResource) -> This {                                    \
        return res;                                                                                \
    }                                                                                              \
}                                                                                                  \
func uniform##TYPE<T>(index: uint) -> TYPE<T, IS_MS> {                                             \
    return _uniform_input_attachment<TYPE<T, IS_MS>>(index);                                       \
}                                                                                                  \
func nonuniform##TYPE<T>(index: uint) -> TYPE<T, IS_MS> {                                          \
    return _nonuniform_input_attachment<TYPE<T, IS_MS>>(index);                                    \
}

_VKO_DEFINE_INPUT_ATTACHMENT(SubpassInput, subpassInput, 0)
_VKO_DEFINE_INPUT_ATTACHMENT(SubpassInputMS, subpassInputMS, 1)

#undef _VKO_DEFINE_INPUT_ATTACHMENT

#endif // VKO_INPUT_ATTACHMENT_ENABLED
#endif // VKO_INPUT_ATTACHMENT_ENABLED

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

#undef _VKO_DEFINE_RESOURCE

} // namespace vko
