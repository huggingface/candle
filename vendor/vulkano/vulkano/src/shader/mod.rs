//! A program that is run on the device.
//!
//! In Vulkan, shaders are grouped in *shader modules*. Each shader module is built from SPIR-V
//! code and can contain one or more entry points. Note that for the moment the official
//! GLSL-to-SPIR-V compiler does not support multiple entry points.
//!
//! The vulkano library can parse and introspect SPIR-V code, but it does not fully validate the
//! code. You are encouraged to use the `vulkano-shaders` crate that will generate Rust code that
//! wraps around vulkano's shaders API.
//!
//! # Shader interface
//!
//! Vulkan has specific rules for interfacing shaders with each other, and with other parts
//! of a program.
//!
//! ## Endianness
//!
//! The Vulkan specification requires that a Vulkan implementation has runtime support for the
//! types [`u8`], [`u16`], [`u32`], [`u64`] as well as their signed versions, as well as [`f32`]
//! and [`f64`] on the host, and that the representation and endianness of these types matches
//! those on the device. This means that if you have for example a `Subbuffer<u32>`, you can be
//! sure that it is represented the same way on the host as it is on the device, and you don't need
//! to worry about converting the endianness.
//!
//! ## Layout of data
//!
//! When buffers, push constants or other user-provided data are accessed in shaders,
//! the shader expects the values inside to be laid out in a specific way. For every uniform
//! buffer, storage buffer or push constant block, the SPIR-V specification requires the SPIR-V
//! code to provide the `Offset` decoration for every member of a struct, indicating where it is
//! placed relative to the start of the struct. If there are arrays or matrices among the
//! variables, the SPIR-V code must also provide an `ArrayStride` or `MatrixStride` decoration for
//! them, indicating the number of bytes between the start of each element in the array or column
//! in the matrix. When providing data to shaders, you must make sure that your data is placed at
//! the locations indicated within the SPIR-V code, or the shader will read the wrong data and
//! produce nonsense.
//!
//! GLSL does not require you to give explicit offsets and/or strides to your variables (although
//! it has the option to provide them if you wish). Instead, the shader compiler automatically
//! assigns every variable an offset, increasing in the order you declare them in.
//! To know the exact offsets that will be used, so that you can lay out your data appropriately,
//! you must know the alignment rules that the shader compiler uses. The shader compiler will
//! always give a variable the smallest offset that fits the alignment rules and doesn't overlap
//! with the previous variable. The shader compiler uses default alignment rules depending on the
//! type of block, but you can specify another layout by using the `layout` qualifier.
//!
//! ## Alignment rules
//!
//! The offset of each variable from the start of a block, matrix or array must be a
//! multiple of a certain number, which is called its *alignment*. The stride of an array or matrix
//! must likewise be a multiple of this number. An alignment is always a power-of-two value.
//! Regardless of whether the offset/stride is provided manually in the compiled SPIR-V code,
//! or assigned automatically by the shader compiler, all variable offsets/strides in a shader must
//! follow these alignment rules.
//!
//! Three sets of [alignment rules] are supported by Vulkan. Each one has a GLSL qualifier that
//! you can place in front of a block, to make the shader compiler use that layout for the block.
//! If you don't provide this qualifier, it will use a default alignment.
//!
//! - **Scalar alignment** (GLSL qualifier: `layout(scalar)`, requires the
//!   [`GL_EXT_scalar_block_layout`] GLSL extension). This is the same as the C alignment,
//!   expressed in Rust with the
//!   [`#[repr(C)]`](https://doc.rust-lang.org/nomicon/other-reprs.html#reprc) attribute.
//!   The shader compiler does not use this alignment by default, so you must use the GLSL
//!   qualifier. You must also enable the [`scalar_block_layout`] feature in Vulkan.
//! - **Base alignment**, also known as **std430** (GLSL qualifier: `layout(std430)`). The shader
//!   compiler uses this alignment by default for all shader data except uniform buffers. If you
//!   use the base alignment for a uniform buffer, you must also enable the
//!   [`uniform_buffer_standard_layout`] feature in Vulkan.
//! - **Extended alignment**, also known as **std140** (GLSL qualifier: `layout(std140)`). The
//!   shader compiler uses this alignment by default for uniform buffers.
//!
//! Each alignment type is a subset of the ones above it, so if something adheres to the extended
//! alignment rules, it also follows the rules for the base and scalar alignments.
//!
//! In all three of these alignment rules, a primitive/scalar value with a size of N bytes has an
//! alignment of N, meaning that it must have an offset that is a multiple of its size,
//! like in C or Rust. For example, a `float` (like a Rust `f32`) has a size of 4 bytes,
//! and an alignment of 4.
//!
//! The differences between the alignment rules are in how compound types (vectors, matrices,
//! arrays and structs) are expected to be laid out. For a compound type with an element whose
//! alignment is N, the scalar alignment considers the alignment of the compound type to be also N.
//! However, the base and extended alignments are stricter:
//!
//! | GLSL type | Scalar          | Base            | Extended                 |
//! |-----------|-----------------|-----------------|--------------------------|
//! | primitive | N               | N               | N                        |
//! | `vec2`    | N               | N * 2           | N * 2                    |
//! | `vec3`    | N               | N * 4           | N * 4                    |
//! | `vec4`    | N               | N * 4           | N * 4                    |
//! | array     | N               | N               | max(N, 16)               |
//! | `struct`  | N<sub>max</sub> | N<sub>max</sub> | max(N<sub>max</sub>, 16) |
//!
//! In the base and extended alignment, the alignment of a vector is the size of the whole vector,
//! rather than the size of its individual elements as is the case in the scalar alignment.
//! But note that, because alignment must be a power of two, the alignment of `vec3` cannot be
//! N * 3; it must be N * 4, the same alignment as `vec4`. This means that it is not possible to
//! tightly pack multiple `vec3` values (e.g. in an array); there will always be empty padding
//! between them.
//!
//! In both the scalar and base alignment, the alignment of arrays and their elements is equal to
//! the alignment of the contained type. In the extended alignment, however, the alignment is
//! always at least 16 (the size of a `vec4`). Therefore, the minimum stride of the array can be
//! much greater than the element size. For example, in an array of `float`, the stride must be at
//! least 16, even though a `float` itself is only 4 bytes in size. Every `float` element will be
//! followed by at least 12 bytes of unused space.
//!
//! A matrix `matCxR` is considered equivalent to an array of column vectors `vecR[C]`.
//! In the base and extended alignments, that means that if the matrix has 3 rows, there will be
//! one element's worth of padding between the column vectors. In the extended alignment,
//! the alignment is also at least 16, further increasing the amount of padding between the
//! column vectors.
//!
//! The rules for `struct`s are similar to those of arrays. When the members of the struct have
//! different alignment requirements, the alignment of the struct as a whole is the maximum
//! of the alignments of its members. As with arrays, in the extended alignment, the alignment
//! of a struct is at least 16.
//!
//! # Safety
//!
//! The following general safety requirements apply to the descriptors in a shader, and to the
//! resources that were bound to them. They apply to all shader types, and must be met at the
//! moment the shader executes on the device.
//!
//! Vulkano will validate many of these requirements, but it is only able to do so when the
//! resources involved are statically known. This means that either the descriptor binding must not
//! be arrayed, or if it is arrayed, that the array must be indexed only by constants. If the
//! array index is dynamic (meaning that it depends on values that are inputs to the shader),
//! then Vulkano cannot check these requirements, and you must ensure them yourself.
//!
//! Some requirements, such as the validity of pointers to device memory, cannot be validated
//! by Vulkano at all.
//!
//! ## Descriptors
//!
//! - If a descriptor set binding was created with [`DescriptorBindingFlags::PARTIALLY_BOUND`],
//!   then if the shader accesses a descriptor in that binding, the descriptor must be initialized
//!   and contain a valid resource.
//!
//! ## Buffers and memory accesses
//!
//! - If the [`robust_buffer_access`](DeviceFeatures::robust_buffer_access) feature is not enabled
//!   on the device, then the shader must not access any values outside the range of the buffer, as
//!   specified when writing the descriptor set. <sup>[\[06935\]] [\[06936\]]</sup>
//! - If any `PhysicalStorageBuffer` pointers to device memory are dereferenced in the shader,
//!   then:
//!   - The pointer must point to valid memory of the correct type.
//!   - The pointer must be aligned to a multiple of the largest scalar type within the type that
//!     it points to. <sup>[\[06314\]]</sup>
//!   - If the instruction has `Aligned` as one of its memory operands, the pointer must be aligned
//!     to the specified alignment. <sup>[\[06315\]]</sup>
//! - For `OpCooperativeMatrixLoadKHR`, `OpCooperativeMatrixStoreKHR`, `OpCooperativeMatrixLoadNV`
//!   and `OpCooperativeMatrixStoreNV` instructions, the `Pointer` and `Stride` operands must both
//!   be aligned to the minimum of either 16 bytes or the number of bytes per row/column of the
//!   matrix (depending on the `ColumnMajor` and `RowMajor` decorations). <sup>[\[06324\]]
//!   [\[08986\]]</sup>
//!
//! ## Image views and buffer views
//!
//! - The [`view_type`](ImageView::view_type) of the bound image view must match the `Dim` operand
//!   of the `OpImageType`. <sup>[\[07752\]]</sup>
//! - The numeric type of the [`format`](ImageView::format) of the bound image view must match the
//!   `Sampled Type` operand of the `OpImageType`. <sup>[\[07753\]]</sup>
//! - For every `OpImageWrite` instruction, the type of the `Texel` operand must have at least as
//!   many components as the format of the bound image view or buffer view. If the bound image
//!   view's format is [`Format::A8_UNORM`], then the type of the `Texel` operand must have four
//!   components. <sup>[\[04469\]] [\[08795\]] [\[08796\]]</sup>
//! - The `Sampled Type` operand of the `OpTypeImage` declaration must have a `Width` of 64, if and
//!   only if the format of the bound image view or buffer view also has a 64-bit component.
//!   Otherwise, it must have a `Width` of 32. <sup>[\[04470\]] [\[04471\]] [\[04472\]]
//!   [\[04473\]]</sup>
//! - The [`samples`](Image::samples) of the underlying image of the bound image view must match
//!   the `MS` operand of the `OpImageType`. <sup>[\[08725\]] [\[08726\]]</sup>
//! - For a storage image/texel buffer declared with `OpTypeImage` with an `Unknown` format:
//!   - If it is written to in the shader, the format of the bound image view or buffer view must
//!     have the [`FormatFeatures::STORAGE_WRITE_WITHOUT_FORMAT`] format feature. <sup>[\[07027\]]
//!     [\[07029\]]</sup>
//!   - If it is read from in the shader, the format of the bound image view or buffer view must
//!     have the [`FormatFeatures::STORAGE_READ_WITHOUT_FORMAT`] format feature. <sup>[\[07028\]]
//!     [\[07030\]]</sup>
//! - If atomic operations are used on a storage image/texel buffer:
//!   - The bound image view's format must have the [`FormatFeatures::STORAGE_IMAGE_ATOMIC`] format
//!     feature. <sup>[\[02691\]]</sup>
//!   - The bound buffer view's format must have the
//!     [`FormatFeatures::STORAGE_TEXEL_BUFFER_ATOMIC`] format feature. <sup>[\[07888\]]</sup>
//!
//! ## Image sampling
//!
//! If the bound sampler uses [`Filter::Linear`] or [`SamplerMipmapMode::Linear`]:
//! - The bound image view's format must have the [`FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR`]
//!   format feature. <sup>[\[04553\]] [\[04770\]]</sup>
//!
//! If the bound sampler uses [`Filter::Cubic`]:
//! - The bound image view's format must have the [`FormatFeatures::SAMPLED_IMAGE_FILTER_CUBIC`]
//!   format feature. <sup>[\[02692\]]</sup>
//! - The bound image view's type and format must support cubic filtering, as indicated in
//!   [`ImageFormatProperties::filter_cubic`] returned from
//!   [`PhysicalDevice::image_format_properties`]. <sup>[\[02694\]]</sup>
//! - If the sampler's reduction mode is [`SamplerReductionMode::Min`] or
//!   [`SamplerReductionMode::Max`], the image view type and format must support cubic min-max
//!   filtering, as indicated in [`ImageFormatProperties::filter_cubic_minmax`] returned from
//!   [`PhysicalDevice::image_format_properties`]. <sup>[\[02695\]]</sup>
//!
//! If the bound sampler uses [depth comparison](SamplerCreateInfo::compare):
//! - The bound image view's format must have the
//!   [`FormatFeatures::SAMPLED_IMAGE_DEPTH_COMPARISON`] format feature. <sup>[\[06479\]]</sup>
//!
//! If the bound sampler uses [unnormalized
//! coordinates](SamplerCreateInfo::unnormalized_coordinates):
//! - The bound image view must have a type of [`ImageViewType::Dim1d`] or
//!   [`ImageViewType::Dim2d`]. <sup>[\[08609\]]</sup>
//! - The sampler must not be used in any `OpImageSample*` or `OpImageSparseSample*` instructions,
//!   that contain `ImplicitLod`, `Dref` or `Proj` in their name. <sup>[\[08610\]]</sup>
//! - The sampler must not be used in any `OpImageSample*` or `OpImageSparseSample*` instructions,
//!   that include an LOD bias or offset operand. <sup>[\[08611\]]</sup>
//!
//! If the bound sampler has a [sampler YCbCr conversion](crate::image::sampler::ycbcr):
//! - The sampler must only be used in `OpImageSample*` or `OpImageSparseSample*` instructions.
//!   <sup>[\[06550\]]</sup>
//! - The sampler must not be used with the `ConstOffset` or `Offset` image operands.
//!   <sup>[\[06551\]]</sup>
//!
//! ## Mesh shading
//!
//! - If the shader declares the `OutputPoints` execution mode with a value greater than 0, and the
//!   [`maintenance5`](DeviceFeatures::maintenance5) feature is not enabled on the device, then the
//!   shader must write to a variable decorated with `PointSize` for each output point.
//!   <sup>[\[09218\]]</sup>
//!
//! For `OpSetMeshOutputsEXT` instructions:
//!
//! - The `Vertex Count` operand must be less than or equal to the value declared with the shader's
//!   `OutputVertices` execution mode. <sup>[\[07332\]]</sup>
//! - The `Primitive Count` operand must be less than or equal to the value declared with the
//!   shader's `OutputPrimitivesEXT` execution mode. <sup>[\[07333\]]</sup>
//!
//! ## Acceleration structures, ray queries and ray tracing
//!
//! - Acceleration structures that are used as operands to an instruction must have been built as a
//!   top-level acceleration structure. <sup>[\[06352\]] [\[06359\]] [\[06365\]] [\[07709\]]</sup>
//! - In any top-level acceleration structure, the pointers that refer to the contained
//!   bottom-level acceleration structure instances must point to valid bottom-level acceleration
//!   structures.
//!
//! For `OpRayQueryInitializeKHR` and `OpTraceRayKHR` instructions:
//!
//! - The `Rayflags` operand must not contain more than one of:
//!   - `SkipTrianglesKHR`, `CullBackFacingTrianglesKHR` and `CullFrontFacingTrianglesKHR`
//!     <sup>[\[06889\]] [\[06892\]]</sup>
//!   - `SkipTrianglesKHR` and `SkipAABBsKHR` <sup>[\[06890\]] [\[06552\]] [\[07712\]]</sup>
//!   - `OpaqueKHR`, `NoOpaqueKHR`, `CullOpaqueKHR`, and `CullNoOpaqueKHR` <sup>[\[06891\]]
//!     [\[06893\]]</sup>
//! - The `RayOrigin` and `RayDirection` operands must not contain infinite or NaN values. <sup>
//!   [\[06348\]] [\[06351\]] [\[06355\]] [\[06358\]] </sup>
//! - The `RayTmin` and `RayTmax` operands must not contain negative or NaN values, and `RayTmin`
//!   must be less than or equal to `RayTmax`. <sup> [\[06349\]] [\[06350\]] [\[06351\]]
//!   [\[06356\]] [\[06357\]] [\[06358\]] </sup>
//!
//! For `OpRayQueryGenerateIntersectionKHR` instructions:
//!
//! - The `Hit T` operand must be greater than or equal to the value that would be returned by
//!   `OpRayQueryGetRayTMinKHR`. <sup>[\[06353\]]</sup>
//! - The `Hit T` operand must be less than or equal to the value that would be returned by
//!   `OpRayQueryGetIntersectionTKHR` for the current committed intersection.
//!   <sup>[\[06353\]]</sup>
//!
//! For `OpReportIntersectionKHR` instructions:
//!
//! - The `Hit Kind` operand must be between 0 and 127 inclusive. <sup>[\[06998\]]</sup>
//!
//! ## Dynamically uniform values and control flow
//!
//! In a shader, a value (expression, variable) is *[dynamically uniform]* if its value is the same
//! for all shader invocations within an *invocation group*. What counts as an invocation group
//! depends on the type of shader being executed:
//!
//! - For compute, task and mesh shaders, an invocation group is the same as the (local) workgroup.
//!   A single `dispatch` command value spawns one distinct invocation group for every element in
//!   the product of the given `group_counts` argument.
//! - For all other graphics shaders, an invocation group is all shaders invoked by a single draw
//!   command. For indirect draws, each element of the indirect buffer creates one draw call.
//! - For ray tracing shaders, an invocation group is an implementation-dependent subset of the
//!   shaders invoked by a single ray tracing command.
//!
//! Vulkan and SPIR-V assume that certain values within a shader are dynamically uniform, and will
//! optimize the generated shader code accordingly. If such a value is not actually dynamically
//! uniform, this results in undefined behavior. This concerns the following values:
//!
//! - The index into an arrayed descriptor binding. If the index is not dynamically uniform, you
//!   must explicitly mark it with the `NonUniform` decoration in SPIR-V, or the `nonuniformEXT`
//!   function in GLSL. <sup>[\[06274\]]</sup>
//! - The `Index` argument of the `OpGroupNonUniformQuadBroadcast` instruction.
//!   <sup>[\[06276\]]</sup>
//! - The `Id` argument of the `OpGroupNonUniformBroadcast` instruction. <sup>[\[06277\]]</sup>
//! - The arguments of the `OpEmitMeshTasksEXT` and `OpSetMeshOutputsEXT` instructions.
//!   <sup>[\[07117\]] [\[07118\]]</sup>
//! - The `Texture Sampled Image` and `Weight Image` arguments of the `OpImageWeightedSampleQCOM`
//!   instruction. <sup>[\[06979\]]</sup>
//! - The `Texture Sampled Image`, `Reference Sampled Image` and `Block Size` arguments of the
//!   `OpImageBlockMatchSADQCOM` and `OpImageBlockMatchSSDQCOM` instructions.
//!   <sup>[\[06982\]]</sup>
//! - The `Sampled Texture Image` and `Box Size` arguments of the `OpImageBoxFilterQCOM`
//!   instruction. <sup>[\[06990\]]</sup>
//! - The `Target Sampled Image`, `Reference Sampled Image` and `Block Size` arguments of any
//!   `OpImageBlockMatchWindow*QCOM` or `OpImageBlockMatchGather*QCOM` instructions.
//!   <sup>[\[09219\]]</sup>
//!
//! Some operations have specific requirements for control flow within the shader:
//!
//! - The `OpEmitMeshTasksEXT` and `OpSetMeshOutputsEXT` instructions must be executed uniformly
//!   within the invocation group. That means that, either all shader invocations within the
//!   invocation group must execute the instruction, or none of them must execute it.
//!   <sup>[\[07117\]] [\[07118\]]</sup>
//! - If the `PointSize` built-in is written to, then all execution paths must write to it.
//!   <sup>[\[09190\]]</sup>
//!
//! [alignment rules]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap15.html#interfaces-resources-layout
//! [`GL_EXT_scalar_block_layout`]: https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_scalar_block_layout.txt
//! [`scalar_block_layout`]: DeviceFeatures::scalar_block_layout
//! [`uniform_buffer_standard_layout`]: DeviceFeatures::uniform_buffer_standard_layout
//! [dynamically uniform]: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_uniformity
//! [\[02691\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-02691
//! [\[02692\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-02692
//! [\[02694\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-filterCubic-02694
//! [\[02695\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-filterCubicMinmax-02695
//! [\[04469\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-OpImageWrite-04469
//! [\[04470\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-SampledType-04470
//! [\[04471\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-SampledType-04471
//! [\[04472\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-SampledType-04472
//! [\[04473\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-SampledType-04473
//! [\[04553\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-magFilter-04553
//! [\[04770\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-mipmapMode-04770
//! [\[06274\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-NonUniform-06274
//! [\[06276\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-subgroupBroadcastDynamicId-06276
//! [\[06277\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-subgroupBroadcastDynamicId-06277
//! [\[06314\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-PhysicalStorageBuffer64-06314
//! [\[06315\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-PhysicalStorageBuffer64-06315
//! [\[06324\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpCooperativeMatrixLoadNV-06324
//! [\[06348\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06348
//! [\[06349\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06349
//! [\[06350\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06350
//! [\[06351\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06351
//! [\[06352\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06352
//! [\[06353\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryGenerateIntersectionKHR-06353
//! [\[06355\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06355
//! [\[06356\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06356
//! [\[06357\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06357
//! [\[06358\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06358
//! [\[06359\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06359
//! [\[06361\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayMotionNV-06361
//! [\[06362\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayMotionNV-06362
//! [\[06363\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayMotionNV-06363
//! [\[06364\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayMotionNV-06364
//! [\[06365\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayMotionNV-06365
//! [\[06366\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayMotionNV-06366
//! [\[06479\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-06479
//! [\[06550\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-06550
//! [\[06551\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-ConstOffset-06551
//! [\[06552\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06552
//! [\[06889\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06889
//! [\[06890\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06890
//! [\[06891\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpRayQueryInitializeKHR-06891
//! [\[06892\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06892
//! [\[06893\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpTraceRayKHR-06893
//! [\[06935\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-uniformBuffers-06935
//! [\[06936\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-storageBuffers-06936
//! [\[06979\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpImageWeightedSampleQCOM-06979
//! [\[06982\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpImageBlockMatchSADQCOM-06982
//! [\[06990\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpImageBoxFilterQCOM-06990
//! [\[06998\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpReportIntersectionKHR-06998
//! [\[07027\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-OpTypeImage-07027
//! [\[07029\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-OpTypeImage-07029
//! [\[07028\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-OpTypeImage-07028
//! [\[07030\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-OpTypeImage-07030
//! [\[07117\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-TaskEXT-07117
//! [\[07118\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-MeshEXT-07118
//! [\[07332\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-MeshEXT-07332
//! [\[07333\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-MeshEXT-07333
//! [\[07705\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07705
//! [\[07706\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07706
//! [\[07707\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07707
//! [\[07708\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07708
//! [\[07709\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayMotionNV-07709
//! [\[07710\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07710
//! [\[07712\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07712
//! [\[07713\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07713
//! [\[07714\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07714
//! [\[07752\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-viewType-07752
//! [\[07753\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-format-07753
//! [\[07888\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-07888
//! [\[08609\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-08609
//! [\[08610\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-08610
//! [\[08611\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-None-08611
//! [\[08725\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-samples-08725
//! [\[08726\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-samples-08726
//! [\[08795\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-OpImageWrite-08795
//! [\[08796\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdDispatch-OpImageWrite-08796
//! [\[08986\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986
//! [\[09190\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-maintenance5-09190
//! [\[09218\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-MeshEXT-09218
//! [\[09219\]]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-RuntimeSpirv-OpImageBlockMatchWindow-09219

use self::spirv::{Id, Instruction};
#[cfg(doc)]
use crate::{
    acceleration_structure::BuildAccelerationStructureFlags,
    descriptor_set::layout::DescriptorBindingFlags,
    device::{physical::PhysicalDevice, DeviceFeatures, DeviceProperties},
    format::FormatFeatures,
    image::{
        sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode, SamplerReductionMode},
        view::ImageView,
        Image, ImageFormatProperties,
    },
};
use crate::{
    descriptor_set::layout::DescriptorType,
    device::{Device, DeviceOwned},
    format::{Format, NumericType},
    image::view::ImageViewType,
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags_enum},
    pipeline::layout::PushConstantRange,
    shader::spirv::{Capability, Spirv},
    sync::PipelineStages,
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ash::vk;
use bytemuck::bytes_of;
use foldhash::{HashMap, HashSet};
use half::f16;
use smallvec::SmallVec;
use spirv::ExecutionModel;
use std::{
    collections::hash_map::Entry,
    mem::{discriminant, MaybeUninit},
    num::NonZero,
    ptr,
    sync::Arc,
};

pub mod reflect;
pub mod spirv;

include!(crate::autogen_output!("spirv_reqs.rs"));

/// Contains SPIR-V code with one or more entry points.
#[derive(Debug)]
pub struct ShaderModule {
    handle: vk::ShaderModule,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZero<u64>,

    spirv: Spirv,
    specialization_constants: Vec<(u32, SpecializationConstant)>,
}

impl ShaderModule {
    /// Creates a new shader module, panicking on a validation error.
    ///
    /// This is a shortcut for `try_new().map_err(Validated::unwrap)`.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code in `create_info.code` must be valid.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub unsafe fn new(
        device: &Arc<Device>,
        create_info: &ShaderModuleCreateInfo<'_>,
    ) -> Result<Arc<ShaderModule>, VulkanError> {
        match unsafe { Self::try_new(device, create_info) } {
            Ok(res) => Ok(res),
            Err(err) => Err(err.unwrap()),
        }
    }

    /// Creates a new shader module.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code in `create_info.code` must be valid.
    #[inline]
    pub unsafe fn try_new(
        device: &Arc<Device>,
        create_info: &ShaderModuleCreateInfo<'_>,
    ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
        let spirv = Spirv::new(create_info.code).map_err(|err| {
            Box::new(ValidationError {
                context: "create_info.code".into(),
                problem: format!("error while parsing: {}", err).into(),
                ..Default::default()
            })
        })?;

        Self::validate_new(device, create_info, &spirv)?;

        Ok(unsafe { Self::new_with_spirv_unchecked(device, create_info, spirv) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &ShaderModuleCreateInfo<'_>,
        spirv: &Spirv,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device, spirv)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: &Arc<Device>,
        create_info: &ShaderModuleCreateInfo<'_>,
    ) -> Result<Arc<ShaderModule>, VulkanError> {
        let spirv = Spirv::new(create_info.code).unwrap();

        unsafe { Self::new_with_spirv_unchecked(device, create_info, spirv) }
    }

    unsafe fn new_with_spirv_unchecked(
        device: &Arc<Device>,
        create_info: &ShaderModuleCreateInfo<'_>,
        spirv: Spirv,
    ) -> Result<Arc<ShaderModule>, VulkanError> {
        let create_info_vk = create_info.to_vk();

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.v1_0.create_shader_module)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(unsafe { Self::from_handle_with_spirv(device, handle, create_info, spirv) })
    }

    /// Creates a new `ShaderModule` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: &Arc<Device>,
        handle: vk::ShaderModule,
        create_info: &ShaderModuleCreateInfo<'_>,
    ) -> Arc<ShaderModule> {
        let spirv = Spirv::new(create_info.code).unwrap();

        unsafe { Self::from_handle_with_spirv(device, handle, create_info, spirv) }
    }

    unsafe fn from_handle_with_spirv(
        device: &Arc<Device>,
        handle: vk::ShaderModule,
        create_info: &ShaderModuleCreateInfo<'_>,
        spirv: Spirv,
    ) -> Arc<ShaderModule> {
        let &ShaderModuleCreateInfo { code: _, _ne: _ } = create_info;
        let specialization_constants = reflect::specialization_constants(&spirv);

        Arc::new(ShaderModule {
            handle,
            device: InstanceOwnedDebugWrapper(device.clone()),
            id: Self::next_id(),

            spirv,
            specialization_constants,
        })
    }

    /// Builds a new shader module from SPIR-V 32-bit words. The shader code is parsed and the
    /// necessary information is extracted from it.
    ///
    /// # Safety
    ///
    /// - The SPIR-V code is not validated beyond the minimum needed to extract the information.
    #[deprecated(since = "0.34.0", note = "use `new` instead")]
    #[inline]
    pub unsafe fn from_words(
        device: Arc<Device>,
        words: &[u32],
    ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
        unsafe { Self::try_new(&device, &ShaderModuleCreateInfo::new(words)) }
    }

    /// As `from_words`, but takes a slice of bytes.
    ///
    /// # Panics
    ///
    /// - Panics if `bytes` is not aligned to 4.
    /// - Panics if the length of `bytes` is not a multiple of 4.
    #[deprecated(
        since = "0.34.0",
        note = "use `shader::spirv::bytes_to_words`, and then use `new` instead"
    )]
    #[inline]
    pub unsafe fn from_bytes(
        device: Arc<Device>,
        bytes: &[u8],
    ) -> Result<Arc<ShaderModule>, Validated<VulkanError>> {
        let words = spirv::bytes_to_words(bytes).unwrap();

        unsafe { Self::try_new(&device, &ShaderModuleCreateInfo::new(&words)) }
    }

    /// Returns the specialization constants that are defined in the module,
    /// along with their default values.
    ///
    /// Specialization constants are constants whose value can be overridden when you create
    /// a pipeline. They are indexed by their `constant_id`.
    #[inline]
    pub fn specialization_constants(&self) -> &[(u32, SpecializationConstant)] {
        &self.specialization_constants
    }

    /// Applies the specialization constants to the shader module, and returns a specialized
    /// version of the module, panicking on a validation error.
    ///
    /// Constants that are not given a value here will have the default value that was specified
    /// for them in the shader code.
    /// When provided, they must have the same type as defined in the shader (as returned by
    /// [`specialization_constants`]).
    ///
    /// This is a shortcut for `try_specialize().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_specialize`] returns a [`ValidationError`].
    ///
    /// [`specialization_constants`]: Self::specialization_constants
    /// [`try_specialize`]: Self::try_specialize
    #[inline]
    #[track_caller]
    pub fn specialize(
        self: &Arc<Self>,
        specialization_info: &[(u32, SpecializationConstant)],
    ) -> Arc<SpecializedShaderModule> {
        self.try_specialize(specialization_info).unwrap()
    }

    /// Applies the specialization constants to the shader module, and returns a specialized
    /// version of the module.
    ///
    /// Constants that are not given a value here will have the default value that was specified
    /// for them in the shader code.
    /// When provided, they must have the same type as defined in the shader (as returned by
    /// [`specialization_constants`]).
    ///
    /// [`specialization_constants`]: Self::specialization_constants
    #[inline]
    pub fn try_specialize(
        self: &Arc<Self>,
        specialization_info: &[(u32, SpecializationConstant)],
    ) -> Result<Arc<SpecializedShaderModule>, Box<ValidationError>> {
        SpecializedShaderModule::try_new(self, specialization_info)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn specialize_unchecked(
        self: &Arc<Self>,
        specialization_info: &[(u32, SpecializationConstant)],
    ) -> Arc<SpecializedShaderModule> {
        unsafe { SpecializedShaderModule::new_unchecked(self, specialization_info) }
    }

    /// Equivalent to calling [`specialize`] with empty specialization info,
    /// and then calling [`SpecializedShaderModule::entry_point`].
    ///
    /// [`specialize`]: Self::specialize
    #[inline]
    pub fn entry_point(self: &Arc<Self>, name: &str) -> Option<EntryPoint> {
        unsafe { self.specialize_unchecked(&[]) }.entry_point(name)
    }

    /// Equivalent to calling [`specialize`] with empty specialization info,
    /// and then calling [`SpecializedShaderModule::entry_point_with_execution`].
    ///
    /// [`specialize`]: Self::specialize
    #[inline]
    pub fn entry_point_with_execution(
        self: &Arc<Self>,
        name: &str,
        execution: ExecutionModel,
    ) -> Option<EntryPoint> {
        unsafe { self.specialize_unchecked(&[]) }.entry_point_with_execution(name, execution)
    }

    /// Equivalent to calling [`specialize`] with empty specialization info,
    /// and then calling [`SpecializedShaderModule::single_entry_point`].
    ///
    /// [`specialize`]: Self::specialize
    #[inline]
    pub fn single_entry_point(self: &Arc<Self>) -> Option<EntryPoint> {
        unsafe { self.specialize_unchecked(&[]) }.single_entry_point()
    }

    /// Equivalent to calling [`specialize`] with empty specialization info,
    /// and then calling [`SpecializedShaderModule::single_entry_point_with_execution`].
    ///
    /// [`specialize`]: Self::specialize
    #[inline]
    pub fn single_entry_point_with_execution(
        self: &Arc<Self>,
        execution: ExecutionModel,
    ) -> Option<EntryPoint> {
        unsafe { self.specialize_unchecked(&[]) }.single_entry_point_with_execution(execution)
    }
}

impl Drop for ShaderModule {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe { (fns.v1_0.destroy_shader_module)(self.device.handle(), self.handle, ptr::null()) };
    }
}

unsafe impl VulkanObject for ShaderModule {
    type Handle = vk::ShaderModule;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for ShaderModule {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(ShaderModule);

#[derive(Clone, Debug)]
pub struct ShaderModuleCreateInfo<'a> {
    /// The SPIR-V code, in the form of 32-bit words.
    ///
    /// There is no default value.
    pub code: &'a [u32],

    pub _ne: crate::NonExhaustive<'a>,
}

impl<'a> ShaderModuleCreateInfo<'a> {
    /// Returns a default `ShaderModuleCreateInfo` with the provided `code`.
    #[inline]
    pub const fn new(code: &'a [u32]) -> Self {
        Self {
            code,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(
        &self,
        device: &Device,
        spirv: &Spirv,
    ) -> Result<(), Box<ValidationError>> {
        let &Self { code, _ne: _ } = self;

        if code.is_empty() {
            return Err(Box::new(ValidationError {
                context: "code".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkShaderModuleCreateInfo-codeSize-01085"],
                ..Default::default()
            }));
        }

        let spirv_version = Version {
            patch: 0, // Ignore the patch version
            ..spirv.version()
        };

        {
            match spirv_version {
                Version::V1_0 => None,
                Version::V1_1 | Version::V1_2 | Version::V1_3 => {
                    (!(device.api_version() >= Version::V1_1)).then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    ]))
                }
                Version::V1_4 => (!(device.api_version() >= Version::V1_2
                    || device.enabled_extensions().khr_spirv_1_4))
                    .then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_spirv_1_4")]),
                    ])),
                Version::V1_5 => {
                    (!(device.api_version() >= Version::V1_2)).then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                    ]))
                }
                Version::V1_6 => {
                    (!(device.api_version() >= Version::V1_3)).then_some(RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    ]))
                }
                _ => {
                    return Err(Box::new(ValidationError {
                        context: "code".into(),
                        problem: format!(
                            "uses SPIR-V version {}.{}, which is not supported by Vulkan",
                            spirv_version.major, spirv_version.minor
                        )
                        .into(),
                        // vuids?
                        ..Default::default()
                    }));
                }
            }
        }
        .map_or(Ok(()), |requires_one_of| {
            Err(Box::new(ValidationError {
                context: "code".into(),
                problem: format!(
                    "uses SPIR-V version {}.{}",
                    spirv_version.major, spirv_version.minor
                )
                .into(),
                requires_one_of,
                ..Default::default()
            }))
        })?;

        for &capability in spirv
            .capabilities()
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Capability { capability } => Some(capability),
                _ => None,
            })
        {
            validate_spirv_capability(device, capability).map_err(|err| err.add_context("code"))?;
        }

        for extension in spirv
            .extensions()
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Extension { name } => Some(name.as_str()),
                _ => None,
            })
        {
            validate_spirv_extension(device, extension).map_err(|err| err.add_context("code"))?;
        }

        // VUID-VkShaderModuleCreateInfo-pCode-08736
        // VUID-VkShaderModuleCreateInfo-pCode-08737
        // VUID-VkShaderModuleCreateInfo-pCode-08738
        // Unsafe

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::ShaderModuleCreateInfo<'_> {
        let &Self { code, _ne: _ } = self;

        vk::ShaderModuleCreateInfo::default()
            .flags(vk::ShaderModuleCreateFlags::empty())
            .code(code)
    }
}

/// The value to provide for a specialization constant, when creating a pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecializationConstant {
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(f16),
    F32(f32),
    F64(f64),
}

impl SpecializationConstant {
    /// Returns the value as a byte slice. Booleans are expanded to a `VkBool32` value.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Bool(false) => bytes_of(&vk::FALSE),
            Self::Bool(true) => bytes_of(&vk::TRUE),
            Self::U8(value) => bytes_of(value),
            Self::U16(value) => bytes_of(value),
            Self::U32(value) => bytes_of(value),
            Self::U64(value) => bytes_of(value),
            Self::I8(value) => bytes_of(value),
            Self::I16(value) => bytes_of(value),
            Self::I32(value) => bytes_of(value),
            Self::I64(value) => bytes_of(value),
            Self::F16(value) => bytes_of(value),
            Self::F32(value) => bytes_of(value),
            Self::F64(value) => bytes_of(value),
        }
    }

    /// Returns whether `self` and `other` have the same type, ignoring the value.
    #[inline]
    pub fn eq_type(&self, other: &Self) -> bool {
        discriminant(self) == discriminant(other)
    }
}

impl From<bool> for SpecializationConstant {
    #[inline]
    fn from(value: bool) -> Self {
        SpecializationConstant::Bool(value)
    }
}

impl From<i8> for SpecializationConstant {
    #[inline]
    fn from(value: i8) -> Self {
        SpecializationConstant::I8(value)
    }
}

impl From<i16> for SpecializationConstant {
    #[inline]
    fn from(value: i16) -> Self {
        SpecializationConstant::I16(value)
    }
}

impl From<i32> for SpecializationConstant {
    #[inline]
    fn from(value: i32) -> Self {
        SpecializationConstant::I32(value)
    }
}

impl From<i64> for SpecializationConstant {
    #[inline]
    fn from(value: i64) -> Self {
        SpecializationConstant::I64(value)
    }
}

impl From<u8> for SpecializationConstant {
    #[inline]
    fn from(value: u8) -> Self {
        SpecializationConstant::U8(value)
    }
}

impl From<u16> for SpecializationConstant {
    #[inline]
    fn from(value: u16) -> Self {
        SpecializationConstant::U16(value)
    }
}

impl From<u32> for SpecializationConstant {
    #[inline]
    fn from(value: u32) -> Self {
        SpecializationConstant::U32(value)
    }
}

impl From<u64> for SpecializationConstant {
    #[inline]
    fn from(value: u64) -> Self {
        SpecializationConstant::U64(value)
    }
}

impl From<f16> for SpecializationConstant {
    #[inline]
    fn from(value: f16) -> Self {
        SpecializationConstant::F16(value)
    }
}

impl From<f32> for SpecializationConstant {
    #[inline]
    fn from(value: f32) -> Self {
        SpecializationConstant::F32(value)
    }
}

impl From<f64> for SpecializationConstant {
    #[inline]
    fn from(value: f64) -> Self {
        SpecializationConstant::F64(value)
    }
}

/// A shader module with specialization constants applied.
#[derive(Debug)]
pub struct SpecializedShaderModule {
    base_module: Arc<ShaderModule>,
    specialization_info: Vec<(u32, SpecializationConstant)>,
    spirv: Option<Spirv>,
    entry_point_infos: SmallVec<[(Id, EntryPointInfo); 1]>,
}

impl SpecializedShaderModule {
    /// Returns `base_module` specialized with `specialization_info`, panicking on a validation
    /// error.
    ///
    /// This is a shortcut for `try_new().unwrap()`.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_new`] returns a [`ValidationError`].
    ///
    /// [`try_new`]: Self::try_new
    #[inline]
    #[track_caller]
    pub fn new(
        base_module: &Arc<ShaderModule>,
        specialization_info: &[(u32, SpecializationConstant)],
    ) -> Arc<Self> {
        Self::try_new(base_module, specialization_info).unwrap()
    }

    /// Returns `base_module` specialized with `specialization_info`.
    #[inline]
    pub fn try_new(
        base_module: &Arc<ShaderModule>,
        specialization_info: &[(u32, SpecializationConstant)],
    ) -> Result<Arc<Self>, Box<ValidationError>> {
        Self::validate_new(base_module, specialization_info)?;

        Ok(unsafe { Self::new_unchecked(base_module, specialization_info) })
    }

    fn validate_new(
        base_module: &ShaderModule,
        specialization_info: &[(u32, SpecializationConstant)],
    ) -> Result<(), Box<ValidationError>> {
        for (index, &(constant_id, provided_value)) in specialization_info.iter().enumerate() {
            // Per `VkSpecializationMapEntry` spec:
            // "If a constantID value is not a specialization constant ID used in the shader,
            // that map entry does not affect the behavior of the pipeline."
            // We *may* want to be stricter than this for the sake of catching user errors?
            if let Some((_, default_value)) = base_module
                .specialization_constants
                .iter()
                .find(|(id, _)| *id == constant_id)
            {
                // Check for equal types rather than only equal size.
                if !provided_value.eq_type(default_value) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`specialization_info[{0}]` does not have the same type as \
                            `base_module.specialization_constants()[{0}]`",
                            constant_id
                        )
                        .into(),
                        vuids: &["VUID-VkSpecializationMapEntry-constantID-00776"],
                        ..Default::default()
                    }));
                }
            }

            for (other_index, &(other_constant_id, _)) in
                specialization_info.iter().enumerate().skip(index + 1)
            {
                if constant_id == other_constant_id {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`specialization_info[{}].0` and `specialization_info[{}].0` have the \
                            same value",
                            index, other_index,
                        )
                        .into(),
                        vuids: &["VUID-VkSpecializationInfo-constantID-04911"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        base_module: &Arc<ShaderModule>,
        specialization_info: &[(u32, SpecializationConstant)],
    ) -> Arc<Self> {
        let spirv = (!base_module.specialization_constants.is_empty()).then(|| {
            let mut spirv = base_module.spirv.clone();
            spirv.apply_specialization(specialization_info);
            spirv
        });
        let entry_point_infos =
            reflect::entry_points(spirv.as_ref().unwrap_or(&base_module.spirv)).collect();

        Arc::new(Self {
            base_module: base_module.clone(),
            specialization_info: specialization_info.to_owned(),
            spirv,
            entry_point_infos,
        })
    }

    /// Returns the base module, without specialization applied.
    #[inline]
    pub fn base_module(&self) -> &Arc<ShaderModule> {
        &self.base_module
    }

    /// Returns the specialization constants that have been applied to the module.
    #[inline]
    pub fn specialization_info(&self) -> &[(u32, SpecializationConstant)] {
        &self.specialization_info
    }

    /// Returns the SPIR-V code of this module.
    #[inline]
    pub(crate) fn spirv(&self) -> &Spirv {
        self.spirv.as_ref().unwrap_or(&self.base_module.spirv)
    }

    /// Returns information about the entry point with the provided name. Returns `None` if no
    /// entry point with that name exists in the shader module or if multiple entry points with
    /// the same name exist.
    #[inline]
    pub fn entry_point(self: &Arc<Self>, name: &str) -> Option<EntryPoint> {
        self.single_entry_point_filter(|info| info.name == name)
    }

    /// Returns information about the entry point with the provided name and execution model.
    /// Returns `None` if no entry and execution model exists in the shader module.
    #[inline]
    pub fn entry_point_with_execution(
        self: &Arc<Self>,
        name: &str,
        execution: ExecutionModel,
    ) -> Option<EntryPoint> {
        self.single_entry_point_filter(|info| {
            info.name == name && info.execution_model == execution
        })
    }

    /// checks for *exactly* one entry point matching the `filter`, otherwise returns `None`
    #[inline]
    fn single_entry_point_filter<P>(self: &Arc<Self>, mut filter: P) -> Option<EntryPoint>
    where
        P: FnMut(&EntryPointInfo) -> bool,
    {
        let mut iter = self
            .entry_point_infos
            .iter()
            .enumerate()
            .filter(|(_, (_, infos))| filter(infos))
            .map(|(x, _)| x);
        let info_index = iter.next()?;
        iter.next().is_none().then(|| EntryPoint {
            module: self.clone(),
            id: self.entry_point_infos[info_index].0,
            info_index,
        })
    }

    /// Returns information about the entry point if `self` only contains a single entry point,
    /// `None` otherwise.
    #[inline]
    pub fn single_entry_point(self: &Arc<Self>) -> Option<EntryPoint> {
        self.single_entry_point_filter(|_| true)
    }

    /// Returns information about the entry point if `self` only contains a single entry point
    /// with the provided `ExecutionModel`. Returns `None` if no entry point was found or multiple
    /// entry points have been found matching the provided `ExecutionModel`.
    #[inline]
    pub fn single_entry_point_with_execution(
        self: &Arc<Self>,
        execution: ExecutionModel,
    ) -> Option<EntryPoint> {
        self.single_entry_point_filter(|info| info.execution_model == execution)
    }
}

unsafe impl VulkanObject for SpecializedShaderModule {
    type Handle = vk::ShaderModule;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.base_module.handle
    }
}

unsafe impl DeviceOwned for SpecializedShaderModule {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.base_module.device
    }
}

/// The information associated with a single entry point in a shader.
#[derive(Clone, Debug)]
pub struct EntryPointInfo {
    pub name: String,
    pub execution_model: ExecutionModel,
    pub descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    pub push_constant_requirements: Option<PushConstantRange>,
}

/// Represents a shader entry point in a shader module.
///
/// Can be obtained by calling [`entry_point`](ShaderModule::entry_point) on the shader module.
#[derive(Clone, Debug)]
pub struct EntryPoint {
    module: Arc<SpecializedShaderModule>,
    id: Id,
    info_index: usize,
}

impl EntryPoint {
    /// Returns the module this entry point comes from.
    #[inline]
    pub fn module(&self) -> &Arc<SpecializedShaderModule> {
        &self.module
    }

    /// Returns the Id of the entry point function.
    pub(crate) fn id(&self) -> Id {
        self.id
    }

    /// Returns information about the entry point.
    #[inline]
    pub fn info(&self) -> &EntryPointInfo {
        &self.module.entry_point_infos[self.info_index].1
    }
}

/// The requirements imposed by a shader on a binding within a descriptor set layout, and on any
/// resource that is bound to that binding.
#[derive(Clone, Debug, Default)]
pub struct DescriptorBindingRequirements {
    /// The descriptor types that are allowed.
    pub descriptor_types: Vec<DescriptorType>,

    /// The number of descriptors (array elements) that the shader requires. The descriptor set
    /// layout can declare more than this, but never less.
    ///
    /// `None` means that the shader declares this as a runtime-sized array, and could potentially
    /// access every array element provided in the descriptor set.
    pub descriptor_count: Option<u32>,

    /// The image format that is required for image views bound to this binding. If this is
    /// `None`, then any image format is allowed.
    pub image_format: Option<Format>,

    /// Whether image views bound to this binding must have multisampling enabled or disabled.
    pub image_multisampled: bool,

    /// The base scalar type required for the format of image views bound to this binding.
    /// This is `None` for non-image bindings.
    pub image_scalar_type: Option<NumericType>,

    /// The view type that is required for image views bound to this binding.
    /// This is `None` for non-image bindings.
    pub image_view_type: Option<ImageViewType>,

    /// The shader stages that the binding must be declared for.
    pub stages: ShaderStages,

    /// The requirements for individual descriptors within a binding.
    ///
    /// Keys with `Some` hold requirements for a specific descriptor index, if it is statically
    /// known in the shader (a constant). The key `None` holds requirements for indices that are
    /// not statically known, but determined only at runtime (calculated from an input variable).
    pub descriptors: HashMap<Option<u32>, DescriptorRequirements>,
}

/// The requirements imposed by a shader on resources bound to a descriptor.
#[derive(Clone, Debug, Default)]
pub struct DescriptorRequirements {
    /// For buffers and images, which shader stages perform read operations.
    pub memory_read: ShaderStages,

    /// For buffers and images, which shader stages perform write operations.
    pub memory_write: ShaderStages,

    /// For sampler bindings, whether the shader performs depth comparison operations.
    pub sampler_compare: bool,

    /// For sampler bindings, whether the shader performs sampling operations that are not
    /// permitted with unnormalized coordinates. This includes sampling with `ImplicitLod`,
    /// `Dref` or `Proj` SPIR-V instructions or with an LOD bias or offset.
    pub sampler_no_unnormalized_coordinates: bool,

    /// For sampler bindings, whether the shader performs sampling operations that are not
    /// permitted with a sampler YCbCr conversion. This includes sampling with `Gather` SPIR-V
    /// instructions or with an offset.
    pub sampler_no_ycbcr_conversion: bool,

    /// For sampler bindings, the sampled image descriptors that are used in combination with this
    /// sampler.
    pub sampler_with_images: HashSet<DescriptorIdentifier>,

    /// For storage image bindings, whether the shader performs atomic operations.
    pub storage_image_atomic: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DescriptorIdentifier {
    pub set: u32,
    pub binding: u32,
    pub index: u32,
}

impl DescriptorBindingRequirements {
    /// Merges `other` into `self`, so that `self` satisfies the requirements of both.
    /// An error is returned if the requirements conflict.
    #[inline]
    pub fn merge(&mut self, other: &Self) -> Result<(), Box<ValidationError>> {
        let Self {
            descriptor_types,
            descriptor_count,
            image_format,
            image_multisampled: _,
            image_scalar_type,
            image_view_type,
            stages,
            descriptors,
        } = self;

        /* Checks */

        if !descriptor_types
            .iter()
            .any(|ty| other.descriptor_types.contains(ty))
        {
            return Err(Box::new(ValidationError {
                problem: "the allowed descriptor types of the two descriptors do not overlap"
                    .into(),
                ..Default::default()
            }));
        }

        /* Merge */

        descriptor_types.retain(|ty| other.descriptor_types.contains(ty));

        *descriptor_count = (*descriptor_count).max(other.descriptor_count);
        *image_format = image_format.or(other.image_format);
        *image_scalar_type = image_scalar_type.or(other.image_scalar_type);
        *image_view_type = image_view_type.or(other.image_view_type);
        *stages |= other.stages;

        for (&index, other) in &other.descriptors {
            match descriptors.entry(index) {
                Entry::Vacant(entry) => {
                    entry.insert(other.clone());
                }
                Entry::Occupied(entry) => {
                    entry.into_mut().merge(other);
                }
            }
        }

        Ok(())
    }
}

impl DescriptorRequirements {
    /// Merges `other` into `self`, so that `self` satisfies the requirements of both.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        let Self {
            memory_read,
            memory_write,
            sampler_compare,
            sampler_no_unnormalized_coordinates,
            sampler_no_ycbcr_conversion,
            sampler_with_images,
            storage_image_atomic,
        } = self;

        *memory_read |= other.memory_read;
        *memory_write |= other.memory_write;
        *sampler_compare |= other.sampler_compare;
        *sampler_no_unnormalized_coordinates |= other.sampler_no_unnormalized_coordinates;
        *sampler_no_ycbcr_conversion |= other.sampler_no_ycbcr_conversion;
        sampler_with_images.extend(&other.sampler_with_images);
        *storage_image_atomic |= other.storage_image_atomic;
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`ShaderStage`] values.
    ShaderStages impl {
        /// Creates a `ShaderStages` struct with all graphics stages set to `true`.
        #[inline]
        pub const fn all_graphics() -> ShaderStages {
            ShaderStages::VERTEX
                .union(ShaderStages::TESSELLATION_CONTROL)
                .union(ShaderStages::TESSELLATION_EVALUATION)
                .union(ShaderStages::GEOMETRY)
                .union(ShaderStages::FRAGMENT)
        }
    },

    /// A shader stage within a pipeline.
    ShaderStage,

    = ShaderStageFlags(u32);

    // TODO: document
    VERTEX, Vertex = VERTEX,

    // TODO: document
    TESSELLATION_CONTROL, TessellationControl = TESSELLATION_CONTROL,

    // TODO: document
    TESSELLATION_EVALUATION, TessellationEvaluation = TESSELLATION_EVALUATION,

    // TODO: document
    GEOMETRY, Geometry = GEOMETRY,

    // TODO: document
    FRAGMENT, Fragment = FRAGMENT,

    // TODO: document
    COMPUTE, Compute = COMPUTE,

    // TODO: document
    RAYGEN, Raygen = RAYGEN_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    ANY_HIT, AnyHit = ANY_HIT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    CLOSEST_HIT, ClosestHit = CLOSEST_HIT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    MISS, Miss = MISS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    INTERSECTION, Intersection = INTERSECTION_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    CALLABLE, Callable = CALLABLE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    // TODO: document
    TASK, Task = TASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    // TODO: document
    MESH, Mesh = MESH_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    // TODO: document
    SUBPASS_SHADING, SubpassShading = SUBPASS_SHADING_HUAWEI
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(huawei_subpass_shading)]),
    ]),
}

impl From<ExecutionModel> for ShaderStage {
    #[inline]
    fn from(value: ExecutionModel) -> Self {
        match value {
            ExecutionModel::Vertex => ShaderStage::Vertex,
            ExecutionModel::TessellationControl => ShaderStage::TessellationControl,
            ExecutionModel::TessellationEvaluation => ShaderStage::TessellationEvaluation,
            ExecutionModel::Geometry => ShaderStage::Geometry,
            ExecutionModel::Fragment => ShaderStage::Fragment,
            ExecutionModel::GLCompute => ShaderStage::Compute,
            ExecutionModel::Kernel => {
                unimplemented!("the `Kernel` execution model is not supported by Vulkan")
            }
            ExecutionModel::TaskNV | ExecutionModel::TaskEXT => ShaderStage::Task,
            ExecutionModel::MeshNV | ExecutionModel::MeshEXT => ShaderStage::Mesh,
            ExecutionModel::RayGenerationKHR => ShaderStage::Raygen,
            ExecutionModel::IntersectionKHR => ShaderStage::Intersection,
            ExecutionModel::AnyHitKHR => ShaderStage::AnyHit,
            ExecutionModel::ClosestHitKHR => ShaderStage::ClosestHit,
            ExecutionModel::MissKHR => ShaderStage::Miss,
            ExecutionModel::CallableKHR => ShaderStage::Callable,
        }
    }
}

impl From<ShaderStages> for PipelineStages {
    #[inline]
    fn from(stages: ShaderStages) -> PipelineStages {
        let mut result = PipelineStages::empty();

        if stages.intersects(ShaderStages::VERTEX) {
            result |= PipelineStages::VERTEX_SHADER
        }

        if stages.intersects(ShaderStages::TESSELLATION_CONTROL) {
            result |= PipelineStages::TESSELLATION_CONTROL_SHADER
        }

        if stages.intersects(ShaderStages::TESSELLATION_EVALUATION) {
            result |= PipelineStages::TESSELLATION_EVALUATION_SHADER
        }

        if stages.intersects(ShaderStages::GEOMETRY) {
            result |= PipelineStages::GEOMETRY_SHADER
        }

        if stages.intersects(ShaderStages::FRAGMENT) {
            result |= PipelineStages::FRAGMENT_SHADER
        }

        if stages.intersects(ShaderStages::COMPUTE) {
            result |= PipelineStages::COMPUTE_SHADER
        }

        if stages.intersects(
            ShaderStages::RAYGEN
                | ShaderStages::ANY_HIT
                | ShaderStages::CLOSEST_HIT
                | ShaderStages::MISS
                | ShaderStages::INTERSECTION
                | ShaderStages::CALLABLE,
        ) {
            result |= PipelineStages::RAY_TRACING_SHADER
        }

        if stages.intersects(ShaderStages::TASK) {
            result |= PipelineStages::TASK_SHADER;
        }

        if stages.intersects(ShaderStages::MESH) {
            result |= PipelineStages::MESH_SHADER;
        }

        if stages.intersects(ShaderStages::SUBPASS_SHADING) {
            result |= PipelineStages::SUBPASS_SHADER;
        }

        result
    }
}
