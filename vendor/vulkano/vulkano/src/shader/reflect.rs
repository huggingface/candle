//! Extraction of information from SPIR-V modules, that is needed by the rest of Vulkano.

use super::DescriptorBindingRequirements;
use crate::{
    descriptor_set::layout::DescriptorType,
    image::view::ImageViewType,
    pipeline::layout::PushConstantRange,
    shader::{
        spirv::{Decoration, Dim, Id, Instruction, Spirv, StorageClass},
        DescriptorIdentifier, DescriptorRequirements, EntryPointInfo, NumericType, ShaderStage,
        ShaderStages, SpecializationConstant,
    },
    DeviceSize, Version,
};
use foldhash::{HashMap, HashSet};
use half::f16;
use smallvec::{smallvec, SmallVec};

/// Returns an iterator over all entry points in `spirv`, with information about the entry point.
#[inline]
pub fn entry_points(spirv: &Spirv) -> impl Iterator<Item = (Id, EntryPointInfo)> + use<'_> {
    let interface_variables = interface_variables(spirv);

    spirv.entry_points().iter().filter_map(move |instruction| {
        let &Instruction::EntryPoint {
            execution_model,
            entry_point,
            ref name,
            ..
        } = instruction
        else {
            return None;
        };

        let stage = ShaderStage::from(execution_model);

        let descriptor_binding_requirements = inspect_entry_point(
            &interface_variables.descriptor_binding,
            spirv,
            stage,
            entry_point,
        );
        let push_constant_requirements = push_constant_requirements(
            &interface_variables.push_constant,
            spirv,
            stage,
            entry_point,
        );

        Some((
            entry_point,
            EntryPointInfo {
                name: name.clone(),
                execution_model,
                descriptor_binding_requirements,
                push_constant_requirements,
            },
        ))
    })
}

#[derive(Clone, Debug, Default)]
struct InterfaceVariables {
    descriptor_binding: HashMap<Id, DescriptorBindingVariable>,
    push_constant: HashMap<Id, PushConstantRange>,
}

// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface.
#[derive(Clone, Debug)]
struct DescriptorBindingVariable {
    set: u32,
    binding: u32,
    reqs: DescriptorBindingRequirements,
}

fn interface_variables(spirv: &Spirv) -> InterfaceVariables {
    let mut variables = InterfaceVariables::default();

    for instruction in spirv.global_variables() {
        if let Instruction::Variable {
            result_id,
            result_type_id,
            storage_class,
            ..
        } = instruction
        {
            match storage_class {
                StorageClass::StorageBuffer
                | StorageClass::Uniform
                | StorageClass::UniformConstant => {
                    variables.descriptor_binding.insert(
                        *result_id,
                        descriptor_binding_requirements_of(spirv, *result_id),
                    );
                }
                StorageClass::PushConstant => {
                    variables.push_constant.insert(
                        *result_id,
                        push_constant_requirements_of(spirv, *result_type_id),
                    );
                }
                _ => (),
            }
        }
    }

    variables
}

fn inspect_entry_point(
    global: &HashMap<Id, DescriptorBindingVariable>,
    spirv: &Spirv,
    stage: ShaderStage,
    entry_point: Id,
) -> HashMap<(u32, u32), DescriptorBindingRequirements> {
    struct Context<'a> {
        global: &'a HashMap<Id, DescriptorBindingVariable>,
        spirv: &'a Spirv,
        stage: ShaderStage,
        inspected_functions: HashSet<Id>,
        result: HashMap<Id, DescriptorBindingVariable>,
    }

    impl Context<'_> {
        fn instruction_chain<const N: usize>(
            &mut self,
            chain: [fn(&Spirv, Id) -> Option<Id>; N],
            id: Id,
        ) -> Option<(&mut DescriptorBindingVariable, Option<u64>)> {
            let mut id = chain
                .into_iter()
                .try_fold(id, |id, func| func(self.spirv, id))?;

            if let Some(variable) = self.global.get(&id) {
                // Variable was accessed without an access chain, return with index 0.
                let variable = self.result.entry(id).or_insert_with(|| variable.clone());
                variable.reqs.stages = self.stage.into();
                return Some((variable, Some(0)));
            }

            while let Instruction::AccessChain {
                base, ref indexes, ..
            }
            | Instruction::InBoundsAccessChain {
                base, ref indexes, ..
            } = *self.spirv.id(id).instruction()
            {
                id = base;

                if let Some(variable) = self.global.get(&id) {
                    // Variable was accessed with an access chain.
                    // Retrieve index from instruction if it's a constant value.
                    // TODO: handle a `None` index too?
                    let index = get_constant(self.spirv, *indexes.first().unwrap());
                    let variable = self.result.entry(id).or_insert_with(|| variable.clone());
                    variable.reqs.stages = self.stage.into();
                    return Some((variable, index));
                }
            }

            None
        }

        fn inspect_entry_point_r(&mut self, function: Id) {
            fn desc_reqs(
                descriptor_variable: Option<(&mut DescriptorBindingVariable, Option<u64>)>,
            ) -> Option<&mut DescriptorRequirements> {
                descriptor_variable.map(|(variable, index)| {
                    variable
                        .reqs
                        .descriptors
                        .entry(index.map(|index| index.try_into().unwrap()))
                        .or_default()
                })
            }

            fn inst_image_texel_pointer(spirv: &Spirv, id: Id) -> Option<Id> {
                match *spirv.id(id).instruction() {
                    Instruction::ImageTexelPointer { image, .. } => Some(image),
                    _ => None,
                }
            }

            fn inst_load(spirv: &Spirv, id: Id) -> Option<Id> {
                match *spirv.id(id).instruction() {
                    Instruction::Load { pointer, .. } => Some(pointer),
                    _ => None,
                }
            }

            fn inst_sampled_image(spirv: &Spirv, id: Id) -> Option<Id> {
                match *spirv.id(id).instruction() {
                    Instruction::SampledImage { sampler, .. } => Some(sampler),
                    _ => Some(id),
                }
            }

            self.inspected_functions.insert(function);

            for instruction in self.spirv.function(function).instructions() {
                let stage = self.stage;

                match *instruction {
                    Instruction::AtomicLoad { pointer, .. } => {
                        // Storage buffer
                        if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer)) {
                            desc_reqs.memory_read = stage.into();
                        }

                        // Storage image
                        if let Some(desc_reqs) =
                            desc_reqs(self.instruction_chain([inst_image_texel_pointer], pointer))
                        {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.storage_image_atomic = true;
                        }
                    }

                    Instruction::AtomicStore { pointer, .. } => {
                        // Storage buffer
                        if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer)) {
                            desc_reqs.memory_write = stage.into();
                        }

                        // Storage image
                        if let Some(desc_reqs) =
                            desc_reqs(self.instruction_chain([inst_image_texel_pointer], pointer))
                        {
                            desc_reqs.memory_write = stage.into();
                            desc_reqs.storage_image_atomic = true;
                        }
                    }

                    Instruction::AtomicExchange { pointer, .. }
                    | Instruction::AtomicCompareExchange { pointer, .. }
                    | Instruction::AtomicCompareExchangeWeak { pointer, .. }
                    | Instruction::AtomicIIncrement { pointer, .. }
                    | Instruction::AtomicIDecrement { pointer, .. }
                    | Instruction::AtomicIAdd { pointer, .. }
                    | Instruction::AtomicISub { pointer, .. }
                    | Instruction::AtomicSMin { pointer, .. }
                    | Instruction::AtomicUMin { pointer, .. }
                    | Instruction::AtomicSMax { pointer, .. }
                    | Instruction::AtomicUMax { pointer, .. }
                    | Instruction::AtomicAnd { pointer, .. }
                    | Instruction::AtomicOr { pointer, .. }
                    | Instruction::AtomicXor { pointer, .. }
                    | Instruction::AtomicFlagTestAndSet { pointer, .. }
                    | Instruction::AtomicFlagClear { pointer, .. }
                    | Instruction::AtomicFMinEXT { pointer, .. }
                    | Instruction::AtomicFMaxEXT { pointer, .. }
                    | Instruction::AtomicFAddEXT { pointer, .. } => {
                        // Storage buffer
                        if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer)) {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.memory_write = stage.into();
                        }

                        // Storage image
                        if let Some(desc_reqs) =
                            desc_reqs(self.instruction_chain([inst_image_texel_pointer], pointer))
                        {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.memory_write = stage.into();
                            desc_reqs.storage_image_atomic = true;
                        }
                    }

                    Instruction::CopyMemory { target, source, .. } => {
                        self.instruction_chain([], target);
                        self.instruction_chain([], source);
                    }

                    Instruction::CopyObject { operand, .. } => {
                        self.instruction_chain([], operand);
                    }

                    Instruction::ExtInst { ref operands, .. } => {
                        // We don't know which extended instructions take pointers,
                        // so we must interpret every operand as a pointer.
                        for &operand in operands {
                            self.instruction_chain([], operand);
                        }
                    }

                    Instruction::FunctionCall {
                        function,
                        ref arguments,
                        ..
                    } => {
                        // Rather than trying to figure out the type of each argument, we just
                        // try all of them as pointers.
                        for &argument in arguments {
                            self.instruction_chain([], argument);
                        }

                        if !self.inspected_functions.contains(&function) {
                            self.inspect_entry_point_r(function);
                        }
                    }

                    Instruction::ImageGather {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseGather {
                        sampled_image,
                        image_operands,
                        ..
                    } => {
                        if let Some(desc_reqs) = desc_reqs(
                            self.instruction_chain([inst_sampled_image, inst_load], sampled_image),
                        ) {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.sampler_no_ycbcr_conversion = true;

                            if image_operands.as_ref().is_some_and(|image_operands| {
                                image_operands.bias.is_some()
                                    || image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                            }) {
                                desc_reqs.sampler_no_unnormalized_coordinates = true;
                            }
                        }
                    }

                    Instruction::ImageDrefGather { sampled_image, .. }
                    | Instruction::ImageSparseDrefGather { sampled_image, .. } => {
                        if let Some(desc_reqs) = desc_reqs(
                            self.instruction_chain([inst_sampled_image, inst_load], sampled_image),
                        ) {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.sampler_no_unnormalized_coordinates = true;
                            desc_reqs.sampler_no_ycbcr_conversion = true;
                        }
                    }

                    Instruction::ImageSampleImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSampleProjImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleProjImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    } => {
                        if let Some(desc_reqs) = desc_reqs(
                            self.instruction_chain([inst_sampled_image, inst_load], sampled_image),
                        ) {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.sampler_no_unnormalized_coordinates = true;

                            if image_operands.as_ref().is_some_and(|image_operands| {
                                image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                            }) {
                                desc_reqs.sampler_no_ycbcr_conversion = true;
                            }
                        }
                    }

                    Instruction::ImageSampleProjExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleProjExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    } => {
                        if let Some(desc_reqs) = desc_reqs(
                            self.instruction_chain([inst_sampled_image, inst_load], sampled_image),
                        ) {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.sampler_no_unnormalized_coordinates = true;

                            if image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                desc_reqs.sampler_no_ycbcr_conversion = true;
                            }
                        }
                    }

                    Instruction::ImageSampleDrefImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSampleProjDrefImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleDrefImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleProjDrefImplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    } => {
                        if let Some(desc_reqs) = desc_reqs(
                            self.instruction_chain([inst_sampled_image, inst_load], sampled_image),
                        ) {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.sampler_no_unnormalized_coordinates = true;
                            desc_reqs.sampler_compare = true;

                            if image_operands.as_ref().is_some_and(|image_operands| {
                                image_operands.const_offset.is_some()
                                    || image_operands.offset.is_some()
                            }) {
                                desc_reqs.sampler_no_ycbcr_conversion = true;
                            }
                        }
                    }

                    Instruction::ImageSampleDrefExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSampleProjDrefExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleDrefExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleProjDrefExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    } => {
                        if let Some(desc_reqs) = desc_reqs(
                            self.instruction_chain([inst_sampled_image, inst_load], sampled_image),
                        ) {
                            desc_reqs.memory_read = stage.into();
                            desc_reqs.sampler_no_unnormalized_coordinates = true;
                            desc_reqs.sampler_compare = true;

                            if image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                desc_reqs.sampler_no_ycbcr_conversion = true;
                            }
                        }
                    }

                    Instruction::ImageSampleExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    }
                    | Instruction::ImageSparseSampleExplicitLod {
                        sampled_image,
                        image_operands,
                        ..
                    } => {
                        if let Some(desc_reqs) = desc_reqs(
                            self.instruction_chain([inst_sampled_image, inst_load], sampled_image),
                        ) {
                            desc_reqs.memory_read = stage.into();

                            if image_operands.bias.is_some()
                                || image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                desc_reqs.sampler_no_unnormalized_coordinates = true;
                            }

                            if image_operands.const_offset.is_some()
                                || image_operands.offset.is_some()
                            {
                                desc_reqs.sampler_no_ycbcr_conversion = true;
                            }
                        }
                    }

                    Instruction::ImageTexelPointer { image, .. } => {
                        self.instruction_chain([], image);
                    }

                    Instruction::ImageRead { image, .. } => {
                        if let Some(desc_reqs) =
                            desc_reqs(self.instruction_chain([inst_load], image))
                        {
                            desc_reqs.memory_read = stage.into();
                        }
                    }

                    Instruction::ImageWrite { image, .. } => {
                        if let Some(desc_reqs) =
                            desc_reqs(self.instruction_chain([inst_load], image))
                        {
                            desc_reqs.memory_write = stage.into();
                        }
                    }

                    Instruction::Load { pointer, .. } => {
                        if let Some((binding_variable, index)) = self.instruction_chain([], pointer)
                        {
                            // Only loads on buffers access memory directly.
                            // Loads on images load the image object itself, but don't touch
                            // the texels in memory yet.
                            if binding_variable.reqs.descriptor_types.iter().any(|ty| {
                                matches!(
                                    ty,
                                    DescriptorType::UniformBuffer
                                        | DescriptorType::UniformBufferDynamic
                                        | DescriptorType::StorageBuffer
                                        | DescriptorType::StorageBufferDynamic
                                        | DescriptorType::InlineUniformBlock
                                )
                            }) {
                                if let Some(desc_reqs) = desc_reqs(Some((binding_variable, index)))
                                {
                                    desc_reqs.memory_read = stage.into();
                                }
                            }
                        }
                    }

                    Instruction::SampledImage { image, sampler, .. } => {
                        let identifier = match self.instruction_chain([inst_load], image) {
                            Some((variable, Some(index))) => DescriptorIdentifier {
                                set: variable.set,
                                binding: variable.binding,
                                index: index.try_into().unwrap(),
                            },
                            _ => continue,
                        };

                        if let Some(desc_reqs) =
                            desc_reqs(self.instruction_chain([inst_load], sampler))
                        {
                            desc_reqs.sampler_with_images.insert(identifier);
                        }
                    }

                    Instruction::Store { pointer, .. } => {
                        // This can only apply to buffers, right?
                        if let Some(desc_reqs) = desc_reqs(self.instruction_chain([], pointer)) {
                            desc_reqs.memory_write = stage.into();
                        }
                    }

                    _ => (),
                }
            }
        }
    }

    let mut context = Context {
        global,
        spirv,
        stage,
        inspected_functions: HashSet::default(),
        result: HashMap::default(),
    };
    context.inspect_entry_point_r(entry_point);

    context
        .result
        .into_values()
        .map(|variable| ((variable.set, variable.binding), variable.reqs))
        .collect()
}

/// Returns a `DescriptorBindingRequirements` value for the pointed type.
///
/// See also section 14.5.2 of the Vulkan specs: Descriptor Set Interface
fn descriptor_binding_requirements_of(spirv: &Spirv, variable_id: Id) -> DescriptorBindingVariable {
    let variable_id_info = spirv.id(variable_id);

    let mut reqs = DescriptorBindingRequirements {
        descriptor_count: Some(1),
        ..Default::default()
    };

    let (mut next_type_id, is_storage_buffer) = {
        let variable_type_id = match *variable_id_info.instruction() {
            Instruction::Variable { result_type_id, .. } => result_type_id,
            _ => panic!("Id {} is not a variable", variable_id),
        };

        match *spirv.id(variable_type_id).instruction() {
            Instruction::TypePointer {
                ty, storage_class, ..
            } => (Some(ty), storage_class == StorageClass::StorageBuffer),
            _ => panic!(
                "Variable {} result_type_id does not refer to a TypePointer instruction",
                variable_id
            ),
        }
    };

    while let Some(id) = next_type_id {
        let id_info = spirv.id(id);

        next_type_id = match *id_info.instruction() {
            Instruction::TypeStruct { .. } => {
                let decoration_block = id_info.decorations().iter().any(|instruction| {
                    matches!(
                        instruction,
                        Instruction::Decorate {
                            decoration: Decoration::Block,
                            ..
                        }
                    )
                });

                let decoration_buffer_block = id_info.decorations().iter().any(|instruction| {
                    matches!(
                        instruction,
                        Instruction::Decorate {
                            decoration: Decoration::BufferBlock,
                            ..
                        }
                    )
                });

                assert!(
                    decoration_block ^ decoration_buffer_block,
                    "Structs in shader interface are expected to be decorated with one of Block or \
                    BufferBlock",
                );

                if decoration_buffer_block || decoration_block && is_storage_buffer {
                    reqs.descriptor_types = vec![
                        DescriptorType::StorageBuffer,
                        DescriptorType::StorageBufferDynamic,
                    ];
                } else {
                    reqs.descriptor_types = vec![
                        DescriptorType::UniformBuffer,
                        DescriptorType::UniformBufferDynamic,
                        DescriptorType::InlineUniformBlock,
                    ];
                };

                None
            }

            Instruction::TypeImage {
                sampled_type,
                dim,
                arrayed,
                ms,
                sampled,
                image_format,
                ..
            } => {
                assert_ne!(
                    sampled, 0,
                    "Vulkan requires that variables of type OpTypeImage have a Sampled operand of \
                    1 or 2",
                );
                reqs.image_format = image_format.into();
                reqs.image_multisampled = ms != 0;
                reqs.image_scalar_type = Some(match *spirv.id(sampled_type).instruction() {
                    Instruction::TypeInt {
                        width, signedness, ..
                    } => {
                        assert_eq!(width, 32); // TODO: 64-bit components
                        match signedness {
                            0 => NumericType::Uint,
                            1 => NumericType::Int,
                            _ => unreachable!(),
                        }
                    }
                    Instruction::TypeFloat { width, .. } => {
                        assert_eq!(width, 32); // TODO: 64-bit components
                        NumericType::Float
                    }
                    _ => unreachable!(),
                });

                match dim {
                    Dim::SubpassData => {
                        assert!(
                            reqs.image_format.is_none(),
                            "If Dim is SubpassData, Image Format must be Unknown",
                        );
                        assert_eq!(sampled, 2, "If Dim is SubpassData, Sampled must be 2");
                        assert_eq!(arrayed, 0, "If Dim is SubpassData, Arrayed must be 0");

                        reqs.descriptor_types = vec![DescriptorType::InputAttachment];
                    }
                    Dim::Buffer => {
                        if sampled == 1 {
                            reqs.descriptor_types = vec![DescriptorType::UniformTexelBuffer];
                        } else {
                            reqs.descriptor_types = vec![DescriptorType::StorageTexelBuffer];
                        }
                    }
                    _ => {
                        reqs.image_view_type = Some(match (dim, arrayed) {
                            (Dim::Dim1D, 0) => ImageViewType::Dim1d,
                            (Dim::Dim1D, 1) => ImageViewType::Dim1dArray,
                            (Dim::Dim2D, 0) => ImageViewType::Dim2d,
                            (Dim::Dim2D, 1) => ImageViewType::Dim2dArray,
                            (Dim::Dim3D, 0) => ImageViewType::Dim3d,
                            (Dim::Dim3D, 1) => {
                                panic!("Vulkan doesn't support arrayed 3D textures")
                            }
                            (Dim::Cube, 0) => ImageViewType::Cube,
                            (Dim::Cube, 1) => ImageViewType::CubeArray,
                            (Dim::Rect, _) => {
                                panic!("Vulkan doesn't support rectangle textures")
                            }
                            _ => unreachable!(),
                        });

                        if reqs.descriptor_types.is_empty() {
                            if sampled == 1 {
                                reqs.descriptor_types = vec![DescriptorType::SampledImage];
                            } else {
                                reqs.descriptor_types = vec![DescriptorType::StorageImage];
                            }
                        }
                    }
                }

                None
            }

            Instruction::TypeSampler { .. } => {
                reqs.descriptor_types = vec![DescriptorType::Sampler];

                None
            }

            Instruction::TypeSampledImage { image_type, .. } => {
                reqs.descriptor_types = vec![DescriptorType::CombinedImageSampler];

                Some(image_type)
            }

            Instruction::TypeAccelerationStructureKHR { .. } => {
                reqs.descriptor_types = vec![DescriptorType::AccelerationStructure];

                None
            }

            Instruction::TypeArray {
                element_type,
                length,
                ..
            } => {
                // Inline uniform blocks can't be arrayed.
                reqs.descriptor_types
                    .retain(|&d| d != DescriptorType::InlineUniformBlock);

                let len = get_constant(spirv, length).expect("failed to find array length");

                if let Some(count) = reqs.descriptor_count.as_mut() {
                    *count *= len as u32
                }

                Some(element_type)
            }

            Instruction::TypeRuntimeArray { element_type, .. } => {
                // Inline uniform blocks can't be arrayed.
                reqs.descriptor_types
                    .retain(|&d| d != DescriptorType::InlineUniformBlock);

                reqs.descriptor_count = None;

                Some(element_type)
            }

            _ => {
                let name = variable_id_info
                    .names()
                    .iter()
                    .find_map(|instruction| match *instruction {
                        Instruction::Name { ref name, .. } => Some(name.as_str()),
                        _ => None,
                    })
                    .unwrap_or("__unnamed");

                panic!(
                    "Couldn't find relevant type for global variable `{}` (id {}, maybe \
                    unimplemented)",
                    name, variable_id,
                );
            }
        };
    }

    DescriptorBindingVariable {
        set: variable_id_info
            .decorations()
            .iter()
            .find_map(|instruction| match *instruction {
                Instruction::Decorate {
                    decoration: Decoration::DescriptorSet { descriptor_set },
                    ..
                } => Some(descriptor_set),
                _ => None,
            })
            .unwrap(),
        binding: variable_id_info
            .decorations()
            .iter()
            .find_map(|instruction| match *instruction {
                Instruction::Decorate {
                    decoration: Decoration::Binding { binding_point },
                    ..
                } => Some(binding_point),
                _ => None,
            })
            .unwrap(),
        reqs,
    }
}

fn push_constant_requirements_of(spirv: &Spirv, pointer_type_id: Id) -> PushConstantRange {
    let struct_type_id = match *spirv.id(pointer_type_id).instruction() {
        Instruction::TypePointer {
            ty,
            storage_class: StorageClass::PushConstant,
            ..
        } => ty,
        _ => unreachable!(),
    };

    assert!(
        matches!(
            spirv.id(struct_type_id).instruction(),
            Instruction::TypeStruct { .. }
        ),
        "VUID-StandaloneSpirv-PushConstant-06808"
    );
    let start = offset_of_struct(spirv, struct_type_id);
    let end =
        size_of_type(spirv, struct_type_id).expect("Found runtime-sized push constants") as u32;

    PushConstantRange {
        stages: ShaderStages::default(),
        offset: start,
        size: end - start,
    }
}

/// Extracts the `PushConstantRange` from `spirv`.
fn push_constant_requirements(
    global: &HashMap<Id, PushConstantRange>,
    spirv: &Spirv,
    stage: ShaderStage,
    function_id: Id,
) -> Option<PushConstantRange> {
    fn find_variables_used(
        function_id: Id,
        global: &HashMap<Id, PushConstantRange>,
        spirv: &Spirv,
        visited_fns: &mut HashSet<Id>,
        variables: &mut HashSet<Id>,
    ) {
        visited_fns.insert(function_id);
        let function_info = spirv.function(function_id);
        for instruction in function_info.instructions() {
            match instruction {
                Instruction::FunctionCall {
                    function,
                    arguments,
                    ..
                } => {
                    for arg in arguments {
                        if global.contains_key(arg) {
                            variables.insert(*arg);
                        }
                    }
                    if !visited_fns.contains(function) {
                        find_variables_used(*function, global, spirv, visited_fns, variables);
                    }
                }
                Instruction::AccessChain {
                    base: variable_id, ..
                }
                | Instruction::InBoundsAccessChain {
                    base: variable_id, ..
                }
                | Instruction::PtrAccessChain {
                    base: variable_id, ..
                }
                | Instruction::InBoundsPtrAccessChain {
                    base: variable_id, ..
                }
                | Instruction::Load {
                    pointer: variable_id,
                    ..
                }
                | Instruction::CopyMemory {
                    source: variable_id,
                    ..
                }
                | Instruction::CopyObject {
                    result_id: variable_id,
                    ..
                } => {
                    if global.contains_key(variable_id) {
                        variables.insert(*variable_id);
                    }
                }
                _ => (),
            }
        }
    }

    if global.is_empty() {
        return None;
    }
    let mut variables = HashSet::default();
    if spirv.version() < Version::V1_4 {
        let mut visited_fns = HashSet::default();
        find_variables_used(function_id, global, spirv, &mut visited_fns, &mut variables);
    } else if let Instruction::EntryPoint { interface, .. } =
        spirv.function(function_id).entry_point().unwrap()
    {
        for id in interface {
            if global.contains_key(id) {
                variables.insert(*id);
            }
        }
    } else {
        unreachable!();
    }
    assert!(
        variables.len() <= 1,
        "VUID-StandaloneSpirv-OpEntryPoint-06674"
    );
    let variable_id = variables.into_iter().next()?;
    let mut push_constant_range = global.get(&variable_id).copied().unwrap();
    push_constant_range.stages = stage.into();
    Some(push_constant_range)
}

/// Extracts the `SpecializationConstant` map from `spirv`.
pub(super) fn specialization_constants(spirv: &Spirv) -> Vec<(u32, SpecializationConstant)> {
    let get_constant_id = |result_id| {
        spirv
            .id(result_id)
            .decorations()
            .iter()
            .find_map(|instruction| match *instruction {
                Instruction::Decorate {
                    decoration:
                        Decoration::SpecId {
                            specialization_constant_id,
                        },
                    ..
                } => Some(specialization_constant_id),
                _ => None,
            })
    };

    spirv
        .constants()
        .iter()
        .filter_map(|instruction| match *instruction {
            Instruction::SpecConstantFalse { result_id, .. } => get_constant_id(result_id)
                .map(|constant_id| (constant_id, SpecializationConstant::Bool(false))),
            Instruction::SpecConstantTrue { result_id, .. } => get_constant_id(result_id)
                .map(|constant_id| (constant_id, SpecializationConstant::Bool(true))),
            Instruction::SpecConstant {
                result_type_id,
                result_id,
                ref value,
            } => get_constant_id(result_id).map(|constant_id| {
                let value = match *spirv.id(result_type_id).instruction() {
                    Instruction::TypeInt {
                        width, signedness, ..
                    } => {
                        if width == 64 {
                            assert_eq!(value.len(), 2);
                        } else {
                            assert_eq!(value.len(), 1);
                        }

                        match (signedness, width) {
                            (0, 8) => SpecializationConstant::U8(value[0] as u8),
                            (0, 16) => SpecializationConstant::U16(value[0] as u16),
                            (0, 32) => SpecializationConstant::U32(value[0]),
                            (0, 64) => SpecializationConstant::U64(
                                (value[0] as u64) | ((value[1] as u64) << 32),
                            ),
                            (1, 8) => SpecializationConstant::I8(value[0] as i8),
                            (1, 16) => SpecializationConstant::I16(value[0] as i16),
                            (1, 32) => SpecializationConstant::I32(value[0] as i32),
                            (1, 64) => SpecializationConstant::I64(
                                (value[0] as i64) | ((value[1] as i64) << 32),
                            ),
                            _ => unimplemented!(),
                        }
                    }
                    Instruction::TypeFloat { width, .. } => {
                        if width == 64 {
                            assert_eq!(value.len(), 2);
                        } else {
                            assert_eq!(value.len(), 1);
                        }

                        match width {
                            16 => SpecializationConstant::F16(f16::from_bits(value[0] as u16)),
                            32 => SpecializationConstant::F32(f32::from_bits(value[0])),
                            64 => SpecializationConstant::F64(f64::from_bits(
                                (value[0] as u64) | ((value[1] as u64) << 32),
                            )),
                            _ => unimplemented!(),
                        }
                    }
                    _ => panic!(
                        "Specialization constant {} has a non-scalar type",
                        constant_id
                    ),
                };

                (constant_id, value)
            }),
            _ => None,
        })
        .collect()
}

/// Returns the size of a type, or `None` if its size cannot be determined.
pub(crate) fn size_of_type(spirv: &Spirv, id: Id) -> Option<DeviceSize> {
    let id_info = spirv.id(id);

    match *id_info.instruction() {
        Instruction::TypeVoid { .. } => Some(0),
        Instruction::TypeBool { .. } => Some(4),
        Instruction::TypeInt { width, .. } | Instruction::TypeFloat { width, .. } => {
            assert_eq!(width % 8, 0);
            Some(width as DeviceSize / 8)
        }
        Instruction::TypePointer {
            storage_class, ty, ..
        } => match storage_class {
            StorageClass::PhysicalStorageBuffer => Some(8),
            _ => size_of_type(spirv, ty),
        },
        Instruction::TypeVector {
            component_type,
            component_count,
            ..
        } => size_of_type(spirv, component_type)
            .map(|component_size| component_size * component_count as DeviceSize),
        Instruction::TypeMatrix {
            column_type,
            column_count,
            ..
        } => {
            // FIXME: row-major or column-major
            // FIXME: `MatrixStride` applies to a struct member containing the matrix, not the
            // matrix type itself.
            id_info
                .decorations()
                .iter()
                .find_map(|instruction| match *instruction {
                    Instruction::Decorate {
                        decoration: Decoration::MatrixStride { matrix_stride },
                        ..
                    } => Some(matrix_stride as DeviceSize),
                    _ => None,
                })
                .or_else(|| size_of_type(spirv, column_type))
                .map(|stride| stride * column_count as DeviceSize)
        }
        Instruction::TypeArray {
            element_type,
            length,
            ..
        } => id_info
            .decorations()
            .iter()
            .find_map(|instruction| match *instruction {
                Instruction::Decorate {
                    decoration: Decoration::ArrayStride { array_stride },
                    ..
                } => Some(array_stride as DeviceSize),
                _ => None,
            })
            .or_else(|| size_of_type(spirv, element_type))
            .map(|stride| {
                let length = get_constant(spirv, length).unwrap();
                stride * length
            }),
        Instruction::TypeRuntimeArray { .. } => None,
        Instruction::TypeStruct {
            ref member_types, ..
        } => {
            member_types.iter().zip(id_info.members()).try_fold(
                0,
                |end_of_struct, (&member, member_info)| {
                    let offset = member_info
                        .decorations()
                        .iter()
                        .find_map(|instruction| {
                            match *instruction {
                                // Built-ins have an unknown size.
                                Instruction::MemberDecorate {
                                    decoration: Decoration::BuiltIn { .. },
                                    ..
                                } => Some(None),
                                Instruction::MemberDecorate {
                                    decoration: Decoration::Offset { byte_offset },
                                    ..
                                } => Some(Some(byte_offset as DeviceSize)),
                                _ => None,
                            }
                        })
                        .unwrap_or(Some(end_of_struct))?;
                    let size = size_of_type(spirv, member)?;
                    Some(end_of_struct.max(offset + size))
                },
            )
        }
        ref instruction => todo!(
            "An unknown type was passed to `size_of_type`. \
            This is a Vulkano bug and should be reported.\n
            Instruction::{:?}",
            instruction
        ),
    }
}

/// Returns the smallest offset of all members of a struct, or 0 if `id` is not a struct.
fn offset_of_struct(spirv: &Spirv, id: Id) -> u32 {
    spirv
        .id(id)
        .members()
        .iter()
        .filter_map(|member_info| {
            member_info
                .decorations()
                .iter()
                .find_map(|instruction| match *instruction {
                    Instruction::MemberDecorate {
                        decoration: Decoration::Offset { byte_offset },
                        ..
                    } => Some(byte_offset),
                    _ => None,
                })
        })
        .min()
        .unwrap_or(0)
}

pub(crate) fn get_constant(spirv: &Spirv, id: Id) -> Option<u64> {
    match spirv.id(id).instruction() {
        Instruction::Constant { value, .. } => match value.len() {
            1 => Some(value[0] as u64),
            2 => Some(value[0] as u64 | ((value[1] as u64) << 32)),
            _ => panic!("constant {} is larger than 64 bits", id),
        },
        _ => None,
    }
}

pub(crate) fn get_constant_composite(spirv: &Spirv, id: Id) -> Option<SmallVec<[u64; 4]>> {
    match spirv.id(id).instruction() {
        Instruction::ConstantComposite { constituents, .. } => Some(
            constituents
                .iter()
                .map(|&id| match spirv.id(id).instruction() {
                    Instruction::Constant { value, .. } => match value.len() {
                        1 => value[0] as u64,
                        2 => value[0] as u64 | ((value[1] as u64) << 32),
                        _ => panic!("constant {} is larger than 64 bits", id),
                    },
                    _ => unreachable!(),
                })
                .collect(),
        ),
        _ => None,
    }
}

pub(crate) fn get_constant_float_composite(spirv: &Spirv, id: Id) -> Option<SmallVec<[f64; 4]>> {
    match spirv.id(id).instruction() {
        Instruction::ConstantComposite { constituents, .. } => Some(
            constituents
                .iter()
                .map(|&id| match spirv.id(id).instruction() {
                    Instruction::Constant { value, .. } => match value.len() {
                        1 => f32::from_bits(value[0]) as f64,
                        2 => f64::from_bits(value[0] as u64 | ((value[1] as u64) << 32)),
                        _ => panic!("constant {} is larger than 64 bits", id),
                    },
                    _ => unreachable!(),
                })
                .collect(),
        ),
        _ => None,
    }
}

fn integer_constant_to_i64(spirv: &Spirv, value: &[u32], result_type_id: Id) -> i64 {
    let type_id_instruction = spirv.id(result_type_id).instruction();
    match type_id_instruction {
        Instruction::TypeInt {
            width, signedness, ..
        } => {
            if *width == 64 {
                assert_eq!(value.len(), 2);
            } else {
                assert_eq!(value.len(), 1);
            }

            match (signedness, width) {
                (0, 8) => value[0] as u8 as i64,
                (0, 16) => value[0] as u16 as i64,
                (0, 32) => value[0] as i64,
                (0, 64) => i64::try_from((value[0] as u64) | ((value[1] as u64) << 32)).unwrap(),
                (1, 8) => value[0] as i8 as i64,
                (1, 16) => value[0] as i16 as i64,
                (1, 32) => value[0] as i32 as i64,
                (1, 64) => (value[0] as i64) | ((value[1] as i64) << 32),
                _ => unimplemented!(),
            }
        }
        _ => unreachable!(),
    }
}

pub(crate) fn get_constant_signed_maybe_composite(
    spirv: &Spirv,
    id: Id,
) -> Option<SmallVec<[i64; 4]>> {
    match spirv.id(id).instruction() {
        Instruction::Constant {
            value,
            result_type_id,
            ..
        } => Some(smallvec![integer_constant_to_i64(
            spirv,
            value,
            *result_type_id
        )]),
        Instruction::ConstantComposite { constituents, .. } => Some(
            constituents
                .iter()
                .map(|&id| match spirv.id(id).instruction() {
                    Instruction::Constant {
                        value,
                        result_type_id,
                        ..
                    } => integer_constant_to_i64(spirv, value, *result_type_id),
                    _ => unreachable!(),
                })
                .collect(),
        ),
        _ => None,
    }
}

pub(crate) fn get_constant_signed_composite_composite(
    spirv: &Spirv,
    id: Id,
) -> Option<SmallVec<[SmallVec<[i64; 4]>; 4]>> {
    match spirv.id(id).instruction() {
        Instruction::ConstantComposite { constituents, .. } => Some(
            constituents
                .iter()
                .map(|&id| match spirv.id(id).instruction() {
                    Instruction::ConstantComposite { constituents, .. } => constituents
                        .iter()
                        .map(|&id| match spirv.id(id).instruction() {
                            Instruction::Constant {
                                value,
                                result_type_id,
                                ..
                            } => integer_constant_to_i64(spirv, value, *result_type_id),
                            _ => unreachable!(),
                        })
                        .collect(),
                    _ => unreachable!(),
                })
                .collect(),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{HashMap, PushConstantRange, ShaderStages, Version};

    #[test]
    fn push_constant_range() {
        /*
            ; SPIR-V
            ; Version: 1.0
            ; Generator: Google Shaderc over Glslang; 10
            ; Bound: 27
            ; Schema: 0
            OpCapability Shader
            %glsl_std450 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint GLCompute %main_cs "main_cs" %push_cs
            OpEntryPoint Fragment %main_fs "main_fs" %push_fs
            OpExecutionMode %main_cs LocalSize 1 1 1
            OpExecutionMode %main_fs OriginUpperLeft
            OpName %main_cs "main_cs"
            OpName %PushConstsCS "PushCS"
            OpMemberName %PushConstsCS 0 "a"
            OpMemberName %PushConstsCS 1 "b"
            OpName %main_fs "main_fs"
            OpName %push_fs "PushFS"
            OpMemberName %PushConstsFS 0 "a"
            OpMemberDecorate %PushConstsCS 0 Offset 0
            OpMemberDecorate %PushConstsCS 1 Offset 4
            OpDecorate %PushConstsCS Block
            OpMemberDecorate %PushConstsFS 0 Offset 0
            OpDecorate %PushConstsFS Block
            %void = OpTypeVoid
            %fn_void = OpTypeFunction %void
            %uint = OpTypeInt 32 0
            %int = OpTypeInt 32 1
            %float = OpTypeFloat 32
            %PushConstsCS = OpTypeStruct %uint %float
            %_ptr_PushConstant_PushConstsCS = OpTypePointer PushConstant %PushConstsCS
            %push_cs = OpVariable %_ptr_PushConstant_PushConstsCS PushConstant
            %_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
            %int_0 = OpConstant %int 0
            %int_1 = OpConstant %int 1
            %_ptr_PushConstant_float = OpTypePointer PushConstant %float
            %PushConstsFS = OpTypeStruct %float
            %_ptr_PushConstant_PushConstsFS = OpTypePointer PushConstant %PushConstsFS
            %push_fs = OpVariable %_ptr_PushConstant_PushConstsFS PushConstant
            %main_cs = OpFunction %void None %fn_void
                %main_cs_label = OpLabel
                %push_cs_access_0 = OpAccessChain %_ptr_PushConstant_uint %push_cs %int_0
                %push_cs_access_1 = OpAccessChain %_ptr_PushConstant_float %push_cs %int_1
                %push_cs_load_0 = OpLoad %uint %push_cs_access_0
                %push_cs_load_1 = OpLoad %float %push_cs_access_1
                OpReturn
            OpFunctionEnd
            %main_fs = OpFunction %void None %fn_void
                %main_fs_label = OpLabel
                %push_fs_access_0 = OpAccessChain %_ptr_PushConstant_float %push_fs %int_0
                %push_fs_load_0 = OpLoad %float %push_fs_access_0
                OpReturn
            OpFunctionEnd
        */
        const MODULE: [u32; 186] = [
            119734787, 65536, 458752, 27, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
            808793134, 0, 196622, 0, 1, 393231, 5, 2, 1852399981, 7562079, 3, 393231, 4, 4,
            1852399981, 7562847, 5, 393232, 2, 17, 1, 1, 1, 196624, 4, 7, 262149, 2, 1852399981,
            7562079, 262149, 6, 1752397136, 21315, 262150, 6, 0, 97, 262150, 6, 1, 98, 262149, 4,
            1852399981, 7562847, 262149, 5, 1752397136, 21318, 262150, 7, 0, 97, 327752, 6, 0, 35,
            0, 327752, 6, 1, 35, 4, 196679, 6, 2, 327752, 7, 0, 35, 0, 196679, 7, 2, 131091, 8,
            196641, 9, 8, 262165, 10, 32, 0, 262165, 11, 32, 1, 196630, 12, 32, 262174, 6, 10, 12,
            262176, 13, 9, 6, 262203, 13, 3, 9, 262176, 14, 9, 10, 262187, 11, 15, 0, 262187, 11,
            16, 1, 262176, 17, 9, 12, 196638, 7, 12, 262176, 18, 9, 7, 262203, 18, 5, 9, 327734, 8,
            2, 0, 9, 131320, 19, 327745, 14, 20, 3, 15, 327745, 17, 21, 3, 16, 262205, 10, 22, 20,
            262205, 12, 23, 21, 65789, 65592, 327734, 8, 4, 0, 9, 131320, 24, 327745, 17, 25, 5,
            15, 262205, 12, 26, 25, 65789, 65592,
        ];
        let spirv = crate::shader::spirv::Spirv::new(&MODULE).unwrap();
        assert_eq!(spirv.version(), Version::V1_0);
        let entry_points: HashMap<_, _> = super::entry_points(&spirv)
            .map(|(_, v)| (v.name.clone(), v))
            .collect();
        assert_eq!(entry_points.len(), 2);
        let main_cs = &entry_points["main_cs"];
        assert_eq!(
            main_cs.push_constant_requirements,
            Some(PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: 8,
            })
        );
        let main_fs = &entry_points["main_fs"];
        assert_eq!(
            main_fs.push_constant_requirements,
            Some(PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                offset: 0,
                size: 4,
            })
        );
    }

    #[test]
    fn push_constant_range_spirv_1_4() {
        /*
            ; SPIR-V
            ; Version: 1.4
            ; Generator: Google Shaderc over Glslang; 10
            ; Bound: 27
            ; Schema: 0
            OpCapability Shader
            %glsl_std450 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint GLCompute %main_cs "main_cs" %push_cs
            OpEntryPoint Fragment %main_fs "main_fs" %push_fs
            OpExecutionMode %main_cs LocalSize 1 1 1
            OpExecutionMode %main_fs OriginUpperLeft
            OpName %main_cs "main_cs"
            OpName %PushConstsCS "PushCS"
            OpMemberName %PushConstsCS 0 "a"
            OpMemberName %PushConstsCS 1 "b"
            OpName %main_fs "main_fs"
            OpName %push_fs "PushFS"
            OpMemberName %PushConstsFS 0 "a"
            OpMemberDecorate %PushConstsCS 0 Offset 0
            OpMemberDecorate %PushConstsCS 1 Offset 4
            OpDecorate %PushConstsCS Block
            OpMemberDecorate %PushConstsFS 0 Offset 0
            OpDecorate %PushConstsFS Block
            %void = OpTypeVoid
            %fn_void = OpTypeFunction %void
            %uint = OpTypeInt 32 0
            %int = OpTypeInt 32 1
            %float = OpTypeFloat 32
            %PushConstsCS = OpTypeStruct %uint %float
            %_ptr_PushConstant_PushConstsCS = OpTypePointer PushConstant %PushConstsCS
            %push_cs = OpVariable %_ptr_PushConstant_PushConstsCS PushConstant
            %_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
            %int_0 = OpConstant %int 0
            %int_1 = OpConstant %int 1
            %_ptr_PushConstant_float = OpTypePointer PushConstant %float
            %PushConstsFS = OpTypeStruct %float
            %_ptr_PushConstant_PushConstsFS = OpTypePointer PushConstant %PushConstsFS
            %push_fs = OpVariable %_ptr_PushConstant_PushConstsFS PushConstant
            %main_cs = OpFunction %void None %fn_void
                %main_cs_label = OpLabel
                %push_cs_access_0 = OpAccessChain %_ptr_PushConstant_uint %push_cs %int_0
                %push_cs_access_1 = OpAccessChain %_ptr_PushConstant_float %push_cs %int_1
                %push_cs_load_0 = OpLoad %uint %push_cs_access_0
                %push_cs_load_1 = OpLoad %float %push_cs_access_1
                OpReturn
            OpFunctionEnd
            %main_fs = OpFunction %void None %fn_void
                %main_fs_label = OpLabel
                %push_fs_access_0 = OpAccessChain %_ptr_PushConstant_float %push_fs %int_0
                %push_fs_load_0 = OpLoad %float %push_fs_access_0
                OpReturn
            OpFunctionEnd
        */
        const MODULE: [u32; 186] = [
            119734787, 66560, 458752, 27, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
            808793134, 0, 196622, 0, 1, 393231, 5, 2, 1852399981, 7562079, 3, 393231, 4, 4,
            1852399981, 7562847, 5, 393232, 2, 17, 1, 1, 1, 196624, 4, 7, 262149, 2, 1852399981,
            7562079, 262149, 6, 1752397136, 21315, 262150, 6, 0, 97, 262150, 6, 1, 98, 262149, 4,
            1852399981, 7562847, 262149, 5, 1752397136, 21318, 262150, 7, 0, 97, 327752, 6, 0, 35,
            0, 327752, 6, 1, 35, 4, 196679, 6, 2, 327752, 7, 0, 35, 0, 196679, 7, 2, 131091, 8,
            196641, 9, 8, 262165, 10, 32, 0, 262165, 11, 32, 1, 196630, 12, 32, 262174, 6, 10, 12,
            262176, 13, 9, 6, 262203, 13, 3, 9, 262176, 14, 9, 10, 262187, 11, 15, 0, 262187, 11,
            16, 1, 262176, 17, 9, 12, 196638, 7, 12, 262176, 18, 9, 7, 262203, 18, 5, 9, 327734, 8,
            2, 0, 9, 131320, 19, 327745, 14, 20, 3, 15, 327745, 17, 21, 3, 16, 262205, 10, 22, 20,
            262205, 12, 23, 21, 65789, 65592, 327734, 8, 4, 0, 9, 131320, 24, 327745, 17, 25, 5,
            15, 262205, 12, 26, 25, 65789, 65592,
        ];
        let spirv = crate::shader::spirv::Spirv::new(&MODULE).unwrap();
        assert_eq!(spirv.version(), Version::V1_4);
        let entry_points: HashMap<_, _> = super::entry_points(&spirv)
            .map(|(_, v)| (v.name.clone(), v))
            .collect();
        assert_eq!(entry_points.len(), 2);
        let main_cs = &entry_points["main_cs"];
        assert_eq!(
            main_cs.push_constant_requirements,
            Some(PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: 8,
            })
        );
        let main_fs = &entry_points["main_fs"];
        assert_eq!(
            main_fs.push_constant_requirements,
            Some(PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                offset: 0,
                size: 4,
            })
        );
    }
}
