use crate::{
    descriptor_set::layout::DescriptorType,
    device::{physical::ShaderFloatControlsIndependence, Device, DeviceFeatures},
    pipeline::inout_interface::{
        input_output_map, shader_interface_analyze_type, InputOutputData, InputOutputKey,
    },
    shader::{
        reflect::{
            get_constant, get_constant_composite, get_constant_float_composite,
            get_constant_signed_composite_composite, get_constant_signed_maybe_composite,
            size_of_type,
        },
        spirv::{
            Capability, Decoration, Dim, ExecutionMode, ExecutionModel, FunctionInfo, Id,
            ImageFormat, Instruction, Scope, Spirv, StorageClass,
        },
        ShaderStage,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version,
};
use foldhash::HashMap;
use std::{cmp::max, convert::Infallible};

pub(crate) fn validate_runtime(
    device: &Device,
    spirv: &Spirv,
    entry_point: Id,
) -> Result<(), Box<ValidationError>> {
    let entry_point_info = spirv.function(entry_point);
    let Instruction::EntryPoint {
        execution_model,
        ref interface,
        ..
    } = *entry_point_info.entry_point().unwrap()
    else {
        unreachable!()
    };

    let mut validator = RuntimeValidator {
        device,
        spirv,
        entry_point_info,
        execution_model,
        interface,

        first_emitted_stream: None,
        output_primitives: None,
    };

    // Ordering is important
    validator.validate_capabilities()?;
    validator.validate_decorations()?;
    validator.validate_execution_modes()?;
    validator.validate_types()?;
    // FIXME: validator.validate_global_variables()?;
    validator.validate_functions()?;

    Ok(())
}

struct RuntimeValidator<'a> {
    device: &'a Device,
    spirv: &'a Spirv,
    entry_point_info: &'a FunctionInfo,
    execution_model: ExecutionModel,
    interface: &'a [Id],

    first_emitted_stream: Option<u32>,
    output_primitives: Option<&'a ExecutionMode>,
}

impl RuntimeValidator<'_> {
    fn validate_capabilities(&self) -> Result<(), Box<ValidationError>> {
        for instruction in self.spirv.capabilities() {
            let capability = match *instruction {
                Instruction::Capability { capability } => capability,
                _ => continue,
            };

            #[allow(clippy::single_match)]
            match capability {
                Capability::InterpolationFunction => {
                    if self.device.enabled_extensions().khr_portability_subset
                        && !self
                            .device
                            .enabled_features()
                            .shader_sample_rate_interpolation_functions
                    {
                        return Err(Box::new(ValidationError {
                            problem: "this device is a portability subset device, and the shader \
                                uses the `InterpolationFunction` capability"
                                .into(),
                            vuids: &[
                                "VUID-RuntimeSpirv-shaderSampleRateInterpolationFunctions-06325",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                _ => (),
            }
        }

        Ok(())
    }

    fn validate_decorations(&self) -> Result<(), Box<ValidationError>> {
        let properties = self.device.physical_device().properties();

        for instruction in self.spirv.decorations() {
            let decoration = match instruction {
                Instruction::Decorate { decoration, .. }
                | Instruction::DecorateId { decoration, .. }
                | Instruction::DecorateString { decoration, .. }
                | Instruction::MemberDecorate { decoration, .. }
                | Instruction::MemberDecorateString { decoration, .. } => decoration,
                _ => continue,
            };

            match *decoration {
                Decoration::XfbStride { xfb_stride } => {
                    if xfb_stride
                        > properties
                            .max_transform_feedback_buffer_data_stride
                            .unwrap_or_default()
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the `XfbStride` decoration is used, but its `xfb_stride` \
                                value is greater than the \
                                `max_transform_feedback_buffer_data_stride` device limit"
                                .into(),
                            vuids: &["VUID-RuntimeSpirv-XfbStride-06313"],
                            ..Default::default()
                        }));
                    }
                }
                Decoration::Stream { stream_number } => {
                    if stream_number
                        >= properties
                            .max_transform_feedback_streams
                            .unwrap_or_default()
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the `Stream` decoration is used, but its `stream_number` \
                                value is not less than the `max_transform_feedback_streams` \
                                device limit"
                                .into(),
                            vuids: &["VUID-RuntimeSpirv-Stream-06312"],
                            ..Default::default()
                        }));
                    }
                }
                _ => (),
            }
        }

        Ok(())
    }

    fn validate_execution_modes(&mut self) -> Result<(), Box<ValidationError>> {
        let properties = self.device.physical_device().properties();

        #[derive(PartialEq, Eq)]
        enum DenormMode {
            Preserve,
            FlushToZero,
        }

        #[allow(clippy::upper_case_acronyms)]
        #[derive(PartialEq, Eq)]
        enum RoundingMode {
            RTE,
            RTZ,
        }

        let mut denorm_mode_16 = None;
        let mut denorm_mode_32 = None;
        let mut denorm_mode_64 = None;
        let mut rounding_mode_16 = None;
        let mut rounding_mode_32 = None;
        let mut rounding_mode_64 = None;

        for instruction in self.entry_point_info.execution_modes() {
            let execution_mode = match instruction {
                Instruction::ExecutionMode { mode, .. }
                | Instruction::ExecutionModeId { mode, .. } => mode,
                _ => continue,
            };

            match *execution_mode {
                ExecutionMode::SignedZeroInfNanPreserve { target_width } => match target_width {
                    16 => {
                        if !properties
                            .shader_signed_zero_inf_nan_preserve_float16
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `SignedZeroInfNanPreserve` \
                                    execution mode with a `target_width` of 16, but the \
                                    `shader_signed_zero_inf_nan_preserve_float16` \
                                    device property is `false`"
                                    .into(),
                                vuids: &[
                                    "VUID-RuntimeSpirv-shaderSignedZeroInfNanPreserveFloat16-06293",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    32 => {
                        if !properties
                            .shader_signed_zero_inf_nan_preserve_float32
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `SignedZeroInfNanPreserve` \
                                    execution mode with a `target_width` of 32, but the \
                                    `shader_signed_zero_inf_nan_preserve_float32` \
                                    device property is `false`"
                                    .into(),
                                vuids: &[
                                    "VUID-RuntimeSpirv-shaderSignedZeroInfNanPreserveFloat32-06294",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    64 => {
                        if !properties
                            .shader_signed_zero_inf_nan_preserve_float64
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `SignedZeroInfNanPreserve` \
                                    execution mode with a `target_width` of 64, but the \
                                    `shader_signed_zero_inf_nan_preserve_float64` \
                                    device property is `false`"
                                    .into(),
                                vuids: &[
                                    "VUID-RuntimeSpirv-shaderSignedZeroInfNanPreserveFloat64-06295",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                },
                ExecutionMode::DenormPreserve { target_width } => match target_width {
                    16 => {
                        denorm_mode_16 = Some(DenormMode::Preserve);

                        if !properties
                            .shader_denorm_preserve_float16
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `DenormPreserve` \
                                    execution mode with a `target_width` of 16, but the \
                                    `shader_denorm_preserve_float16` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderDenormPreserveFloat16-06296"],
                                ..Default::default()
                            }));
                        }
                    }
                    32 => {
                        denorm_mode_32 = Some(DenormMode::Preserve);

                        if !properties
                            .shader_denorm_preserve_float32
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `DenormPreserve` \
                                    execution mode with a `target_width` of 32, but the \
                                    `shader_denorm_preserve_float32` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderDenormPreserveFloat32-06297"],
                                ..Default::default()
                            }));
                        }
                    }
                    64 => {
                        denorm_mode_64 = Some(DenormMode::Preserve);

                        if !properties
                            .shader_denorm_preserve_float64
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `DenormPreserve` \
                                    execution mode with a `target_width` of 64, but the \
                                    `shader_denorm_preserve_float64` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderDenormPreserveFloat64-06298"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                },
                ExecutionMode::DenormFlushToZero { target_width } => match target_width {
                    16 => {
                        denorm_mode_16 = Some(DenormMode::FlushToZero);

                        if !properties
                            .shader_denorm_flush_to_zero_float16
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `DenormFlushToZero` \
                                    execution mode with a `target_width` of 16, but the \
                                    `shader_denorm_flush_to_zero_float16` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderDenormFlushToZeroFloat16-06299"],
                                ..Default::default()
                            }));
                        }
                    }
                    32 => {
                        denorm_mode_32 = Some(DenormMode::FlushToZero);

                        if !properties
                            .shader_denorm_flush_to_zero_float32
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `DenormFlushToZero` \
                                    execution mode with a `target_width` of 32, but the \
                                    `shader_denorm_flush_to_zero_float32` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderDenormFlushToZeroFloat32-06300"],
                                ..Default::default()
                            }));
                        }
                    }
                    64 => {
                        denorm_mode_64 = Some(DenormMode::FlushToZero);

                        if !properties
                            .shader_denorm_flush_to_zero_float64
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `DenormFlushToZero` \
                                    execution mode with a `target_width` of 64, but the \
                                    `shader_denorm_flush_to_zero_float64` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderDenormFlushToZeroFloat64-06301"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                },
                ExecutionMode::RoundingModeRTE { target_width } => match target_width {
                    16 => {
                        rounding_mode_16 = Some(RoundingMode::RTE);

                        if !properties
                            .shader_rounding_mode_rte_float16
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `RoundingModeRTE` \
                                    execution mode with a `target_width` of 16, but the \
                                    `shader_rounding_mode_rte_float16` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderRoundingModeRTEFloat16-06302"],
                                ..Default::default()
                            }));
                        }
                    }
                    32 => {
                        rounding_mode_32 = Some(RoundingMode::RTE);

                        if !properties
                            .shader_rounding_mode_rte_float32
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `RoundingModeRTE` \
                                    execution mode with a `target_width` of 32, but the \
                                    `shader_rounding_mode_rte_float32` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderRoundingModeRTEFloat32-06303"],
                                ..Default::default()
                            }));
                        }
                    }
                    64 => {
                        rounding_mode_64 = Some(RoundingMode::RTE);

                        if !properties
                            .shader_rounding_mode_rte_float64
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `RoundingModeRTE` \
                                    execution mode with a `target_width` of 64, but the \
                                    `shader_rounding_mode_rte_float64` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderRoundingModeRTEFloat64-06304"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                },
                ExecutionMode::RoundingModeRTZ { target_width } => match target_width {
                    16 => {
                        rounding_mode_16 = Some(RoundingMode::RTZ);

                        if !properties
                            .shader_rounding_mode_rtz_float16
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `RoundingModeRTZ` \
                                    execution mode with a `target_width` of 16, but the \
                                    `shader_rounding_mode_rtz_float16` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderRoundingModeRTZFloat16-06305"],
                                ..Default::default()
                            }));
                        }
                    }
                    32 => {
                        rounding_mode_32 = Some(RoundingMode::RTZ);

                        if !properties
                            .shader_rounding_mode_rtz_float32
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `RoundingModeRTZ` \
                                    execution mode with a `target_width` of 32, but the \
                                    `shader_rounding_mode_rtz_float32` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderRoundingModeRTZFloat32-06306"],
                                ..Default::default()
                            }));
                        }
                    }
                    64 => {
                        rounding_mode_64 = Some(RoundingMode::RTZ);

                        if !properties
                            .shader_rounding_mode_rtz_float64
                            .unwrap_or_default()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point has a `RoundingModeRTZ` \
                                    execution mode with a `target_width` of 64, but the \
                                    `shader_rounding_mode_rtz_float64` \
                                    device property is `false`"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-shaderRoundingModeRTZFloat64-06307"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                },
                ExecutionMode::Isolines => {
                    if self.device.enabled_extensions().khr_portability_subset
                        && self.device.enabled_features().tessellation_shader
                        && !self.device.enabled_features().tessellation_isolines
                    {
                        return Err(Box::new(ValidationError {
                            problem: "this device is a portability subset device, and \
                                the entry point has an `IsoLines` execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("tessellation_isolines"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-tessellationShader-06326"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::PointMode => {
                    if self.device.enabled_extensions().khr_portability_subset
                        && self.device.enabled_features().tessellation_shader
                        && !self.device.enabled_features().tessellation_point_mode
                    {
                        return Err(Box::new(ValidationError {
                            problem: "this device is a portability subset device, and \
                                the entry point has an `PointMode` execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("tessellation_point_mode"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-tessellationShader-06327"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::LocalSize { .. } | ExecutionMode::LocalSizeId { .. } => {
                    let local_size = match *execution_mode {
                        ExecutionMode::LocalSize {
                            x_size,
                            y_size,
                            z_size,
                        } => [x_size as u64, y_size as u64, z_size as u64],
                        ExecutionMode::LocalSizeId {
                            x_size,
                            y_size,
                            z_size,
                        } => {
                            if !self.device.enabled_features().maintenance4 {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point has a `LocalSizeId` execution mode"
                                        .into(),
                                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                        Requires::DeviceFeature("maintenance4"),
                                    ])]),
                                    vuids: &["VUID-RuntimeSpirv-LocalSizeId-06434"],
                                    ..Default::default()
                                }));
                            }

                            let x_size = get_constant(self.spirv, x_size).unwrap();
                            let y_size = get_constant(self.spirv, y_size).unwrap();
                            let z_size = get_constant(self.spirv, z_size).unwrap();

                            [x_size, y_size, z_size]
                        }
                        _ => unreachable!(),
                    };
                    let workgroup_size = local_size.into_iter().try_fold(1, |t, x| {
                        u32::try_from(x).ok().and_then(|x| x.checked_mul(t))
                    });

                    match self.execution_model {
                        ExecutionModel::GLCompute => {
                            if u32::try_from(local_size[0]).map_or(true, |size| {
                                size > properties.max_compute_work_group_size[0]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `GLCompute`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_x` is greater than the \
                                        `max_compute_work_group_size[0]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-x-06429"],
                                    ..Default::default()
                                }));
                            }

                            if u32::try_from(local_size[1]).map_or(true, |size| {
                                size > properties.max_compute_work_group_size[1]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `GLCompute`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_y` is greater than the \
                                        `max_compute_work_group_size[1]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-x-06430"],
                                    ..Default::default()
                                }));
                            }

                            if u32::try_from(local_size[2]).map_or(true, |size| {
                                size > properties.max_compute_work_group_size[2]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `GLCompute`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_z` is greater than the \
                                        `max_compute_work_group_size[2]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-x-06431"],
                                    ..Default::default()
                                }));
                            }

                            if workgroup_size.is_none_or(|size| {
                                size > properties.max_compute_work_group_invocations
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `GLCompute`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but the product of its `size_x`, `size_y` and `size_z` is \
                                        greater than the `max_compute_work_group_invocations` \
                                        device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-x-06432"],
                                    ..Default::default()
                                }));
                            }
                        }
                        ExecutionModel::TaskEXT => {
                            if u32::try_from(local_size[0]).map_or(true, |size| {
                                size > properties.max_task_work_group_size.unwrap_or_default()[0]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `TaskEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_x` is greater than the \
                                        `max_task_work_group_size[0]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-TaskEXT-07291"],
                                    ..Default::default()
                                }));
                            }

                            if u32::try_from(local_size[1]).map_or(true, |size| {
                                size > properties.max_task_work_group_size.unwrap_or_default()[1]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `TaskEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_y` is greater than the \
                                        `max_task_work_group_size[1]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-TaskEXT-07292"],
                                    ..Default::default()
                                }));
                            }

                            if u32::try_from(local_size[2]).map_or(true, |size| {
                                size > properties.max_task_work_group_size.unwrap_or_default()[2]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `TaskEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_z` is greater than the \
                                        `max_task_work_group_size[2]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-TaskEXT-07293"],
                                    ..Default::default()
                                }));
                            }

                            if workgroup_size.is_none_or(|size| {
                                size > properties
                                    .max_task_work_group_invocations
                                    .unwrap_or_default()
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `TaskEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but the product of its `size_x`, `size_y` and `size_z` is \
                                        greater than the `max_task_work_group_invocations` \
                                        device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-TaskEXT-07294"],
                                    ..Default::default()
                                }));
                            }
                        }
                        ExecutionModel::MeshEXT => {
                            if u32::try_from(local_size[0]).map_or(true, |size| {
                                size > properties.max_mesh_work_group_size.unwrap_or_default()[0]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `MeshEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_x` is greater than the \
                                        `max_mesh_work_group_size[0]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-MeshEXT-07295"],
                                    ..Default::default()
                                }));
                            }

                            if u32::try_from(local_size[1]).map_or(true, |size| {
                                size > properties.max_mesh_work_group_size.unwrap_or_default()[1]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `MeshEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_y` is greater than the \
                                        `max_mesh_work_group_size[1]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-MeshEXT-07296"],
                                    ..Default::default()
                                }));
                            }

                            if u32::try_from(local_size[2]).map_or(true, |size| {
                                size > properties.max_mesh_work_group_size.unwrap_or_default()[2]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `MeshEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but its `size_z` is greater than the \
                                    `max_mesh_work_group_size[2]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-MeshEXT-07297"],
                                    ..Default::default()
                                }));
                            }

                            if workgroup_size.is_none_or(|size| {
                                size > properties
                                    .max_mesh_work_group_invocations
                                    .unwrap_or_default()
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `MeshEXT`, \
                                        and it has a `LocalSize` or `LocalSizeId` execution mode, \
                                        but the product of its `size_x`, `size_y` and `size_z` is \
                                        greater than the `max_mesh_work_group_invocations` \
                                        device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-MeshEXT-07298"],
                                    ..Default::default()
                                }));
                            }
                        }
                        _ => (),
                    }
                }
                ExecutionMode::SubgroupUniformControlFlowKHR => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_subgroup_uniform_control_flow
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `SubgroupUniformControlFlowKHR` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_subgroup_uniform_control_flow"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-SubgroupUniformControlFlowKHR-06379"],
                            ..Default::default()
                        }));
                    }

                    if !properties
                        .supported_stages
                        .unwrap_or_default()
                        .contains_enum(ShaderStage::from(self.execution_model))
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `SubgroupUniformControlFlowKHR` \
                                execution mode, but the `supported_stages` device property does \
                                not contain the shader stage of the entry point's execution model"
                                .into(),
                            vuids: &["VUID-RuntimeSpirv-SubgroupUniformControlFlowKHR-06379"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::EarlyAndLateFragmentTestsAMD => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_early_and_late_fragment_tests
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `EarlyAndLateFragmentTestsAMD` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_early_and_late_fragment_tests"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderEarlyAndLateFragmentTests-06767"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::StencilRefUnchangedFrontAMD => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_early_and_late_fragment_tests
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `StencilRefUnchangedFrontAMD` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_early_and_late_fragment_tests"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderEarlyAndLateFragmentTests-06768"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::StencilRefUnchangedBackAMD => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_early_and_late_fragment_tests
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `StencilRefUnchangedBackAMD` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_early_and_late_fragment_tests"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderEarlyAndLateFragmentTests-06769"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::StencilRefGreaterFrontAMD => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_early_and_late_fragment_tests
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `StencilRefGreaterFrontAMD` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_early_and_late_fragment_tests"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderEarlyAndLateFragmentTests-06770"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::StencilRefGreaterBackAMD => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_early_and_late_fragment_tests
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `StencilRefGreaterBackAMD` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_early_and_late_fragment_tests"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderEarlyAndLateFragmentTests-06771"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::StencilRefLessFrontAMD => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_early_and_late_fragment_tests
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `StencilRefLessFrontAMD` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_early_and_late_fragment_tests"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderEarlyAndLateFragmentTests-06772"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::StencilRefLessBackAMD => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_early_and_late_fragment_tests
                    {
                        return Err(Box::new(ValidationError {
                            problem: "the entry point has a `StencilRefLessBackAMD` \
                                execution mode"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_early_and_late_fragment_tests"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderEarlyAndLateFragmentTests-06773"],
                            ..Default::default()
                        }));
                    }
                }
                ExecutionMode::OutputPoints
                | ExecutionMode::OutputLineStrip
                | ExecutionMode::OutputTriangleStrip => {
                    self.output_primitives = Some(execution_mode);
                }
                ExecutionMode::OutputVertices { vertex_count } => {
                    match self.execution_model {
                        ExecutionModel::MeshNV => {
                            // TODO: needs VK_NV_mesh_shader support
                            // VUID-RuntimeSpirv-MeshNV-07113
                        }
                        ExecutionModel::MeshEXT => {
                            if vertex_count
                                > properties.max_mesh_output_vertices.unwrap_or_default()
                            {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `MeshEXT`, and \
                                        it has an `OutputVertices` execution mode, but its \
                                        `vertex_count` is greater than the \
                                        `max_mesh_output_vertices` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-MeshEXT-07115"],
                                    ..Default::default()
                                }));
                            }
                        }
                        _ => (),
                    }
                }
                ExecutionMode::OutputPrimitivesEXT { primitive_count } => {
                    match self.execution_model {
                        ExecutionModel::MeshNV => {
                            // TODO: needs VK_NV_mesh_shader support
                            // VUID-RuntimeSpirv-MeshNV-07114
                        }
                        ExecutionModel::MeshEXT => {
                            if primitive_count
                                > properties.max_mesh_output_primitives.unwrap_or_default()
                            {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `MeshEXT`, and \
                                        it has an `OutputPrimitivesEXT` execution mode, but its \
                                        `primitive_count` is greater than the \
                                        `max_mesh_output_primitives` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-MeshEXT-07116"],
                                    ..Default::default()
                                }));
                            }
                        }
                        _ => (),
                    }
                }
                _ => (),
            }
        }

        match properties.denorm_behavior_independence {
            Some(ShaderFloatControlsIndependence::Float32Only) => {
                if denorm_mode_16 != denorm_mode_64 {
                    return Err(Box::new(ValidationError {
                        problem: "the `denorm_behavior_independence` device property is \
                            `ShaderFloatControlsIndependence::Float32Only`, but the entry point \
                            does not have the same denormals execution mode for \
                            both 16-bit and 64-bit values"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-denormBehaviorIndependence-06289"],
                        ..Default::default()
                    }));
                }
            }
            Some(ShaderFloatControlsIndependence::None) => {
                if denorm_mode_16 != denorm_mode_32 || denorm_mode_16 != denorm_mode_64 {
                    return Err(Box::new(ValidationError {
                        problem: "the `denorm_behavior_independence` device property is \
                            `ShaderFloatControlsIndependence::None`, but the entry point \
                            does not have the same denormals execution mode for \
                            16-bit, 32-bit and 64-bit values"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-denormBehaviorIndependence-06290"],
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        match properties.rounding_mode_independence {
            Some(ShaderFloatControlsIndependence::Float32Only) => {
                if rounding_mode_16 != rounding_mode_64 {
                    return Err(Box::new(ValidationError {
                        problem: "the `rounding_mode_independence` device property is \
                            `ShaderFloatControlsIndependence::Float32Only`, but the entry point \
                            does not have the same rounding execution mode for \
                            both 16-bit and 64-bit values"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-roundingModeIndependence-06291"],
                        ..Default::default()
                    }));
                }
            }
            Some(ShaderFloatControlsIndependence::None) => {
                if rounding_mode_16 != rounding_mode_32 || rounding_mode_16 != rounding_mode_64 {
                    return Err(Box::new(ValidationError {
                        problem: "the `rounding_mode_independence` device property is \
                            `ShaderFloatControlsIndependence::None`, but the entry point \
                            does not have the same rounding execution mode for \
                            16-bit, 32-bit and 64-bit values"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-roundingModeIndependence-06292"],
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        Ok(())
    }

    fn validate_types(&self) -> Result<(), Box<ValidationError>> {
        let properties = self.device.physical_device().properties();

        for instruction in self.spirv.types() {
            match *instruction {
                Instruction::TypeCooperativeMatrixKHR { .. } => {
                    // TODO: needs VK_KHR_cooperative_matrix support
                    // VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-08974

                    if !properties
                        .cooperative_matrix_supported_stages
                        .unwrap()
                        .contains_enum(ShaderStage::from(self.execution_model))
                    {
                        return Err(Box::new(ValidationError {
                            problem: "an `OpTypeCooperativeMatrixKHR` type is declared, but \
                                the `cooperative_matrix_supported_stages` device property does \
                                not contain the shader stage of the entry point's execution model"
                                .into(),
                            vuids: &["VUID-RuntimeSpirv-cooperativeMatrixSupportedStages-08985"],
                            ..Default::default()
                        }));
                    }
                }
                Instruction::TypeCooperativeMatrixNV { .. } => {
                    // TODO: needs VK_NV_cooperative_matrix support
                    // VUID-RuntimeSpirv-OpTypeCooperativeMatrixNV-06316
                    // VUID-RuntimeSpirv-OpTypeCooperativeMatrixNV-06322
                }
                _ => (),
            }
        }

        Ok(())
    }

    // FIXME: this needs to use only the global variables that are statically used by the
    // entry point's call tree. That requires analysing the call tree during `Spirv::new`.
    #[allow(dead_code)]
    fn validate_global_variables(&self) -> Result<(), Box<ValidationError>> {
        let properties = self.device.physical_device().properties();

        // Graphics stages except task.
        let mut input_locations_required = 0;
        let mut output_locations_required = 0;
        let mut per_patch_output_locations_required = 0;

        // Only TaskEXT and MeshEXT.
        let mut task_payload_workgroup_memory_size = 0;

        // Only GLCompute, TaskEXT and MeshEXT.
        let mut workgroup_memory_size = 0;

        let mut stream_sizes: HashMap<u32, HashMap<u32, DeviceSize>> = HashMap::default();

        for instruction in self.spirv.global_variables() {
            let Instruction::Variable {
                result_type_id,
                result_id,
                storage_class,
                initializer,
            } = *instruction
            else {
                continue;
            };

            let is_in_interface = self.interface.contains(&result_id);
            let Instruction::TypePointer { ty: type_id, .. } =
                *self.spirv.id(result_type_id).instruction()
            else {
                unreachable!()
            };

            fn get_bits(has_8bit: &mut bool, has_16bit: &mut bool, spirv: &Spirv, ty: Id) {
                match *spirv.id(ty).instruction() {
                    Instruction::TypeInt { width, .. } | Instruction::TypeFloat { width, .. } => {
                        match width {
                            8 => *has_8bit = true,
                            16 => *has_16bit = true,
                            _ => (),
                        }
                    }
                    Instruction::TypePointer { ty, .. }
                    | Instruction::TypeArray {
                        element_type: ty, ..
                    }
                    | Instruction::TypeRuntimeArray {
                        element_type: ty, ..
                    }
                    | Instruction::TypeVector {
                        component_type: ty, ..
                    }
                    | Instruction::TypeMatrix {
                        column_type: ty, ..
                    } => get_bits(has_8bit, has_16bit, spirv, ty),
                    Instruction::TypeStruct {
                        ref member_types, ..
                    } => {
                        for &ty in member_types {
                            get_bits(has_8bit, has_16bit, spirv, ty)
                        }
                    }
                    _ => (),
                }
            }

            let mut has_8bit = false;
            let mut has_16bit = false;
            get_bits(&mut has_8bit, &mut has_16bit, self.spirv, type_id);

            let mut has_aliased = false;
            let mut has_block = false;
            let mut has_buffer_block = false;
            let mut has_non_readable = false;
            let mut has_non_writable = false;
            let mut has_patch = true;

            let mut offset = None;
            let mut stream = 0;
            let mut xfb_buffer = None;

            for instruction in self.spirv.id(result_id).decorations() {
                if let Instruction::Decorate { decoration, .. } = instruction {
                    match *decoration {
                        Decoration::Aliased => has_aliased = true,
                        Decoration::Block => has_block = true,
                        Decoration::BufferBlock => has_buffer_block = true,
                        Decoration::NonReadable => has_non_readable = true,
                        Decoration::NonWritable => has_non_writable = true,
                        Decoration::Offset { byte_offset } => offset = Some(byte_offset),
                        Decoration::Patch => has_patch = true,
                        Decoration::Stream { stream_number } => stream = stream_number,
                        Decoration::XfbBuffer { xfb_buffer_number } => {
                            xfb_buffer = Some(xfb_buffer_number)
                        }
                        _ => (),
                    }
                }
            }

            match storage_class {
                StorageClass::Workgroup => {
                    if matches!(
                        self.execution_model,
                        ExecutionModel::GLCompute
                            | ExecutionModel::TaskEXT
                            | ExecutionModel::MeshEXT
                    ) {
                        if let Some(size) = size_of_type(self.spirv, type_id) {
                            if has_aliased {
                                workgroup_memory_size = workgroup_memory_size.max(size);
                            } else {
                                workgroup_memory_size += size;
                            }
                        }
                    }

                    if initializer.is_some()
                        && !self
                            .device
                            .enabled_features()
                            .shader_zero_initialize_workgroup_memory
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} has a storage class of `StorageClass::Workgroup`, \
                                and has an initializer operand",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_zero_initialize_workgroup_memory"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderZeroInitializeWorkgroupMemory-06372"],
                            ..Default::default()
                        }));
                    }
                }

                StorageClass::TaskPayloadWorkgroupEXT => {
                    if matches!(
                        self.execution_model,
                        ExecutionModel::TaskEXT | ExecutionModel::MeshEXT
                    ) {
                        if let Some(size) = size_of_type(self.spirv, type_id) {
                            task_payload_workgroup_memory_size += size;
                        }
                    }
                }

                StorageClass::StorageBuffer
                | StorageClass::ShaderRecordBufferKHR
                | StorageClass::PhysicalStorageBuffer => {
                    if has_8bit && !self.device.enabled_features().storage_buffer8_bit_access {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} contains an 8-bit integer value, \
                                and has a storage class of `StorageClass::StorageBuffer`, \
                                `StorageClass::ShaderRecordBufferKHR` or \
                                `StorageClass::PhysicalStorageBuffer`",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("storage_buffer8_bit_access"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-storageBuffer8BitAccess-06328"],
                            ..Default::default()
                        }));
                    }

                    if has_16bit && !self.device.enabled_features().storage_buffer16_bit_access {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} contains an 16-bit integer or floating point value, \
                                and has a storage class of `StorageClass::StorageBuffer`, \
                                `StorageClass::ShaderRecordBufferKHR` or \
                                `StorageClass::PhysicalStorageBuffer`",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("storage_buffer16_bit_access"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-storageBuffer16BitAccess-06331"],
                            ..Default::default()
                        }));
                    }
                }

                StorageClass::Uniform => {
                    if has_block {
                        if has_8bit
                            && !self
                                .device
                                .enabled_features()
                                .uniform_and_storage_buffer8_bit_access
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "variable {} contains an 8-bit integer value, and has a \
                                    storage class of `StorageClass::Uniform` and \
                                    is decorated with `Decoration::Block`",
                                    result_id,
                                )
                                .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature(
                                        "uniform_and_storage_buffer8_bit_access",
                                    ),
                                ])]),
                                vuids: &[
                                    "VUID-RuntimeSpirv-uniformAndStorageBuffer8BitAccess-06329",
                                ],
                                ..Default::default()
                            }));
                        }

                        if has_16bit
                            && !self
                                .device
                                .enabled_features()
                                .uniform_and_storage_buffer16_bit_access
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "variable {} contains an 16-bit integer or floating point \
                                    value, and has a storage class of `StorageClass::Uniform` and \
                                    is decorated with `Decoration::Block`",
                                    result_id,
                                )
                                .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature(
                                        "uniform_and_storage_buffer16_bit_access",
                                    ),
                                ])]),
                                vuids: &[
                                    "VUID-RuntimeSpirv-uniformAndStorageBuffer16BitAccess-06332",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                }

                StorageClass::PushConstant => {
                    if has_8bit && !self.device.enabled_features().storage_push_constant8 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} contains an 8-bit integer value, and \
                                has a storage class of `StorageClass::PushConstant`",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("storage_push_constant8"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-storagePushConstant8-06330"],
                            ..Default::default()
                        }));
                    }

                    if has_16bit && !self.device.enabled_features().storage_push_constant16 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} contains an 16-bit integer or floating point value, \
                                and has a storage class of `StorageClass::PushConstant`",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("storage_push_constant16"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-storagePushConstant16-06333"],
                            ..Default::default()
                        }));
                    }
                }

                StorageClass::Input | StorageClass::Output => {
                    if has_16bit && !self.device.enabled_features().storage_input_output16 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} contains an 16-bit integer or floating point value, \
                                and has a storage class of `StorageClass::Input` or \
                                `StorageClass::Output`",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("storage_input_output16"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-storageInputOutput16-06334"],
                            ..Default::default()
                        }));
                    }

                    if matches!(
                        self.execution_model,
                        ExecutionModel::Vertex
                            | ExecutionModel::TessellationControl
                            | ExecutionModel::TessellationEvaluation
                            | ExecutionModel::Geometry
                            | ExecutionModel::Fragment
                    ) || self.execution_model == ExecutionModel::MeshEXT
                        && storage_class == StorageClass::Output
                    {
                        let locations_required = match storage_class {
                            StorageClass::Input => &mut input_locations_required,
                            StorageClass::Output => {
                                if has_patch {
                                    &mut per_patch_output_locations_required
                                } else {
                                    &mut output_locations_required
                                }
                            }
                            _ => unreachable!(),
                        };
                        let mut dummy_scalar_func = |_, _, _| {};

                        input_output_map(
                            self.spirv,
                            self.execution_model,
                            result_id,
                            storage_class,
                            |key, data| -> Result<(), Infallible> {
                                let InputOutputData { type_id, .. } = data;

                                match key {
                                    InputOutputKey::User(key) => {
                                        let (locations, _) = shader_interface_analyze_type(
                                            self.spirv,
                                            type_id,
                                            key,
                                            &mut dummy_scalar_func,
                                        );

                                        *locations_required =
                                            max(*locations_required, key.location + locations);
                                    }
                                    InputOutputKey::BuiltIn(_) => {
                                        // TODO: The spec doesn't currently say how to count this.
                                        // https://github.com/KhronosGroup/Vulkan-Docs/issues/2293
                                    }
                                }

                                Ok(())
                            },
                        )
                        .unwrap();
                    }

                    if is_in_interface && storage_class == StorageClass::Output {
                        let mut check = |offset: u32, xfb_buffer: u32, stream: u32| {
                            if let Some(size) = size_of_type(self.spirv, result_id) {
                                let required_size = offset as DeviceSize + size;

                                if required_size
                                    > properties
                                        .max_transform_feedback_buffer_data_size
                                        .unwrap_or_default()
                                        as DeviceSize
                                {
                                    return Err(Box::new(ValidationError {
                                        problem: format!(
                                            "for the value written to transform feedback buffer \
                                            {} at offset {}, the offset plus the size of the \
                                            value is greater than the \
                                            `max_transform_feedback_buffer_data_size` device limit",
                                            xfb_buffer, offset,
                                        )
                                        .into(),
                                        vuids: &["VUID-RuntimeSpirv-Offset-06308"],
                                        ..Default::default()
                                    }));
                                }

                                let buffer_data_size = stream_sizes
                                    .entry(stream)
                                    .or_default()
                                    .entry(xfb_buffer)
                                    .or_insert(0);
                                *buffer_data_size = max(*buffer_data_size, required_size);
                            }

                            Ok(())
                        };

                        if let (Some(offset), Some(xfb_buffer)) = (offset, xfb_buffer) {
                            check(offset, xfb_buffer, stream)?;
                        } else if let Instruction::TypeStruct { .. } =
                            self.spirv.id(type_id).instruction()
                        {
                            for member_info in self.spirv.id(type_id).members() {
                                let mut member_offset = None;
                                let mut member_stream = None;
                                let mut member_xfb_buffer = None;

                                for instruction in member_info.decorations() {
                                    if let Instruction::Decorate { decoration, .. } = instruction {
                                        match *decoration {
                                            Decoration::Offset { byte_offset } => {
                                                member_offset = Some(byte_offset)
                                            }
                                            Decoration::Stream { stream_number } => {
                                                member_stream = Some(stream_number)
                                            }
                                            Decoration::XfbBuffer { xfb_buffer_number } => {
                                                member_xfb_buffer = Some(xfb_buffer_number)
                                            }
                                            _ => (),
                                        }
                                    }
                                }

                                // Inherit the XfbBuffer and Stream of the parent variable if there
                                // is one.
                                if let (Some(offset), Some(xfb_buffer)) =
                                    (member_offset, member_xfb_buffer.or(xfb_buffer))
                                {
                                    check(offset, xfb_buffer, member_stream.unwrap_or(stream))?;
                                }
                            }
                        }
                    }
                }

                _ => (),
            }

            let descriptor_type = match storage_class {
                StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer => {
                    match has_block {
                        true => Some(DescriptorType::StorageBuffer),
                        false => Some(DescriptorType::UniformBuffer),
                    }
                }
                StorageClass::Uniform => match has_buffer_block {
                    true => Some(DescriptorType::StorageBuffer),
                    false => Some(DescriptorType::UniformBuffer),
                },
                StorageClass::UniformConstant => {
                    let base_type = match *self.spirv.id(type_id).instruction() {
                        Instruction::TypeArray { element_type, .. } => element_type,
                        _ => type_id,
                    };

                    match *self.spirv.id(base_type).instruction() {
                        Instruction::TypeImage { dim, sampled, .. } => match dim {
                            Dim::Dim1D | Dim::Dim2D | Dim::Dim3D | Dim::Cube | Dim::Rect => {
                                if sampled == 1 {
                                    Some(DescriptorType::SampledImage)
                                } else {
                                    Some(DescriptorType::StorageImage)
                                }
                            }
                            Dim::Buffer => {
                                if sampled == 1 {
                                    Some(DescriptorType::UniformTexelBuffer)
                                } else {
                                    Some(DescriptorType::StorageTexelBuffer)
                                }
                            }
                            Dim::SubpassData => Some(DescriptorType::InputAttachment),
                            Dim::TileImageDataEXT => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            };

            if !has_non_writable
                && matches!(
                    descriptor_type,
                    Some(
                        DescriptorType::StorageImage
                            | DescriptorType::StorageTexelBuffer
                            | DescriptorType::StorageBuffer
                    )
                )
            {
                match self.execution_model {
                    ExecutionModel::Fragment => {
                        if !self.device.enabled_features().fragment_stores_and_atomics {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the entry point's execution model is `Fragment`, and \
                                    variable {} is a storage image, storage texel buffer or \
                                    storage buffer variable, and does not have a `NonWritable` \
                                    decoration",
                                    result_id,
                                )
                                .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature("fragment_stores_and_atomics"),
                                ])]),
                                vuids: &["VUID-RuntimeSpirv-NonWritable-06340"],
                                ..Default::default()
                            }));
                        }
                    }
                    ExecutionModel::Vertex
                    | ExecutionModel::TessellationControl
                    | ExecutionModel::TessellationEvaluation
                    | ExecutionModel::Geometry => {
                        if !self
                            .device
                            .enabled_features()
                            .vertex_pipeline_stores_and_atomics
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "the entry point's execution model is `Vertex`, \
                                    `TessellationControl`, `TessellationEvaluation` or \
                                    `Geometry`, and variable {} is a storage image, storage texel \
                                    buffer or storage buffer variable, and does not have a \
                                    `NonWritable` decoration",
                                    result_id,
                                )
                                .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature("vertex_pipeline_stores_and_atomics"),
                                ])]),
                                vuids: &["VUID-RuntimeSpirv-NonWritable-06341"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                }
            }

            if self.device.physical_device().api_version() < Version::V1_3
                && !self
                    .device
                    .physical_device()
                    .supported_extensions()
                    .khr_format_feature_flags2
            {
                let base_type = match *self.spirv.id(type_id).instruction() {
                    Instruction::TypeArray { element_type, .. } => element_type,
                    _ => type_id,
                };

                if matches!(
                    *self.spirv.id(base_type).instruction(),
                    Instruction::TypeImage {
                        sampled: 2,
                        image_format: ImageFormat::Unknown,
                        ..
                    }
                ) {
                    if !has_non_writable
                        && self
                            .device
                            .enabled_features()
                            .shader_storage_image_write_without_format
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} is an image with a `sampled` operand of 2, and an \
                                `image_format` operand of `ImageFormat::Unknown`, and does not \
                                have a `NonWritable` decoration",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[
                                RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                                RequiresAllOf(&[Requires::DeviceExtension(
                                    "khr_format_feature_flags2",
                                )]),
                                RequiresAllOf(&[Requires::DeviceFeature("storage_input_output16")]),
                            ]),
                            vuids: &["VUID-RuntimeSpirv-apiVersion-07954"],
                            ..Default::default()
                        }));
                    }

                    if !has_non_readable
                        && self
                            .device
                            .enabled_features()
                            .shader_storage_image_read_without_format
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "variable {} is an image with a `sampled` operand of 2, and an \
                                `image_format` operand of `ImageFormat::Unknown`, and does not \
                                have a `NonReadable` decoration",
                                result_id,
                            )
                            .into(),
                            requires_one_of: RequiresOneOf(&[
                                RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                                RequiresAllOf(&[Requires::DeviceExtension(
                                    "khr_format_feature_flags2",
                                )]),
                                RequiresAllOf(&[Requires::DeviceFeature("storage_input_output16")]),
                            ]),
                            vuids: &["VUID-RuntimeSpirv-apiVersion-07955"],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        for (stream, buffer_data_sizes) in stream_sizes {
            let required_size: DeviceSize = buffer_data_sizes.values().sum();

            if required_size
                > properties
                    .max_transform_feedback_buffer_data_size
                    .unwrap_or_default() as DeviceSize
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "the sum of the data sizes of all transform feedback buffers \
                        associated with stream {} is greater than the \
                        `max_transform_feedback_buffer_data_size` device limit",
                        stream,
                    )
                    .into(),
                    vuids: &["VUID-RuntimeSpirv-XfbBuffer-06309"],
                    ..Default::default()
                }));
            }
        }

        match self.execution_model {
            ExecutionModel::Vertex => {
                if input_locations_required > properties.max_vertex_input_attributes {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `Vertex`, but \
                            the number of input locations required is greater than the \
                            `max_vertex_input_attributes` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if output_locations_required > properties.max_vertex_output_components / 4 {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `Vertex`, but \
                            the number of output components required is greater than the \
                            `max_vertex_output_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }
            }
            ExecutionModel::TessellationControl => {
                if input_locations_required
                    > properties.max_tessellation_control_per_vertex_input_components / 4
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TessellationControl`, but \
                            the number of input components required is greater than the \
                            `max_tessellation_control_per_vertex_input_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if output_locations_required
                    > properties.max_tessellation_control_per_vertex_output_components / 4
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TessellationControl`, but \
                            the number of per-vertex output components required is greater than \
                            the `max_tessellation_control_per_vertex_output_components` device \
                            limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if per_patch_output_locations_required
                    > properties.max_tessellation_control_per_patch_output_components / 4
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TessellationControl`, but \
                            the number of per-patch output components required is greater than \
                            the `max_tessellation_control_per_patch_output_components` device \
                            limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if max(
                    output_locations_required,
                    per_patch_output_locations_required,
                ) > properties.max_tessellation_control_total_output_components / 4
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TessellationControl`, but \
                            the number of output components required is greater than the \
                            `max_tessellation_control_total_output_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }
            }
            ExecutionModel::TessellationEvaluation => {
                if input_locations_required
                    > properties.max_tessellation_evaluation_input_components / 4
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TessellationEvaluation`, \
                            but the number of input components required is greater than the \
                            `max_tessellation_evaluation_input_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if output_locations_required
                    > properties.max_tessellation_evaluation_output_components / 4
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TessellationEvaluation`, \
                            but the number of output components required is greater than the \
                            `max_tessellation_evaluation_output_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }
            }
            ExecutionModel::Geometry => {
                if input_locations_required > properties.max_geometry_input_components / 4 {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `Geometry`, but \
                            the number of input components required is greater than the \
                            `max_geometry_input_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if output_locations_required > properties.max_geometry_output_components / 4 {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `Geometry`, but \
                            the number of output components required is greater than the \
                            `max_geometry_output_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                // TODO: max_geometry_total_output_components
            }
            ExecutionModel::Fragment => {
                if input_locations_required > properties.max_fragment_input_components / 4 {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `Fragment`, but \
                            the number of input components required is greater than the \
                            `max_fragment_input_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if output_locations_required > properties.max_fragment_output_attachments {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `Fragment`, but \
                            the number of output locations required is greater than the \
                            `max_fragment_output_attachments` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }
            }
            ExecutionModel::GLCompute => {
                if workgroup_memory_size > properties.max_compute_shared_memory_size as u64 {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `GLCompute`, but \
                            the total size of all variables in the `Workgroup` storage class is \
                            greater than the `max_compute_shared_memory_size` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Workgroup-06530"],
                        ..Default::default()
                    }));
                }
            }
            ExecutionModel::MeshEXT => {
                if output_locations_required
                    > properties.max_mesh_output_components.unwrap_or_default() / 4
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `MeshEXT`, but \
                            the number of output components required is greater than the \
                            `max_mesh_output_components` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-Location-06272"],
                        ..Default::default()
                    }));
                }

                if workgroup_memory_size
                    > properties.max_mesh_shared_memory_size.unwrap_or_default() as u64
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `MeshEXT`, but \
                            the total size of all variables in the `Workgroup` storage class is \
                            greater than the `max_mesh_shared_memory_size` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-maxMeshSharedMemorySize-08754"],
                        ..Default::default()
                    }));
                }

                if task_payload_workgroup_memory_size + workgroup_memory_size
                    > properties
                        .max_mesh_payload_and_shared_memory_size
                        .unwrap_or_default() as u64
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `MeshEXT`, but \
                            the total size of all variables in the `TaskPayloadWorkGroupEXT` or \
                            `Workgroup` storage classes is greater than the \
                            `max_mesh_payload_and_shared_memory_size` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-maxMeshPayloadAndSharedMemorySize-08755"],
                        ..Default::default()
                    }));
                }
            }
            ExecutionModel::TaskEXT => {
                if task_payload_workgroup_memory_size
                    > properties.max_task_payload_size.unwrap_or_default() as u64
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TaskEXT`, but \
                            the total size of all variables in the `TaskPayloadWorkgroupEXT` \
                            storage class is greater than the `max_task_payload_size` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-maxTaskPayloadSize-08758"],
                        ..Default::default()
                    }));
                }

                if workgroup_memory_size
                    > properties.max_task_shared_memory_size.unwrap_or_default() as u64
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TaskEXT`, but \
                            the total size of all variables in the `Workgroup` storage class is \
                            greater than the `max_task_shared_memory_size` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-maxTaskSharedMemorySize-08759"],
                        ..Default::default()
                    }));
                }

                if task_payload_workgroup_memory_size + workgroup_memory_size
                    > properties
                        .max_task_payload_and_shared_memory_size
                        .unwrap_or_default() as u64
                {
                    return Err(Box::new(ValidationError {
                        problem: "the entry point's execution model is `TaskEXT`, but \
                            the total size of all variables in the `TaskPayloadWorkgroupEXT` or \
                            `Workgroup` storage class is greater than the \
                            `max_task_payload_and_shared_memory_size` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-maxTaskPayloadAndSharedMemorySize-08760"],
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        Ok(())
    }

    fn validate_functions(&mut self) -> Result<(), Box<ValidationError>> {
        for &function in self.spirv.functions().keys() {
            self.validate_function(function)?;
        }

        Ok(())
    }

    fn validate_function(&mut self, function: Id) -> Result<(), Box<ValidationError>> {
        let properties = self.device.physical_device().properties();

        for instruction in self.spirv.function(function).instructions() {
            if let Some(pointer) = instruction.atomic_pointer_id() {
                let (storage_class, ty) = match self
                    .spirv
                    .id(pointer)
                    .instruction()
                    .result_type_id()
                    .map(|id| self.spirv.id(id).instruction())
                {
                    Some(&Instruction::TypePointer {
                        storage_class, ty, ..
                    }) => (storage_class, ty),
                    _ => unreachable!(),
                };

                match *self.spirv.id(ty).instruction() {
                    Instruction::TypeInt { width: 64, .. } => match storage_class {
                        StorageClass::StorageBuffer | StorageClass::Uniform => {
                            if !self.device.enabled_features().shader_buffer_int64_atomics {
                                return Err(Box::new(ValidationError {
                                    problem: "an atomic operation is performed on a \
                                        64-bit integer value with a storage class of \
                                        `StorageClass::StorageBuffer` or `StorageClass::Uniform`"
                                        .into(),
                                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                        Requires::DeviceFeature("shader_buffer_int64_atomics"),
                                    ])]),
                                    vuids: &["VUID-RuntimeSpirv-None-06278"],
                                    ..Default::default()
                                }));
                            }
                        }
                        StorageClass::Workgroup => {
                            if !self.device.enabled_features().shader_shared_int64_atomics {
                                return Err(Box::new(ValidationError {
                                    problem: "an atomic operation is performed on a \
                                        64-bit integer value with a storage class of \
                                        `StorageClass::Workgroup`"
                                        .into(),
                                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                        Requires::DeviceFeature("shader_shared_int64_atomics"),
                                    ])]),
                                    vuids: &["VUID-RuntimeSpirv-None-06279"],
                                    ..Default::default()
                                }));
                            }
                        }
                        StorageClass::Image => {
                            if !self.device.enabled_features().shader_image_int64_atomics {
                                return Err(Box::new(ValidationError {
                                    problem: "an atomic operation is performed on a \
                                        64-bit integer value with a storage class of \
                                        `StorageClass::Image`"
                                        .into(),
                                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                        Requires::DeviceFeature("shader_image_int64_atomics"),
                                    ])]),
                                    vuids: &["VUID-RuntimeSpirv-None-06288"],
                                    ..Default::default()
                                }));
                            }
                        }
                        _ => (),
                    },
                    Instruction::TypeFloat { width, .. } => {
                        match width {
                            16 => {
                                if !self.device.enabled_features().intersects(&DeviceFeatures {
                                    shader_buffer_float16_atomics: true,
                                    shader_buffer_float16_atomic_add: true,
                                    shader_buffer_float16_atomic_min_max: true,
                                    shader_shared_float16_atomics: true,
                                    shader_shared_float16_atomic_add: true,
                                    shader_shared_float16_atomic_min_max: true,
                                    ..DeviceFeatures::empty()
                                }) {
                                    return Err(Box::new(ValidationError {
                                        problem: "an atomic operation is performed on a \
                                            16-bit floating point value"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float16_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float16_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float16_atomic_min_max",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float16_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float16_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float16_atomic_min_max",
                                            )]),
                                        ]),
                                        vuids: &["VUID-RuntimeSpirv-None-06337"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            32 => {
                                if !self.device.enabled_features().intersects(&DeviceFeatures {
                                    shader_buffer_float32_atomics: true,
                                    shader_buffer_float32_atomic_add: true,
                                    shader_buffer_float32_atomic_min_max: true,
                                    shader_shared_float32_atomics: true,
                                    shader_shared_float32_atomic_add: true,
                                    shader_shared_float32_atomic_min_max: true,
                                    ..DeviceFeatures::empty()
                                }) {
                                    return Err(Box::new(ValidationError {
                                        problem: "an atomic operation is performed on a \
                                            32-bit floating point value"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float32_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float32_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float32_atomic_min_max",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float32_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float32_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float32_atomic_min_max",
                                            )]),
                                        ]),
                                        vuids: &["VUID-RuntimeSpirv-None-06338"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            64 => {
                                if !self.device.enabled_features().intersects(&DeviceFeatures {
                                    shader_buffer_float64_atomics: true,
                                    shader_buffer_float64_atomic_add: true,
                                    shader_buffer_float64_atomic_min_max: true,
                                    shader_shared_float64_atomics: true,
                                    shader_shared_float64_atomic_add: true,
                                    shader_shared_float64_atomic_min_max: true,
                                    ..DeviceFeatures::empty()
                                }) {
                                    return Err(Box::new(ValidationError {
                                        problem: "an atomic operation is performed on a \
                                            64-bit floating point value"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float64_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float64_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float64_atomic_min_max",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float64_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float64_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float64_atomic_min_max",
                                            )]),
                                        ]),
                                        vuids: &["VUID-RuntimeSpirv-None-06339"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            _ => (),
                        }

                        match storage_class {
                            StorageClass::StorageBuffer => {
                                if !self.device.enabled_features().intersects(&DeviceFeatures {
                                    shader_buffer_float16_atomics: true,
                                    shader_buffer_float16_atomic_add: true,
                                    shader_buffer_float16_atomic_min_max: true,
                                    shader_buffer_float32_atomics: true,
                                    shader_buffer_float32_atomic_add: true,
                                    shader_buffer_float32_atomic_min_max: true,
                                    shader_buffer_float64_atomics: true,
                                    shader_buffer_float64_atomic_add: true,
                                    shader_buffer_float64_atomic_min_max: true,
                                    ..DeviceFeatures::empty()
                                }) {
                                    return Err(Box::new(ValidationError {
                                        problem: "an atomic operation is performed on a \
                                            floating point value with a storage class of \
                                            `StorageClass::StorageBuffer`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float16_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float16_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float16_atomic_min_max",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float32_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float32_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float32_atomic_min_max",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float64_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float64_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_buffer_float64_atomic_min_max",
                                            )]),
                                        ]),
                                        vuids: &["VUID-RuntimeSpirv-None-06284"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            StorageClass::Workgroup => {
                                if !self.device.enabled_features().intersects(&DeviceFeatures {
                                    shader_shared_float16_atomics: true,
                                    shader_shared_float16_atomic_add: true,
                                    shader_shared_float16_atomic_min_max: true,
                                    shader_shared_float32_atomics: true,
                                    shader_shared_float32_atomic_add: true,
                                    shader_shared_float32_atomic_min_max: true,
                                    shader_shared_float64_atomics: true,
                                    shader_shared_float64_atomic_add: true,
                                    shader_shared_float64_atomic_min_max: true,
                                    ..DeviceFeatures::empty()
                                }) {
                                    return Err(Box::new(ValidationError {
                                        problem: "an atomic operation is performed on a \
                                            floating point value with a storage class of \
                                            `StorageClass::Workgroup`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float16_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float16_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float16_atomic_min_max",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float32_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float32_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float32_atomic_min_max",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float64_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float64_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_shared_float64_atomic_min_max",
                                            )]),
                                        ]),
                                        vuids: &["VUID-RuntimeSpirv-None-06285"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            StorageClass::Image => {
                                if width == 32
                                    && !self.device.enabled_features().intersects(&DeviceFeatures {
                                        shader_image_float32_atomics: true,
                                        shader_image_float32_atomic_add: true,
                                        shader_image_float32_atomic_min_max: true,
                                        ..DeviceFeatures::empty()
                                    })
                                {
                                    return Err(Box::new(ValidationError {
                                        problem: "an atomic operation is performed on a \
                                            32-bit floating point value with a storage \
                                            class of `StorageClass::Image`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_image_float32_atomics",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_image_float32_atomic_add",
                                            )]),
                                            RequiresAllOf(&[Requires::DeviceFeature(
                                                "shader_image_float32_atomic_min_max",
                                            )]),
                                        ]),
                                        vuids: &["VUID-RuntimeSpirv-None-06286"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                }
            }

            if instruction.is_cooperative_matrix() {
                if !properties
                    .cooperative_matrix_supported_stages
                    .unwrap()
                    .contains_enum(ShaderStage::from(self.execution_model))
                {
                    return Err(Box::new(ValidationError {
                        problem: "a cooperative matrix operation is performed, but \
                            the `cooperative_matrix_supported_stages` device property does not \
                            contain the shader stage of the entry point's execution model"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-cooperativeMatrixSupportedStages-08985"],
                        ..Default::default()
                    }));
                }
            } else if instruction.is_cooperative_matrix_nv() {
                // TODO: needs VK_NV_cooperative_matrix support
                // VUID-RuntimeSpirv-OpTypeCooperativeMatrixNV-06322
                // OpTypeCooperativeMatrixNV and OpCooperativeMatrix* instructions
                // must not be used in shader stages not included in
                // VkPhysicalDeviceCooperativeMatrixPropertiesNV::cooperativeMatrixSupportedStages
            }

            if let Some(scope) = instruction
                .memory_scope_id()
                .and_then(|scope| get_constant(self.spirv, scope))
                .and_then(|scope| Scope::try_from(scope as u32).ok())
            {
                match scope {
                    Scope::Device => {
                        if self.device.enabled_features().vulkan_memory_model
                            && !self
                                .device
                                .enabled_features()
                                .vulkan_memory_model_device_scope
                        {
                            return Err(Box::new(ValidationError {
                                problem: "an instruction uses `Device` as the memory scope, and \
                                    the `vulkan_memory_model` feature is enabled"
                                    .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature("vulkan_memory_model_device_scope"),
                                ])]),
                                vuids: &["VUID-RuntimeSpirv-vulkanMemoryModel-06265"],
                                ..Default::default()
                            }));
                        }
                    }
                    Scope::QueueFamily => {
                        if !self.device.enabled_features().vulkan_memory_model {
                            return Err(Box::new(ValidationError {
                                problem: "an instruction uses `QueueFamily` as the memory scope"
                                    .into(),
                                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                    Requires::DeviceFeature("vulkan_memory_model"),
                                ])]),
                                vuids: &["VUID-RuntimeSpirv-vulkanMemoryModel-06266"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                }
            }

            if instruction.is_group_operation() {
                if !properties
                    .subgroup_supported_stages
                    .unwrap_or_default()
                    .contains_enum(ShaderStage::from(self.execution_model))
                {
                    let execution_scope = if let Some(scope) = instruction.execution_scope_id() {
                        get_constant(self.spirv, scope)
                            .and_then(|scope| Scope::try_from(scope as u32).ok())
                    } else if matches!(instruction, Instruction::GroupNonUniformPartitionEXT { .. })
                    {
                        Some(Scope::Subgroup)
                    } else {
                        todo!(
                            "Encountered an unknown group instruction without an `execution` \
                            operand. This is a Vulkano bug and should be reported.\n\
                            Instruction::{:?}",
                            instruction
                        )
                    };

                    if let Some(scope) = execution_scope {
                        if scope == Scope::Subgroup {
                            return Err(Box::new(ValidationError {
                                problem: "a group operation instruction is performed \
                                with an execution scope of `Scope::Subgroup`, but \
                                the `subgroup_supported_stages` device property does not contain \
                                the shader stage of the entry point's execution model"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-None-06343"],
                                ..Default::default()
                            }));
                        }
                    }
                }

                if !self
                    .device
                    .enabled_features()
                    .shader_subgroup_extended_types
                {
                    if let Some(mut result_type_id) = instruction.result_type_id() {
                        if let Instruction::TypeVector { component_type, .. } =
                            *self.spirv.id(result_type_id).instruction()
                        {
                            result_type_id = component_type;
                        }

                        match *self.spirv.id(result_type_id).instruction() {
                            Instruction::TypeInt { width, .. } => match width {
                                8 => {
                                    return Err(Box::new(ValidationError {
                                        problem: "a group operation instruction is performed \
                                            on an 8-bit integer or vector"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature(
                                                "shader_subgroup_extended_types",
                                            ),
                                        ])]),
                                        vuids: &["VUID-RuntimeSpirv-None-06275"],
                                        ..Default::default()
                                    }));
                                }
                                16 => {
                                    return Err(Box::new(ValidationError {
                                        problem: "a group operation instruction is performed \
                                            on a 16-bit integer or vector"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature(
                                                "shader_subgroup_extended_types",
                                            ),
                                        ])]),
                                        vuids: &["VUID-RuntimeSpirv-None-06275"],
                                        ..Default::default()
                                    }));
                                }
                                64 => {
                                    return Err(Box::new(ValidationError {
                                        problem: "a group operation instruction is performed \
                                            on an 64-bit integer or vector"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature(
                                                "shader_subgroup_extended_types",
                                            ),
                                        ])]),
                                        vuids: &["VUID-RuntimeSpirv-None-06275"],
                                        ..Default::default()
                                    }));
                                }
                                _ => (),
                            },
                            Instruction::TypeFloat { width: 16, .. } => {
                                return Err(Box::new(ValidationError {
                                    problem: "a group operation instruction is performed \
                                        on an 16-bit floating point scalar or vector"
                                        .into(),
                                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                        Requires::DeviceFeature("shader_subgroup_extended_types"),
                                    ])]),
                                    vuids: &["VUID-RuntimeSpirv-None-06275"],
                                    ..Default::default()
                                }));
                            }
                            _ => (),
                        }
                    }
                }

                if instruction.is_quad_group_operation()
                    && !properties
                        .subgroup_quad_operations_in_all_stages
                        .unwrap_or_default()
                {
                    return Err(Box::new(ValidationError {
                        problem: "a quad group operation instruction is performed, and \
                            the `subgroup_quad_operations_in_all_stages` device property is \
                            `false`, but entry point's execution model is not `Fragment` or \
                            `GLCompute`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-None-06342"],
                        ..Default::default()
                    }));
                }
            }

            if instruction.is_image_gather() {
                if let Some(image_operands) = instruction.image_operands() {
                    if let Some(components) = image_operands
                        .const_offset
                        .or(image_operands.offset)
                        .and_then(|offset| get_constant_signed_maybe_composite(self.spirv, offset))
                    {
                        for offset in components {
                            if offset < properties.min_texel_gather_offset as i64 {
                                return Err(Box::new(ValidationError {
                                    problem: "an `OpImage*Gather` instruction is performed, but \
                                        its `Offset`, `ConstOffset` or `ConstOffsets` \
                                        image operand contains a value that is less than the \
                                        `min_texel_gather_offset` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-OpImage-06376"],
                                    ..Default::default()
                                }));
                            }

                            if offset > properties.max_texel_gather_offset as i64 {
                                return Err(Box::new(ValidationError {
                                    problem: "an `OpImage*Gather` instruction is performed, but \
                                        its `Offset`, `ConstOffset` or `ConstOffsets` \
                                        image operand contains a value that is greater than the \
                                        `max_texel_gather_offset` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-OpImage-06377"],
                                    ..Default::default()
                                }));
                            }
                        }
                    } else if let Some(elements) = image_operands
                        .const_offsets
                        .and_then(|id| get_constant_signed_composite_composite(self.spirv, id))
                    {
                        for components in elements {
                            for offset in components {
                                if offset < properties.min_texel_gather_offset as i64 {
                                    return Err(Box::new(ValidationError {
                                        problem: "an `OpImage*Gather` instruction is performed, \
                                            but its `Offset`, `ConstOffset` or `ConstOffsets` \
                                            image operand contains a value that is less than \
                                            the `min_texel_gather_offset` device limit"
                                            .into(),
                                        vuids: &["VUID-RuntimeSpirv-OpImage-06376"],
                                        ..Default::default()
                                    }));
                                }

                                if offset > properties.max_texel_gather_offset as i64 {
                                    return Err(Box::new(ValidationError {
                                        problem: "an `OpImage*Gather` instruction is performed, \
                                            but its `Offset`, `ConstOffset` or `ConstOffsets` \
                                            image operand contains a value that is greater than \
                                            the `max_texel_gather_offset` device limit"
                                            .into(),
                                        vuids: &["VUID-RuntimeSpirv-OpImage-06377"],
                                        ..Default::default()
                                    }));
                                }
                            }
                        }
                    }
                }
            }

            if instruction.is_image_sample() || instruction.is_image_fetch() {
                if let Some(image_operands) = instruction.image_operands() {
                    if let Some(components) = image_operands
                        .const_offset
                        .and_then(|offset| get_constant_signed_maybe_composite(self.spirv, offset))
                    {
                        for offset in components {
                            if offset < properties.min_texel_offset as i64 {
                                return Err(Box::new(ValidationError {
                                    problem: "an `OpImageSample*` or `OpImageFetch*` instruction \
                                        is performed, but its `ConstOffset` image operand \
                                        contains a value that is less than the \
                                        `min_texel_offset` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-OpImageSample-06435"],
                                    ..Default::default()
                                }));
                            }

                            if offset > properties.max_texel_offset as i64 {
                                return Err(Box::new(ValidationError {
                                    problem: "an `OpImageSample*` or `OpImageFetch*` instruction \
                                        is performed, but its `ConstOffset` image operand \
                                        contains a value that is greater than the \
                                        `max_texel_offset` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-OpImageSample-06436"],
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            }

            match *instruction {
                Instruction::ReadClockKHR { scope, .. } => {
                    let scope = get_constant(self.spirv, scope)
                        .and_then(|scope| Scope::try_from(scope as u32).ok());

                    if let Some(scope) = scope {
                        match scope {
                            Scope::Subgroup => {
                                if !self.device.enabled_features().shader_subgroup_clock {
                                    return Err(Box::new(ValidationError {
                                        problem: "an `OpReadClockKHR` instruction is performed \
                                            with a scope of `Scope::Subgroup`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature("shader_subgroup_clock"),
                                        ])]),
                                        vuids: &["VUID-RuntimeSpirv-shaderSubgroupClock-06267"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            Scope::Device => {
                                if !self.device.enabled_features().shader_device_clock {
                                    return Err(Box::new(ValidationError {
                                        problem: "an `OpReadClockKHR` instruction is performed \
                                        with a scope of `Scope::Device`"
                                            .into(),
                                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                            Requires::DeviceFeature("shader_device_clock"),
                                        ])]),
                                        vuids: &["VUID-RuntimeSpirv-shaderDeviceClock-06268"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            _ => (),
                        }
                    }
                }
                Instruction::GroupNonUniformQuadBroadcast { index, .. } => {
                    if !self.device.enabled_features().subgroup_broadcast_dynamic_id
                        && !matches!(
                            self.spirv.id(index).instruction(),
                            Instruction::Constant { .. }
                        )
                    {
                        return Err(Box::new(ValidationError {
                            problem: "an `OpGroupNonUniformQuadBroadcast` instruction is \
                                performed, and its `index` operand is not a constant"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("subgroup_broadcast_dynamic_id"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-subgroupBroadcastDynamicId-06276"],
                            ..Default::default()
                        }));
                    }
                }
                Instruction::GroupNonUniformBroadcast { invocation_id, .. } => {
                    if !self.device.enabled_features().subgroup_broadcast_dynamic_id
                        && !matches!(
                            self.spirv.id(invocation_id).instruction(),
                            Instruction::Constant { .. }
                        )
                    {
                        return Err(Box::new(ValidationError {
                            problem: "an `OpGroupNonUniformBroadcast` instruction is \
                                performed, and its `id` operand is not a constant"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("subgroup_broadcast_dynamic_id"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-subgroupBroadcastDynamicId-06277"],
                            ..Default::default()
                        }));
                    }
                }
                Instruction::CooperativeMatrixMulAddNV { .. } => {
                    // TODO: needs VK_NV_cooperative_matrix support
                    // VUID-RuntimeSpirv-OpCooperativeMatrixMulAddNV-06317
                    // VUID-RuntimeSpirv-OpCooperativeMatrixMulAddNV-06318
                    // VUID-RuntimeSpirv-OpCooperativeMatrixMulAddNV-06319
                    // VUID-RuntimeSpirv-OpCooperativeMatrixMulAddNV-06320
                    // VUID-RuntimeSpirv-OpCooperativeMatrixMulAddNV-06321
                }
                Instruction::CooperativeMatrixMulAddKHR { .. } => {
                    // TODO: needs VK_KHR_cooperative_matrix support
                    // VUID-RuntimeSpirv-MSize-08975
                    // VUID-RuntimeSpirv-KSize-08977
                    // VUID-RuntimeSpirv-MSize-08979
                    // VUID-RuntimeSpirv-MSize-08981
                    // VUID-RuntimeSpirv-saturatingAccumulation-08983
                }
                Instruction::EmitStreamVertex { stream }
                | Instruction::EndStreamPrimitive { stream } => {
                    let stream = get_constant(self.spirv, stream).unwrap();

                    if u32::try_from(stream).map_or(true, |stream| {
                        stream
                            >= properties
                                .max_transform_feedback_streams
                                .unwrap_or_default()
                    }) {
                        return Err(Box::new(ValidationError {
                            problem: "an `OpEmitStreamVertex` or `OpEndStreamPrimitive` \
                                instruction is performed, but the value of the `stream` operand \
                                is not less than the `max_transform_feedback_streams` device limit"
                                .into(),
                            vuids: &["VUID-RuntimeSpirv-OpEmitStreamVertex-06310"],
                            ..Default::default()
                        }));
                    }

                    if let (&Instruction::EmitStreamVertex { stream }, false) = (
                        instruction,
                        matches!(self.output_primitives, Some(ExecutionMode::OutputPoints)),
                    ) {
                        let stream = get_constant(self.spirv, stream).unwrap() as u32;

                        match self.first_emitted_stream {
                            Some(first_emitted_stream) => {
                                if stream != first_emitted_stream
                                    && !properties
                                        .transform_feedback_streams_lines_triangles
                                        .unwrap_or_default()
                                {
                                    return Err(Box::new(ValidationError {
                                        problem: "the shader emits to more than one vertex \
                                            stream, and the entry point does not have an \
                                            `OutputPoints` execution mode, but the \
                                            `transform_feedback_streams_lines_triangles` device \
                                            property is `false`".into(),
                                        vuids: &["VUID-RuntimeSpirv-transformFeedbackStreamsLinesTriangles-06311"],
                                        ..Default::default()
                                    }));
                                }
                            }
                            None => self.first_emitted_stream = Some(stream),
                        }
                    }
                }
                Instruction::ImageBoxFilterQCOM { box_size, .. } => {
                    if let Some(box_size) = get_constant_float_composite(self.spirv, box_size) {
                        if box_size[1]
                            > properties.max_box_filter_block_size.unwrap_or_default()[1] as f64
                        {
                            return Err(Box::new(ValidationError {
                                problem: "an `OpImageBoxFilterQCOM` instruction is performed, but \
                                    the `y` component of the `box_size` operand is greater than \
                                    the `max_box_filter_block_size[1]` device limit"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-OpImageBoxFilterQCOM-06989"],
                                ..Default::default()
                            }));
                        }
                    }
                }
                Instruction::EmitMeshTasksEXT {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    ..
                } => {
                    // TODO: If the shader has multiple entry points with different execution
                    // models, then we really need to use the entry point's call
                    // tree, instead of a flat iteration over all functions.
                    if self.execution_model == ExecutionModel::MeshEXT {
                        let group_count_x = get_constant(self.spirv, group_count_x);
                        let group_count_y = get_constant(self.spirv, group_count_y);
                        let group_count_z = get_constant(self.spirv, group_count_z);
                        let mut product: Option<u32> = Some(1);

                        if let Some(count) = group_count_x {
                            product = product
                                .zip(count.try_into().ok())
                                .and_then(|(product, count)| product.checked_mul(count));

                            if u32::try_from(count).map_or(true, |count| {
                                count > properties.max_mesh_work_group_count.unwrap_or_default()[0]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `TaskEXT`, and \
                                        an `OpEmitMeshTasksEXT` instruction is performed, but \
                                        the value of the `group_count_x` operand is greater than \
                                        the `max_mesh_work_group_count[0]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-TaskEXT-07299"],
                                    ..Default::default()
                                }));
                            }
                        }

                        if let Some(count) = group_count_y {
                            product = product
                                .zip(count.try_into().ok())
                                .and_then(|(product, count)| product.checked_mul(count));

                            if u32::try_from(count).map_or(true, |count| {
                                count > properties.max_mesh_work_group_count.unwrap_or_default()[1]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `TaskEXT`, and \
                                        an `OpEmitMeshTasksEXT` instruction is performed, but \
                                        the value of the `group_count_y` operand is greater than \
                                        the `max_mesh_work_group_count[1]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-TaskEXT-07300"],
                                    ..Default::default()
                                }));
                            }
                        }

                        if let Some(count) = group_count_z {
                            product = product
                                .zip(count.try_into().ok())
                                .and_then(|(product, count)| product.checked_mul(count));

                            if u32::try_from(count).map_or(true, |count| {
                                count > properties.max_mesh_work_group_count.unwrap_or_default()[2]
                            }) {
                                return Err(Box::new(ValidationError {
                                    problem: "the entry point's execution model is `TaskEXT`, and \
                                        an `OpEmitMeshTasksEXT` instruction is performed, but \
                                        the value of the `group_count_z` operand is greater than \
                                        the `max_mesh_work_group_count[2]` device limit"
                                        .into(),
                                    vuids: &["VUID-RuntimeSpirv-TaskEXT-07301"],
                                    ..Default::default()
                                }));
                            }
                        }

                        if product.is_none_or(|product| {
                            product
                                > properties
                                    .max_mesh_work_group_total_count
                                    .unwrap_or_default()
                        }) {
                            return Err(Box::new(ValidationError {
                                problem: "the entry point's execution model is `TaskEXT`, and \
                                    an `OpEmitMeshTasksEXT` instruction is performed, but \
                                    the product of its `group_count_x`, `group_count_y` and \
                                    `group_count_z` operands is greater than the \
                                    `max_mesh_work_group_total_count` device limit"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-TaskEXT-07302"],
                                ..Default::default()
                            }));
                        }
                    }
                }
                Instruction::ColorAttachmentReadEXT { .. } => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_tile_image_color_read_access
                    {
                        return Err(Box::new(ValidationError {
                            problem: "an `OpColorAttachmentReadEXT` instruction is performed"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_tile_image_color_read_access"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderTileImageColorReadAccess-08728"],
                            ..Default::default()
                        }));
                    }
                }
                Instruction::DepthAttachmentReadEXT { .. } => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_tile_image_depth_read_access
                    {
                        return Err(Box::new(ValidationError {
                            problem: "an `OpDepthAttachmentReadEXT` instruction is performed"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_tile_image_depth_read_access"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderTileImageDepthReadAccess-08729"],
                            ..Default::default()
                        }));
                    }
                }
                Instruction::StencilAttachmentReadEXT { .. } => {
                    if !self
                        .device
                        .enabled_features()
                        .shader_tile_image_stencil_read_access
                    {
                        return Err(Box::new(ValidationError {
                            problem: "an `OpStencilAttachmentReadEXT` instruction is performed"
                                .into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                                Requires::DeviceFeature("shader_tile_image_stencil_read_access"),
                            ])]),
                            vuids: &["VUID-RuntimeSpirv-shaderTileImageStencilReadAccess-08730"],
                            ..Default::default()
                        }));
                    }
                }
                Instruction::ImageBlockMatchSADQCOM { block_size, .. }
                | Instruction::ImageBlockMatchSSDQCOM { block_size, .. } => {
                    let block_size = get_constant_composite(self.spirv, block_size);

                    if let Some(block_size) = block_size {
                        let max_block_match_region =
                            properties.max_block_match_region.unwrap_or_default();

                        if block_size[0] > max_block_match_region[0] as u64 {
                            return Err(Box::new(ValidationError {
                                problem: "an `OpImageBlockMatchSADQCOM` or \
                                    `OpImageBlockMatchSSDQCOM` instruction is performed, but \
                                    the `x` component of the `block_size` operand is greater than \
                                    the `max_block_match_region[0]` device limit"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-maxBlockMatchRegion-09225"],
                                ..Default::default()
                            }));
                        }

                        if block_size[1] > max_block_match_region[1] as u64 {
                            return Err(Box::new(ValidationError {
                                problem: "an `OpImageBlockMatchSADQCOM` or \
                                    `OpImageBlockMatchSSDQCOM` instruction is performed, but \
                                    the `y` component of the `block_size` operand is greater than \
                                    the `max_block_match_region[1]` device limit"
                                    .into(),
                                vuids: &["VUID-RuntimeSpirv-maxBlockMatchRegion-09225"],
                                ..Default::default()
                            }));
                        }
                    }
                }
                _ => (),
            }
        }

        Ok(())
    }
}

// TODO: spec clarification

// VUID-RuntimeSpirv-maxMeshOutputMemorySize-08756
// VUID-RuntimeSpirv-maxMeshPayloadAndOutputMemorySize-08757
// VUID-RuntimeSpirv-Location-06428
// VUID-RuntimeSpirv-maxExecutionGraphShaderPayloadSize-09193
// VUID-RuntimeSpirv-maxExecutionGraphShaderPayloadSize-09194
// VUID-RuntimeSpirv-maxExecutionGraphShaderPayloadSize-09195
// VUID-RuntimeSpirv-maxExecutionGraphShaderPayloadCount-09196
// VUID-RuntimeSpirv-maxExecutionGraphShaderOutputNodes-09197

// TODO: depends on descriptor resources

// VUID-RuntimeSpirv-None-06287
// VUID-RuntimeSpirv-OpEntryPoint-08727

// TODO: requires items that are not implemented

// VUID-RuntimeSpirv-OpTraceRayMotionNV-06367
// VUID-RuntimeSpirv-OpHitObjectTraceRayMotionNV-07711
// VUID-RuntimeSpirv-OpHitObjectTraceRayMotionNV-07704
// VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07715
// VUID-RuntimeSpirv-OpHitObjectTraceRayNV-07716
// VUID-RuntimeSpirv-flags-08761
// VUID-RuntimeSpirv-OpImageBlockMatchSSDQCOM-06985
// VUID-RuntimeSpirv-OpImageBlockMatchSSDQCOM-06986
// VUID-RuntimeSpirv-OpImageBlockMatchSSDQCOM-06987
// VUID-RuntimeSpirv-OpImageBlockMatchSSDQCOM-06988
// VUID-RuntimeSpirv-OpImageBlockMatchWindow-09223
// VUID-RuntimeSpirv-OpImageBlockMatchWindow-09224
// VUID-RuntimeSpirv-pNext-09226
// VUID-RuntimeSpirv-minSampleShading-08731
// VUID-RuntimeSpirv-minSampleShading-08732
