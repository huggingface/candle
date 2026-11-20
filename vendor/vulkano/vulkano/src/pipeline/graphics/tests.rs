//! This module contains the unit tests of `GraphicsPipeline`.

#![cfg(test)]

use format::Format;
use framebuffer::Subpass;
use pipeline::blend::Blend;
use pipeline::depth_stencil::DepthStencil;
use pipeline::input_assembly::InputAssembly;
use pipeline::input_assembly::PrimitiveTopology;
use pipeline::layout::PipelineLayoutDesc;
use pipeline::multisample::Multisample;
use pipeline::shader::ShaderInterface;
use pipeline::shader::ShaderModule;
use pipeline::vertex::SingleBufferDefinition;
use pipeline::viewport::ViewportsState;
use pipeline::GraphicsPipeline;
use pipeline::GraphicsPipelineCreationError;
use pipeline::GraphicsPipelineParams;
use std::ffi::CString;

#[test]
fn create() {
    let (device, _) = gfx_dev_and_queue!();

    let vs = unsafe { ShaderModule::new(device.clone(), &BASIC_VS).unwrap() };
    let fs = unsafe { ShaderModule::new(device.clone(), &BASIC_FS).unwrap() };

    let _ = GraphicsPipeline::new(
        &device,
        GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 1 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            render_pass: Subpass::from(
                simple_rp::CustomRenderPass::new(&device, &{
                    simple_rp::Formats {
                        color: (Format::R8G8B8A8_UNORM, 1),
                    }
                })
                .unwrap(),
                0,
            )
            .unwrap(),
        },
    )
    .unwrap();
}

#[test]
fn bad_primitive_restart() {
    let (device, _) = gfx_dev_and_queue!();

    let vs = unsafe { ShaderModule::new(device.clone(), &BASIC_VS).unwrap() };
    let fs = unsafe { ShaderModule::new(device.clone(), &BASIC_FS).unwrap() };

    let result = GraphicsPipeline::new(
        &device,
        GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            input_assembly: InputAssembly {
                topology: PrimitiveTopology::TriangleList,
                primitive_restart_enable: true,
            },
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 1 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            render_pass: Subpass::from(
                simple_rp::CustomRenderPass::new(&device, &{
                    simple_rp::Formats {
                        color: (Format::R8G8B8A8_UNORM, 1),
                    }
                })
                .unwrap(),
                0,
            )
            .unwrap(),
        },
    );

    match result {
        Err(GraphicsPipelineCreationError::RequirementNotMet { .. }) => (),
        _ => panic!(),
    }
}

#[test]
fn multi_viewport_feature() {
    let (device, _) = gfx_dev_and_queue!();

    let vs = unsafe { ShaderModule::new(device.clone(), &BASIC_VS).unwrap() };
    let fs = unsafe { ShaderModule::new(device.clone(), &BASIC_FS).unwrap() };

    let result = GraphicsPipeline::new(
        &device,
        GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 2 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            render_pass: Subpass::from(
                simple_rp::CustomRenderPass::new(&device, &{
                    simple_rp::Formats {
                        color: (Format::R8G8B8A8_UNORM, 1),
                    }
                })
                .unwrap(),
                0,
            )
            .unwrap(),
        },
    );

    match result {
        Err(GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled) => (),
        _ => panic!(),
    }
}

#[test]
fn max_viewports() {
    let (device, _) = gfx_dev_and_queue!(multi_viewport);

    let vs = unsafe { ShaderModule::new(device.clone(), &BASIC_VS).unwrap() };
    let fs = unsafe { ShaderModule::new(device.clone(), &BASIC_FS).unwrap() };

    let result = GraphicsPipeline::new(
        &device,
        GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: !0 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            render_pass: Subpass::from(
                simple_rp::CustomRenderPass::new(&device, &{
                    simple_rp::Formats {
                        color: (Format::R8G8B8A8_UNORM, 1),
                    }
                })
                .unwrap(),
                0,
            )
            .unwrap(),
        },
    );

    match result {
        Err(GraphicsPipelineCreationError::MaxViewportsExceeded { .. }) => (),
        _ => panic!(),
    }
}

#[test]
fn no_depth_attachment() {
    let (device, _) = gfx_dev_and_queue!();

    let vs = unsafe { ShaderModule::new(device.clone(), &BASIC_VS).unwrap() };
    let fs = unsafe { ShaderModule::new(device.clone(), &BASIC_FS).unwrap() };

    let result = GraphicsPipeline::new(
        &device,
        GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 1 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(
                    &CString::new("main").unwrap(),
                    ShaderInterface::empty(),
                    ShaderInterface::empty(),
                    PipelineLayoutDesc::empty(),
                )
            },
            depth_stencil: DepthStencil::simple_depth_test(),
            blend: Blend::pass_through(),
            render_pass: Subpass::from(
                simple_rp::CustomRenderPass::new(&device, &{
                    simple_rp::Formats {
                        color: (Format::R8G8B8A8_UNORM, 1),
                    }
                })
                .unwrap(),
                0,
            )
            .unwrap(),
        },
    );

    match result {
        Err(GraphicsPipelineCreationError::NoDepthAttachment) => (),
        _ => panic!(),
    }
}

mod simple_rp {
    use format::Format;

    single_pass_renderpass! {
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    }
}

/*
    #version 450

    #extension GL_ARB_separate_shader_objects : enable
    #extension GL_ARB_shading_language_420pack : enable

    layout(location = 0) in vec2 position;

    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
    }
*/
const BASIC_VS: [u8; 912] = [
    3, 2, 35, 7, 0, 0, 1, 0, 1, 0, 8, 0, 27, 0, 0, 0, 0, 0, 0, 0, 17, 0, 2, 0, 1, 0, 0, 0, 17, 0,
    2, 0, 32, 0, 0, 0, 17, 0, 2, 0, 33, 0, 0, 0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115,
    116, 100, 46, 52, 53, 48, 0, 0, 0, 0, 14, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 7, 0, 0, 0,
    0, 0, 4, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 13, 0, 0, 0, 18, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0,
    0, 194, 1, 0, 0, 4, 0, 9, 0, 71, 76, 95, 65, 82, 66, 95, 115, 101, 112, 97, 114, 97, 116, 101,
    95, 115, 104, 97, 100, 101, 114, 95, 111, 98, 106, 101, 99, 116, 115, 0, 0, 4, 0, 9, 0, 71, 76,
    95, 65, 82, 66, 95, 115, 104, 97, 100, 105, 110, 103, 95, 108, 97, 110, 103, 117, 97, 103, 101,
    95, 52, 50, 48, 112, 97, 99, 107, 0, 5, 0, 4, 0, 4, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 5,
    0, 6, 0, 11, 0, 0, 0, 103, 108, 95, 80, 101, 114, 86, 101, 114, 116, 101, 120, 0, 0, 0, 0, 6,
    0, 6, 0, 11, 0, 0, 0, 0, 0, 0, 0, 103, 108, 95, 80, 111, 115, 105, 116, 105, 111, 110, 0, 6, 0,
    7, 0, 11, 0, 0, 0, 1, 0, 0, 0, 103, 108, 95, 80, 111, 105, 110, 116, 83, 105, 122, 101, 0, 0,
    0, 0, 6, 0, 7, 0, 11, 0, 0, 0, 2, 0, 0, 0, 103, 108, 95, 67, 108, 105, 112, 68, 105, 115, 116,
    97, 110, 99, 101, 0, 6, 0, 7, 0, 11, 0, 0, 0, 3, 0, 0, 0, 103, 108, 95, 67, 117, 108, 108, 68,
    105, 115, 116, 97, 110, 99, 101, 0, 5, 0, 3, 0, 13, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 18, 0, 0,
    0, 112, 111, 115, 105, 116, 105, 111, 110, 0, 0, 0, 0, 72, 0, 5, 0, 11, 0, 0, 0, 0, 0, 0, 0,
    11, 0, 0, 0, 0, 0, 0, 0, 72, 0, 5, 0, 11, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0, 0, 1, 0, 0, 0, 72, 0,
    5, 0, 11, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 3, 0, 0, 0, 72, 0, 5, 0, 11, 0, 0, 0, 3, 0, 0, 0,
    11, 0, 0, 0, 4, 0, 0, 0, 71, 0, 3, 0, 11, 0, 0, 0, 2, 0, 0, 0, 71, 0, 4, 0, 18, 0, 0, 0, 30, 0,
    0, 0, 0, 0, 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2, 0, 0, 0, 22, 0, 3, 0, 6,
    0, 0, 0, 32, 0, 0, 0, 23, 0, 4, 0, 7, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 21, 0, 4, 0, 8, 0, 0, 0,
    32, 0, 0, 0, 0, 0, 0, 0, 43, 0, 4, 0, 8, 0, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 28, 0, 4, 0, 10, 0,
    0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 30, 0, 6, 0, 11, 0, 0, 0, 7, 0, 0, 0, 6, 0, 0, 0, 10, 0, 0, 0,
    10, 0, 0, 0, 32, 0, 4, 0, 12, 0, 0, 0, 3, 0, 0, 0, 11, 0, 0, 0, 59, 0, 4, 0, 12, 0, 0, 0, 13,
    0, 0, 0, 3, 0, 0, 0, 21, 0, 4, 0, 14, 0, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 43, 0, 4, 0, 14, 0, 0,
    0, 15, 0, 0, 0, 0, 0, 0, 0, 23, 0, 4, 0, 16, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 32, 0, 4, 0, 17,
    0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 59, 0, 4, 0, 17, 0, 0, 0, 18, 0, 0, 0, 1, 0, 0, 0, 43, 0, 4,
    0, 6, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 43, 0, 4, 0, 6, 0, 0, 0, 21, 0, 0, 0, 0, 0, 128, 63,
    32, 0, 4, 0, 25, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
    0, 3, 0, 0, 0, 248, 0, 2, 0, 5, 0, 0, 0, 61, 0, 4, 0, 16, 0, 0, 0, 19, 0, 0, 0, 18, 0, 0, 0,
    81, 0, 5, 0, 6, 0, 0, 0, 22, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 81, 0, 5, 0, 6, 0, 0, 0, 23, 0,
    0, 0, 19, 0, 0, 0, 1, 0, 0, 0, 80, 0, 7, 0, 7, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 23, 0, 0, 0,
    20, 0, 0, 0, 21, 0, 0, 0, 65, 0, 5, 0, 25, 0, 0, 0, 26, 0, 0, 0, 13, 0, 0, 0, 15, 0, 0, 0, 62,
    0, 3, 0, 26, 0, 0, 0, 24, 0, 0, 0, 253, 0, 1, 0, 56, 0, 1, 0,
];

/*
    #version 450

    #extension GL_ARB_separate_shader_objects : enable
    #extension GL_ARB_shading_language_420pack : enable

    layout(location = 0) out vec4 f_color;

    void main() {
        f_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
*/
const BASIC_FS: [u8; 420] = [
    3, 2, 35, 7, 0, 0, 1, 0, 1, 0, 8, 0, 13, 0, 0, 0, 0, 0, 0, 0, 17, 0, 2, 0, 1, 0, 0, 0, 11, 0,
    6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115, 116, 100, 46, 52, 53, 48, 0, 0, 0, 0, 14, 0, 3, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 6, 0, 4, 0, 0, 0, 4, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 9,
    0, 0, 0, 16, 0, 3, 0, 4, 0, 0, 0, 7, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0, 0, 4, 0, 9, 0,
    71, 76, 95, 65, 82, 66, 95, 115, 101, 112, 97, 114, 97, 116, 101, 95, 115, 104, 97, 100, 101,
    114, 95, 111, 98, 106, 101, 99, 116, 115, 0, 0, 4, 0, 9, 0, 71, 76, 95, 65, 82, 66, 95, 115,
    104, 97, 100, 105, 110, 103, 95, 108, 97, 110, 103, 117, 97, 103, 101, 95, 52, 50, 48, 112, 97,
    99, 107, 0, 5, 0, 4, 0, 4, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 5, 0, 4, 0, 9, 0, 0, 0, 102,
    95, 99, 111, 108, 111, 114, 0, 71, 0, 4, 0, 9, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 19, 0, 2, 0,
    2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2, 0, 0, 0, 22, 0, 3, 0, 6, 0, 0, 0, 32, 0, 0, 0, 23, 0,
    4, 0, 7, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 32, 0, 4, 0, 8, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 59,
    0, 4, 0, 8, 0, 0, 0, 9, 0, 0, 0, 3, 0, 0, 0, 43, 0, 4, 0, 6, 0, 0, 0, 10, 0, 0, 0, 0, 0, 128,
    63, 43, 0, 4, 0, 6, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 44, 0, 7, 0, 7, 0, 0, 0, 12, 0, 0, 0, 10,
    0, 0, 0, 11, 0, 0, 0, 11, 0, 0, 0, 10, 0, 0, 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
    0, 3, 0, 0, 0, 248, 0, 2, 0, 5, 0, 0, 0, 62, 0, 3, 0, 9, 0, 0, 0, 12, 0, 0, 0, 253, 0, 1, 0,
    56, 0, 1, 0,
];
