/// Builds a `RenderPass` object whose template parameter is of indeterminate type.
#[macro_export]
macro_rules! single_pass_renderpass {
    (
        $device:expr,
        attachments: { $($a:tt)* },
        pass: {
            color: [$($color_atch:ident),* $(,)?]
            $(, color_resolve: [$($color_resolve_atch:ident),* $(,)?])?
            , depth_stencil: {$($depth_stencil_atch:ident)?}
            $(
                , depth_stencil_resolve: {$depth_stencil_resolve_atch:ident}
                $(, depth_resolve_mode: $depth_resolve_mode:ident)?
                $(, stencil_resolve_mode: $stencil_resolve_mode:ident)?
            )?
            $(,)?
        } $(,)?
    ) => (
        $crate::ordered_passes_renderpass!(
            $device,
            attachments: { $($a)* },
            passes: [
                {
                    color: [$($color_atch),*]
                    $(, color_resolve: [$($color_resolve_atch),*])?
                    , depth_stencil: {$($depth_stencil_atch)?}
                    $(
                        , depth_stencil_resolve: {$depth_stencil_resolve_atch}
                        $(, depth_resolve_mode: $depth_resolve_mode)?
                        $(, stencil_resolve_mode: $stencil_resolve_mode)?
                    )?
                    , input: [],
                }
            ]
        )
    )
}

/// Builds a `RenderPass` object whose template parameter is of indeterminate type.
#[macro_export]
macro_rules! ordered_passes_renderpass {
    (
        $device:expr,
        attachments: {
            $(
                $atch_name:ident: {
                    format: $format:expr,
                    samples: $samples:expr,
                    load_op: $load_op:ident,
                    store_op: $store_op:ident
                    $(,initial_layout: $init_layout:expr)?
                    $(,final_layout: $final_layout:expr)?
                    $(,)?
                }
            ),* $(,)?
        },
        passes: [
            $(
                {
                    color: [$($color_atch:ident),* $(,)?]
                    $(, color_resolve: [$($color_resolve_atch:ident),* $(,)?])?
                    , depth_stencil: {$($depth_stencil_atch:ident)?}
                    $(
                        , depth_stencil_resolve: {$depth_stencil_resolve_atch:ident}
                        $(, depth_resolve_mode: $depth_resolve_mode:ident)?
                        $(, stencil_resolve_mode: $stencil_resolve_mode:ident)?
                    )?
                    , input: [$($input_atch:ident),* $(,)?]
                    $(,)*
                }
            ),* $(,)?
        ] $(,)?
    ) => ({
        #[allow(unused)]
        let mut attachment_num = 0;
        $(
            let $atch_name = attachment_num;
            attachment_num += 1;
        )*

        #[allow(unused)]
        #[derive(Clone, Copy, Default)]
        struct Layouts {
            initial_layout: Option<$crate::image::ImageLayout>,
            final_layout: Option<$crate::image::ImageLayout>,
        }

        #[allow(unused)]
        let mut layouts: Vec<Layouts> = vec![Layouts::default(); attachment_num as usize];
        let mut subpass_count: u32 = 0;

        $(
            $({
                let layouts = &mut layouts[$color_atch as usize];
                layouts.initial_layout = layouts.initial_layout.or(Some($crate::image::ImageLayout::ColorAttachmentOptimal));
                layouts.final_layout = Some($crate::image::ImageLayout::ColorAttachmentOptimal);
            })*
            $($({
                let layouts = &mut layouts[$color_resolve_atch as usize];
                layouts.final_layout = Some($crate::image::ImageLayout::TransferDstOptimal);
                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);
            })*)?
            $({
                let layouts = &mut layouts[$depth_stencil_atch as usize];
                layouts.final_layout = Some($crate::image::ImageLayout::DepthStencilAttachmentOptimal);
                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);
            })?
            $({
                let layouts = &mut layouts[$depth_stencil_resolve_atch as usize];
                layouts.final_layout = Some($crate::image::ImageLayout::TransferDstOptimal);
                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);
            })?
            $({
                let layouts = &mut layouts[$input_atch as usize];
                layouts.final_layout = Some($crate::image::ImageLayout::ShaderReadOnlyOptimal);
                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);
            })*
            subpass_count += 1;
        )+

        $({
            $(layouts[$atch_name as usize].initial_layout = Some($init_layout);)?
            $(layouts[$atch_name as usize].final_layout = Some($final_layout);)?
        })*

        $crate::render_pass::RenderPass::try_new(
            $device,
            &$crate::render_pass::RenderPassCreateInfo {
                attachments: &[$(
                    $crate::render_pass::AttachmentDescription {
                        format: $format,
                        samples: $crate::image::SampleCount::try_from($samples).unwrap(),
                        load_op: $crate::render_pass::AttachmentLoadOp::$load_op,
                        store_op: $crate::render_pass::AttachmentStoreOp::$store_op,
                        initial_layout: layouts[$atch_name as usize].initial_layout.expect(
                            format!(
                                "attachment {} is missing initial_layout; this is normally \
                                automatically determined but you can manually specify it for an \
                                individual attachment in the single_pass_renderpass! macro",
                                attachment_num,
                            )
                            .as_ref(),
                        ),
                        final_layout: layouts[$atch_name as usize].final_layout.expect(
                            format!(
                                "attachment {} is missing final_layout; this is normally \
                                automatically determined but you can manually specify it for an \
                                individual attachment in the single_pass_renderpass! macro",
                                attachment_num,
                            )
                            .as_ref(),
                        ),
                        ..Default::default()
                    },
                )*],
                subpasses: &[$(
                    $crate::render_pass::SubpassDescription {
                        input_attachments: &[$(
                            Some($crate::render_pass::AttachmentReference {
                                attachment: $input_atch,
                                layout: $crate::image::ImageLayout::ShaderReadOnlyOptimal,
                                ..Default::default()
                            }),
                        )*],
                        color_attachments: &[$(
                            Some($crate::render_pass::AttachmentReference {
                                attachment: $color_atch,
                                layout: $crate::image::ImageLayout::ColorAttachmentOptimal,
                                ..Default::default()
                            }),
                        )*],
                        color_resolve_attachments: &[$($(
                            Some($crate::render_pass::AttachmentReference {
                                attachment: $color_resolve_atch,
                                layout: $crate::image::ImageLayout::TransferDstOptimal,
                                ..Default::default()
                            }),
                        )*)?],
                        $(depth_stencil_attachment: Some(
                            &Some($crate::render_pass::AttachmentReference {
                                attachment: $depth_stencil_atch,
                                layout: $crate::image::ImageLayout::DepthStencilAttachmentOptimal,
                                ..Default::default()
                            }),
                        ),)?
                        $(depth_stencil_resolve_attachment: Some(
                            &Some($crate::render_pass::AttachmentReference {
                                attachment: $depth_stencil_resolve_atch,
                                layout: $crate::image::ImageLayout::TransferDstOptimal,
                                ..Default::default()
                            }),
                        ),)?
                        $($(depth_resolve_mode: Some(
                            $crate::render_pass::ResolveMode::$depth_resolve_mode,
                        ),)?)?
                        $($(stencil_resolve_mode: Some(
                            $crate::render_pass::ResolveMode::$stencil_resolve_mode,
                        ),)?)?
                        preserve_attachments: &(0..attachment_num)
                            .filter(|a| {
                                ![
                                    $($input_atch,)*
                                    $($color_atch,)*
                                    $($($color_resolve_atch,)*)?
                                    $($depth_stencil_atch,)?
                                    $($depth_stencil_resolve_atch,)?
                                ]
                                .contains(a)
                            })
                            .collect::<Vec<_>>(),
                        ..Default::default()
                    },
                )*],
                dependencies: &(0..subpass_count.saturating_sub(1))
                    .map(|id| {
                        // TODO: correct values
                        let src_stages = $crate::sync::PipelineStages::ALL_GRAPHICS;
                        let dst_stages = $crate::sync::PipelineStages::ALL_GRAPHICS;
                        let src_access = $crate::sync::AccessFlags::MEMORY_READ
                            | $crate::sync::AccessFlags::MEMORY_WRITE;
                        let dst_access = $crate::sync::AccessFlags::MEMORY_READ
                            | $crate::sync::AccessFlags::MEMORY_WRITE;

                        $crate::render_pass::SubpassDependency {
                            src_subpass: id.into(),
                            dst_subpass: (id + 1).into(),
                            src_stages,
                            dst_stages,
                            src_access,
                            dst_access,
                            // TODO: correct values
                            dependency_flags: $crate::sync::DependencyFlags::BY_REGION,
                            ..Default::default()
                        }
                    })
                    .collect::<Vec<_>>(),
                ..Default::default()
            },
        )
    });
}

#[cfg(test)]
mod tests {
    use crate::format::Format;

    #[test]
    fn single_pass_resolve() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = single_pass_renderpass!(
            &device,
            attachments: {
                a: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },
                b: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [a],
                color_resolve: [b],
                depth_stencil: {},
            },
        )
        .unwrap();
    }
}
