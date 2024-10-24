//The BindgroupLayouts-structs creates all commonly used BindgroupAlignments and Pipeline layouts in advance.

use std::num::NonZeroU64;

use tracing::instrument;
use wgpu::BindGroupLayoutDescriptor;

use super::BindgroupAlignment;

#[derive(Debug)]
pub(crate) struct BindgroupLayoutAndPipeline(pub wgpu::BindGroupLayout, pub wgpu::PipelineLayout);

#[derive(Debug, Clone, PartialEq, Eq, Hash, std::marker::Copy)]
#[cfg_attr(
    any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum BindgroupAlignmentLayout {
    Bindgroup0(BindgroupAlignment),
    Bindgroup1(BindgroupAlignment, BindgroupAlignment),
    Bindgroup2(BindgroupAlignment, BindgroupAlignment, BindgroupAlignment),
    Bindgroup3(
        BindgroupAlignment,
        BindgroupAlignment,
        BindgroupAlignment,
        BindgroupAlignment,
    ),
}

impl BindgroupAlignmentLayout {
    fn values() -> [BindgroupAlignmentLayout; 23] {
        use BindgroupAlignment::{Aligned16, Aligned4, Aligned8};
        use BindgroupAlignmentLayout::{Bindgroup0, Bindgroup1, Bindgroup2, Bindgroup3};
        [
            Bindgroup0(Aligned4),
            Bindgroup0(Aligned8),
            Bindgroup0(Aligned16),
            Bindgroup1(Aligned4, Aligned4),
            Bindgroup1(Aligned8, Aligned8),
            Bindgroup1(Aligned16, Aligned16),
            Bindgroup1(Aligned4, Aligned8),
            Bindgroup1(Aligned4, Aligned16),
            Bindgroup1(Aligned8, Aligned4),
            Bindgroup1(Aligned8, Aligned16),
            Bindgroup1(Aligned16, Aligned4),
            Bindgroup1(Aligned16, Aligned8),
            Bindgroup2(Aligned4, Aligned4, Aligned4),
            Bindgroup2(Aligned8, Aligned8, Aligned8),
            Bindgroup2(Aligned16, Aligned16, Aligned16),
            Bindgroup2(Aligned8, Aligned4, Aligned8),
            Bindgroup2(Aligned4, Aligned8, Aligned4),
            Bindgroup2(Aligned4, Aligned16, Aligned16),
            Bindgroup3(Aligned4, Aligned4, Aligned4, Aligned4),
            Bindgroup3(Aligned8, Aligned8, Aligned8, Aligned8),
            Bindgroup3(Aligned16, Aligned16, Aligned16, Aligned16),
            Bindgroup3(Aligned4, Aligned8, Aligned4, Aligned4),
            Bindgroup3(Aligned8, Aligned4, Aligned8, Aligned8),
        ]
    }

    pub fn validate(&self){
        _= self.get_index();
    }

    pub fn get_index(&self) -> usize {
        use BindgroupAlignment::{Aligned16, Aligned4, Aligned8};
        use BindgroupAlignmentLayout::{Bindgroup0, Bindgroup1, Bindgroup2, Bindgroup3};
        match self {
            Bindgroup0(Aligned4) => 0,
            Bindgroup0(Aligned8) => 1,
            Bindgroup0(Aligned16) => 2,

            Bindgroup1(Aligned4, Aligned4) => 3,
            Bindgroup1(Aligned8, Aligned8) => 4,
            Bindgroup1(Aligned16, Aligned16) => 5,
            Bindgroup1(Aligned4, Aligned8) => 6,
            Bindgroup1(Aligned4, Aligned16) => 7,
            Bindgroup1(Aligned8, Aligned4) => 8,
            Bindgroup1(Aligned8, Aligned16) => 9,
            Bindgroup1(Aligned16, Aligned4) => 10,
            Bindgroup1(Aligned16, Aligned8) => 11,

            Bindgroup2(Aligned4, Aligned4, Aligned4) => 12,
            Bindgroup2(Aligned8, Aligned8, Aligned8) => 13,
            Bindgroup2(Aligned16, Aligned16, Aligned16) => 14,
            Bindgroup2(Aligned8, Aligned4, Aligned8) => 15,
            Bindgroup2(Aligned4, Aligned8, Aligned4) => 16,
            Bindgroup2(Aligned4, Aligned16, Aligned16) => 17,

            Bindgroup3(Aligned4, Aligned4, Aligned4, Aligned4) => 18,
            Bindgroup3(Aligned8, Aligned8, Aligned8, Aligned8) => 19,
            Bindgroup3(Aligned16, Aligned16, Aligned16, Aligned16) => 20,
            Bindgroup3(Aligned4, Aligned8, Aligned4, Aligned4) => 21,
            Bindgroup3(Aligned8, Aligned4, Aligned8, Aligned8) => 22,
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct BindgroupLayouts {
    data: [BindgroupLayoutAndPipeline; 23],
}

impl std::ops::Index<BindgroupAlignmentLayout> for BindgroupLayouts {
    type Output = BindgroupLayoutAndPipeline;

    fn index(&self, index: BindgroupAlignmentLayout) -> &Self::Output {
        &self.data[index.get_index()]
    }
}

impl std::ops::IndexMut<BindgroupAlignmentLayout> for BindgroupLayouts {
    fn index_mut(&mut self, index: BindgroupAlignmentLayout) -> &mut Self::Output {
        &mut self.data[index.get_index()]
    }
}

impl BindgroupLayouts {
    #[instrument(skip(dev))]
    pub(crate) fn new(dev: &wgpu::Device) -> Self {
        fn create_bingroup_entry(
            binding: u32,
            alignment: u32,
            read_only: bool,
        ) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(alignment.into()).unwrap()),
                },
                count: None,
            }
        }

        fn create_bindgroup_layout_and_pipeline(
            dev: &wgpu::Device,
            entries: &[wgpu::BindGroupLayoutEntry],
        ) -> BindgroupLayoutAndPipeline {
            let bindgroup_layout = dev.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries,
            });

            let pipeline_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bindgroup_layout],
                push_constant_ranges: &[],
            });

            BindgroupLayoutAndPipeline(bindgroup_layout, pipeline_layout)
        }

        let meta_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let dest_entries = [2, 4, 8, 16].map(|a| create_bingroup_entry(0, a, false));
        let input1_entries = [2, 4, 8, 16].map(|a| create_bingroup_entry(2, a, true));
        let input2_entries = [2, 4, 8, 16].map(|a| create_bingroup_entry(3, a, true));
        let input3_entries = [2, 4, 8, 16].map(|a| create_bingroup_entry(4, a, true));

        let data = BindgroupAlignmentLayout::values().map(|v| match v {
            BindgroupAlignmentLayout::Bindgroup0(a) => create_bindgroup_layout_and_pipeline(
                dev,
                &[dest_entries[a.get_index()], meta_entry],
            ),
            BindgroupAlignmentLayout::Bindgroup1(a1, a2) => create_bindgroup_layout_and_pipeline(
                dev,
                &[
                    dest_entries[a1.get_index()],
                    meta_entry,
                    input1_entries[a2.get_index()],
                ],
            ),
            BindgroupAlignmentLayout::Bindgroup2(a1, a2, a3) => {
                create_bindgroup_layout_and_pipeline(
                    dev,
                    &[
                        dest_entries[a1.get_index()],
                        meta_entry,
                        input1_entries[a2.get_index()],
                        input2_entries[a3.get_index()],
                    ],
                )
            }
            BindgroupAlignmentLayout::Bindgroup3(a1, a2, a3, a4) => {
                create_bindgroup_layout_and_pipeline(
                    dev,
                    &[
                        dest_entries[a1.get_index()],
                        meta_entry,
                        input1_entries[a2.get_index()],
                        input2_entries[a3.get_index()],
                        input3_entries[a4.get_index()],
                    ],
                )
            }
        });

        return BindgroupLayouts { data };
    }
}
