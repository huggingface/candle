use wgpu_compute_layer::{
    DType, LoaderIndex, OpIsInplaceable, PipelineIndex, ShaderIndex, WgpuDevice, cache::{BindGroupReference, BindgroupAlignmentLayout, BindgroupInputBase, BindgroupReferenceInput, BufferReferenceId}
};

// Simple shader loader that returns a no-op shader and can reply with a
// rewrite plan to trigger copy-inplace optimizations.
#[derive(Debug)]
struct ElideLoader;

wgpu_compute_layer::create_loader!(ElideLoader);

fn create_dummy_shader_index() -> ShaderIndex {
    ShaderIndex::new(LoaderIndex(ElideLoader::LOADER_INDEX.0), 0)
}

fn create_dummy_pipeline_index(index: u8) -> PipelineIndex {
    PipelineIndex::new(create_dummy_shader_index(), index)
}

impl wgpu_compute_layer::ShaderLoader for ElideLoader {
    fn load(&self, _index: wgpu_compute_layer::ShaderIndex) -> &str {
        "
        @group(0) @binding(0) var<storage, read_write> output : array<u32>;
        @group(0) @binding(1) var<storage> op_meta : array<u32>;
        @group(0) @binding(2) var<storage> input : array<u32>;
        @compute @workgroup_size(1) fn main() { output[0] = input[0]; }
        @compute @workgroup_size(1) fn main_no_input() { output[0] += 1; }
        "
    }

    fn get_entry_point(&self, index: PipelineIndex) -> &str {
        if index.1 == 10{
            "main_no_input"
        }
        else{
            "main"
        }
    }

    fn rewrite_plan(
        &self,
        desc: wgpu_compute_layer::InplaceRewriteDesc,
    ) -> Option<wgpu_compute_layer::RewritePlan> {
        match desc.pipeline.1 {
            //Copy Inplace
            0 if desc.inplace_flags.input1_inplaceable => Some(wgpu_compute_layer::RewritePlan::ElideDispatch {
                replaced_input: wgpu_compute_layer::ReplacedInput::Input1,
            }),

            //Unary Inplace
            1 if desc.inplace_flags.input1_inplaceable  =>{
                let BindgroupReferenceInput::Bindgroup1(v1, layout) = desc.bindgroup.get_input()
                else {
                    return None;
                };

                Some(wgpu_compute_layer::RewritePlan::InplaceDispatch {
                    new_pipeline: create_dummy_pipeline_index(10),
                    new_bindgroup: BindGroupReference::new(
                        *v1,
                        BindgroupInputBase::Bindgroup0(BindgroupAlignmentLayout::Bindgroup0(
                            layout.get_dest(),
                        )),
                    ),
                    replaced_input: wgpu_compute_layer::ReplacedInput::Input1,
                })
            } 
            2 if desc.inplace_flags.input1_inplaceable => {
                let BindgroupReferenceInput::Bindgroup2(v1, v2,  layout) = desc.bindgroup.get_input()
                else {
                    return None;
                };

                Some(wgpu_compute_layer::RewritePlan::InplaceDispatch {
                    new_pipeline: create_dummy_pipeline_index(11),
                    new_bindgroup: BindGroupReference::new(
                        *v1,
                        BindgroupInputBase::Bindgroup1(*v2, BindgroupAlignmentLayout::Bindgroup1(
                            layout.get_dest(),
                            layout.get_dest(),
                        )),
                    ),
                    replaced_input: wgpu_compute_layer::ReplacedInput::Input1,
                })
            } 
            3 if desc.inplace_flags.input2_inplaceable => {
                let BindgroupReferenceInput::Bindgroup2(v1, v2,  layout) = desc.bindgroup.get_input()
                else {
                    return None;
                };

                Some(wgpu_compute_layer::RewritePlan::InplaceDispatch {
                    new_pipeline: create_dummy_pipeline_index(11),
                    new_bindgroup: BindGroupReference::new(
                        *v2,
                        BindgroupInputBase::Bindgroup1(*v1, BindgroupAlignmentLayout::Bindgroup1(
                            layout.get_dest(),
                            layout.get_dest(),
                        )),
                    ),
                    replaced_input: wgpu_compute_layer::ReplacedInput::Input1,
                })
            } 
            _ => panic!()
        }

    }
}

fn create_device() -> WgpuDevice {
    let dev = WgpuDevice::create(Default::default()).expect("create device");
    dev.add_wgpu_shader_loader(ElideLoader::LOADER_INDEX, || ElideLoader {});
    dev
}

fn dummy_copy_inplace(dev: &WgpuDevice, input: BufferReferenceId, output: BufferReferenceId) {
    let mut q = dev.get_queue();
    let pipeline = q.get_pipeline_inplace(
        create_dummy_pipeline_index(0),
        OpIsInplaceable {
            input1_inplaceable: true,
            input2_inplaceable: false,
        },
    );

    let bind = dev.create_bind_group_input1(output, input, DType::U32.into());
    q.enqueue_workgroups(pipeline, bind, 1, 1, 1, 1);
}

fn dummy_copy(dev: &WgpuDevice, input: BufferReferenceId, output: BufferReferenceId) {
    let mut q = dev.get_queue();
    let pipeline = q.get_pipeline_inplace(
        create_dummy_pipeline_index(0),
        OpIsInplaceable {
            input1_inplaceable: false,
            input2_inplaceable: false,
        },
    );

    let bind = dev.create_bind_group_input1(output, input, DType::U32.into());
    q.enqueue_workgroups(pipeline, bind, 1, 1, 1, 1);
}

fn dummy_unary_rewrite(dev: &WgpuDevice, input: BufferReferenceId, output: BufferReferenceId) {
    let mut q = dev.get_queue();
    let pipeline = q.get_pipeline_inplace(
        create_dummy_pipeline_index(1),
        OpIsInplaceable {
            input1_inplaceable: true,
            input2_inplaceable: false,
        },
    );

    let bind = dev.create_bind_group_input1(output, input, DType::U32.into());
    q.enqueue_workgroups(pipeline, bind, 1, 1, 1, 1);
}

#[test]
fn copy_inplace() {
    let dev = create_device();
    {
        let out = dev.alloc_uninit_size(DType::U32, 1);
        let input = dev
            .alloc_from_slice(DType::U32, &[1, 2, 3, 4])
            .expect("alloc from slice");

        dummy_copy_inplace(&dev, input.buffer(), out.buffer());
        dev.synchronize().expect("sync");
        assert_eq!(
            dev.get_internal_counters().copy_inplace_counter,
            0,
            "copy should not be inplaced while input is still alive"
        );
    }

    //now the same test, but we drop the input, so now the call should be inplaced
    let out = dev.alloc_uninit_size(DType::U32, 1);
    let input = dev
        .alloc_from_slice(DType::U32, &[1, 2, 3, 4])
        .expect("alloc from slice");
    assert_eq!(dev.get_internal_counters().copy_inplace_counter, 0);
    dummy_copy_inplace(&dev, input.buffer(), out.buffer());

    drop(input);
    dev.synchronize().expect("sync");
    assert_eq!(
        dev.get_internal_counters().copy_inplace_counter,
        1,
        "copy should be inplaced once input is dropped before sync"
    );


    //now the same test, but we set is_inplacecable to false, this shouldnt increase the copy_inplace_counter
    let out = dev.alloc_uninit_size(DType::U32, 1);
    let input = dev
        .alloc_from_slice(DType::U32, &[1, 2, 3, 4])
        .expect("alloc from slice");
    assert_eq!(dev.get_internal_counters().copy_inplace_counter, 1);
    dummy_copy(&dev, input.buffer(), out.buffer());

    drop(input);
    dev.synchronize().expect("sync");
    assert_eq!(
        dev.get_internal_counters().copy_inplace_counter,
        1,
        "copy should be inplaced once input is dropped before sync"
    );
}

#[test]
fn counters_buffer_creation_and_reuse() {
    //test buffer reuse between different WgpuStorages
    {
        let dev = create_device();
        assert_eq!(dev.get_internal_counters().buffer_counter, 0);

        let a = dev
            .alloc_from_bytes(DType::U32, bytemuck::cast_slice(&[0u32; 4]))
            .expect("alloc");
        assert_eq!(dev.get_internal_counters().buffer_counter, 1);
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 0); 
        drop(a);
        dev.synchronize().expect("sync");
        let b = dev
            .alloc_from_bytes(DType::U32, bytemuck::cast_slice(&[0u32; 4]))
            .expect("alloc");
        assert_eq!(dev.get_internal_counters().buffer_counter, 1); //we should reuse the wgpu::Buffer, created with 'a'
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 1);
        drop(b);
    }

    //test buffer reuse with intermediate storage
    {
        let dev = create_device();
        assert_eq!(dev.get_internal_counters().buffer_counter, 0);

        let a = dev
            .alloc_from_bytes(DType::U32, bytemuck::cast_slice(&[0u32; 4]))
            .expect("alloc");
        assert_eq!(dev.get_internal_counters().buffer_counter, 1);
      
        
        let out1 = dev.alloc_uninit_size(DType::U32, 4);
        dummy_copy(&dev, a.buffer(), out1.buffer());
        assert_eq!(dev.get_internal_counters().buffer_counter, 1, "creating a WgpuStorage with alloc_uninit_size should not create a buffer until we synchronize");
        dev.synchronize().expect("sync");
        assert_eq!(dev.get_internal_counters().buffer_counter, 2);

        let out2 = dev.alloc_uninit_size(DType::U32, 4);
        dummy_copy(&dev, out1.buffer(), out2.buffer());
        drop(out1);
        assert_eq!(dev.get_internal_counters().buffer_counter, 2, "creating a WgpuStorage with alloc_uninit_size should not create a buffer until we synchronize");
        dev.synchronize().expect("sync");
        assert_eq!(dev.get_internal_counters().buffer_counter, 3);

        let out3 = dev.alloc_uninit_size(DType::U32, 4);
        dummy_copy(&dev, out2.buffer(), out3.buffer()); //as out1 has droped we expect to reuse the buffer of out1 for the output here
        
        assert_eq!(dev.get_internal_counters().buffer_counter, 3, "creating a WgpuStorage with alloc_uninit_size should not create a buffer until we synchronize");
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 0, "buffer_reuse_counter has been calculated incorrectly");
        dev.synchronize().expect("sync");
        assert_eq!(dev.get_internal_counters().buffer_counter, 3, "as 'out1' has beed dropped we should have reused that buffer");
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 1, "buffer_reuse_counter has been calculated incorrectly");
    }
    
    //test buffer reuse with intermediate storage with multiple commands between synchronization
    {
        let dev = create_device();
        assert_eq!(dev.get_internal_counters().buffer_counter, 0);

        let a = dev
            .alloc_from_bytes(DType::U32, bytemuck::cast_slice(&[0u32; 4]))
            .expect("alloc");
        assert_eq!(dev.get_internal_counters().buffer_counter, 1);
      
        let out1 = dev.alloc_uninit_size(DType::U32, 4);
        dummy_copy(&dev, a.buffer(), out1.buffer());

        let out2 = dev.alloc_uninit_size(DType::U32, 4);
        dummy_copy(&dev, out1.buffer(), out2.buffer());
        drop(out1);

        let out3 = dev.alloc_uninit_size(DType::U32, 4);
        dummy_copy(&dev, out2.buffer(), out3.buffer()); //as out1 has droped we expect to reuse the buffer of out1 for the output here
        
        assert_eq!(dev.get_internal_counters().buffer_counter, 1);
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 0, "buffer_reuse_counter has been calculated incorrectly");
        dev.synchronize().expect("sync");
        assert_eq!(dev.get_internal_counters().buffer_counter, 3);
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 1, "buffer_reuse_counter has been calculated incorrectly");
    }

    //test buffer reuse when using create_buffer_reference temporary Buffers
    {
        let dev = create_device();
        assert_eq!(dev.get_internal_counters().buffer_counter, 0);

        let a = dev
            .alloc_from_bytes(DType::U32, bytemuck::cast_slice(&[0u32; 4]))
            .expect("alloc");
        assert_eq!(dev.get_internal_counters().buffer_counter, 1);
      
        let temp1 = dev.create_buffer_reference(16u64, false);
        dummy_copy(&dev, a.buffer(), temp1);

        let temp2 = dev.create_buffer_reference(16u64, false);
        dummy_copy(&dev, temp1, temp2);

        let out3 = dev.alloc_uninit_size(DType::U32, 4);
        dummy_copy(&dev, temp2, out3.buffer()); //as out1 has droped we expect to reuse the buffer of out1 for the output here
        
        assert_eq!(dev.get_internal_counters().buffer_counter, 1);
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 0, "buffer_reuse_counter has been calculated incorrectly");
        dev.synchronize().expect("sync");
        assert_eq!(dev.get_internal_counters().buffer_counter, 3);
        assert_eq!(dev.get_internal_counters().buffer_reuse_counter, 1, "buffer_reuse_counter has been calculated incorrectly");
    }
}

#[test]
fn counters_unary_and_binary_inplace() {
   let dev = create_device();
   
    let input = dev.alloc_from_slice(DType::U32, &[1, 2, 3, 4]).expect("alloc");
    let out = dev.alloc_uninit_size(DType::U32, 4);

    assert_eq!(dev.get_internal_counters().unary_inplace_counter, 0);

    dummy_unary_rewrite(&dev, input.buffer(), out.buffer());
    drop(input);
    dev.synchronize().expect("sync");

    assert_eq!(dev.get_internal_counters().unary_inplace_counter, 1);
}
