use anyhow::Result;
use candle_core::{backend::BackendStorage, CustomOp1, Device, Tensor};
use wgpu_compute_engine::{PipelineIndex, ShaderIndex};

//this demonstrates, how a custom wgpu kernel can be used:
#[derive(Debug)]
struct MyCustomLoader{}

wgpu_compute_engine::create_loader!(MyCustomLoader);

impl wgpu_compute_engine::ShaderLoader for MyCustomLoader {
    //define the shader:
    fn load(&self, _: wgpu_compute_engine::ShaderIndex) -> &str {
        "
//Binding Order: Dest, Meta, Input1, Input2, Input3
@group(0) @binding(0)
var<storage, read_write> v_dest: array<u32>;

@group(0) @binding(1)
var<storage> op_meta : array<u32>;

@group(0) @binding(2)
var<storage> v_input1: array<u32>;

@compute @workgroup_size(1)
fn main1() {
    v_dest[0] = 2 * op_meta[0];
}
@compute @workgroup_size(1)
fn main2() {
    v_dest[0] = v_input1[0] * op_meta[0];
}
        "
    }

    //define the entry point:
    fn get_entry_point(&self, index: wgpu_compute_engine::PipelineIndex) -> &str {
        match index.get_index() {
            0 => "main1",
            1 => "main2",
            _ => {
                todo!()
            }
        }
    }
}

#[cfg(feature = "wgpu")]
fn main() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
    let wgpu_device= device.as_wgpu_device()?;

    //0. add the custom loader to the device(this must be done only once)
    wgpu_device.add_wgpu_shader_loader(MyCustomLoader::LOADER_INDEX, || MyCustomLoader {});

    let mut queue = wgpu_device.get_queue();
    let output_buffer = wgpu_device.alloc_uninit_size(candle_core::DType::U32, 1);
    
    //1. add optional data for the next shader call
    queue.add(42);

    //2. define the pipeline to use:
    let pipeline = queue.get_pipeline(PipelineIndex::new(
        ShaderIndex::new(MyCustomLoader::LOADER_INDEX, 0),
        0,
    ));
   
    //3. define the bindgroup to use (defines dest, input buffer and the alignment)
    let bind_group = wgpu_device.create_bind_group_input0(
        output_buffer.buffer(),
        candle_core::DType::U32.into(),
    );

    //4. add the command to the queue:
    queue.enqueue_workgroups(pipeline, bind_group, 1, 1, 1, 1);

    let cpu_storage_data = output_buffer.to_cpu_storage()?;

    match cpu_storage_data {
        candle_core::CpuStorage::U32(vec) => {
            assert_eq!(vec[0], 42 * 2);
        }
        _ => todo!(),
    }

    let input = Tensor::from_slice(&[17u32], (), device)?;
    let output = input.apply_op1(CustomExampleOp {})?;

    assert_eq!(output.to_vec0::<u32>()?, 17 * 42u32);

    Ok(())
}

struct CustomExampleOp {}

impl CustomOp1 for CustomExampleOp {
    fn name(&self) -> &'static str {
        "CustomExampleOp"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        todo!()
    }

    fn wgpu_fwd(
        &self,
        storage: &candle_core::WgpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::WgpuStorage, candle_core::Shape)> {
        //1. add the custom loader to the device
        storage
            .device()
            .add_wgpu_shader_loader(MyCustomLoader::LOADER_INDEX, || MyCustomLoader {});

        //2. add optional data to the meta - structure
        let mut queue = storage.device().get_queue();
        queue.add(42);

        //3. define the pipeline to use:
        let pipeline = queue.get_pipeline(PipelineIndex::new(
            ShaderIndex::new(MyCustomLoader::LOADER_INDEX, 0),
            1,
        ));

        let output_buffer = storage.device().alloc_uninit_size(
            candle_core::DType::U32,
            1,
        );

        //4. define the bindgroup to use (defines dest, input buffer and the alignment)
        let bind_group = storage.device().create_bind_group_input1(
            output_buffer.buffer(),
            storage.buffer(),
            candle_core::DType::U32.into(),
        );

        //5. queue the command to the queue:
        queue.enqueue_workgroups(pipeline, bind_group, 1, 1, 1, 1);

        Ok((output_buffer, ().into()))
    }
}
