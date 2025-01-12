use anyhow::Result;
use candle_core::{backend::BackendStorage, CustomOp1, Device, Tensor};
use candle_wgpu_kernels::{PipelineIndex, ShaderIndex};

//this demonstrates, how a custom wgpu kernel can be used:
candle_wgpu_kernels::create_loader!(MyCustomLoader);

impl candle_wgpu_kernels::ShaderLoader for MyCustomLoader {
    //define the shader:
    fn load(&self, _: candle_wgpu_kernels::ShaderIndex) -> &str {
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
    fn get_entry_point(&self, index: candle_wgpu_kernels::PipelineIndex) -> &str {
        match index.get_index() {
            0 => "main1",
            1 => "main2",
            _ => {
                todo!()
            }
        }
    }
}

fn main() -> Result<()> {
    let device = &Device::new_wgpu(0)?;

    match &device {
        Device::Wgpu(wgpu_device) => {
            //1. add the custom loader to the device
            wgpu_device.add_wgpu_shader_loader(MyCustomLoader::LOADER_INDEX, || MyCustomLoader {});

            //2. add optional data to the meta - structure
            let mut meta = candle_core::wgpu::wgpu_functions::get_queue(wgpu_device);
            meta.add(42);

            //3. define the pipeline to use:
            let pipeline = meta.get_pipeline(PipelineIndex::new(
                ShaderIndex::new(MyCustomLoader::LOADER_INDEX, 0),
                0,
            ));

            let output_buffer = candle_core::wgpu::create_wgpu_storage(
                wgpu_device,
                candle_core::DType::U32,
                candle_core::DType::U32.size_in_bytes(),
            );

            //4. define the bindgroup to use (defines dest, input buffer and the alignment)
            let bind_group = candle_core::wgpu::wgpu_functions::create_bind_group_input0(
                *output_buffer.buffer(),
                candle_core::DType::U32.into(),
            );

            //5. add the command to the queue:
            candle_core::wgpu::wgpu_functions::enqueue_64(meta, pipeline, bind_group, 1, 1);

            let cpu_storage_data = output_buffer.to_cpu_storage()?;

            match cpu_storage_data {
                candle_core::CpuStorage::U32(vec) => {
                    assert_eq!(vec[0], 42 * 2);
                }
                _ => todo!(),
            }
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
        let mut meta = candle_core::wgpu::wgpu_functions::get_queue(storage.device());
        meta.add(42);

        //3. define the pipeline to use:
        let pipeline = meta.get_pipeline(PipelineIndex::new(
            ShaderIndex::new(MyCustomLoader::LOADER_INDEX, 0),
            1,
        ));

        let output_buffer = candle_core::wgpu::create_wgpu_storage(
            storage.device(),
            candle_core::DType::U32,
            candle_core::DType::U32.size_in_bytes(),
        );

        //4. define the bindgroup to use (defines dest, input buffer and the alignment)
        let bind_group = candle_core::wgpu::wgpu_functions::create_bind_group_input1(
            *output_buffer.buffer(),
            *storage.buffer(),
            candle_core::DType::U32.into(),
        );

        //5. queue the command to the queue:
        candle_core::wgpu::wgpu_functions::enqueue_64(meta, pipeline, bind_group, 1, 1);

        Ok((output_buffer, ().into()))
    }
}
