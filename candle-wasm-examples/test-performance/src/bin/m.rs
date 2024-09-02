use candle::{backend::BackendDevice, wgpu::{self, wgpu_functions::Pipelines, MatmulAlgorithm}, Device, Shape, Tensor, WgpuStorage};

mod utils;

use safetensors::Dtype;
use utils::{bench_function_max_time_async, MeasurementInfo};
use web_time::Duration;



fn main() -> Result<(), Box<dyn std::error::Error>>{
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Warn)
            .format_target(false)
            .format_timestamp(None)
            .init();
        pollster::block_on(test_main());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        wasm_logger::init(wasm_logger::Config::new(log::Level::Warn).message_on_new_line());
        wasm_bindgen_futures::spawn_local(test_main());
    }
    Ok(())
}

async fn test_main(){
    test().await.expect("Error while Executing");
}

async fn test() -> Result<(), Box<dyn std::error::Error>>{
    //performance_test2().await?;
    test_matmul().await?;
    Ok(())
}

fn load_recording_consts(device : &Device) -> Result<(), Box<dyn std::error::Error>>{
    let debug_recordings_consts = include_str!("wgpu_stable_diffusion_test_1_e.json");
    let debug_recordings_consts :  Vec<std::collections::HashMap<String, f64>> = serde_json::from_str(debug_recordings_consts)?;
    match &device{
        Device::WebGpu(wgpu) => {
            wgpu.load_debug_info(debug_recordings_consts);
           
        },
        _ => todo!(),
    }
    Ok(())
}

async fn test_matmul() -> Result<(), Box<dyn std::error::Error>>{
    let device = candle::Device::new_webgpu(0).await?;

    //load_recording_consts(&device)?;
    //let buffers = create_buffers(&device)?;

    //let command = r#"{"x":1,"y":64,"z":16,"pipeline":[{"Matmul64x648x8":["F32","Matmul"]},37],"meta":[16,4096,4096,64,16777216,0,262144,0],"bindgroup":[{"Bindgroup2":[true]}],"count":150}"#;
    //let command: candle::wgpu_backend::DebugPipelineRecording = serde_json::from_str(command)?;

    let b = 16;
    let m = 4096;
    let k = 4096;
    let n = 64;



    let algs = vec![
       // MatmulAlgorithm::Matmul1,
        MatmulAlgorithm::Matmul64_64(false, false),
        MatmulAlgorithm::Matmul64_64_8_8(false, false),
        MatmulAlgorithm::Matmul64_64_4_8(false, false),
       
        // MatmulAlgorithm::MatmulX,
        // MatmulAlgorithm::Matmul7,
        // MatmulAlgorithm::Matmul16_16,
        // MatmulAlgorithm::Matmul32_32(false, false, true, true),
        // MatmulAlgorithm::Matmul24_24(false,false, true, true),
        // MatmulAlgorithm::Matmul24_48(false,false, true, true),
        // MatmulAlgorithm::Matmul64_64(false, false),
       
        // MatmulAlgorithm::Matmul64_128(false, false),
        // MatmulAlgorithm::Matmul64_128_8_8(false, false),
        // MatmulAlgorithm::Matmul128_128(false, false),









        // MatmulAlgorithm::Matmul5_32_32(true, false, true, true),
        // MatmulAlgorithm::Matmul5_64_64(true, false),
        // MatmulAlgorithm::Matmul5_64_64_8_8(true, false),
        // MatmulAlgorithm::Matmul5_128_128(true, false),
     
        //MatmulAlgorithm::Matmul5_32_32(true, false, true, true),
        //MatmulAlgorithm::Matmul5_64_64(true, false),
        //MatmulAlgorithm::Matmul5_64_64_8_8(true, false),
        //MatmulAlgorithm::Matmul5_128_128(true, false),  

        // MatmulAlgorithm::Matmul5_32_32(false, true, true, true),
        // MatmulAlgorithm::Matmul5_64_64(false, true),
        // MatmulAlgorithm::Matmul5_64_64_8_8(false, true),
        // MatmulAlgorithm::Matmul5_128_128(false, true),
     
        // MatmulAlgorithm::Matmul5_32_32(true, true, true, true),
        // MatmulAlgorithm::Matmul5_64_64(true, true),
        // MatmulAlgorithm::Matmul5_64_64_8_8(true, true),
        // MatmulAlgorithm::Matmul5_128_128(true, true),  
       ];

   
    let dtype = candle::DType::F32;
    let buffer_a = Tensor::ones((b, m, k), dtype, &device)?;
    let buffer_b = Tensor::ones((b, k, n), dtype, &device)?;

    buffer_a.matmul(&buffer_b).unwrap();
    device.synchronize_async().await?;

    let mut measurements : Vec<MeasurementInfo> = vec![];
    match &device{
        Device::WebGpu(wgpu) => {
            for alg in algs{
                *wgpu.matmul_alg.lock().unwrap() = alg.clone();

                test_func(&device, 10, || {
                    buffer_a.matmul(&buffer_b).unwrap();
                    return Ok(())}, &format!("{:?}:",alg), &mut measurements, 0).await;    
            }
        },
        _ => {todo!()}
    }

    Ok(())
}

fn create_buffers(device : &Device) -> Result<[WgpuStorage;4], Box<dyn std::error::Error>>{
    let shape = Shape::from_dims(&[1000, 1000, 250]);
    let dtype = candle::DType::F32;
    match &device{
        Device::WebGpu(wgpu) => {
            let buf1 = wgpu.ones_impl(&shape, dtype)?;
            let buf2 = wgpu.ones_impl(&shape, dtype)?;
            let buf3 = wgpu.ones_impl(&shape, dtype)?;
            let buf4 = wgpu.ones_impl(&shape, dtype)?;
           return Ok([buf1, buf2, buf3, buf4]);
        },
        _ => todo!(),
    }
}

async fn performance_test2() -> Result<(), Box<dyn std::error::Error>>{
    let device = candle::Device::new_webgpu(0).await?;

    load_recording_consts(&device)?;
    let buffers = create_buffers(&device)?;

    let debug_recordings = include_str!("wgpu_stable_diffusion_test_1_d.json");
    let debug_recordings : Vec<candle::wgpu_backend::DebugPipelineRecording> = serde_json::from_str(debug_recordings)?;
    let debug_recordings : Vec<_> = debug_recordings.iter().filter(|v| if let Pipelines::Matmul64x648x8(_,_) = &v.pipeline.0 {return true;} else {return false;}).collect();


    let mut measurements : Vec<MeasurementInfo> = vec![];

    match &device{
        Device::WebGpu(wgpu) => {
            let total = debug_recordings.len();
            for (index, command) in debug_recordings.iter().enumerate(){
                
                log::warn!("progress: {index}/{total}");
                test_func(&device, 10, || {
                    wgpu.simulate_command(command, &buffers[0], &buffers[1], &buffers[2], &buffers[3]); return Ok(())}, &format!("{}", serde_json::to_string(command).unwrap()), &mut measurements, command.count).await;    
            }

        },
        _ => todo!(),
    }

    
    measurements.sort_by(|a, b| {(a.result.mean * (a.count as f64)).partial_cmp(&(b.result.mean * (b.count as f64))).unwrap_or(std::cmp::Ordering::Equal)});


    let total_time : f64 = measurements.iter().map(|v| {v.count as f64 * v.result.mean}).sum();

    for measure in measurements.iter(){
        log::warn!("{}:\nDuration: {:.3}s {:.2}%", measure.name, measure.result.mean * measure.count as f64, 100.0 * measure.result.mean * measure.count as f64 / total_time);
    }

    log::warn!("total_time: {total_time:.3}s");

    #[cfg(not(target_arch="wasm32"))]{
        utils::save_list(&measurements, "performance6.json")?;
    }
    
   
    return Ok(());
}

async fn test_func<'a,F>(device : &Device, count: u32, func : F, name : &str, measures : &mut Vec<MeasurementInfo>, total_counts : u32)
where   F: Fn() -> Result<(), candle::Error>,
{
        let device_name = match device{
            Device::Cpu => "CPU:",
            Device::WebGpu(_) => "GPU:",
            _ => todo!(),
        };

        let res = bench_function_max_time_async(&format!("{device_name}: {name}"), || async {
            for _ in 0..count{
                func().unwrap();
            }
            device.synchronize_async().await.unwrap();
        }, Duration::from_secs_f32(10.0), 10, count as usize).await;

        let m = MeasurementInfo{ name : name.to_owned(), result: res, device: device_name.to_owned(), size: 0, count : total_counts};
        measures.push(m);
}