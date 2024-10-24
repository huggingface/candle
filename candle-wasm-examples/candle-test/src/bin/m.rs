use candle::{backend::BackendDevice, wgpu::{wgpu_functions::Pipelines, MatmulAlgorithm}, Device, Shape, Tensor, WgpuStorage};

mod utils;

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
    //performance_test().await?;
    test_matmul().await?;
    Ok(())
}

fn load_recording_consts(device : &Device) -> Result<(), Box<dyn std::error::Error>>{
    let debug_recordings_consts = include_str!("wgpu_stable_diffusion_test_1_e.json");
    let debug_recordings_consts :  Vec<std::collections::HashMap<String, f64>> = serde_json::from_str(debug_recordings_consts)?;
    match &device{
        Device::Wgpu(wgpu) => {
            wgpu.load_debug_info(debug_recordings_consts);
           
        },
        _ => todo!(),
    }
    Ok(())
}

fn format_bytes(bytes: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    if bytes >= TB {
        format!("{:.2} TiB", bytes / TB)
    } else if bytes >= GB {
        format!("{:.2} GiB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes / KB)
    } else {
        format!("{} bytes", bytes)
    }
}

async fn test_matmul() -> Result<(), Box<dyn std::error::Error>>{
    let device = candle::Device::new_wgpu(0).await?;

    let b = 1;
    let m = 2048;
    let k = 2048;
    let n = 2048;

    let flops = b*m*k*n;

    let algs = vec![
       // MatmulAlgorithm::Matmul1
        MatmulAlgorithm::Matmul32_64,
        //MatmulAlgorithm::Matmul64_64_8_8,
        //MatmulAlgorithm::Matmul64_64_4_8,
       
        // MatmulAlgorithm::MatmulX,
        // MatmulAlgorithm::Matmul7,
        // MatmulAlgorithm::Matmul16_16,
        // MatmulAlgorithm::Matmul32_32,
        // MatmulAlgorithm::Matmul24_24,
        // MatmulAlgorithm::Matmul24_48,
        // MatmulAlgorithm::Matmul64_64,
       ];

   
    let dtype = candle::DType::F32;
    let buffer_a = Tensor::ones((b, m, k), dtype, &device)?;
    let buffer_b = Tensor::ones((b, k, n), dtype, &device)?;

    buffer_a.matmul(&buffer_b).unwrap();
    device.synchronize_async().await?;

    let mut measurements : Vec<MeasurementInfo> = vec![];
    match &device{
        Device::Wgpu(wgpu) => {
            for alg in algs{
                *wgpu.matmul_alg.lock().unwrap() = alg.clone();

                test_func(&device, 1000, || {
                    buffer_a.matmul(&buffer_b).unwrap();
                    Ok(())}, &format!("{:?}:",alg), &mut measurements, 0).await;    
                }

                if let Some(l) = measurements.last()
                {
                    log::warn!("throughput: {}",  format_bytes(flops as f64 / l.result.mean))
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
        Device::Wgpu(wgpu) => {
            let buf1 = wgpu.ones_impl(&shape, dtype)?;
            let buf2 = wgpu.ones_impl(&shape, dtype)?;
            let buf3 = wgpu.ones_impl(&shape, dtype)?;
            let buf4 = wgpu.ones_impl(&shape, dtype)?;
           Ok([buf1, buf2, buf3, buf4])
        },
        _ => todo!(),
    }
}

pub async fn performance_test() -> Result<(), Box<dyn std::error::Error>>{
    let device = candle::Device::new_wgpu(0).await?;

    load_recording_consts(&device)?;
    let buffers = create_buffers(&device)?;

    let debug_recordings = include_str!("wgpu_stable_diffusion_test_1_d.json");
    let debug_recordings : Vec<candle::wgpu_backend::DebugPipelineRecording> = serde_json::from_str(debug_recordings)?;
    let debug_recordings : Vec<_> = debug_recordings.iter().filter(|v| matches!(&v.pipeline.0.into(), Pipelines::Matmul64x648x8(_,_))).collect();


    let mut measurements : Vec<MeasurementInfo> = vec![];

    match &device{
        Device::Wgpu(wgpu) => {
            let total = debug_recordings.len();
            for (index, command) in debug_recordings.iter().enumerate(){
                
                log::warn!("progress: {index}/{total}");
                test_func(&device, 10, || {
                    wgpu.simulate_command(command, &buffers[0], &buffers[1], &buffers[2], &buffers[3]); Ok(())}, &serde_json::to_string(command).unwrap().to_string(), &mut measurements, command.count).await;    
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
    
   
    Ok(())
}

async fn test_func<'a,F>(device : &Device, count: u32, func : F, name : &str, measures : &mut Vec<MeasurementInfo>, total_counts : u32)
where   F: Fn() -> Result<(), candle::Error>,
{
        let device_name = match device{
            Device::Cpu => "CPU:",
            Device::Wgpu(_) => "GPU:",
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