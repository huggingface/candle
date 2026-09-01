use std::future::IntoFuture;

use log::warn;
use web_time::{Duration, Instant};

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct MeasurementInfo {
    pub result: Measurement,
    pub name : String,
    pub device : String,
    pub size : u32,
    pub count : u32,
}

impl MeasurementInfo {
    pub fn new<S1 : Into<String>, S2 : Into<String>>(result: Measurement, name: S1, device: S2, size: usize, count : u32) -> Self {
        Self { result, name : name.into(), device : device.into(), size : size as u32, count  }
    }
}


#[derive(Serialize, Deserialize)]
pub struct Measurement {
    pub label: String,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std: f64,
    pub count : u32
}

#[cfg(not(target_arch="wasm32"))]
pub fn save_list<T: Serialize>(measurements : &T, file_name : &str) -> Result<(),Box<dyn std::error::Error>>{
    let file = std::fs::File::create(file_name)?;
    serde_json::to_writer(file, measurements)?;
    Ok(())
}

// Stolen from `bencher`, where it's known as `black_box`.
//
// NOTE: We don't have a proper black box in stable Rust. This is
// a workaround implementation, that may have a too big performance overhead,
// depending on operation, or it may fail to properly avoid having code
// optimized out. It is good enough that it is used by default.
//
// A function that is opaque to the optimizer, to allow benchmarks to
// pretend to use outputs to assist in avoiding dead-code
// elimination.
fn pretend_to_use<T>(dummy: T) -> T {
    unsafe {
        let ret = ::std::ptr::read_volatile(&dummy);
        ::std::mem::forget(dummy);
        ret
    }
}

pub async fn bench_function_single_async<F : Fn() -> T, T : IntoFuture>(func : F) -> web_time::Duration{
    let start = Instant::now();
    let val = func().await;
    let end = Instant::now();
    pretend_to_use(val);
    end - start
}


pub async fn bench_function_max_time_async<F : Fn() -> T, T : IntoFuture>(id : &str, func : F, max_time : Duration, max_step : u32, iterations_per_step : usize) -> Measurement{
    warn!("Running Benchmark {id}");

    let mut durations = vec![];

    let start = Instant::now();

    for _ in 0..max_step{
        durations.push(bench_function_single_async(&func).await);
    
        if Instant::now() - start > max_time{
            break;
        }
    }
    let durations_f64 : Vec<f64> = durations.iter().map(|c| c.as_secs_f64() / iterations_per_step as f64).collect();
    let count = durations.len();

    let d_min =  durations_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let d_max =  durations_f64.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    let d_mean : f64 = durations_f64.iter().sum();
    let d_mean = d_mean / count as f64;

    let variance = durations_f64.iter().map(|value| {
        let diff = d_mean - *value;

        diff * diff
    }).sum::<f64>() / count as f64;
    let d_std = variance.sqrt();

    let result = Measurement{ 
        label: id.to_owned(), 
        mean: d_mean, 
        min: d_min,
        max: d_max, 
        std: d_std,
        count : count as u32 };

    warn!("time: [{:?} {:?} {:?}] +-{:?} ({count} iterations)", Duration::from_secs_f64(d_min), Duration::from_secs_f64(d_mean) ,Duration::from_secs_f64(d_max), Duration::from_secs_f64(d_std));
    result
}
