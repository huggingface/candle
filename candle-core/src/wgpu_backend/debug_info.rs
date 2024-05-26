use std::{collections::HashMap, fs::File, sync::{atomic::AtomicU32, Arc, Mutex}};
use wgpu::Device;

use serde::{Deserialize, Serialize};
use serde_json::to_writer;

#[derive(Debug, Clone)]
pub struct DebugInfo{
    pub (crate) query_set_buffer : Arc<wgpu::Buffer>,
    pub (crate) counter :  Arc<AtomicU32>,
    pub (crate) shader_pipeline : Arc<Mutex<HashMap<u32, String>>>,
}

impl DebugInfo {
    pub (crate) fn new(device : &Device) -> Self{
        // Create a buffer to store the query results
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 10000000 as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });
        return DebugInfo{
            counter : Arc::new(AtomicU32::new(0)), 
            shader_pipeline : Arc::new(Mutex::new(HashMap::new())), 
            query_set_buffer : Arc::new(query_buffer)};
    }

    pub (crate) fn insert_info(&self, index : u32, info : String){
        //let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        //let filtered_lines: Vec<&str> = backtrace
        //.lines()
        //.filter(|line| line.contains("candle")).collect();

        //let lines = filtered_lines.join("\n");
        self.shader_pipeline.lock().expect("could not lock debug info").insert(index, info);     
        //self.shader_pipeline.lock().expect("could not lock debug info").insert(index,format!("{info}\nTrace{}", lines));     
    }

}


#[derive(Serialize, Deserialize, Clone)]
pub struct MInfo{
    pub label: String,
    pub start_time : u64,
    pub end_time : u64
}

impl MInfo {
    pub fn new(label: String, start_time: u64, end_time: u64) -> Self {
        Self { label, start_time, end_time }
    }
}

#[derive(Serialize, Deserialize)]

pub struct Measurements{
    pub data : Vec<MInfo>,
    timestamp_period : f32
}

impl Measurements {
    pub fn new(timestamp_period: f32) -> Self {
        Self { data : vec![], timestamp_period }
    }
}


#[derive(Serialize, Deserialize)]
pub struct MeasurementInfo {
    pub result: Measurement,
    pub name : String,
    pub device : String,
    pub size : u32
}

impl MeasurementInfo {
    pub fn new<S1 : Into<String>, S2 : Into<String>>(result: Measurement, name: S1, device: S2, size: usize) -> Self {
        Self { result, name : name.into(), device : device.into(), size : size as u32 }
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

pub fn calulate_measurment(map: &HashMap<String, Vec<u64>>) -> Vec<Measurement>{
    const NANO : f64 = 1e9;
    
    return map.iter().map(|(k,data)| {
        let count = data.len();
        let sum: u64 = data.iter().sum();
        let mean = (sum as f64 / count as f64) / NANO;
        let max = *data.iter().max().unwrap() as f64 / NANO;
        let min = *data.iter().min().unwrap() as f64  / NANO;

        let variance = data.iter().map(|&value| {
            let diff = mean - (value as f64 / NANO) as f64;
            diff * diff
        }).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        return Measurement{ label: k.to_owned(), mean, min, max, std: std_dev, count: count as u32 }

    }).collect();
}

pub fn save_list<T: Serialize>(measurements : &T, file_name : &str) -> Result<(),Box<dyn std::error::Error>>{
    let file = File::create(file_name)?;
    to_writer(file, measurements)?;
    Ok(())
}
