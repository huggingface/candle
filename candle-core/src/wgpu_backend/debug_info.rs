use std::{collections::HashMap, fs::File, sync::{atomic::AtomicU32, Mutex}};
use wgpu::Device;

use serde::{Deserialize, Serialize};
use serde_json::to_writer;

#[derive(Debug)]
pub struct DebugInfo{
    pub (crate) query_set_buffer : wgpu::Buffer,
    pub (crate) query_set : wgpu::QuerySet,
    pub (crate) counter :  AtomicU32,
    pub (crate) shader_pipeline : Mutex<HashMap<u32, (String, u64, u32, u32, u32)>>,
}

impl DebugInfo {
    pub (crate) fn new(device : &Device) -> Self{
        // Create a buffer to store the query results
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 100000 as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            count: 4096 * 2, // We need 2 queries: one for start and one for end
            ty: wgpu::QueryType::Timestamp,
            label: None,
        });

        return DebugInfo{
            counter : AtomicU32::new(0), 
            query_set : query_set, 
            shader_pipeline : Mutex::new(HashMap::new()), 
            query_set_buffer : query_buffer};
    }

    pub (crate) fn insert_info(&self, index : u32, info : (String, u64, u32, u32, u32)){
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
    pub end_time : u64,
    pub output_size : u64,
    pub x : u32,
    pub y : u32,
    pub z : u32,
}

impl MInfo {
    pub fn new(label: String, start_time: u64, end_time: u64, output_size: u64, x: u32, y: u32, z: u32) -> Self {
        Self { label, start_time, end_time, output_size, x, y, z }
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
pub enum MeasurementType{
    Duration, 
    OutputSize,
    DispatchSize
}


#[derive(Serialize, Deserialize)]
pub struct Measurement {
    pub label: String,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std: f64,
    pub count : u32,
    pub m_type : MeasurementType,
}

#[derive(Serialize, Deserialize)]
pub struct PipelineInfo{
    pub name : String,
    pub consts : Vec<f64>
}

#[derive(Serialize, Deserialize)]
pub struct ShaderInfo {
    pub name : String,
    pub pipelines : Vec<PipelineInfo>
}




pub fn calulate_measurment(map: &HashMap<String, Vec<(u64, u64, u32, u32, u32)>>) -> Vec<Measurement>{
    const NANO : f64 = 1e9;
    
    return map.iter().flat_map(|(k,data)| {
        let count = data.len();
        
        //dur
        let iter = data.iter().map(|f| f.0);
       
        let sum: u64 = iter.clone().sum();
        let mean = (sum as f64 / count as f64) / NANO;
        let max = iter.clone().max().unwrap() as f64 / NANO;
        let min = iter.clone().min().unwrap() as f64  / NANO;

        let variance = iter.clone().map(|value| {
            let diff = mean - (value as f64 / NANO) as f64;
            diff * diff
        }).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let m1 = Measurement{ label: k.to_owned(), mean , min , max , std: std_dev, count: count as u32, m_type : MeasurementType::Duration};
        
        //out_size
        let iter = data.iter().map(|f| f.1);

        let sum: u64 = iter.clone().sum();
        let mean = sum as f64 / count as f64;
        let max = iter.clone().max().unwrap() as f64;
        let min = iter.clone().min().unwrap() as f64;

        let variance = iter.clone().map(|value| {
            let diff = mean - (value as f64) as f64;
            diff * diff
        }).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        let m2 = Measurement{ label: k.to_owned(), mean , min , max , std: std_dev, count: count as u32, m_type : MeasurementType::OutputSize};
        

        //dispatch size
        let iter = data.iter().map(|f| f.2 * f.3 * f.4);

        let sum: u32 = iter.clone().sum();
        let mean = sum as f64 / count as f64;
        let max = iter.clone().max().unwrap() as f64;
        let min = iter.clone().min().unwrap() as f64;

        let variance = iter.clone().map(|value| {
            let diff = mean - (value as f64) as f64;
            diff * diff
        }).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        let m3 = Measurement{ label: k.to_owned(), mean , min , max , std: std_dev, count: count as u32, m_type : MeasurementType::DispatchSize};
        
        return [m1, m2, m3];
    }).collect();
}

pub fn save_list<T: Serialize>(measurements : &T, file_name : &str) -> Result<(),Box<dyn std::error::Error>>{
    let file = File::create(file_name)?;
    to_writer(file, measurements)?;
    Ok(())
}
