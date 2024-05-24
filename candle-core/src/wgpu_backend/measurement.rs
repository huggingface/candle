
use std::{collections::HashMap, fs::File};
use serde_json::to_writer;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct MeasurementInfo {
    pub result: Measurement,
    pub name : String,
    pub device : String,
    pub  size : u32
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

