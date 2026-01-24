use std::{
    collections::HashMap, fs::File, sync::{Mutex, atomic::AtomicU32}
};
use wgpu::Device;

use serde::{Deserialize, Serialize};
use serde_json::{to_string, to_writer};

use crate::WgpuDevice;

use super::device::WgpuDebugInfo;


#[derive(Debug, Clone)]
pub(crate) struct ShaderPerformanceMeasurmentDebugInfo {
    pub pipeline: String,
    pub workload_size: u64,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug)]

pub(crate) struct PerformanceMeasurmentDebugInfo {
    pub(crate) query_set_buffer: wgpu::Buffer,
    pub(crate) query_set: wgpu::QuerySet,
    pub(crate) counter: AtomicU32,
    pub(crate) shader_pipeline: Mutex<HashMap<u32, ShaderPerformanceMeasurmentDebugInfo>>,
}

impl PerformanceMeasurmentDebugInfo {
    pub(crate) fn new(device: &Device) -> Self {
        // Create a buffer to store the query results
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 1_000_000_u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            count: wgpu::QUERY_SET_MAX_QUERIES,
            ty: wgpu::QueryType::Timestamp,
            label: None,
        });

        PerformanceMeasurmentDebugInfo {
            counter: AtomicU32::new(0),
            query_set,
            shader_pipeline: Mutex::new(HashMap::new()),
            query_set_buffer: query_buffer,
        }
    }

    pub(crate) fn insert_info(&self, index: u32, info: ShaderPerformanceMeasurmentDebugInfo) {
        self.shader_pipeline
            .lock()
            .expect("could not lock debug info")
            .insert(index, info);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MInfo {
    pub label: String,
    pub start_time: u64,
    pub end_time: u64,
    pub output_size: u64,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl MInfo {
    pub fn new(
        label: String,
        start_time: u64,
        end_time: u64,
        output_size: u64,
        x: u32,
        y: u32,
        z: u32,
    ) -> Self {
        Self {
            label,
            start_time,
            end_time,
            output_size,
            x,
            y,
            z,
        }
    }
}

#[derive(Serialize, Deserialize)]

pub struct Measurements {
    pub data: Vec<MInfo>,
    timestamp_period: f32,
}

impl Measurements {
    pub fn new(timestamp_period: f32) -> Self {
        Self {
            data: vec![],
            timestamp_period,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct MeasurementInfo {
    pub result: Measurement,
    pub name: String,
    pub device: String,
    pub size: u32,
}

impl MeasurementInfo {
    pub fn new<S1: Into<String>, S2: Into<String>>(
        result: Measurement,
        name: S1,
        device: S2,
        size: usize,
    ) -> Self {
        Self {
            result,
            name: name.into(),
            device: device.into(),
            size: size as u32,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum MeasurementType {
    Duration,
    OutputSize,
    DispatchSize,
}

#[derive(Serialize, Deserialize)]
pub struct Measurement {
    pub label: String,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std: f64,
    pub count: u32,
    pub m_type: MeasurementType,
}

#[derive(Serialize, Deserialize)]
pub struct PipelineInfo {
    pub name: String,
    pub consts: Vec<(String, f64)>,
}

#[derive(Serialize, Deserialize)]
pub struct ShaderInfo {
    pub name: String,
    pub pipelines: Vec<PipelineInfo>,
}

pub fn calulate_measurment(map: &HashMap<String, Vec<WgpuDebugInfo>>) -> Vec<Measurement> {
    const NANO: f64 = 1e9;
    map
        .iter()
        .flat_map(|(k, data)| {
            let count = data.len();

            //dur
            let iter = data.iter().map(|f| f.duration);

            let sum: u64 = iter.clone().sum();
            let mean = (sum as f64 / count as f64) / NANO;
            let max = iter.clone().max().unwrap() as f64 / NANO;
            let min = iter.clone().min().unwrap() as f64 / NANO;

            let variance = iter
                .clone()
                .map(|value| {
                    let diff = mean - (value as f64 / NANO);
                    diff * diff
                })
                .sum::<f64>()
                / count as f64;
            let std_dev = variance.sqrt();

            let m1 = Measurement {
                label: k.to_owned(),
                mean,
                min,
                max,
                std: std_dev,
                count: count as u32,
                m_type: MeasurementType::Duration,
            };

            //out_size
            let iter = data.iter().map(|f| f.output_size);

            let sum: u64 = iter.clone().sum();
            let mean = sum as f64 / count as f64;
            let max = iter.clone().max().unwrap() as f64;
            let min = iter.clone().min().unwrap() as f64;

            let variance = iter
                .clone()
                .map(|value| {
                    let diff = mean - (value as f64);
                    diff * diff
                })
                .sum::<f64>()
                / count as f64;
            let std_dev = variance.sqrt();
            let m2 = Measurement {
                label: k.to_owned(),
                mean,
                min,
                max,
                std: std_dev,
                count: count as u32,
                m_type: MeasurementType::OutputSize,
            };

            //dispatch size
            let iter = data.iter().map(|f| f.x * f.y * f.z);

            let sum: u32 = iter.clone().sum();
            let mean = sum as f64 / count as f64;
            let max = iter.clone().max().unwrap() as f64;
            let min = iter.clone().min().unwrap() as f64;

            let variance = iter
                .clone()
                .map(|value| {
                    let diff = mean - (value as f64);
                    diff * diff
                })
                .sum::<f64>()
                / count as f64;
            let std_dev = variance.sqrt();
            let m3 = Measurement {
                label: k.to_owned(),
                mean,
                min,
                max,
                std: std_dev,
                count: count as u32,
                m_type: MeasurementType::DispatchSize,
            };

            [m1, m2, m3]
        })
        .collect()
}

pub fn save_list<T: Serialize>(
    measurements: &T,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(file_name)?;
    to_writer(file, measurements)?;
    Ok(())
}

pub fn get_list<T: Serialize>(measurements: &T) -> Result<String, Box<dyn std::error::Error>> {
    Ok(to_string(measurements)?)
}





/*****  Helper data structures for recording full debug data with input and output buffers *****/


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum NumericArray {
    U8(Vec<u8>),
    F32(Vec<f32>),
    U32(Vec<u32>),
    I32(Vec<i32>),
}

#[derive(Debug)]
pub (crate) struct DebugRecordingWithData{
    pub recordings : Vec<DebugPipelineRecordingWithData>,
    pub should_record : bool,
}



#[derive(Debug)]
#[cfg(feature = "wgpu_debug")]
pub(crate) struct DebugPipelineRecordingWithData{
    pub recording : super::DebugPipelineRecording,
    pub(crate) v_dest : NumericArray, 
    pub(crate) v_input1 : Option<NumericArray>, 
    pub(crate) v_input2 : Option<NumericArray>,
    pub(crate) v_input3 : Option<NumericArray>
}

#[derive(Serialize)]
struct DispatchJson<'a> {
    compute_entry_point: &'a str,
    shader_debug_name: &'a str,
    x: u32,
    y: u32,
    z: u32,
    meta: &'a [u32],
    const_array: &'a [(&'a str, f64)],
    v_dest: ArrayRef<'a>,
    v_input1: Option<ArrayRef<'a>>,
    v_input2: Option<ArrayRef<'a>>,
    v_input3: Option<ArrayRef<'a>>,
}

#[derive(Serialize)]
struct ArrayRef<'a> {
    #[serde(rename = "type")]
    ty: &'a str,
    file: &'a str,
}


fn write_numeric_array<W: Write>(
    writer: &mut W,
    array: &NumericArray,
) -> crate::Result<()> {
    match array {
        NumericArray::U8(v) => {
            for x in v {
                writer.write_all(&x.to_le_bytes())?;
            }
        }
        NumericArray::F32(v) => {
            for x in v {
                writer.write_all(&x.to_le_bytes())?;
            }
        }
        NumericArray::U32(v) => {
            for x in v {
                writer.write_all(&x.to_le_bytes())?;
            }
        }
        NumericArray::I32(v) => {
            for x in v {
                writer.write_all(&x.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn array_type_name(array: &NumericArray) -> &'static str {
    match array {
        NumericArray::U8(_) => "U8",
        NumericArray::F32(_) => "F32",
        NumericArray::U32(_) => "U32",
        NumericArray::I32(_) => "I32",
    }
}


#[cfg(feature = "wgpu_debug")]
use zip::{ZipWriter, write::FileOptions};
use std::io::Write;
use serde_json::to_vec_pretty;

#[cfg(feature = "wgpu_debug")]
pub (crate) fn create_dispatch_zip(
    wgpu : &WgpuDevice,
    output_path: &str
) -> crate::Result<()> {
    let file = File::create(output_path)?;
    let mut zip = ZipWriter::new(file);
    let options : FileOptions<()> = FileOptions::default();
    let cache = wgpu.cache.lock().unwrap();
    let dispatches = &cache.full_recording.recordings;
    let loader_cache = &cache.shader.loader_cache;
    let queue: std::sync::MutexGuard<'_, super::queue_buffer::QueueBufferInner> = wgpu.command_queue.lock().unwrap();
     let consts = &queue.id_to_const_array;

    for (i, dispatch) in dispatches.iter().enumerate() {
        let base = format!("dispatch_{}/", i);


        // shader.wgsl
        zip.start_file(format!("{base}shader.wgsl"), options)?;
        zip.write_all( loader_cache.get_shader(dispatch.recording.pipeline.0.get_shader()).as_bytes())?;

        // v_dest.bin
        zip.start_file(format!("{base}v_dest.bin"), options)?;
        write_numeric_array(&mut zip, &dispatch.v_dest)?;

        // optional inputs
        if let Some(input) = &dispatch.v_input1 {
            zip.start_file(format!("{base}v_input1.bin"), options)?;
            write_numeric_array(&mut zip, input)?;
        }

        if let Some(input) = &dispatch.v_input2 {
            zip.start_file(format!("{base}v_input2.bin"), options)?;
            write_numeric_array(&mut zip, input)?;
        }

        if let Some(input) = &dispatch.v_input3 {
            zip.start_file(format!("{base}v_input3.bin"), options)?;
            write_numeric_array(&mut zip, input)?;
        }

        // dispatch.json
        let json = DispatchJson {
            compute_entry_point: loader_cache.get_entry_point(dispatch.recording.pipeline.0),
            shader_debug_name : &loader_cache.get_shader_name(dispatch.recording.pipeline.0.get_shader()),
            x: dispatch.recording.x,
            y: dispatch.recording.y,
            z: dispatch.recording.z,
            meta: &dispatch.recording.meta,
            const_array: &consts[dispatch.recording.pipeline.1][..],
            v_dest: ArrayRef {
                ty: array_type_name(&dispatch.v_dest),
                file: "v_dest.bin",
            },
            v_input1: dispatch.v_input1.as_ref().map(|a| ArrayRef {
                ty: array_type_name(a),
                file: "v_input1.bin",
            }),
            v_input2: dispatch.v_input2.as_ref().map(|a| ArrayRef {
                ty: array_type_name(a),
                file: "v_input2.bin",
            }),
            v_input3: dispatch.v_input3.as_ref().map(|a| ArrayRef {
                ty: array_type_name(a),
                file: "v_input3.bin",
            }),
        };

        zip.start_file(format!("{base}dispatch.json"), options)?;
        zip.write_all(&to_vec_pretty(&json).map_err(|e| crate::Error::Msg(format!("{:?}", e)))?)?;
    }

    zip.finish()?;
    Ok(())
}