//! Support for the GGUF file format.
//!
//! Spec: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md

use super::GgmlDType;
use crate::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => crate::bail!("unknown magic {value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            _ => crate::bail!("ggml: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: crate::Shape,
    pub offset: u64,
}

impl TensorInfo {
    pub fn read<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
    ) -> Result<super::QTensor> {
        let tensor_elems = self.shape.elem_count();
        let size_in_bytes =
            tensor_elems * self.ggml_dtype.type_size() / self.ggml_dtype.blck_size();
        let mut raw_data = vec![0u8; size_in_bytes];
        reader.seek(std::io::SeekFrom::Start(tensor_data_offset + self.offset))?;
        reader.read_exact(&mut raw_data)?;
        super::ggml_file::qtensor_from_ggml(self.ggml_dtype, &raw_data, self.shape.dims().to_vec())
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u32::<LittleEndian>()?;
    let mut v = vec![0u8; len as usize];
    reader.read_exact(&mut v)?;
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    fn read<R: std::io::Read>(reader: &mut R, value_type: ValueType) -> Result<Self> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader)?),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type)?)
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            v => crate::bail!("unrecognized value-type {v}"),
        };
        Ok(v)
    }
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(reader)?;
        let tensor_count = reader.read_u32::<LittleEndian>()? as usize;
        let metadata_kv_count = reader.read_u32::<LittleEndian>()?;
        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader)?;
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type)?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()?;
            let mut dimensions = vec![0u32; n_dimensions as usize];
            reader.read_u32_into::<LittleEndian>(&mut dimensions)?;
            dimensions.reverse();
            let dimensions: Vec<usize> = dimensions.into_iter().map(|c| c as usize).collect();
            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: crate::Shape::from(dimensions),
                    offset,
                    ggml_dtype,
                },
            );
        }
        let position = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = (position + alignment - 1) / alignment * alignment;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }
}
