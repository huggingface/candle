//! Support for the [GGUF file format](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
//!
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md  

use super::{GgmlDType, QTensor};
use crate::{Context, Device, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;

pub const DEFAULT_ALIGNMENT: u64 = 32;

// GGUF stores several lengths and counts as file-controlled integers. Keep these
// checks close to the parser so invalid files fail before allocation or seeking.
const GGUF_MAX_STRING_LENGTH: u64 = 1 << 30;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const GGUF_MAX_METADATA_KV_COUNT: u64 = 1 << 20;
const GGUF_MAX_TENSOR_COUNT: u64 = 1 << 20;
const GGUF_MAX_TENSOR_DIMS: u32 = 4;
const GGUF_MAX_VALUE_DEPTH: usize = 64;
const GGUF_MAX_ALIGNMENT: u64 = 1 << 20;

fn reader_len<R: std::io::Seek>(reader: &mut R) -> Result<u64> {
    let current = reader.stream_position()?;
    let len = reader.seek(std::io::SeekFrom::End(0))?;
    reader.seek(std::io::SeekFrom::Start(current))?;
    Ok(len)
}

fn read_len<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> Result<u64> {
    match magic {
        VersionedMagic::GgufV1 => Ok(reader.read_u32::<LittleEndian>()? as u64),
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => Ok(reader.read_u64::<LittleEndian>()?),
    }
}

fn checked_len(value: u64, max: u64, what: &str) -> Result<usize> {
    if value > max {
        crate::bail!("gguf: {what} {value} exceeds maximum {max}")
    }
    usize::try_from(value)
        .map_err(|_| crate::Error::msg(format!("gguf: {what} {value} does not fit in usize")))
}

fn checked_add_u64(lhs: u64, rhs: u64, what: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| crate::Error::msg(format!("gguf: overflow while computing {what}")))
}

fn checked_mul_u64(lhs: u64, rhs: u64, what: &str) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| crate::Error::msg(format!("gguf: overflow while computing {what}")))
}

fn checked_mul_usize(lhs: usize, rhs: usize, what: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| crate::Error::msg(format!("gguf: overflow while computing {what}")))
}

fn ensure_remaining<R: std::io::Seek>(
    reader: &mut R,
    file_len: u64,
    needed: u64,
    what: &str,
) -> Result<()> {
    let pos = reader.stream_position()?;
    let remaining = file_len
        .checked_sub(pos)
        .ok_or_else(|| crate::Error::msg("gguf: reader position is past end of file"))?;
    if needed > remaining {
        crate::bail!("gguf: need {needed} bytes for {what}, only {remaining} remain")
    }
    Ok(())
}

fn len_size(magic: &VersionedMagic) -> u64 {
    match magic {
        VersionedMagic::GgufV1 => 4,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => 8,
    }
}

fn value_type_min_size(value_type: ValueType, magic: &VersionedMagic) -> u64 {
    match value_type {
        ValueType::U8 | ValueType::I8 | ValueType::Bool => 1,
        ValueType::U16 | ValueType::I16 => 2,
        ValueType::U32 | ValueType::I32 | ValueType::F32 => 4,
        ValueType::U64 | ValueType::I64 | ValueType::F64 => 8,
        ValueType::String => len_size(magic),
        ValueType::Array => 4 + len_size(magic),
    }
}

fn checked_align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() || alignment > GGUF_MAX_ALIGNMENT {
        crate::bail!("gguf: invalid alignment {alignment}")
    }
    let value = checked_add_u64(value, alignment - 1, "aligned tensor data offset")?;
    Ok(value / alignment * alignment)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => crate::bail!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => crate::bail!("gguf: unsupported magic/version {magic:?}/{version}"),
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
    fn size_in_bytes(&self) -> Result<usize> {
        let block_size = self.ggml_dtype.block_size();
        super::check_shape(&self.shape, block_size)?;
        let tensor_elems = self.shape.dims().iter().try_fold(1usize, |acc, dim| {
            checked_mul_usize(acc, *dim, "tensor element count")
        })?;
        checked_mul_usize(
            tensor_elems / block_size,
            self.ggml_dtype.type_size(),
            "tensor byte size",
        )
    }

    fn validate_data_range(&self, tensor_data_offset: u64, file_len: u64) -> Result<()> {
        let size_in_bytes = self.size_in_bytes()? as u64;
        let offset = checked_add_u64(tensor_data_offset, self.offset, "tensor offset")?;
        let end = checked_add_u64(offset, size_in_bytes, "tensor end offset")?;
        if end > file_len {
            crate::bail!("gguf: tensor data range {offset}..{end} exceeds file length {file_len}")
        }
        Ok(())
    }

    pub fn read<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
        device: &Device,
    ) -> Result<QTensor> {
        let size_in_bytes = self.size_in_bytes()?;
        let mut raw_data = Vec::new();
        raw_data
            .try_reserve(size_in_bytes)
            .map_err(|e| crate::Error::msg(format!("gguf: failed to reserve tensor data: {e}")))?;
        raw_data.resize(size_in_bytes, 0);
        let offset = checked_add_u64(tensor_data_offset, self.offset, "tensor offset")?;
        reader.seek(std::io::SeekFrom::Start(offset))?;
        reader.read_exact(&mut raw_data)?;
        super::ggml_file::qtensor_from_ggml(
            self.ggml_dtype,
            &raw_data,
            self.shape.dims().to_vec(),
            device,
        )
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    magic: &VersionedMagic,
    file_len: u64,
) -> Result<String> {
    let len = read_len(reader, magic)?;
    let len = checked_len(len, GGUF_MAX_STRING_LENGTH, "string length")?;
    ensure_remaining(reader, file_len, len as u64, "string")?;
    let mut v = Vec::new();
    v.try_reserve(len)
        .map_err(|e| crate::Error::msg(format!("gguf: failed to reserve string: {e}")))?;
    v.resize(len, 0);
    reader.read_exact(&mut v)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
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
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn to_u8(&self) -> Result<u8> {
        match self {
            Self::U8(v) => Ok(*v),
            v => crate::bail!("not a u8 {v:?}"),
        }
    }

    pub fn to_i8(&self) -> Result<i8> {
        match self {
            Self::I8(v) => Ok(*v),
            v => crate::bail!("not a i8 {v:?}"),
        }
    }

    pub fn to_u16(&self) -> Result<u16> {
        match self {
            Self::U16(v) => Ok(*v),
            v => crate::bail!("not a u16 {v:?}"),
        }
    }

    pub fn to_i16(&self) -> Result<i16> {
        match self {
            Self::I16(v) => Ok(*v),
            v => crate::bail!("not a i16 {v:?}"),
        }
    }

    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => crate::bail!("not a u32 {v:?}"),
        }
    }

    pub fn to_i32(&self) -> Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            v => crate::bail!("not a i32 {v:?}"),
        }
    }

    /// This will also automatically upcast any integral types which will not truncate.
    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            // Autoupcast cases here
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::Bool(v) => Ok(*v as u64),
            v => crate::bail!("not a u64 or upcastable to u64 {v:?}"),
        }
    }

    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            v => crate::bail!("not a i64 {v:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            v => crate::bail!("not a f32 {v:?}"),
        }
    }

    pub fn to_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            v => crate::bail!("not a f64 {v:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => crate::bail!("not a bool {v:?}"),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => crate::bail!("not a vec {v:?}"),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => crate::bail!("not a string {v:?}"),
        }
    }

    fn read<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
        file_len: u64,
        depth: usize,
    ) -> Result<Self> {
        if depth > GGUF_MAX_VALUE_DEPTH {
            crate::bail!("gguf: value nesting depth exceeds maximum {GGUF_MAX_VALUE_DEPTH}")
        }
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic, file_len)?),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = read_len(reader, magic)?;
                let len = checked_len(len, GGUF_MAX_ARRAY_ELEMENTS, "array length")?;
                let min_size = checked_mul_u64(
                    len as u64,
                    value_type_min_size(value_type, magic),
                    "array byte size",
                )?;
                ensure_remaining(reader, file_len, min_size, "array values")?;
                let mut vs = Vec::new();
                vs.try_reserve(len).map_err(|e| {
                    crate::Error::msg(format!("gguf: failed to reserve array: {e}"))
                })?;
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type, magic, file_len, depth + 1)?)
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }

    fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        match self {
            &Self::U8(v) => w.write_u8(v)?,
            &Self::I8(v) => w.write_i8(v)?,
            &Self::U16(v) => w.write_u16::<LittleEndian>(v)?,
            &Self::I16(v) => w.write_i16::<LittleEndian>(v)?,
            &Self::U32(v) => w.write_u32::<LittleEndian>(v)?,
            &Self::I32(v) => w.write_i32::<LittleEndian>(v)?,
            &Self::U64(v) => w.write_u64::<LittleEndian>(v)?,
            &Self::I64(v) => w.write_i64::<LittleEndian>(v)?,
            &Self::F32(v) => w.write_f32::<LittleEndian>(v)?,
            &Self::F64(v) => w.write_f64::<LittleEndian>(v)?,
            &Self::Bool(v) => w.write_u8(u8::from(v))?,
            Self::String(v) => write_string(w, v.as_str())?,
            Self::Array(v) => {
                // The `Value` type does not enforce that all the values in an Array have the same
                // type.
                let value_type = if v.is_empty() {
                    // Doesn't matter, the array is empty.
                    ValueType::U32
                } else {
                    let value_type: std::collections::HashSet<_> =
                        v.iter().map(|elem| elem.value_type()).collect();
                    if value_type.len() != 1 {
                        crate::bail!("multiple value-types in the same array {value_type:?}")
                    }
                    value_type.into_iter().next().context("empty value_type")?
                };
                w.write_u32::<LittleEndian>(value_type.to_u32())?;
                w.write_u64::<LittleEndian>(v.len() as u64)?;
                for elem in v.iter() {
                    elem.write(w)?
                }
            }
        }
        Ok(())
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
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => crate::bail!("unrecognized value-type {v:#08x}"),
        };
        Ok(v)
    }

    fn to_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self> {
        let file_len = reader_len(reader)?;
        let magic = VersionedMagic::read(reader)?;

        let tensor_count = checked_len(
            read_len(reader, &magic)?,
            GGUF_MAX_TENSOR_COUNT,
            "tensor count",
        )?;
        let metadata_kv_count = checked_len(
            read_len(reader, &magic)?,
            GGUF_MAX_METADATA_KV_COUNT,
            "metadata count",
        )?;

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic, file_len)?;
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic, file_len, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                crate::bail!("gguf: duplicate metadata key {key}")
            }
        }
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic, file_len)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()?;
            if n_dimensions > GGUF_MAX_TENSOR_DIMS {
                crate::bail!(
                    "gguf: tensor rank {n_dimensions} exceeds maximum {GGUF_MAX_TENSOR_DIMS}"
                )
            }

            let mut dimensions = Vec::new();
            dimensions.try_reserve(n_dimensions as usize).map_err(|e| {
                crate::Error::msg(format!("gguf: failed to reserve tensor dimensions: {e}"))
            })?;
            for _ in 0..n_dimensions {
                let dim = read_len(reader, &magic)?;
                dimensions.push(checked_len(dim, usize::MAX as u64, "tensor dimension")?);
            }

            dimensions.reverse();
            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            let info = TensorInfo {
                shape: crate::Shape::from(dimensions),
                offset,
                ggml_dtype,
            };
            if tensor_infos.insert(tensor_name.clone(), info).is_some() {
                crate::bail!("gguf: duplicate tensor name {tensor_name}")
            }
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
        let tensor_data_offset = checked_align_up(position, alignment)?;
        for tensor_info in tensor_infos.values() {
            tensor_info.validate_data_range(tensor_data_offset, file_len)?
        }
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    pub fn tensor<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<QTensor> {
        let tensor_info = match self.tensor_infos.get(name) {
            Some(tensor_info) => tensor_info,
            None => crate::bail!("cannot find tensor info for {name}"),
        };
        tensor_info.read(reader, self.tensor_data_offset, device)
    }
}

fn write_string<W: std::io::Write>(w: &mut W, str: &str) -> Result<()> {
    let bytes = str.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64)?;
    w.write_all(bytes)?;
    Ok(())
}

pub fn write<W: std::io::Seek + std::io::Write>(
    w: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[(&str, &QTensor)],
) -> Result<()> {
    w.write_u32::<LittleEndian>(0x46554747)?;
    w.write_u32::<LittleEndian>(2)?; // version 2.
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (name, value) in metadata.iter() {
        write_string(w, name)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(w)?;
    }
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors.iter() {
        write_string(w, name)?;
        let dims = tensor.shape().dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }
        w.write_u32::<LittleEndian>(tensor.dtype().to_u32())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offsets.push(offset);
        let size_in_bytes = tensor.storage_size_in_bytes();
        let padding = 31 - (31 + size_in_bytes) % 32;
        offset += size_in_bytes + padding;
    }
    let pos = w.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    w.write_all(&vec![0u8; padding])?;
    let tensor_start_pos = w.stream_position()? as usize;
    for (offset, (_name, tensor)) in offsets.iter().zip(tensors.iter()) {
        let pos = w.stream_position()? as usize;
        if tensor_start_pos + offset != pos {
            crate::bail!(
                "internal error, unexpected current position {tensor_start_pos} {offset} {pos}"
            )
        }
        let data = tensor.data()?;
        let size_in_bytes = data.len();
        w.write_all(&data)?;
        let padding = 31 - (31 + size_in_bytes) % 32;
        w.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}
