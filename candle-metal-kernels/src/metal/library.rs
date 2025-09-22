use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLDataType, MTLFunction, MTLFunctionConstantValues, MTLLibrary};
use std::{ffi::c_void, ptr};

#[derive(Clone, Debug)]
pub struct Library {
    raw: Retained<ProtocolObject<dyn MTLLibrary>>,
}
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLLibrary>>) -> Library {
        Library { raw }
    }

    pub fn get_function(
        &self,
        name: &str,
        constant_values: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let function = match constant_values {
            Some(constant_values) => self
                .raw
                .newFunctionWithName_constantValues_error(
                    &NSString::from_str(name),
                    &constant_values.function_constant_values().raw,
                )
                .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?,
            None => self
                .raw
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(MetalKernelError::LoadFunctionError("".to_string()))?,
        };

        Ok(Function { raw: function })
    }
}

pub struct Function {
    raw: Retained<ProtocolObject<dyn MTLFunction>>,
}

impl Function {
    pub fn as_ref(&self) -> &ProtocolObject<dyn MTLFunction> {
        &*self.raw
    }
}

pub struct FunctionConstantValues {
    raw: Retained<MTLFunctionConstantValues>,
}

impl FunctionConstantValues {
    pub fn new() -> FunctionConstantValues {
        FunctionConstantValues {
            raw: MTLFunctionConstantValues::new(),
        }
    }

    pub fn set_constant_value_at_index<T>(&self, value: &T, dtype: MTLDataType, index: usize) {
        let value = ptr::NonNull::new(value as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setConstantValue_type_atIndex(value, dtype, index) }
    }
}

#[derive(Debug, PartialEq)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::USize(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            // usize is usually u64 aka ulong, but can be u32 on 32-bit systems.
            // https://developer.apple.com/documentation/objectivec/nsuinteger
            Value::USize(_) => MTLDataType::ULong,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
            }
        }
        f
    }
}
