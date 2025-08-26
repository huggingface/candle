use crate::{ConstantValues, MetalKernelError};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCounterSet,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunction, MTLFunctionConstantValues,
    MTLLibrary, MTLResource, MTLResourceUsage, MTLSize,
};
use std::{collections::HashMap, ffi::c_void, ptr, sync::Arc};

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
