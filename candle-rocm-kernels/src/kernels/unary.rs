use crate::error::RocmKernelError;
use crate::manager::KernelManager;
use crate::source::Source;
use rocm_rs::hip::{DeviceMemory, Stream};

pub struct UnaryKernels {
    manager: KernelManager,
}

impl UnaryKernels {
    pub fn new(device: &rocm_rs::hip::Device) -> Result<Self, RocmKernelError> {
        let manager = KernelManager::new(device)?;
        Ok(Self { manager })
    }

    pub fn launch<T: Copy + Send + Sync + 'static>(
        &self,
        stream: &Stream,
        op: UnaryOp,
        numel: usize,
        num_dims: usize,
        dims_and_strides: Option<&DeviceMemory<usize>>,
        input: &DeviceMemory<T>,
        output: &mut DeviceMemory<T>,
    ) -> Result<(), RocmKernelError> {
        let module = self.manager.get_or_compile_module(Source::Unary)?;
        let kernel_name = format!("{}_{}", op.kernel_name(), dtype_suffix::<T>());

        let function = module.get_function(&kernel_name).map_err(|e| {
            RocmKernelError::Launch(format!("Kernel {} not found: {}", kernel_name, e))
        })?;

        let (grid, block) = crate::utils::launch_config(numel);

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&numel) as *const usize as *mut std::ffi::c_void,
            (&num_dims) as *const usize as *mut std::ffi::c_void,
        ];

        if let Some(info) = dims_and_strides {
            args.push(info.as_ptr() as *mut std::ffi::c_void);
        } else {
            args.push(std::ptr::null_mut());
        }

        args.push(input.as_ptr() as *mut std::ffi::c_void);
        args.push(output.as_ptr() as *mut std::ffi::c_void);

        function
            .launch(
                rocm_rs::hip::Dim3 {
                    x: grid.0,
                    y: grid.1,
                    z: grid.2,
                },
                rocm_rs::hip::Dim3 {
                    x: block.0,
                    y: block.1,
                    z: block.2,
                },
                0,
                Some(stream),
                &mut args,
            )
            .map_err(|e| RocmKernelError::Launch(e.to_string()))?;

        Ok(())
    }

    pub fn launch_pow<T: Copy + Send + Sync + 'static>(
        &self,
        stream: &Stream,
        numel: usize,
        num_dims: usize,
        dims_and_strides: Option<&DeviceMemory<usize>>,
        input: &DeviceMemory<T>,
        exp_val: T,
        output: &mut DeviceMemory<T>,
    ) -> Result<(), RocmKernelError> {
        let module = self.manager.get_or_compile_module(Source::Unary)?;
        let kernel_name = format!("upow_{}", dtype_suffix::<T>());

        let function = module.get_function(&kernel_name).map_err(|e| {
            RocmKernelError::Launch(format!("Kernel {} not found: {}", kernel_name, e))
        })?;

        let (grid, block) = crate::utils::launch_config(numel);

        let exp_ptr: *const T = &exp_val;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&numel) as *const usize as *mut std::ffi::c_void,
            (&num_dims) as *const usize as *mut std::ffi::c_void,
        ];

        if let Some(info) = dims_and_strides {
            args.push(info.as_ptr() as *mut std::ffi::c_void);
        } else {
            args.push(std::ptr::null_mut());
        }

        args.push(input.as_ptr() as *mut std::ffi::c_void);
        args.push(exp_ptr as *mut std::ffi::c_void);
        args.push(output.as_ptr() as *mut std::ffi::c_void);

        function
            .launch(
                rocm_rs::hip::Dim3 {
                    x: grid.0,
                    y: grid.1,
                    z: grid.2,
                },
                rocm_rs::hip::Dim3 {
                    x: block.0,
                    y: block.1,
                    z: block.2,
                },
                0,
                Some(stream),
                &mut args,
            )
            .map_err(|e| RocmKernelError::Launch(e.to_string()))?;

        Ok(())
    }

    pub fn manager(&self) -> &KernelManager {
        &self.manager
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Copy,
    Relu,
    Sigmoid,
    Tan,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Abs,
    Neg,
    Recip,
    Floor,
    Ceil,
    Round,
    Gelu,
    Silu,
    Erf,
}

impl UnaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            UnaryOp::Copy => "ucopy",
            UnaryOp::Relu => "urelu",
            UnaryOp::Sigmoid => "usigmoid",
            UnaryOp::Tan => "utan",
            UnaryOp::Exp => "uexp",
            UnaryOp::Log => "ulog",
            UnaryOp::Sin => "usin",
            UnaryOp::Cos => "ucos",
            UnaryOp::Sqrt => "usqrt",
            UnaryOp::Abs => "uabs",
            UnaryOp::Neg => "uneg",
            UnaryOp::Recip => "urecip",
            UnaryOp::Floor => "ufloor",
            UnaryOp::Ceil => "uceil",
            UnaryOp::Round => "uround",
            UnaryOp::Gelu => "ugelu",
            UnaryOp::Silu => "usilu",
            UnaryOp::Erf => "uerf",
        }
    }
}

fn dtype_suffix<T: Copy + Send + Sync + 'static>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f64") {
        "f64"
    } else if type_name.contains("u8") {
        "u8"
    } else if type_name.contains("u32") {
        "u32"
    } else if type_name.contains("i64") {
        "i64"
    } else {
        panic!("Unsupported dtype for unary op: {}", type_name)
    }
}
