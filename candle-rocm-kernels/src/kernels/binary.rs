use crate::error::RocmKernelError;
use crate::manager::KernelManager;
use crate::source::Source;
use rocm_rs::hip::{DeviceMemory, Stream};

pub struct BinaryKernels {
    manager: KernelManager,
}

impl BinaryKernels {
    pub fn new(device: &rocm_rs::hip::Device) -> Result<Self, RocmKernelError> {
        let manager = KernelManager::new(device)?;
        Ok(Self { manager })
    }

    pub fn launch<T: Copy + Send + Sync + 'static>(
        &self,
        stream: &Stream,
        op: BinaryOp,
        numel: usize,
        num_dims: usize,
        dims_and_strides: Option<&DeviceMemory<usize>>,
        lhs: &DeviceMemory<T>,
        rhs: &DeviceMemory<T>,
        output: &mut DeviceMemory<T>,
    ) -> Result<(), RocmKernelError> {
        let module = self.manager.get_or_compile_module(Source::Binary)?;
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

        args.push(lhs.as_ptr() as *mut std::ffi::c_void);
        args.push(rhs.as_ptr() as *mut std::ffi::c_void);
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
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Minimum,
    Maximum,
}

impl BinaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "badd",
            BinaryOp::Sub => "bsub",
            BinaryOp::Mul => "bmul",
            BinaryOp::Div => "bdiv",
            BinaryOp::Minimum => "bminimum",
            BinaryOp::Maximum => "bmaximum",
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
        panic!("Unsupported dtype for binary op: {}", type_name)
    }
}
