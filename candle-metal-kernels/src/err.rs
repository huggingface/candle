use crate::kernels::sdpa::SdpaDType;

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Command buffer had following error: {0}")]
    CommandBufferError(String),
    #[error("Could not lock resource: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0}")]
    LoadFunctionError(String),
    #[error("Failed to create compute function")]
    FailedToCreateComputeFunction,
    #[error("Failed to create metal resource: {0}")]
    FailedToCreateResource(String),
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("Invalid matmul arguments {lhs_stride:?} {rhs_stride:?} {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
    #[error("Sdpa {variation} head size was {got}, expectd {expected:?}")]
    SdpaHeadSizeMismatch {
        variation: &'static str,
        got: usize,
        expected: Vec<usize>,
    },
    #[error("Sdpa {variation} got dtype {got:?}")]
    SdpaHeadDTypeMismatch {
        variation: &'static str,
        got: SdpaDType,
    },
    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },
}

impl MetalKernelError {
    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}
