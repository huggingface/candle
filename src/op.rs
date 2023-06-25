use crate::Tensor;

#[derive(Clone)]
pub(crate) enum Op {
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Div(Tensor, Tensor),
    BroadcastAdd(Tensor, Tensor),
    BroadcastMul(Tensor, Tensor),
    BroadcastSub(Tensor, Tensor),
    BroadcastDiv(Tensor, Tensor),
    Matmul(Tensor, Tensor),
    Embedding(Tensor, Tensor),
    WhereCond(Tensor, Tensor, Tensor),

    Cat(Vec<Tensor>, usize),

    #[allow(dead_code)] // add is currently unused.
    Affine {
        arg: Tensor,
        mul: f64,
        add: f64,
    },
    Sum(Tensor, Vec<usize>),
    ToDType(Tensor),
    Broadcast(Tensor),
    Exp(Tensor),
    Log(Tensor),
    Sin(Tensor),
    Cos(Tensor),
    Abs(Tensor),
    Narrow(Tensor, usize, usize, usize),
    Neg(Tensor),
    Reshape(Tensor),
    Softmax(Tensor, usize),
    Sqr(Tensor),
    Sqrt(Tensor),
    ToDevice(Tensor),
    Transpose(Tensor, usize, usize),
    Gelu(Tensor),
    // TODO: Support for custom ops.
}

pub(crate) trait UnaryOp {
    const NAME: &'static str;
    // TODO: These kernels are compatible with arbitrary strides. We should also consider the
    // contiguous case separately as it's easy to optimize things out there.
    const KERNEL_F32: &'static str;
    const KERNEL_F64: &'static str;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u32(v1: u32) -> u32;
}

pub(crate) trait BinaryOp {
    const NAME: &'static str;
    // TODO: These kernels are compatible with arbitrary strides. We should also consider the
    // contiguous case separately as it's easy to optimize things out there.
    const KERNEL_F32: &'static str;
    const KERNEL_F64: &'static str;
    const KERNEL_U32: &'static str;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u32(v1: u32, v2: u32) -> u32;
}

pub(crate) struct Add;
pub(crate) struct Div;
pub(crate) struct Mul;
pub(crate) struct Sub;
pub(crate) struct Exp;
pub(crate) struct Log;
pub(crate) struct Sin;
pub(crate) struct Cos;
pub(crate) struct Abs;
pub(crate) struct Neg;
pub(crate) struct Sqr;
pub(crate) struct Sqrt;
pub(crate) struct Gelu;

impl BinaryOp for Add {
    const NAME: &'static str = "add";
    const KERNEL_F32: &'static str = "badd_f32";
    const KERNEL_F64: &'static str = "badd_f64";
    const KERNEL_U32: &'static str = "badd_u32";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 + v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 + v2
    }
    fn u32(v1: u32, v2: u32) -> u32 {
        v1 + v2
    }
}

impl BinaryOp for Sub {
    const NAME: &'static str = "sub";
    const KERNEL_F32: &'static str = "bsub_f32";
    const KERNEL_F64: &'static str = "bsub_f64";
    const KERNEL_U32: &'static str = "bsub_u32";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 - v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 - v2
    }
    fn u32(v1: u32, v2: u32) -> u32 {
        v1 - v2
    }
}

impl BinaryOp for Mul {
    const NAME: &'static str = "mul";
    const KERNEL_F32: &'static str = "bmul_f32";
    const KERNEL_F64: &'static str = "bmul_f64";
    const KERNEL_U32: &'static str = "bmul_u32";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 * v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 * v2
    }
    fn u32(v1: u32, v2: u32) -> u32 {
        v1 * v2
    }
}

impl BinaryOp for Div {
    const NAME: &'static str = "div";
    const KERNEL_F32: &'static str = "bdiv_f32";
    const KERNEL_F64: &'static str = "bdiv_f64";
    const KERNEL_U32: &'static str = "bdiv_u32";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 / v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 / v2
    }
    fn u32(v1: u32, v2: u32) -> u32 {
        v1 / v2
    }
}

impl UnaryOp for Exp {
    const NAME: &'static str = "exp";
    fn f32(v1: f32) -> f32 {
        v1.exp()
    }
    fn f64(v1: f64) -> f64 {
        v1.exp()
    }
    fn u32(v1: u32) -> u32 {
        (v1 as f64).exp() as u32
    }
    const KERNEL_F32: &'static str = "uexp_f32";
    const KERNEL_F64: &'static str = "uexp_f64";
}

impl UnaryOp for Log {
    const NAME: &'static str = "log";
    fn f32(v1: f32) -> f32 {
        v1.ln()
    }
    fn f64(v1: f64) -> f64 {
        v1.ln()
    }
    fn u32(v1: u32) -> u32 {
        (v1 as f64).ln() as u32
    }
    const KERNEL_F32: &'static str = "ulog_f32";
    const KERNEL_F64: &'static str = "ulog_f64";
}

impl UnaryOp for Sin {
    const NAME: &'static str = "sin";
    fn f32(v1: f32) -> f32 {
        v1.sin()
    }
    fn f64(v1: f64) -> f64 {
        v1.sin()
    }
    fn u32(_: u32) -> u32 {
        0
    }
    const KERNEL_F32: &'static str = "usin_f32";
    const KERNEL_F64: &'static str = "usin_f64";
}

impl UnaryOp for Cos {
    const NAME: &'static str = "cos";
    fn f32(v1: f32) -> f32 {
        v1.cos()
    }
    fn f64(v1: f64) -> f64 {
        v1.cos()
    }
    fn u32(_: u32) -> u32 {
        0
    }
    const KERNEL_F32: &'static str = "ucos_f32";
    const KERNEL_F64: &'static str = "ucos_f64";
}

impl UnaryOp for Abs {
    const NAME: &'static str = "abs";
    fn f32(v1: f32) -> f32 {
        v1.abs()
    }
    fn f64(v1: f64) -> f64 {
        v1.abs()
    }
    fn u32(v1: u32) -> u32 {
        v1
    }
    const KERNEL_F32: &'static str = "uabs_f32";
    const KERNEL_F64: &'static str = "uabs_f64";
}

impl UnaryOp for Neg {
    const NAME: &'static str = "neg";
    fn f32(v1: f32) -> f32 {
        -v1
    }
    fn f64(v1: f64) -> f64 {
        -v1
    }
    fn u32(_: u32) -> u32 {
        0
    }
    const KERNEL_F32: &'static str = "uneg_f32";
    const KERNEL_F64: &'static str = "uneg_f64";
}

impl UnaryOp for Sqr {
    const NAME: &'static str = "sqr";
    fn f32(v1: f32) -> f32 {
        v1 * v1
    }
    fn f64(v1: f64) -> f64 {
        v1 * v1
    }
    fn u32(v: u32) -> u32 {
        v * v
    }
    const KERNEL_F32: &'static str = "usqr_f32";
    const KERNEL_F64: &'static str = "usqr_f64";
}

impl UnaryOp for Sqrt {
    const NAME: &'static str = "sqrt";
    fn f32(v1: f32) -> f32 {
        v1.sqrt()
    }
    fn f64(v1: f64) -> f64 {
        v1.sqrt()
    }
    fn u32(v: u32) -> u32 {
        (v as f64).sqrt() as u32
    }
    const KERNEL_F32: &'static str = "usqrt_f32";
    const KERNEL_F64: &'static str = "usqrt_f64";
}

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
#[inline]
pub fn gelu_f32(v: f32) -> f32 {
    0.5 * v
        * (1.0 + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
}
/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
#[inline]
pub fn gelu_f64(v: f64) -> f64 {
    0.5 * v
        * (1.0 + f64::tanh((2.0f64 / std::f64::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
}
impl UnaryOp for Gelu {
    const NAME: &'static str = "gelu";
    fn f32(v1: f32) -> f32 {
        gelu_f32(v1)
    }
    fn f64(v1: f64) -> f64 {
        gelu_f64(v1)
    }
    fn u32(v1: u32) -> u32 {
        gelu_f64(v1 as f64) as u32
    }
    const KERNEL_F32: &'static str = "gelu_f32";
    const KERNEL_F64: &'static str = "gelu_f64";
}
