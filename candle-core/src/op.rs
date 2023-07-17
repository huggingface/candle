use crate::Tensor;
use half::{bf16, f16};
use num_traits::float::Float;

#[derive(Clone)]
pub(crate) enum Op {
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Div(Tensor, Tensor),
    Matmul(Tensor, Tensor),
    Embedding(Tensor, Tensor),
    WhereCond(Tensor, Tensor, Tensor),

    #[allow(dead_code)]
    Conv1D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        stride: usize,
    },

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
    Relu(Tensor),
    Elu(Tensor, f64),
    // TODO: Support for custom ops.
}

pub(crate) trait UnaryOp {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16) -> bf16;
    fn f16(v1: f16) -> f16;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u8(v1: u8) -> u8;
    fn u32(v1: u32) -> u32;
}

pub(crate) trait BinaryOp {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16, v2: bf16) -> bf16;
    fn f16(v1: f16, v2: f16) -> f16;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u8(v1: u8, v2: u8) -> u8;
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
pub(crate) struct Relu;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl BinaryOp for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("b", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16(v1: bf16, v2: bf16) -> bf16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f16(v1: f16, v2: f16) -> f16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u8(v1: u8, v2: u8) -> u8 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }
        }
    };
}

bin_op!(Add, "add", |v1, v2| v1 + v2);
bin_op!(Sub, "sub", |v1, v2| v1 - v2);
bin_op!(Mul, "mul", |v1, v2| v1 * v2);
bin_op!(Div, "div", |v1, v2| v1 / v2);

macro_rules! unary_op {
    ($op: ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOp for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp());
unary_op!(Log, "log", v, v.ln());
unary_op!(Sin, "sin", v, v.sin());
unary_op!(Cos, "cos", v, v.cos());
unary_op!(Abs, "abs", v, v.abs());
unary_op!(Neg, "neg", v, -v);
unary_op!(Sqr, "sqr", v, v * v);
unary_op!(Sqrt, "sqrt", v, v.sqrt());

/// `gelu` operation
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
impl UnaryOp for Gelu {
    const NAME: &'static str = "gelu";
    const V: Self = Gelu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f32_const(0.5)
            * v
            * (bf16::ONE
                + bf16::tanh(
                    (bf16::from_f32_const(2.0) / bf16::PI).sqrt()
                        * v
                        * (bf16::ONE + bf16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f32_const(0.5)
            * v
            * (f16::ONE
                + f16::tanh(
                    (f16::from_f32_const(2.0) / f16::PI).sqrt()
                        * v
                        * (f16::ONE + f16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        0.5 * v
            * (1.0
                + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        0.5 * v
            * (1.0
                + f64::tanh((2.0f64 / std::f64::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    const KERNEL: &'static str = "ugelu";
}

impl UnaryOp for Relu {
    const NAME: &'static str = "relu";
    const KERNEL: &'static str = "urelu";
    const V: Self = Relu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.max(bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.max(f16::ZERO)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.max(0f32)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.max(0f64)
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
}
