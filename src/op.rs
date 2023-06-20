use crate::Tensor;

pub(crate) enum Op {
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Sqr(Tensor),
    Sqrt(Tensor),
    // TODO: Support for custom ops.
}
