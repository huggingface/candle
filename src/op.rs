use crate::Tensor;

#[allow(dead_code)]
pub(crate) enum Op {
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
}
