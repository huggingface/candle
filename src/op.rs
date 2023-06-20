use crate::Tensor;

pub(crate) enum Op {
    Add(Tensor, Tensor),
    #[allow(dead_code)] // add is currently unused.
    Affine {
        arg: Tensor,
        mul: f64,
        add: f64,
    },
    Mul(Tensor, Tensor),
    Sqr(Tensor),
    Sqrt(Tensor),
    // TODO: Support for custom ops.
}
