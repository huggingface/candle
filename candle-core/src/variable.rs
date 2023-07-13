// Variables are wrappers around tensors that can be modified, they are typically used for holding
// weights and being modified by gradient descent.
// They are not cloneable by default to avoid having too many potential writers on the data.
// We also do not expose a public way to create variables as this would break the invariant that
// the tensor within a variable is actually with `is_variable` set to `true`.
use crate::Tensor;

/// A variable is a wrapper around a tensor, however variables can have their content modified
/// whereas tensors are immutable.
#[derive(Debug)]
pub struct Variable(Tensor);

impl std::ops::Deref for Variable {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Variable {
    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }

    /// Consumes this `Variable` and return the underlying tensor.
    pub fn into_inner(self) -> Tensor {
        self.0
    }
}
