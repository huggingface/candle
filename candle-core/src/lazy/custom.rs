use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use crate::{
    backend::BackendStorage,
    lazy::{LazyBuffer, LazyStorage},
    DType, Layout, Result, Shape, Tensor,
};

pub trait LazyCustomFn<B: LazyBuffer>: LazyCustomFnClone<B> + Send + Sync {
    fn call(&self, input: &[(&B, &Layout, DType)], dst: &B) -> Result<()>;
}

pub trait LazyCustomOpClone {
    fn clone_box(&self) -> Box<dyn LazyCustomOp>;
}

impl<T> LazyCustomOpClone for T
where
    T: 'static + LazyCustomOp + Clone + ?Sized,
{
    fn clone_box(&self) -> Box<dyn LazyCustomOp> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn LazyCustomOp> {
    fn clone(&self) -> Box<dyn LazyCustomOp> {
        self.clone_box()
    }
}

pub trait LazyCustomFnClone<B: LazyBuffer> {
    fn clone_box(&self) -> Box<dyn LazyCustomFn<B>>;
}

impl<T, B> LazyCustomFnClone<B> for T
where
    T: 'static + LazyCustomFn<B> + Clone + ?Sized,
    B: LazyBuffer + Clone,
{
    fn clone_box(&self) -> Box<dyn LazyCustomFn<B>> {
        Box::new(self.clone())
    }
}

impl<B> Clone for Box<dyn LazyCustomFn<B>>
where
    B: LazyBuffer,
{
    fn clone(&self) -> Box<dyn LazyCustomFn<B>> {
        self.clone_box()
    }
}

impl std::fmt::Debug for Box<dyn LazyCustomOp> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("LazyCustomOp").field(&self.name()).finish()
    }
}
impl PartialEq for Box<dyn LazyCustomOp> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

pub trait LazyCustomOp: LazyCustomOpClone + Send + Sync {
    /// Lazy custom op that can be defined in user-land.
    fn name(&self) -> &'static str;

    fn expected_edges(&self) -> usize;

    /// Forward pass. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn fwd(&self, args: &[(&LazyStorage, &Layout)]) -> Result<(LazyStorage, Shape)> {
        let (first, layout) = args[0];
        let first = LazyStorage::copy(first, layout);
        first
            .custom_op(self.clone_box(), &args[1..])
            .map(|s| (s, layout.shape().clone()))
    }

    fn fallback(&self, _tensors: &[&Tensor]) -> Result<crate::Tensor> {
        Err(crate::Error::Msg(
            format!("no lazy fallback for {}", self.name()).into(),
        ))
    }

    /// This function takes as argument the argument `args` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(&self, _args: &[&Tensor], _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }

    fn extract_lazy(&self, tensor: Tensor) -> Result<(LazyStorage, Layout)> {
        let (storage, layout) = tensor.storage_and_layout();
        let storage = storage.try_clone(layout)?;
        let inner = match storage {
            crate::Storage::Lazy(lazy) => lazy,
            _ => unreachable!(),
        };

        Ok((inner, layout.clone()))
    }
}

#[derive(Clone)]
pub enum CustomOp {
    One(Box<dyn crate::CustomOp1>),
    Two(Box<dyn crate::CustomOp2>),
    Three(Box<dyn crate::CustomOp3>),
}

// TODO: This is not true equality, because there may be values stored inside the custom op that make them different.
impl PartialEq for CustomOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::One(s), Self::One(o)) => s.name() == o.name(),
            (Self::Two(s), Self::Two(o)) => s.name() == o.name(),
            (Self::Three(s), Self::Three(o)) => s.name() == o.name(),
            _ => false,
        }
    }
}

impl From<Box<dyn crate::CustomOp2>> for CustomOp {
    fn from(op: Box<dyn crate::CustomOp2>) -> Self {
        CustomOp::Two(op)
    }
}

impl std::fmt::Debug for CustomOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::One(one) => {
                f.write_str("CustomOp::One(")?;
                f.write_str(one.name())?;
                f.write_str(")")
            }
            Self::Two(two) => {
                f.write_str("CustomOp::Two(")?;
                f.write_str(two.name())?;
                f.write_str(")")
            }
            Self::Three(three) => {
                f.write_str("CustomOp::Three(")?;
                f.write_str(three.name())?;
                f.write_str(")")
            }
        }
    }
}

static CUSTOM_OP_REGISTRY: LazyLock<Mutex<HashMap<String, CustomOp>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn register_custom_ops(ops: Vec<CustomOp>) {
    let mut registry = CUSTOM_OP_REGISTRY.lock().unwrap();
    for op in ops {
        match op {
            CustomOp::One(ref custom_op1) => {
                registry.insert(custom_op1.name().to_string(), op.clone());
            }
            CustomOp::Two(ref custom_op2) => {
                registry.insert(custom_op2.name().to_string(), op.clone());
            }
            CustomOp::Three(ref custom_op3) => {
                registry.insert(custom_op3.name().to_string(), op.clone());
            }
        }
    }
}

pub fn get_custom_op_registry() -> HashMap<String, CustomOp> {
    let registry = CUSTOM_OP_REGISTRY.lock().unwrap().clone();
    registry
}
