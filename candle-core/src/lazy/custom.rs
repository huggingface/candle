use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use crate::{lazy::LazyBuffer, DType, Layout, LazyStorage, Result, Shape, Tensor};

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
        f.debug_struct("Box<LazyCustomOp>")
            .field("op", &self.name())
            .finish()
    }
}
impl PartialEq for Box<dyn LazyCustomOp> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

pub trait LazyCustomOp: LazyCustomOpClone + Send + Sync {
    fn name(&self) -> &'static str;

    // Forward pass
    fn lazy_custom(&self, input: &[(&LazyStorage, &Layout)]) -> Result<(LazyStorage, Shape)>;

    fn lazy_fallback(&self, _tensors: &[&Tensor]) -> Result<crate::Tensor> {
        Err(crate::Error::Msg(
            format!("no lazy fallback for {}", self.name()).into(),
        ))
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

pub trait LazyCustomOp1 {
    fn name(&self) -> &'static str;

    fn lazy_fwd(&self, _: &LazyStorage, _: &Layout) -> Result<(LazyStorage, Shape)>;

    fn fallback(&self) -> Result<crate::Tensor> {
        Err(crate::Error::Msg(
            format!("no lazy fallback for {}", self.name()).into(),
        ))
    }
}

pub trait LazyCustomOp2 {
    fn name(&self) -> &'static str;

    fn lazy_fwd(
        &self,
        _: &LazyStorage,
        _: &Layout,
        _: &LazyStorage,
        _: &Layout,
    ) -> Result<(LazyStorage, Shape)>;

    fn fallback(&self, _: &Tensor, _: &Tensor) -> Result<Tensor> {
        Err(crate::Error::Msg(
            format!("no lazy fallback for {}", self.name()).into(),
        ))
    }
}

#[derive(Clone)]
pub enum CustomOp {
    One(Box<dyn crate::CustomOp1>),
    Two(Box<dyn crate::CustomOp2>),
}

impl From<Box<dyn crate::CustomOp2>> for CustomOp {
    fn from(op: Box<dyn crate::CustomOp2>) -> Self {
        CustomOp::Two(op)
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
        }
    }
}

pub fn get_custom_op_registry() -> HashMap<String, CustomOp> {
    let registry = CUSTOM_OP_REGISTRY.lock().unwrap().clone();
    registry
}
