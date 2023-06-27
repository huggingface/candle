use candle::{DType, Device, Result, Shape, Tensor, WithDType};
use std::collections::HashMap;
use std::sync::Arc;

#[allow(dead_code)]
#[derive(Clone)]
struct NamedVar {
    path: String,
    dtype: DType,
    shape: Shape,
}

#[derive(Clone)]
pub struct VarBuilder {
    path: Vec<String>,
    vars: std::rc::Rc<std::cell::RefCell<Vec<NamedVar>>>,
    default_dtype: DType,
    default_device: Device,
    tensors: Arc<Option<HashMap<String, Tensor>>>,
}

#[allow(dead_code)]
pub struct VarStore {
    vars: Vec<NamedVar>,
}

impl VarBuilder {
    pub fn new<B: WithDType>(device: &Device, tensors: Option<HashMap<String, Tensor>>) -> Self {
        let vars = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        Self {
            path: vec![],
            vars,
            default_dtype: B::DTYPE,
            tensors: Arc::new(tensors),
            default_device: device.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    pub fn var<S: Into<Shape>>(&mut self, s: &str, shape: S) -> Result<Tensor> {
        let shape = shape.into();
        let path = format!("{}.{s}", self.path.join("."));
        let mut vars = self.vars.borrow_mut();
        let parameter = match self.tensors.as_ref() {
            None => Tensor::zeros(&shape, self.default_dtype, &self.default_device)?,
            Some(tensors) => match tensors.get(&path) {
                Some(tensor) => tensor.to_device(&self.default_device)?,
                None => panic!("cannot find tensor for {path}"),
            },
        };
        vars.push(NamedVar {
            path,
            dtype: self.default_dtype,
            shape,
        });
        Ok(parameter)
    }

    pub fn into_store(self) -> VarStore {
        let vars = self.vars.borrow();
        VarStore {
            vars: vars.to_vec(),
        }
    }
}

impl<S: ToString> std::ops::Div<S> for &VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        let mut path = self.path.clone();
        path.push(rhs.to_string());
        VarBuilder {
            path,
            vars: self.vars.clone(),
            default_dtype: self.default_dtype,
            default_device: self.default_device.clone(),
            tensors: self.tensors.clone(),
        }
    }
}

impl<S: ToString> std::ops::Div<S> for VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        &self / rhs
    }
}
