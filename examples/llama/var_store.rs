use candle::{DType, Device, Result, Shape, Tensor, WithDType};

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
}

#[allow(dead_code)]
pub struct VarStore {
    vars: Vec<NamedVar>,
}

impl VarBuilder {
    pub fn new<B: WithDType>() -> Self {
        let vars = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        Self {
            path: vec![],
            vars,
            default_dtype: B::DTYPE,
        }
    }

    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    pub fn var<S: Into<Shape>>(&mut self, s: &str, shape: S) -> Result<Tensor> {
        let shape = shape.into();
        let path = format!("{}.{s}", self.path.join("."));
        let mut vars = self.vars.borrow_mut();
        let parameter = Tensor::zeros(&shape, self.default_dtype, &Device::Cpu);
        vars.push(NamedVar {
            path,
            dtype: self.default_dtype,
            shape,
        });
        parameter
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
        }
    }
}

impl<S: ToString> std::ops::Div<S> for VarBuilder {
    type Output = VarBuilder;

    fn div(self, rhs: S) -> VarBuilder {
        &self / rhs
    }
}
