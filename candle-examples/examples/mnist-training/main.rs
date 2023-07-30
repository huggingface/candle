// This should reach 91.5% accuracy.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::{Parser, ValueEnum};

use candle::{DType, Device, Result, Shape, Tensor, Var, D};
use candle_nn::{loss, ops, Init, Linear};
use std::sync::{Arc, Mutex};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

struct TensorData {
    tensors: std::collections::HashMap<String, Var>,
    pub dtype: DType,
    pub device: Device,
}

// A variant of candle_nn::VarBuilder for initializing variables before training.
#[derive(Clone)]
struct VarStore {
    data: Arc<Mutex<TensorData>>,
    path: Vec<String>,
}

impl VarStore {
    fn new(dtype: DType, device: Device) -> Self {
        let data = TensorData {
            tensors: std::collections::HashMap::new(),
            dtype,
            device,
        };
        Self {
            data: Arc::new(Mutex::new(data)),
            path: vec![],
        }
    }

    fn pp(&self, s: &str) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
        }
    }

    fn get<S: Into<Shape>>(&self, shape: S, tensor_name: &str, init: Init) -> Result<Tensor> {
        let shape = shape.into();
        let path = if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        };
        let mut tensor_data = self.data.lock().unwrap();
        if let Some(tensor) = tensor_data.tensors.get(&path) {
            let tensor_shape = tensor.shape();
            if &shape != tensor_shape {
                candle::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
            }
            return Ok(tensor.as_tensor().clone());
        }
        let var = init.var(shape, tensor_data.dtype, &tensor_data.device)?;
        let tensor = var.as_tensor().clone();
        tensor_data.tensors.insert(path, var);
        Ok(tensor)
    }

    fn all_vars(&self) -> Vec<Var> {
        let tensor_data = self.data.lock().unwrap();
        #[allow(clippy::map_clone)]
        tensor_data
            .tensors
            .values()
            .map(|c| c.clone())
            .collect::<Vec<_>>()
    }

    fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let tensor_data = self.data.lock().unwrap();
        let data = tensor_data.tensors.iter().map(|(k, v)| (k, v.as_tensor()));
        safetensors::tensor::serialize_to_file(data, &None, path.as_ref())?;
        Ok(())
    }
}

fn linear_z(in_dim: usize, out_dim: usize, vs: VarStore) -> Result<Linear> {
    let ws = vs.get((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs = vs.get(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}

fn linear(in_dim: usize, out_dim: usize, vs: VarStore) -> Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vs.get((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vs.get(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

trait Model: Sized {
    fn new(vs: VarStore) -> Result<Self>;
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

struct LinearModel {
    linear: Linear,
}

impl Model for LinearModel {
    fn new(vs: VarStore) -> Result<Self> {
        let linear = linear_z(IMAGE_DIM, LABELS, vs)?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Model for Mlp {
    fn new(vs: VarStore) -> Result<Self> {
        let ln1 = linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = linear(100, LABELS, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}

fn training_loop<M: Model>(
    m: candle_nn::vision::Dataset,
    learning_rate: f64,
    save: Option<String>,
) -> anyhow::Result<()> {
    let dev = candle::Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels
        .to_dtype(DType::U32)?
        .unsqueeze(1)?
        .to_device(&dev)?;

    let vs = VarStore::new(DType::F32, dev.clone());
    let model = M::new(vs.clone())?;

    let all_vars = vs.all_vars();
    let all_vars = all_vars.iter().collect::<Vec<_>>();
    let sgd = candle_nn::SGD::new(&all_vars, learning_rate);
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    for epoch in 1..200 {
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_labels)?;
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            100. * test_accuracy
        );
    }
    if let Some(save) = save {
        println!("saving trained weights in {save}");
        vs.save(&save)?
    }
    Ok(())
}

#[derive(ValueEnum, Clone)]
enum WhichModel {
    Linear,
    Mlp,
}

#[derive(Parser)]
struct Args {
    #[clap(value_enum, default_value_t = WhichModel::Linear)]
    model: WhichModel,

    #[arg(long)]
    learning_rate: Option<f64>,

    /// The file where to save the trained weights, in safetensors format.
    #[arg(long)]
    save: Option<String>,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Load the dataset
    let m = candle_nn::vision::mnist::load_dir("data")?;
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    match args.model {
        WhichModel::Linear => {
            training_loop::<LinearModel>(m, args.learning_rate.unwrap_or(1.), args.save)
        }
        WhichModel::Mlp => training_loop::<Mlp>(m, args.learning_rate.unwrap_or(0.01), args.save),
    }
}
