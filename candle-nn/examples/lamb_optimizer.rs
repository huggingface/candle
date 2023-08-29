use candle::{Tensor, Device, Var, Result, Module, DType};
use candle_nn::{Linear, linear, VarBuilder, VarMap};

fn clamp(x: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    let device = x.device();
    let dtype = x.dtype();
    let min = Tensor::from_slice(&[min], 1, device)?.to_dtype(dtype)?;
    let max = Tensor::from_slice(&[max], 1, device)?.to_dtype(dtype)?;
    let x = x.broadcast_minimum(&max)?.broadcast_maximum(&min)?;
    Ok(x)
}

fn norm_l2(x: &Tensor) -> Result<Tensor> {
    x.sqr()?.sum_all()?.sqrt()
}

#[derive(Clone, Debug)]
pub struct ParamsLamb {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsLamb {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
        }
    }
}

// same as VarAdamW
#[derive(Debug)]
struct VarLamb {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct Lamb {
    vars: Vec<VarLamb>,
    step_t: usize,
    params: ParamsLamb,
}

impl Lamb {
    pub fn new(vars: Vec<Var>, params: ParamsLamb) -> Result<Self> {
        let vars = vars.into_iter().map(|var| {
            let first_moment = Var::zeros(
                var.shape(),
                var.dtype(),
                var.device()
            )?;
            let second_moment = Var::zeros(
                var.shape(),
                var.dtype(),
                var.device()
            )?;
            Ok(VarLamb {
                var,
                first_moment,
                second_moment
            })
        }).collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vars,
            params,
            step_t: 0
        })
    }

    pub fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        self.step_t += 1;

        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;

        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));

        for var in self.vars.iter_mut() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;

            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let v_sqrt = v_hat.sqrt()?;

                let mut r = (m_hat / &(v_sqrt + self.params.eps)?)?;
                if self.params.weight_decay > 0.0 {
                    r = (r + (theta.as_tensor() * lr_lambda)?)?;
                }

                let w_norm = norm_l2(theta)?;
                let g_norm = norm_l2(&r)?;
                let ratio = clamp(&(w_norm / g_norm)?, 0.08, 0.5)?; // clamp values from deepspeed
                let next_theta = (theta.as_tensor() - (ratio.broadcast_mul(&r)? * lr)?)?;

                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
}

fn gen_data() -> Result<(Tensor, Tensor)> {
    // Generate some sample linear data.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;
    Ok((sample_xs, sample_ys))
}

fn main() -> Result<()> {
    let (sample_xs, sample_ys) = gen_data()?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = linear(2, 1, vb.pp("linear"))?;
    let params = ParamsLamb::default();
    let mut opt = Lamb::new(varmap.all_vars(), params)?;

    for step in 0..10000 {
        let ys = model.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
        println!("{step} {}", loss.to_vec0::<f32>()?);
    }

    Ok(())
}

fn test_clamp() -> Result<()> {
    let x = Tensor::from_slice(&[1f32, 2f32, 3f32, 4f32], 4, &Device::Cpu)?;
    let y = clamp(&x, 2.0, 3.0)?;
    assert_eq!(
        y.to_vec1::<f32>()?,
        vec![2.0, 2.0, 3.0, 3.0]
    );
    Ok(())
}