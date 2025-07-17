//! Vectorized version of the gym environment.
use candle::{DType, Device, Result, Tensor};
use pyo3::prelude::*;

#[allow(unused)]
#[derive(Debug)]
pub struct Step {
    pub obs: Tensor,
    pub reward: Tensor,
    pub is_done: Tensor,
}

#[allow(unused)]
pub struct VecGymEnv {
    env: PyObject,
    action_space: usize,
    observation_space: Vec<usize>,
}

fn w(res: PyErr) -> candle::Error {
    candle::Error::wrap(res)
}

#[allow(unused)]
impl VecGymEnv {
    pub fn new(name: &str, img_dir: Option<&str>, nprocesses: usize) -> Result<VecGymEnv> {
        Python::with_gil(|py| {
            let sys = py.import_bound("sys")?;
            let path = sys.getattr("path")?;
            let _ = path.call_method1(
                "append",
                ("candle-examples/examples/reinforcement-learning",),
            )?;
            let gym = py.import_bound("atari_wrappers")?;
            let make = gym.getattr("make")?;
            let env = make.call1((name, img_dir, nprocesses))?;
            let action_space = env.getattr("action_space")?;
            let action_space = action_space.getattr("n")?.extract()?;
            let observation_space = env.getattr("observation_space")?;
            let observation_space: Vec<usize> = observation_space.getattr("shape")?.extract()?;
            let observation_space =
                [vec![nprocesses].as_slice(), observation_space.as_slice()].concat();
            Ok(VecGymEnv {
                env: env.into(),
                action_space,
                observation_space,
            })
        })
        .map_err(w)
    }

    pub fn reset(&self) -> Result<Tensor> {
        let obs = Python::with_gil(|py| {
            let obs = self.env.call_method0(py, "reset")?;
            let obs = obs.call_method0(py, "flatten")?;
            obs.extract::<Vec<f32>>(py)
        })
        .map_err(w)?;
        Tensor::new(obs, &Device::Cpu)?.reshape(self.observation_space.as_slice())
    }

    pub fn step(&self, action: Vec<usize>) -> Result<Step> {
        let (obs, reward, is_done) = Python::with_gil(|py| {
            let step = self.env.call_method_bound(py, "step", (action,), None)?;
            let step = step.bind(py);
            let obs = step.get_item(0)?.call_method("flatten", (), None)?;
            let obs_buffer = pyo3::buffer::PyBuffer::get_bound(&obs)?;
            let obs: Vec<u8> = obs_buffer.to_vec(py)?;
            let reward: Vec<f32> = step.get_item(1)?.extract()?;
            let is_done: Vec<f32> = step.get_item(2)?.extract()?;
            Ok((obs, reward, is_done))
        })
        .map_err(w)?;
        let obs = Tensor::from_vec(obs, self.observation_space.as_slice(), &Device::Cpu)?
            .to_dtype(DType::F32)?;
        let reward = Tensor::new(reward, &Device::Cpu)?;
        let is_done = Tensor::new(is_done, &Device::Cpu)?;
        Ok(Step {
            obs,
            reward,
            is_done,
        })
    }

    pub fn action_space(&self) -> usize {
        self.action_space
    }

    pub fn observation_space(&self) -> &[usize] {
        &self.observation_space
    }
}
