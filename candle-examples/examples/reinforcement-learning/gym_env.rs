//! Wrappers around the Python API of Gymnasium (the new version of OpenAI gym)
use candle::{Device, Result, Tensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The return value for a step.
#[derive(Debug)]
pub struct Step<A> {
    pub state: Tensor,
    pub action: A,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

impl<A: Copy> Step<A> {
    /// Returns a copy of this step changing the observation tensor.
    pub fn copy_with_obs(&self, state: &Tensor) -> Step<A> {
        Step {
            state: state.clone(),
            action: self.action,
            reward: self.reward,
            terminated: self.terminated,
            truncated: self.truncated,
        }
    }
}

/// An OpenAI Gym session.
pub struct GymEnv {
    env: PyObject,
    action_space: usize,
    observation_space: Vec<usize>,
}

fn w(res: PyErr) -> candle::Error {
    candle::Error::wrap(res)
}

impl GymEnv {
    /// Creates a new session of the specified OpenAI Gym environment.
    pub fn new(name: &str) -> Result<GymEnv> {
        Python::with_gil(|py| {
            let gym = py.import_bound("gymnasium")?;
            let make = gym.getattr("make")?;
            let env = make.call1((name,))?;
            let action_space = env.getattr("action_space")?;
            let action_space = if let Ok(val) = action_space.getattr("n") {
                val.extract()?
            } else {
                let action_space: Vec<usize> = action_space.getattr("shape")?.extract()?;
                action_space[0]
            };
            let observation_space = env.getattr("observation_space")?;
            let observation_space = observation_space.getattr("shape")?.extract()?;
            Ok(GymEnv {
                env: env.into(),
                action_space,
                observation_space,
            })
        })
        .map_err(w)
    }

    /// Resets the environment, returning the observation tensor.
    pub fn reset(&self, seed: u64) -> Result<Tensor> {
        let state: Vec<f32> = Python::with_gil(|py| {
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("seed", seed)?;
            let state = self.env.call_method_bound(py, "reset", (), Some(&kwargs))?;
            state.bind(py).get_item(0)?.extract()
        })
        .map_err(w)?;
        Tensor::new(state, &Device::Cpu)
    }

    /// Applies an environment step using the specified action.
    pub fn step<A: pyo3::IntoPy<pyo3::Py<pyo3::PyAny>> + Clone>(
        &self,
        action: A,
    ) -> Result<Step<A>> {
        let (state, reward, terminated, truncated) = Python::with_gil(|py| {
            let step = self
                .env
                .call_method_bound(py, "step", (action.clone(),), None)?;
            let step = step.bind(py);
            let state: Vec<f32> = step.get_item(0)?.extract()?;
            let reward: f64 = step.get_item(1)?.extract()?;
            let terminated: bool = step.get_item(2)?.extract()?;
            let truncated: bool = step.get_item(3)?.extract()?;
            Ok((state, reward, terminated, truncated))
        })
        .map_err(w)?;
        let state = Tensor::new(state, &Device::Cpu)?;
        Ok(Step {
            state,
            action,
            reward,
            terminated,
            truncated,
        })
    }

    /// Returns the number of allowed actions for this environment.
    pub fn action_space(&self) -> usize {
        self.action_space
    }

    /// Returns the shape of the observation tensors.
    pub fn observation_space(&self) -> &[usize] {
        &self.observation_space
    }
}
