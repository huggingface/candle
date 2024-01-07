#![allow(unused)]
//! Wrappers around the Python API of Gymnasium (the new version of OpenAI gym)
use candle::{Device, Result, Tensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The return value for a step.
#[derive(Debug)]
// Each Vec has a length equal to the number of processes
pub struct Step<A> {
    pub states: Vec<Tensor>,
    pub actions: Vec<A>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
}

impl<A: Copy> Step<A> {
    /// Returns a copy of this step changing the observation tensor.
    pub fn copy_with_obs(&self, states: &Vec<Tensor>) -> Step<A> {
        Step {
            states: states.clone(),
            actions: self.actions.clone(),
            rewards: self.rewards.clone(),
            terminated: self.terminated.clone(),
            truncated: self.truncated.clone(),
        }
    }
}

/// An OpenAI Gym session.
pub struct VecGymEnv {
    env: PyObject,
    action_space: usize,
    observation_space: Vec<usize>,
}

fn w(res: PyErr) -> candle::Error {
    candle::Error::wrap(res)
}

impl VecGymEnv {
    /// Creates a new session of the specified OpenAI Gym environment.
    pub fn new(name: &str, num_processes: usize, video_folder: Option<&str>) -> Result<VecGymEnv> {
        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            let _ = path.call_method1(
                "append",
                ("candle-examples/examples/reinforcement-learning",),
            )?;
            let gym = py.import("atari_wrappers")?;
            let make = gym.getattr("make")?;
            let env = make.call1((name, num_processes, video_folder))?;
            let action_space = env.getattr("single_action_space")?;
            let action_space = if let Ok(val) = action_space.getattr("n") {
                val.extract()?
            } else {
                let action_space: Vec<usize> = action_space.getattr("shape")?.extract()?;
                action_space[0]
            };
            let observation_space = env.getattr("single_observation_space")?;
            let observation_space: Vec<usize> = observation_space.getattr("shape")?.extract()?;
            let observation_space =
                [vec![num_processes].as_slice(), observation_space.as_slice()].concat();
            Ok(VecGymEnv {
                env: env.into(),
                action_space,
                observation_space,
            })
        })
        .map_err(w)
    }

    fn states_from_buf(&self, buf: Vec<f32>) -> Result<Vec<Tensor>> {
        let states = Tensor::from_vec(buf, self.observation_space(), &Device::Cpu)?
            .chunk(self.observation_space()[0], 0)?
            .into_iter()
            .map(|state| state.squeeze(0).unwrap())
            .collect();
        Ok((states))
    }

    /// Resets the environment, returning the observation tensor.
    pub fn reset(&self, seed: u64) -> Result<Vec<Tensor>> {
        let states_buf: Vec<f32> = Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let step = self.env.call_method(py, "reset", (), Some(kwargs))?;
            let step = step.as_ref(py);
            step.get_item(0)?.call_method0("flatten")?.extract()
        })
        .map_err(w)?;
        self.states_from_buf(states_buf)
    }

    /// Applies an environment step using the specified action.
    pub fn step<A: pyo3::IntoPy<pyo3::Py<pyo3::PyAny>> + Clone>(
        &self,
        actions: Vec<A>,
    ) -> Result<Step<A>> {
        let (states_buf, rewards, terminated, truncated) = Python::with_gil(|py| {
            let step = self.env.call_method(py, "step", (actions.clone(),), None)?;
            let step = step.as_ref(py);
            let states_buf: Vec<f32> = step.get_item(0)?.call_method0("flatten")?.extract()?;
            let rewards: Vec<f64> = step.get_item(1)?.extract()?;
            let terminated: Vec<bool> = step.get_item(2)?.call_method0("tolist")?.extract()?;
            let truncated: Vec<bool> = step.get_item(3)?.call_method0("tolist")?.extract()?;
            Ok((states_buf, rewards, terminated, truncated))
        })
        .map_err(w)?;
        let states = self.states_from_buf(states_buf)?;
        Ok(Step {
            states,
            actions,
            rewards,
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
