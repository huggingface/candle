use std::collections::VecDeque;

use rand::{distr::Uniform, rng, Rng};

use candle::{DType, Device, Error, Module, Result, Tensor};
use candle_nn::loss::mse;
use candle_nn::{linear, seq, Activation, AdamW, Optimizer, VarBuilder, VarMap};

use crate::gym_env::GymEnv;

const DEVICE: Device = Device::Cpu;
const EPISODES: usize = 200;
const BATCH_SIZE: usize = 64;
const GAMMA: f64 = 0.99;
const LEARNING_RATE: f64 = 0.01;

pub fn run() -> Result<()> {
    let env = GymEnv::new("CartPole-v1")?;

    // Build the model that predicts the estimated rewards given a specific state.
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &DEVICE);
    let observation_space = *env.observation_space().first().unwrap();

    let model = seq()
        .add(linear(observation_space, 64, vb.pp("linear_in"))?)
        .add(Activation::Relu)
        .add(linear(64, env.action_space(), vb.pp("linear_out"))?);

    let mut optimizer = AdamW::new_lr(var_map.all_vars(), LEARNING_RATE)?;

    // Initialize the model's memory.
    let mut memory = VecDeque::with_capacity(10000);

    // Start the training loop.
    let mut state = env.reset(0)?;
    let mut episode = 0;
    let mut accumulate_rewards = 0.0;
    while episode < EPISODES {
        // Given the current state, predict the estimated rewards, and take the
        // action that is expected to return the most rewards.
        let estimated_rewards = model.forward(&state.unsqueeze(0)?)?;
        let action: u32 = estimated_rewards.squeeze(0)?.argmax(0)?.to_scalar()?;

        // Take that action in the environment, and memorize the outcome:
        // - the state for which the action was taken
        // - the action taken
        // - the new state resulting of taking that action
        // - the actual rewards of taking that action
        // - whether the environment reached a terminal state or not (e.g. game over)
        let step = env.step(action)?;
        accumulate_rewards += step.reward;
        memory.push_back((
            state,
            action,
            step.state.clone(),
            step.reward,
            step.terminated || step.truncated,
        ));
        state = step.state;

        // If there's enough entries in the memory, perform a learning step, where
        // BATCH_SIZE transitions will be sampled from the memory and will be
        // fed to the model so that it performs a backward pass.
        if memory.len() > BATCH_SIZE {
            // Sample randomly from the memory.
            let batch = rng()
                .sample_iter(Uniform::try_from(0..memory.len()).map_err(Error::wrap)?)
                .take(BATCH_SIZE)
                .map(|i| memory.get(i).unwrap().clone())
                .collect::<Vec<_>>();

            // Group all the samples together into tensors with the appropriate shape.
            let states: Vec<_> = batch.iter().map(|e| e.0.clone()).collect();
            let states = Tensor::stack(&states, 0)?;

            let actions = batch.iter().map(|e| e.1);
            let actions = Tensor::from_iter(actions, &DEVICE)?.unsqueeze(1)?;

            let next_states: Vec<_> = batch.iter().map(|e| e.2.clone()).collect();
            let next_states = Tensor::stack(&next_states, 0)?;

            let rewards = batch.iter().map(|e| e.3 as f32);
            let rewards = Tensor::from_iter(rewards, &DEVICE)?.unsqueeze(1)?;

            let non_final_mask = batch.iter().map(|e| !e.4 as u8 as f32);
            let non_final_mask = Tensor::from_iter(non_final_mask, &DEVICE)?.unsqueeze(1)?;

            // Get the estimated rewards for the actions that where taken at each step.
            let estimated_rewards = model.forward(&states)?;
            let x = estimated_rewards.gather(&actions, 1)?;

            // Get the maximum expected rewards for the next state, apply them a discount rate
            // GAMMA and add them to the rewards that were actually gathered on the current state.
            // If the next state is a terminal state, just omit maximum estimated
            // rewards for that state.
            let expected_rewards = model.forward(&next_states)?.detach();
            let y = expected_rewards.max_keepdim(1)?;
            let y = (y * GAMMA * non_final_mask + rewards)?;

            // Compare the estimated rewards with the maximum expected rewards and
            // perform the backward step.
            let loss = mse(&x, &y)?;
            optimizer.backward_step(&loss)?;
        }

        // If we are on a terminal state, reset the environment and log how it went.
        if step.terminated || step.truncated {
            episode += 1;
            println!("Episode {episode} | Rewards {}", accumulate_rewards as i64);
            state = env.reset(0)?;
            accumulate_rewards = 0.0;
        }
    }

    Ok(())
}
