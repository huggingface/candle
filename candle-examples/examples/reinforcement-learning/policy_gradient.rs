use super::gym_env::{GymEnv, Step};
use candle::{DType, Device, Error, Module, Result, Tensor};
use candle_nn::{
    linear, ops::log_softmax, ops::softmax, sequential::seq, Activation, AdamW, Optimizer,
    ParamsAdamW, VarBuilder, VarMap,
};
use rand::{distr::Distribution, rngs::ThreadRng, Rng};

fn new_model(
    input_shape: &[usize],
    num_actions: usize,
    dtype: DType,
    device: &Device,
) -> Result<(impl Module, VarMap)> {
    let input_size = input_shape.iter().product();

    let varmap = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&varmap, dtype, device);

    let model = seq()
        .add(linear(input_size, 32, var_builder.pp("lin1"))?)
        .add(Activation::Relu)
        .add(linear(32, num_actions, var_builder.pp("lin2"))?);

    Ok((model, varmap))
}

fn accumulate_rewards(steps: &[Step<i64>]) -> Vec<f64> {
    let mut rewards: Vec<f64> = steps.iter().map(|s| s.reward).collect();
    let mut acc_reward = 0f64;
    for (i, reward) in rewards.iter_mut().enumerate().rev() {
        if steps[i].terminated {
            acc_reward = 0.0;
        }
        acc_reward += *reward;
        *reward = acc_reward;
    }
    rewards
}

fn weighted_sample(probs: Vec<f32>, rng: &mut ThreadRng) -> Result<usize> {
    let distribution = rand::distr::weighted::WeightedIndex::new(probs).map_err(Error::wrap)?;
    let mut rng = rng;
    Ok(distribution.sample(&mut rng))
}

pub fn run() -> Result<()> {
    let env = GymEnv::new("CartPole-v1")?;

    println!("action space: {:?}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let (model, varmap) = new_model(
        env.observation_space(),
        env.action_space(),
        DType::F32,
        &Device::Cpu,
    )?;

    let optimizer_params = ParamsAdamW {
        lr: 0.01,
        weight_decay: 0.01,
        ..Default::default()
    };

    let mut optimizer = AdamW::new(varmap.all_vars(), optimizer_params)?;

    let mut rng = rand::rng();

    for epoch_idx in 0..100 {
        let mut state = env.reset(rng.random::<u64>())?;
        let mut steps: Vec<Step<i64>> = vec![];

        loop {
            let action = {
                let action_probs: Vec<f32> =
                    softmax(&model.forward(&state.detach().unsqueeze(0)?)?, 1)?
                        .squeeze(0)?
                        .to_vec1()?;
                weighted_sample(action_probs, &mut rng)? as i64
            };

            let step = env.step(action)?;
            steps.push(step.copy_with_obs(&state));

            if step.terminated || step.truncated {
                state = env.reset(rng.random::<u64>())?;
                if steps.len() > 5000 {
                    break;
                }
            } else {
                state = step.state;
            }
        }

        let total_reward: f64 = steps.iter().map(|s| s.reward).sum();
        let episodes: i64 = steps
            .iter()
            .map(|s| (s.terminated || s.truncated) as i64)
            .sum();
        println!(
            "epoch: {:<3} episodes: {:<5} avg reward per episode: {:.2}",
            epoch_idx,
            episodes,
            total_reward / episodes as f64
        );

        let batch_size = steps.len();

        let rewards = Tensor::from_vec(accumulate_rewards(&steps), batch_size, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .detach();

        let actions_mask = {
            let actions: Vec<i64> = steps.iter().map(|s| s.action).collect();
            let actions_mask: Vec<Tensor> = actions
                .iter()
                .map(|&action| {
                    // One-hot encoding
                    let mut action_mask = vec![0.0; env.action_space()];
                    action_mask[action as usize] = 1.0;

                    Tensor::from_vec(action_mask, env.action_space(), &Device::Cpu)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                })
                .collect();
            Tensor::stack(&actions_mask, 0)?.detach()
        };

        let states = {
            let states: Vec<Tensor> = steps.into_iter().map(|s| s.state).collect();
            Tensor::stack(&states, 0)?.detach()
        };

        let log_probs = actions_mask
            .mul(&log_softmax(&model.forward(&states)?, 1)?)?
            .sum(1)?;

        let loss = rewards.mul(&log_probs)?.neg()?.mean_all()?;
        optimizer.backward_step(&loss)?;
    }

    Ok(())
}
