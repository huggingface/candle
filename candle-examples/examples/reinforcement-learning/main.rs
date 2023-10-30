#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod gym_env;
mod vec_gym_env;

mod ddpg;

use candle::{Device, Result, Tensor};
use clap::Parser;
use rand::Rng;

// The impact of the q value of the next state on the current state's q value.
const GAMMA: f64 = 0.99;
// The weight for updating the target networks.
const TAU: f64 = 0.005;
// The capacity of the replay buffer used for sampling training data.
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
// The training batch size for each training iteration.
const TRAINING_BATCH_SIZE: usize = 100;
// The total number of episodes.
const MAX_EPISODES: usize = 100;
// The maximum length of an episode.
const EPISODE_LENGTH: usize = 200;
// The number of training iterations after one episode finishes.
const TRAINING_ITERATIONS: usize = 200;

// Ornstein-Uhlenbeck process parameters.
const MU: f64 = 0.0;
const THETA: f64 = 0.15;
const SIGMA: f64 = 0.1;

const ACTOR_LEARNING_RATE: f64 = 1e-4;
const CRITIC_LEARNING_RATE: f64 = 1e-3;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let env = gym_env::GymEnv::new("Pendulum-v1")?;
    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let size_state = env.observation_space().iter().product::<usize>();
    let size_action = env.action_space();

    let mut agent = ddpg::DDPG::new(
        &Device::Cpu,
        size_state,
        size_action,
        true,
        ACTOR_LEARNING_RATE,
        CRITIC_LEARNING_RATE,
        GAMMA,
        TAU,
        REPLAY_BUFFER_CAPACITY,
        ddpg::OuNoise::new(MU, THETA, SIGMA, size_action)?,
    )?;

    let mut rng = rand::thread_rng();

    for episode in 0..MAX_EPISODES {
        // let mut state = env.reset(episode as u64)?;
        let mut state = env.reset(rng.gen::<u64>())?;

        let mut total_reward = 0.0;
        for _ in 0..EPISODE_LENGTH {
            let mut action = 2.0 * agent.actions(&state)?;
            action = action.clamp(-2.0, 2.0);

            let step = env.step(vec![action])?;
            total_reward += step.reward;

            agent.remember(
                &state,
                &Tensor::new(vec![action], &Device::Cpu)?,
                &Tensor::new(vec![step.reward as f32], &Device::Cpu)?,
                &step.state,
                step.terminated,
                step.truncated,
            );

            if step.terminated || step.truncated {
                break;
            }
            state = step.state;
        }

        println!("episode {episode} with total reward of {total_reward}");

        for _ in 0..TRAINING_ITERATIONS {
            agent.train(TRAINING_BATCH_SIZE)?;
        }
    }

    println!("Testing...");
    agent.train = false;
    for episode in 0..10 {
        // let mut state = env.reset(episode as u64)?;
        let mut state = env.reset(rng.gen::<u64>())?;
        let mut total_reward = 0.0;
        for _ in 0..EPISODE_LENGTH {
            let mut action = 2.0 * agent.actions(&state)?;
            action = action.clamp(-2.0, 2.0);

            let step = env.step(vec![action])?;
            total_reward += step.reward;

            if step.terminated || step.truncated {
                break;
            }
            state = step.state;
        }
        println!("episode {episode} with total reward of {total_reward}");
    }
    Ok(())
}
