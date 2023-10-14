#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod gym_env;
mod vec_gym_env;

use candle::Result;
use clap::Parser;
use rand::Rng;

// The total number of episodes.
const MAX_EPISODES: usize = 100;
// The maximum length of an episode.
const EPISODE_LENGTH: usize = 200;

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

    let _num_obs = env.observation_space().iter().product::<usize>();
    let _num_actions = env.action_space();

    let mut rng = rand::thread_rng();

    for episode in 0..MAX_EPISODES {
        let mut obs = env.reset(episode as u64)?;

        let mut total_reward = 0.0;
        for _ in 0..EPISODE_LENGTH {
            let actions = rng.gen_range(-2.0..2.0);

            let step = env.step(vec![actions])?;
            total_reward += step.reward;

            if step.is_done {
                break;
            }
            obs = step.obs;
        }

        println!("episode {episode} with total reward of {total_reward}");
    }
    Ok(())
}
