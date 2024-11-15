#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::Result;
use clap::{Parser, Subcommand};

mod gym_env;
mod vec_gym_env;

mod ddpg;
mod dqn;
mod policy_gradient;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Pg,
    Ddpg,
    Dqn,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Pg => policy_gradient::run()?,
        Command::Ddpg => ddpg::run()?,
        Command::Dqn => dqn::run()?,
    }
    Ok(())
}
