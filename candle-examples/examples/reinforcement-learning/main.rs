#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::Result;

mod gym_env;
mod vec_gym_env;

mod ddpg;
mod policy_gradient;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args
        .iter()
        .map(|x| x.as_str())
        .collect::<Vec<_>>()
        .as_slice()
    {
        [_, "pg"] => policy_gradient::run()?,
        [_, "ddpg"] => ddpg::run()?,
        _ => println!("usage: main pg|ddpg"),
    }
    Ok(())
}
