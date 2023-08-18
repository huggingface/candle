#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let file = std::fs::File::open("/Users/laurent/tmp/linear_layer_state_dict/data.pkl")?;
    let mut br = std::io::BufReader::new(file);
    let mut stack = candle_core::pickle::Stack::empty();
    stack.read_loop(&mut br)?;
    Ok(())
}
