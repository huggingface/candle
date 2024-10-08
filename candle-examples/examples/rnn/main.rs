#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, Tensor};
use candle_nn::{rnn, LSTMConfig, RNN};
use clap::Parser;

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "lstm")]
    LSTM,
    #[value(name = "gru")]
    GRU,
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 10)]
    input_dim: usize,

    #[arg(long, default_value_t = 20)]
    hidden_dim: usize,

    #[arg(long, default_value_t = 1)]
    layers: usize,

    #[arg(long)]
    bidirection: bool,

    #[arg(long, default_value_t = 5)]
    batch_size: usize,

    #[arg(long, default_value_t = 3)]
    seq_len: usize,

    #[arg(long, default_value = "lstm")]
    model: WhichModel,
}

fn lstm_config(layer_idx: usize, direction: rnn::Direction) -> LSTMConfig {
    let mut config = LSTMConfig::default();
    config.layer_idx = layer_idx;
    config.direction = direction;
    config
}

fn gru_config(layer_idx: usize, direction: rnn::Direction) -> rnn::GRUConfig {
    let mut config = rnn::GRUConfig::default();
    config.layer_idx = layer_idx;
    config.direction = direction;
    config
}

fn run_lstm(args: Args) -> Result<Tensor> {
    let device = candle_examples::device(args.cpu)?;
    let map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&map, DType::F32, &device);

    let mut layers = Vec::with_capacity(args.layers);

    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim
        };
        let config = lstm_config(layer_idx, rnn::Direction::Forward);
        let lstm = candle_nn::lstm(input_dim, args.hidden_dim, config, vb.clone())?;
        layers.push(lstm);
    }

    let mut input = Tensor::randn(
        0.0_f32,
        1.0,
        (args.batch_size, args.seq_len, args.input_dim),
        &device,
    )?;

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    Ok(input)
}

fn run_gru(args: Args) -> Result<Tensor> {
    let device = candle_examples::device(args.cpu)?;
    let map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&map, DType::F32, &device);

    let mut layers = Vec::with_capacity(args.layers);

    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim
        };
        let config = gru_config(layer_idx, rnn::Direction::Forward);
        let gru = candle_nn::gru(input_dim, args.hidden_dim, config, vb.clone())?;
        layers.push(gru);
    }

    let mut input = Tensor::randn(
        0.0_f32,
        1.0,
        (args.batch_size, args.seq_len, args.input_dim),
        &device,
    )?;

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    Ok(input)
}

fn run_bidirectional_lstm(args: Args) -> Result<Tensor> {
    let device = candle_examples::device(args.cpu)?;
    let map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&map, DType::F32, &device);

    let mut layers = Vec::with_capacity(args.layers);

    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim * 2
        };

        let forward_config = lstm_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::lstm(input_dim, args.hidden_dim, forward_config, vb.clone())?;

        let backward_config = lstm_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::lstm(input_dim, args.hidden_dim, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    let mut input = Tensor::randn(
        0.0_f32,
        1.0,
        (args.batch_size, args.seq_len, args.input_dim),
        &device,
    )?;

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.combine_states_to_tensor(&forward_states, &backward_states)?;
    }
    Ok(input)
}

fn run_bidirectional_gru(args: Args) -> Result<Tensor> {
    let device = candle_examples::device(args.cpu)?;
    let map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&map, DType::F32, &device);

    let mut layers = Vec::with_capacity(args.layers);
    for layer_idx in 0..args.layers {
        let input_dim = if layer_idx == 0 {
            args.input_dim
        } else {
            args.hidden_dim * 2
        };

        let forward_config = gru_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::gru(input_dim, args.hidden_dim, forward_config, vb.clone())?;

        let backward_config = gru_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::gru(input_dim, args.hidden_dim, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    let mut input = Tensor::randn(
        0.0_f32,
        1.0,
        (args.batch_size, args.seq_len, args.input_dim),
        &device,
    )?;

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.combine_states_to_tensor(&forward_states, &backward_states)?;
    }

    Ok(input)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let runs = if args.bidirection { 2 } else { 1 };
    let batch_size = args.batch_size;
    let seq_len = args.seq_len;
    let hidden_dim = args.hidden_dim;

    println!(
        "Running {:?} bidirection: {} layers: {}",
        args.model, args.bidirection, args.layers
    );

    let output = match (args.model, args.bidirection) {
        (WhichModel::LSTM, false) => run_lstm(args),
        (WhichModel::GRU, false) => run_gru(args),
        (WhichModel::LSTM, true) => run_bidirectional_lstm(args),
        (WhichModel::GRU, true) => run_bidirectional_gru(args),
    }?;

    assert_eq!(output.dims3()?, (batch_size, seq_len, hidden_dim * runs));

    Ok(())
}
