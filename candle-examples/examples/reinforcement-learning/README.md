# candle-reinforcement-learning

Reinforcement Learning examples for candle.

> [!WARNING]  
> uv is not currently compatible with pyo3 as of 2025/3/28.

## System wide python

This has been tested with `gymnasium` version `0.29.1`. You can install the
Python package with:
```bash
pip install "gymnasium[accept-rom-license]"
```

In order to run the examples, use the following commands. Note the additional
`--package` flag to ensure that there is no conflict with the `candle-pyo3`
crate.

For the Policy Gradient example:
```bash
cargo run --example reinforcement-learning --features=pyo3 --package candle-examples -- pg
```

For the Deep Deterministic Policy Gradient example:
```bash
cargo run --example reinforcement-learning --features=pyo3 --package candle-examples -- ddpg
```
