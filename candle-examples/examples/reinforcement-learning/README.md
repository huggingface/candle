# candle-reinforcement-learning

Reinforcement Learning examples for candle.

## System wide python installation

This has been tested with `gymnasium` version `0.29.1`. You can install the
Python package with:
```bash
pip install "gymnasium[accept-rom-license]"
```

## Mac OS uv venv python installation

```bash
$ uv venv
$ source .venv/bin/activate
$ uv pip install gymnasium
$ which python3.xx 
> /path/to/.venv/bin/python3.xx
$ find ~/.local/share/uv/python -name "libpython3.xx.dylib"
> /path/to/pythondir/lib/libpython3.xx.dylib
$ cp /path/to/pythondir/lib/libpython3.xx.dylib ~/libpython3.xx.dylib.backup
$ install_name_tool -id "/path/to/pythondir/lib/libpython3.xx.dylib" "/path/to/pythondir/lib/libpython3.xx.dylib"
$ otool -L /path/to/pythondir/lib/libpython3.xx.dylib
$ cargo clean
$ export PYTHONHOME=/path/to/pythondir
$ export PYTHONPATH=/path/to/pythondir/lib/python3.xx
```
## Running an example

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
