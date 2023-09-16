## Installation 

From the `candle-pyo3` directory, enable a virtual env where you will want the
candle package to be installed then run.

```bash
maturin develop -r 
python test.py
```

## Generating Stub Files for Type Hinting

For type hinting support, the `candle-pyo3` package requires `*.pyi` files. You can automatically generate these files using the `stub.py` script.

### Steps:
1. Install the package using `maturin`.
2. Generate the stub files by running:
   ```
   python stub.py
   ```

### Validation:
To ensure that the stub files match the current implementation, execute:
```
python stub.py --check
```
