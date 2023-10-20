This python module contains external typehinting for certain `candle` classes. This is only necessary for `magic` methodes e.g. `__add__` as their text signature cant be set via pyo3.

The classes in this module will be parsed by the `stub.py` script and interleafed with the signatures of the actual pyo3 `candle.candle` module.