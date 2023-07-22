# candle-kernels

This crate contains CUDA kernels used from candle. Some of these implementations
come from the [dfdx crate](https://github.com/coreylowman/dfdx).

The `ln*` files come from the [flash-attention
repo](https://github.com/Dao-AILab/flash-attention) and have been edited so as
to compile without including the PyTorch codebase.
