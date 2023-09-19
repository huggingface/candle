# Error management

You might have seen in the code base a lot of `.unwrap()` or `?`.
If you're unfamiliar with Rust check out the [Rust book](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html)
for more information.

What's important to know though, is that if you want to know *where* a particular operation failed
You can simply use `RUST_BACKTRACE=1` to get the location of where the model actually failed.

Let's see on failing code:

```rust,ignore
let x = Tensor::zeros((1, 784), DType::F32, &device)?;
let y = Tensor::zeros((1, 784), DType::F32, &device)?;
let z = x.matmul(&y)?;
```

Will print at runtime:

```bash
Error: ShapeMismatchBinaryOp { lhs: [1, 784], rhs: [1, 784], op: "matmul" }
``` 


After adding `RUST_BACKTRACE=1`:


```bash
Error: WithBacktrace { inner: ShapeMismatchBinaryOp { lhs: [1, 784], rhs: [1, 784], op: "matmul" }, backtrace: Backtrace [{ fn: "candle::error::Error::bt", file: "/home/nicolas/.cargo/git/checkouts/candle-5bb8ef7e0626d693/f291065/candle-core/src/error.rs", line: 200 }, { fn: "candle::tensor::Tensor::matmul", file: "/home/nicolas/.cargo/git/checkouts/candle-5bb8ef7e0626d693/f291065/candle-core/src/tensor.rs", line: 816 }, { fn: "myapp::main", file: "./src/main.rs", line: 29 }, { fn: "core::ops::function::FnOnce::call_once", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/ops/function.rs", line: 250 }, { fn: "std::sys_common::backtrace::__rust_begin_short_backtrace", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys_common/backtrace.rs", line: 135 }, { fn: "std::rt::lang_start::{{closure}}", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/rt.rs", line: 166 }, { fn: "core::ops::function::impls::<impl core::ops::function::FnOnce<A> for &F>::call_once", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/ops/function.rs", line: 284 }, { fn: "std::panicking::try::do_call", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs", line: 500 }, { fn: "std::panicking::try", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs", line: 464 }, { fn: "std::panic::catch_unwind", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panic.rs", line: 142 }, { fn: "std::rt::lang_start_internal::{{closure}}", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/rt.rs", line: 148 }, { fn: "std::panicking::try::do_call", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs", line: 500 }, { fn: "std::panicking::try", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs", line: 464 }, { fn: "std::panic::catch_unwind", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panic.rs", line: 142 }, { fn: "std::rt::lang_start_internal", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/rt.rs", line: 148 }, { fn: "std::rt::lang_start", file: "/rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/rt.rs", line: 165 }, { fn: "main" }, { fn: "__libc_start_main" }, { fn: "_start" }] }
```

Not super pretty at the moment, but we can see error occurred on `{ fn: "myapp::main", file: "./src/main.rs", line: 29 }`


Another thing to note, is that since Rust is compiled it is not necessarily as easy to recover proper stacktraces
especially in release builds. We're using [`anyhow`](https://docs.rs/anyhow/latest/anyhow/) for that.
The library is still young, please [report](https://github.com/LaurentMazare/candle/issues) any issues detecting where an error is coming from.

## Cuda error management

When running a model on Cuda, you might get a stacktrace not really representing the error.
The reason is that CUDA is async by nature, and therefore the error might be caught while you were sending totally different kernels.

One way to avoid this is to use `CUDA_LAUNCH_BLOCKING=1` as an environment variable. This will force every kernel to be launched sequentially.
You might still however see the error happening on other kernels as the faulty kernel might exit without an error but spoiling some pointer for which the error will happen when dropping the `CudaSlice` only.


If this occurs, you can use [`compute-sanitizer`](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
This tool is like `valgrind` but for cuda. It will help locate the errors in the kernels.


