# Common Errors

## Missing symbols when compiling with the mkl feature.

If you get some missing symbols when compiling binaries/tests using the mkl
or accelerate features, e.g. for mkl you get:
```
  = note: /usr/bin/ld: (....o): in function `blas::sgemm':
          .../blas-0.22.0/src/lib.rs:1944: undefined reference to `sgemm_' collect2: error: ld returned 1 exit status

  = note: some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified
  = note: use the `-l` flag to specify native libraries to link
  = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo
```
or for accelerate:
```
Undefined symbols for architecture arm64:
            "_dgemm_", referenced from:
                candle_core::accelerate::dgemm::h1b71a038552bcabe in libcandle_core...
            "_sgemm_", referenced from:
                candle_core::accelerate::sgemm::h2cf21c592cba3c47 in libcandle_core...
          ld: symbol(s) not found for architecture arm64
```

This is likely due to a missing linker flag that was needed to enable the mkl library. You
can try adding the following for mkl at the top of your binary:
```rust
extern crate intel_mkl_src;
```
or for accelerate:
```rust
extern crate accelerate_src;
```

## Cannot run the LLaMA examples: access to source requires login credentials

```
Error: request error: https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json: status code 401
```

This is likely because you're not permissioned for the LLaMA-v2 model. To fix
this, you have to register on the huggingface-hub, accept the [LLaMA-v2 model
conditions](https://huggingface.co/meta-llama/Llama-2-7b-hf), and set up your
authentication token. See issue
[#350](https://github.com/huggingface/candle/issues/350) for more details.

## Missing cute/cutlass headers when compiling flash-attn

```
  In file included from kernels/flash_fwd_launch_template.h:11:0,
                   from kernels/flash_fwd_hdim224_fp16_sm80.cu:5:
  kernels/flash_fwd_kernel.h:8:10: fatal error: cute/algorithm/copy.hpp: No such file or directory
   #include <cute/algorithm/copy.hpp>
            ^~~~~~~~~~~~~~~~~~~~~~~~~
  compilation terminated.
  Error: nvcc error while compiling:
```
[cutlass](https://github.com/NVIDIA/cutlass) is provided as a git submodule so you may want to run the following command to check it in properly.
```bash
git submodule update --init
```

## Compiling with flash-attention fails

```
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
```

This is a bug in gcc-11 triggered by the Cuda compiler. To fix this, install a different, supported gcc version - for example gcc-10, and specify the path to the compiler in the CANDLE_NVCC_CCBIN environment variable.
```
env CANDLE_NVCC_CCBIN=/usr/lib/gcc/x86_64-linux-gnu/10 cargo ...
```

## Linking error on windows when running rustdoc or mdbook tests

```
Couldn't compile the test.
---- .\candle-book\src\inference\hub.md - Using_the_hub::Using_in_a_real_model_ (line 50) stdout ----
error: linking with `link.exe` failed: exit code: 1181
//very long chain of linking
 = note: LINK : fatal error LNK1181: cannot open input file 'windows.0.48.5.lib'
```

Make sure you link all native libraries that might be located outside a project target, e.g., to run mdbook tests, you should run:

```
mdbook test candle-book -L .\target\debug\deps\ `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.42.2\lib `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.48.5\lib
```

## Extremely slow model load time with WSL

This may be caused by the models being loaded from `/mnt/c`, more details on
[stackoverflow](https://stackoverflow.com/questions/68972448/why-is-wsl-extremely-slow-when-compared-with-native-windows-npm-yarn-processing).

## Tracking down errors

You can set `RUST_BACKTRACE=1` to be provided with backtraces when a candle
error is generated.