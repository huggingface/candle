# candle
Minimalist ML framework for Rust

## FAQ

### Missing symbols when compiling with the mkl feature.

If you get some missing symbols when compiling binaries/tests using the mkl
features, e.g.:
```
  = note: /usr/bin/ld: (....o): in function `blas::sgemm':
          .../blas-0.22.0/src/lib.rs:1944: undefined reference to `sgemm_' collect2: error: ld returned 1 exit status

  = note: some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified
  = note: use the `-l` flag to specify native libraries to link
  = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargorustc-link-libkindname)
```

This is likely due to some missing linker flag that enable the mkl library. You
can try adding the following at the top of your binary:
```
extern crate intel_mkl_src;
```
