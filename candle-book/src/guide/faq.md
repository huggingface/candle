# FAQ

If you are troubleshooting an error, see the [**common errors**](https://huggingface.github.io/candle/guide/common_errors.html) page for some common errors and fixes.

## Other ML frameworks

- [dfdx](https://github.com/coreylowman/dfdx) is a formidable crate, with shapes being included
  in types. This prevents a lot of headaches by getting the compiler to complain about shape mismatches right off the bat.
  However, we found that some features still require nightly, and writing code can be a bit daunting for non rust experts.

  We're leveraging and contributing to other core crates for the runtime so hopefully both crates can benefit from each
  other.

- [burn](https://github.com/burn-rs/burn) is a general crate that can leverage multiple backends so you can choose the best
  engine for your workload.

- [tch-rs](https://github.com/LaurentMazare/tch-rs.git) Bindings to the torch library in Rust. Extremely versatile, but they 
  bring in the entire torch library into the runtime. The main contributor of `tch-rs` is also involved in the development
  of `candle`.