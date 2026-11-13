# candle-vulkan-kernels

Slang compute shaders in `shaders/` compiled to SPIR-V at build time by
`slangc` (see `build.rs`) and executed through `vulkano`.

Code that is shared by many kernels lives in Slang module files — files with
a `module` declaration and no entry points of their own. Kernels pull the
shared code in with `import`:

- `bf16_emulation.slang`: bf16 <-> float encode/decode helpers.
- `f8e4m3_emulation.slang`: f8e4m3 <-> float encode/decode helpers.
- `dequant_helpers.slang`: f16 bit-pattern decode and int8 helpers shared by
  the dequant and fused GEMM kernels.
- `common_affine.slang`, `common_unary.slang`, `common_const_set.slang`,
  `common_norm.slang`, `common_upsample.slang`: the shared push-constant
  `Params` layouts for each kernel family.

`build.rs` never compiles a module file standalone; it compiles every other
`.slang` file in `shaders/` and passes `shaders/` as an include path when
compiling the test shaders in `shaders/test/`.

## Build-time environment variables

`build.rs` honors the following environment variables when invoking
`slangc`:

- `SLANGC`: path to the slangc executable to use (defaults to `slangc`
  on the PATH).
- `SLANGCFLAGS`: extra flags, space-separated, appended to every slangc
  invocation.
- `CANDLE_VULKAN_SLANG_OPT`: override for the slangc optimization flag.
  The value is passed to slangc as-is (e.g. `-O2`); `none` (case
  insensitive) or an empty value disables the flag. Takes precedence
  over `CANDLE_VULKAN_SLANG_NO_OPT` and the `PROFILE` default below.
- `CANDLE_VULKAN_SLANG_NO_OPT`: set to any value other than `0` to
  disable the optimization flag even when the `PROFILE` default would
  add one.
- `PROFILE` (set by cargo): when it is `release` and neither override is
  set, slangc is invoked with `-O3`; other profiles use the slangc
  default.
