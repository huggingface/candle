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
