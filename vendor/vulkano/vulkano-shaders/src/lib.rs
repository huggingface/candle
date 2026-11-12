//! The procedural macro for vulkano's shader system. Manages the compile-time compilation of
//! shader code into SPIR-V and generation of associated Rust code.
//!
//! # Requirements
//!
//! This crate invokes [`glslc`] and/or [`slangc`] at compile time to compile shaders. `glslc`
//! and/or `slangc` must be installed and available on your `PATH`. Both are included in the
//! [Vulkan SDK] or can be installed separately via your package manager.
//!
//! # Basic usage
//!
//! ```
//! mod vs {
//!     # mod vulkano_shaders { macro_rules! shader { ($($tt:tt)*) => {}; } pub(super) use shader; }
//!     #
//!     vulkano_shaders::shader! {
//!         ty: "vertex",
//!         path: "path/to/vertex_shader.glsl",
//!     }
//! }
//! ```
//!
//! # Details
//!
//! If you want to take a look at what the macro generates, your best option is to use
//! [cargo-expand] to view the expansion of the macro in your own code. On the other hand, if you
//! are looking for a high-level overview, you can see the below section.
//!
//! # Generated code overview
//!
//! The macro generates the following items of interest:
//!
//! - The `load` function. This function takes an `&Arc<Device>`, constructs a [`ShaderModule`]
//!   with the passed-in device and the shader data provided via the macro, and returns
//!   `Result<Arc<ShaderModule>, VulkanError>`. It also generates the `try_load` function, which
//!   works the same, but returns a `Result<Arc<ShaderModule>, Validated<VulkanError>>`. These
//!   functions are both unsafe because they delegate to the unsafe [`ShaderModule::try_new`].
//!   Additionally, a `load_unchecked` function is generated, which skips validation entirely and
//!   delegates to [`ShaderModule::new_unchecked`].
//! - If the `shaders` option is used, then instead of one `load` constructor, there is one for
//!   each shader. They are named based on the provided names, `load_first`, `load_second` etc.
//!   `try_` and `_unchecked` variants are also generated for each shader.
//! - A Rust struct translated from each struct contained in the shader data. By default, each
//!   structure has a `Clone` and a `Copy` implementation. This behavior can be customized through
//!   the `custom_derives` macro option (see below for details). Each struct also has an
//!   implementation of [`BufferContents`], so that it can be read from/written to a buffer.
//!
//! All of these generated items will be accessed through the module where the macro was invoked.
//! If you wanted to store the `ShaderModule` in a struct of your own, you could do something like
//! this:
//!
//! ```
//! # use std::sync::Arc;
//! # use vulkano::{device::Device, shader::ShaderModule, Validated, VulkanError};
//! #
//! # mod vs {
//! #     vulkano_shaders::shader! {
//! #         ty: "vertex",
//! #         src: r"
//! #             #version 450
//! #
//! #             layout(location = 0) in vec3 position;
//! #
//! #             void main() {
//! #                 gl_Position = vec4(position, 1.0);
//! #             }
//! #         ",
//! #     }
//! # }
//! #
//! # fn main() {}
//! #
//! // ...various use statements...
//! // ...`vs` module containing a `shader!` call...
//!
//! pub struct Shaders {
//!     pub vs: Arc<ShaderModule>,
//! }
//!
//! impl Shaders {
//!     pub unsafe fn load(device: &Arc<Device>) -> Result<Self, VulkanError> {
//!         Ok(Self {
//!             vs: unsafe { vs::load(device) }?,
//!         })
//!     }
//! }
//! ```
//!
//! # Options
//!
//! The options available are in the form of the following fields:
//!
//! ## `ty`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     ty: "...",
//!     ...
//! }
//! ```
//!
//! This defines what shader type the given shader source will be compiled into. It can be any of
//! the following:
//!
//! - `vertex`
//! - `tess_ctrl`
//! - `tess_eval`
//! - `geometry`
//! - `task`
//! - `mesh`
//! - `fragment`
//! - `compute`
//! - `raygen`
//! - `anyhit`
//! - `closesthit`
//! - `miss`
//! - `intersection`
//! - `callable`
//!
//! For details on what these shader types mean, [see Vulkano's documentation][pipeline].
//!
//! If unspecified, the behavior is the same as the behavior of the compiler of choice when the
//! shader stage is unspecified. In particular, this means that shaderc will expect a `#pragma
//! shader_stage(<stage>)` directive in the code, and slangc will expect the entry point(s) to be
//! annotated with `[shader("<stage>")]`.
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `path`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     path: "...",
//!     ...
//! }
//! ```
//!
//! Provides the path to the shader source to be compiled, relative to the file invoking the macro.
//!
//! Cannot be defined alongside the [`src`] or [`bytes`] option.
//!
//! ## `src`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     src: "...",
//!     ...
//! }
//! ```
//!
//! Provides the raw shader source to be compiled in the form of a string.
//!
//! Cannot be defined alongside the [`path`] or [`bytes`] option.
//!
//! ## `bytes`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     bytes: "...",
//!     ...
//! }
//! ```
//!
//! Provides the path to precompiled SPIR-V bytecode, relative to the file invoking the macro.
//!
//! Cannot be defined alongside the [`src`] or [`path`] option, and may also not specify the
//! [`ty`], [`lang`], [`entry_point`] or [`compiler`] option. This allows using shaders compiled
//! through a separate build system.
//!
//! ## `lang`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     lang: "...",
//!     ...
//! }
//! ```
//!
//! Provides the language of the shader source. It can be any of the following:
//!
//! - `glsl` (default)
//! - `hlsl`
//! - `slang`
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `entry_point`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     entry_point: "...",
//!     ...
//! }
//! ```
//!
//! Provides the entry point of the shader source to compile.
//!
//! If unspecified, the behavior is the same as the behavior of the compiler of choice when the
//! entry point is unspecified. In particular, this means that the entry point defaults to `main`,
//! except when using slangc and the [`ty`] option is also left out, in which case slangc searches
//! for entry points annotated with `[shader("<stage>")]`.
//!
//! This option cannot be used when compiling with shaderc as it has no support for compiling any
//! entry point other than `main`.
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `compiler`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     compiler: "...",
//!     ...
//! }
//! ```
//!
//! Provides the compiler to invoke. It can be any of the following:
//!
//! - `shaderc` (default for GLSL)
//! - `slangc` (default for HLSL and Slang)
//!
//! Here are the source languages supported for each compiler:
//!
//! - `shaderc`: GLSL
//! - `slangc`: GLSL, HLSL and Slang
//!
//! When using `shaderc`, you must have [`glslc`] available on your `PATH`. When using `slangc`,
//! you must have [`slangc`] available on your `PATH`. The [Vulkan SDK] comes with both.
//!
//! Cannot be defined alongside the [`bytes`] option.
//!
//! ## `shaders`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     shaders: {
//!         first: {
//!             ...
//!         },
//!         ...
//!     },
//!     ...
//! }
//! ```
//!
//! With this option, the user can compile several shaders in a single macro invocation. Each entry
//! key will be the suffix of the generated `load` function (`load_first` in the above example).
//! However, all other Rust structs translated from the shader source will be shared between
//! shaders. The macro checks that the source structs with the same names between different shaders
//! have the same declaration signature, and throws a compile-time error if they don't.
//!
//! Each entry accepts [`ty`], [`path`], [`src`], [`bytes`] and [`entry_point`] options same as
//! above.
//!
//! If an entry defines the [`lang`] option, it overrides a potential global `lang` option:
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     shaders: {
//!         first: {
//!             ty: "...",
//!             path: "...",
//!         },
//!         second: {
//!             ty: "...",
//!             path: "...",
//!             lang: "slang",
//!         },
//!     },
//!     lang: "hlsl",
//! }
//! ```
//!
//! Above, the shaders named `first` and `second` will be compiled as HLSL and Slang, respectively.
//! Without the global `lang` option present, they will be compiled as GLSL and Slang,
//! respectively.
//!
//! If an entry defines the [`compiler`] option, it overrides a potential global `compiler` option
//! in the same way as the `lang` option does.
//!
//! If an entry defines the [`define`] option, it extends a potential global `define` option,
//! overriding any duplicates:
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     shaders: {
//!         first: {
//!             ty: "...",
//!             path: "...",
//!         },
//!         second: {
//!             ty: "...",
//!             path: "...",
//!             define: [("X", "69"), ("Z", "3")],
//!         },
//!     },
//!     define: [("X", "1"), ("Y", "2")],
//! }
//! ```
//!
//! Above, the shaders named `first` and `second` will be compiled with `-DX=1 -DY=2` and
//! `-DX=69 -DY=2 -DZ=3`, respectively. Without the global `define` option present, they will be
//! compiled with no defines and `-DX=69 -DZ=3`, respectively.
//!
//! ## `root_path_env`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     root_path_env: "...",
//!     ...
//! }
//! ```
//!
//! Instead of searching relative to the file invoking the macro, search relative to some other
//! folder specified by this env variable. The intended use case is using `OUT_DIR` to be able to
//! load shaders generated by your build script. You can also set this to `CARGO_MANIFEST_DIR` if
//! you want to search relative to the folder of your `Cargo.toml`.
//!
//! See [`cargo-env-vars`] for a full set of env variables set by cargo. It is also possible to
//! specify env variables from within the build script using the following:
//!
//! ```
//! # let shader_out_dir = "";
//! println!("cargo:rustc-env=SHADER_OUT_DIR={shader_out_dir}");
//! ```
//!
//! ## `include`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     include: ["...", "...", ...],
//!     ...
//! }
//! ```
//!
//! Specifies the standard include directories to be searched through when using the
//! `#include <...>` directive within a shader source. Include directories can be absolute or
//! relative to your `Cargo.toml`. Relative paths are relative to the directory which contains the
//! source file the `#include "..."` directive is declared in. This includes the Rust files with
//! embedded shader source.
//!
//! ## `define`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     define: [("NAME", "VALUE"), ...],
//!     ...
//! }
//! ```
//!
//! Adds the given macro definitions for the pre-processor. This is equivalent to passing the
//! `-DNAME=VALUE` argument on the command line.
//!
//! ## `vulkan_version`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     vulkan_version: "...",
//!     ...
//! }
//! ```
//!
//! Sets the Vulkan version to compile for. It can be any of the following:
//!
//! - `1.0` (default)
//! - `1.1`
//! - `1.2`
//! - `1.3`
//!
//! The generated code must be supported by the device at runtime. If not, then an error will be
//! returned when calling `load`.
//!
//! ## `spirv_version`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     spirv_version: "...",
//!     ...
//! }
//! ```
//!
//! Sets the SPIR-V version to compile for. It can be any of the following:
//!
//! - `1.0`
//! - `1.1`
//! - `1.2`
//! - `1.3`
//! - `1.4`
//! - `1.5`
//! - `1.6`
//!
//! By default, the minimum required SPIR-V version for the [`vulkan_version`] is used.
//!
//! The generated code must be supported by the device at runtime. If not, then an error will be
//! returned when calling `load`.
//!
//! ## `generate_structs`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     generate_structs: false,
//!     ...
//! }
//! ```
//!
//! Generate Rust structs that represent the structs contained in the shader. They all implement
//! [`BufferContents`], which allows them to be passed to the shader without having to worry about
//! the layout of the struct manually. However, some use cases, such as Rust-GPU, may not have any
//! use for such structs and may choose to disable them.
//!
//! By default, structs are generated.
//!
//! ## `custom_derives`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     custom_derives: [Clone, Default, PartialEq, ...],
//!     ...
//! }
//! ```
//!
//! Extends the list of derive macros that are added to the `derive` attribute of Rust structs that
//! represent shader structs.
//!
//! By default, each generated struct derives `Clone` and `Copy`. If the struct has unsized
//! members, none of the derives are applied on the struct, except [`BufferContents`], which is
//! always derived.
//!
//! ## `linalg_type`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     linalg_type: "...",
//!     ...
//! }
//! ```
//!
//! Specifies the way that linear algebra types should be generated. It can be any of the
//! following:
//!
//! - `std` (default)
//! - `cgmath`
//! - `nalgebra`
//!
//! `std` uses arrays to represent vectors and matrices. Note that if the chosen crate doesn't have
//! a type that represents a certain linear algebra type (e.g., `mat3` or a rectangular matrix),
//! then the macro will default back to arrays for that type.
//!
//! If you use linear algebra types from a third-party crate, then you have to have the crate in
//! your dependencies with the appropriate feature enabled that adds `bytemuck` support.
//!
//! ## `dump`
//!
//! ```
//! # macro_rules! shader { ($($tt:tt)*) => {}; }
//! #
//! shader! {
//!     dump: true,
//!     ...
//! }
//! ```
//!
//! The crate fails to compile, but prints the generated Rust code to stdout.
//!
//! # Cargo features
//!
//! | Feature         | Description                                      |
//! |-----------------|--------------------------------------------------|
//! | `shaderc-debug` | Compile shaders with debug information included. |
//!
//! [`glslc`]: https://github.com/google/shaderc/tree/main/glslc
//! [`slangc`]: https://github.com/shader-slang/slang
//! [Vulkan SDK]: https://www.lunarg.com/vulkan-sdk/
//! [cargo-expand]: https://github.com/dtolnay/cargo-expand
//! [`ShaderModule`]: vulkano::shader::ShaderModule
//! [`ShaderModule::try_new`]: vulkano::shader::ShaderModule::try_new
//! [`ShaderModule::new_unchecked`]: vulkano::shader::ShaderModule::new_unchecked
//! [`BufferContents`]: vulkano::buffer::BufferContents
//! [pipeline]: vulkano::pipeline
//! [`cargo-env-vars`]: https://doc.rust-lang.org/cargo/reference/environment-variables.html
//! [`ty`]: self#ty
//! [`path`]: self#path
//! [`src`]: self#src
//! [`bytes`]: self#bytes
//! [`lang`]: self#lang
//! [`entry_point`]: self#entry_point
//! [`compiler`]: self#compiler
//! [`define`]: self#define
//! [`vulkan_version`]: self#vulkan_version

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
#![recursion_limit = "1024"]

use foldhash::HashMap;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::{
    collections::hash_map::Entry,
    env, fs,
    path::{Path, PathBuf},
};
use structs::TypeRegistry;
use syn::{
    braced, bracketed, parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote, Error, Ident, LitBool, LitStr, Path as SynPath, Token,
};
use vulkano::shader::spirv;

mod shaders;
mod structs;
#[cfg(test)]
mod tests;

#[proc_macro]
pub fn include_vulkano_glsl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    parse_macro_input!(input as syn::parse::Nothing);

    let include_string = include_str!("../include/vulkano.glsl");

    quote! { #include_string }.into()
}

#[proc_macro]
pub fn include_vulkano_slang(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    parse_macro_input!(input as syn::parse::Nothing);

    let include_string = include_str!("../include/vulkano.slang");

    quote! { #include_string }.into()
}

#[proc_macro]
pub fn shader(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as MacroInput);

    shader_inner(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn shader_inner(input: MacroInput) -> Result<TokenStream> {
    let MacroInput { shaders, options } = input;

    let mut state = MacroState::new(&options)?;

    match shaders {
        Shaders::Single(shader_fields) => {
            state.process_shader(None, shader_fields)?;
        }
        Shaders::Multiple(shaders) => {
            for (shader_ident, shader_fields) in shaders {
                let shader_name = shader_ident.to_string();
                state.process_shader(Some(shader_name), shader_fields)?;
            }
        }
    }

    let result = state.finalize();

    if options.dump.value {
        println!("{}", result);
        bail!(options.dump, "`shader!` Rust codegen dumped");
    }

    Ok(result)
}

struct MacroState<'a> {
    options: &'a MacroOptions,
    root_path: PathBuf,
    relative_path_error_message: String,
    shaders_code: TokenStream,
    structs_code: TokenStream,
    type_registry: TypeRegistry,
}

impl<'a> MacroState<'a> {
    fn new(options: &'a MacroOptions) -> Result<Self> {
        let (root_path, relative_path_error_message) = match options.root_path_env.as_ref() {
            None => root_path_from_call_site(),
            Some(root_path_env) => root_path_from_env_var(root_path_env),
        }?;

        Ok(MacroState {
            options,
            root_path,
            relative_path_error_message,
            shaders_code: TokenStream::new(),
            structs_code: TokenStream::new(),
            type_registry: TypeRegistry::default(),
        })
    }

    fn process_shader(
        &mut self,
        shader_name: Option<String>,
        shader_fields: ShaderFields,
    ) -> Result {
        let ShaderFields {
            shader_kind,
            source_kind,
            source_language,
            entry_point,
            compiler,
            macro_defines,
        } = shader_fields;

        let (lit, bytes, input_paths) = match source_kind.unwrap() {
            SourceKind::Compiled(compiled_source_kind) => {
                let source_path;
                let source_code;
                let working_dir;

                let lit = match compiled_source_kind {
                    CompiledSourceKind::Path(lit) => {
                        source_path = Some(self.root_path.join(lit.value()));
                        let path = source_path.as_deref().unwrap();

                        self.check_file_exists(&lit, path)?;

                        source_code = read_file_to_string(&lit, path)?;
                        working_dir = path.parent().unwrap();

                        lit
                    }
                    CompiledSourceKind::Inline(lit) => {
                        source_path = None;
                        source_code = lit.value();
                        working_dir = &self.root_path;

                        lit
                    }
                };

                let (bytes, mut input_paths) = shaders::compile_shader(
                    self.options,
                    &source_code,
                    working_dir,
                    shader_kind,
                    source_language,
                    entry_point.as_deref(),
                    compiler,
                    &macro_defines,
                )
                .map_err(|err| Error::new_spanned(&lit, err))?;

                if let Some(source_path) = source_path {
                    input_paths.push(source_path.into_os_string().into_string().unwrap());
                }

                (lit, bytes, input_paths)
            }
            SourceKind::Precompiled(lit) => {
                let path = self.root_path.join(lit.value());

                self.check_file_exists(&lit, &path)?;

                let bytes = read_file(&lit, &path)?;

                let input_paths = vec![path.into_os_string().into_string().unwrap()];

                (lit, bytes, input_paths)
            }
        };

        let words = spirv::bytes_to_words(&bytes)
            .map_err(|err| Error::new_spanned(&lit, format!("malformed SPIR-V: {err}")))?;

        let shaders_code = shaders::generate_shaders(shader_name.as_deref(), &words, input_paths)?;
        let structs_code = structs::generate_structs(
            self.options,
            lit,
            shader_name,
            &words,
            &mut self.type_registry,
        )?;

        self.shaders_code.extend(shaders_code);
        self.structs_code.extend(structs_code);

        Ok(())
    }

    fn check_file_exists(&self, lit: &LitStr, path: &Path) -> Result {
        if !path.is_file() {
            let msg = &self.relative_path_error_message;

            bail!(
                lit,
                "file `{path:?}` was not found; note that the path must be relative {msg}",
            );
        }

        Ok(())
    }

    fn finalize(self) -> TokenStream {
        let shaders_code = self.shaders_code;
        let structs_code = self.structs_code;

        quote! {
            #shaders_code
            #structs_code
        }
    }
}

fn root_path_from_call_site() -> Result<(PathBuf, String)> {
    let Some(file_path) = Span::call_site().unwrap().local_file() else {
        bail!(
            "failed to get the file invoking the macro from the compiler; this only happens when \
            using rust-analyzer with the file with this macro invocation open, and you can ignore \
            this error; if you can't live with it, you can use the `root_path_env: \
            \"CARGO_MANIFEST_DIR\"` option to search relative to your Cargo.toml instead",
        );
    };

    // This comes from cargo, so it better be a real directory path we have access to.
    //
    // TODO: Use a variable set by cargo for the working directory of rustc invocations once it's
    // stable such that this is truly bullet-proof.
    let working_dir = env::current_dir().expect("failed to get the current working directory");

    // The file path should be relative to the current working directory that rustc was invoked
    // with. The proc macros rustc invokes should have the same current working directory as that
    // rustc. So this should always give us the right full path.
    let full_path = working_dir
        .join(&file_path)
        .components()
        .collect::<PathBuf>();

    if !full_path.exists() {
        bail!(
            "the file invoking the macro, `{full_path:?}`, doesn't exist; you can use the \
            `root_path_env: \"CARGO_MANIFEST_DIR\"` option to search relative to your Cargo.toml \
            instead",
        );
    }

    // The file path comes from rustc, so it better be a real file path.
    let root = full_path.parent().expect(
        "the file invoking the macro we got from the compiler doesn't reside in a directory",
    );

    let error = format!("to the file `{file_path:?}` invoking the macro");

    Ok((root.into(), error))
}

fn root_path_from_env_var(root_path_env: &LitStr) -> Result<(PathBuf, String)> {
    let env = root_path_env.value();
    let root = env::var(&env).map_err(|e| {
        Error::new_spanned(
            root_path_env,
            format_args!(
                "failed to fetch the environment variable `{env}`: {e}; typical parameters are \
                `OUT_DIR` to gather results from your build script, `CARGO_MANIFEST_DIR` to \
                search relative to your Cargo.toml, or left default to search relative to the \
                file invoking the macro",
            ),
        )
    })?;
    let error = format!("to the path `{root}` specified by the env variable `{env:?}`");

    Ok((root.into(), error))
}

fn read_file_to_string(lit: &LitStr, path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|err| {
        Error::new_spanned(lit, format_args!("failed to read file `{path:?}`: {err}"))
    })
}

fn read_file(lit: &LitStr, path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|err| {
        Error::new_spanned(lit, format_args!("failed to read file `{path:?}`: {err}"))
    })
}

struct MacroInput {
    shaders: Shaders,
    options: MacroOptions,
}

enum Shaders {
    Single(ShaderFields),
    Multiple(HashMap<Ident, ShaderFields>),
}

#[derive(Default)]
struct ShaderFields {
    shader_kind: Option<ShaderKind>,
    source_kind: Option<SourceKind>,
    source_language: Option<SourceLanguage>,
    entry_point: Option<String>,
    compiler: Option<Compiler>,
    macro_defines: Vec<(String, String)>,
}

enum SourceKind {
    Compiled(CompiledSourceKind),
    Precompiled(LitStr),
}

enum CompiledSourceKind {
    Path(LitStr),
    Inline(LitStr),
}

struct MacroOptions {
    global_source_language: SourceLanguage,
    global_compiler: Option<Compiler>,
    root_path_env: Option<LitStr>,
    include_directories: Vec<PathBuf>,
    global_macro_defines: Vec<(String, String)>,
    spirv_version: SpirvVersion,
    vulkan_version: VulkanVersion,
    generate_structs: bool,
    custom_derives: Vec<SynPath>,
    linalg_type: LinAlgType,
    dump: LitBool,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut parser = MacroInputParser::new();

        parser.parse(input)?;

        let shaders = parser.shaders.unwrap();

        let vulkan_version = parser.vulkan_version.unwrap_or_default();
        let options = MacroOptions {
            global_source_language: parser.global_source_language.unwrap_or_default(),
            global_compiler: parser.global_compiler,
            root_path_env: parser.root_path_env,
            include_directories: parser.include_directories,
            global_macro_defines: parser.global_macro_defines,
            vulkan_version,
            spirv_version: parser
                .spirv_version
                .unwrap_or(vulkan_version.to_spirv_version()),
            generate_structs: parser.generate_structs.unwrap_or(true),
            custom_derives: parser.custom_derives.unwrap_or_else(|| {
                vec![
                    parse_quote! { ::std::clone::Clone },
                    parse_quote! { ::std::marker::Copy },
                ]
            }),
            linalg_type: parser.linalg_type.unwrap_or_default(),
            dump: parser
                .dump
                .unwrap_or_else(|| LitBool::new(false, Span::call_site())),
        };

        Ok(MacroInput { shaders, options })
    }
}

impl MacroOptions {
    #[cfg(test)]
    fn empty() -> Self {
        MacroOptions {
            global_source_language: SourceLanguage::Glsl,
            global_compiler: None,
            root_path_env: None,
            include_directories: Vec::new(),
            global_macro_defines: Vec::new(),
            vulkan_version: VulkanVersion::V1_0,
            spirv_version: SpirvVersion::V1_0,
            generate_structs: true,
            custom_derives: Vec::new(),
            linalg_type: LinAlgType::default(),
            dump: LitBool::new(false, Span::call_site()),
        }
    }
}

struct MacroInputParser {
    manifest_dir: String,
    shaders: Option<Shaders>,
    global_source_language: Option<SourceLanguage>,
    global_compiler: Option<Compiler>,
    root_path_env: Option<LitStr>,
    include_directories: Vec<PathBuf>,
    global_macro_defines: Vec<(String, String)>,
    vulkan_version: Option<VulkanVersion>,
    spirv_version: Option<SpirvVersion>,
    generate_structs: Option<bool>,
    custom_derives: Option<Vec<SynPath>>,
    linalg_type: Option<LinAlgType>,
    dump: Option<LitBool>,
}

impl MacroInputParser {
    fn new() -> Self {
        MacroInputParser {
            manifest_dir: env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()),
            shaders: None,
            global_source_language: None,
            global_compiler: None,
            root_path_env: None,
            include_directories: Vec::new(),
            global_macro_defines: Vec::new(),
            vulkan_version: None,
            spirv_version: None,
            generate_structs: None,
            custom_derives: None,
            linalg_type: None,
            dump: None,
        }
    }

    fn parse(&mut self, input: ParseStream<'_>) -> Result {
        while !input.is_empty() {
            let field_ident = input.parse::<Ident>()?;
            input.parse::<Token![:]>()?;
            let field_name = field_ident.to_string();

            match field_name.as_str() {
                "ty" | "path" | "src" | "bytes" | "entry_point" => {
                    self.parse_shader_field(input, &field_ident, &field_name)?
                }
                "lang" => self.parse_lang(input)?,
                "compiler" => self.parse_compiler(input)?,
                "shaders" => self.parse_shader_entries(input, &field_ident)?,
                "root_path_env" => self.parse_root_path_env(input)?,
                "include" => self.parse_include(input)?,
                "define" => self.parse_define(input)?,
                "vulkan_version" => self.parse_vulkan_version(input)?,
                "spirv_version" => self.parse_spirv_version(input)?,
                "generate_structs" => self.parse_generate_structs(input)?,
                "custom_derives" => self.parse_custom_derives(input)?,
                "linalg_type" => self.parse_linalg_type(input)?,
                "dump" => self.parse_dump(input)?,
                _ => bail!(
                    field_ident,
                    "expected `ty`, `path`, `src`, `bytes`, `entry_point`, `lang`, `compiler`, \
                    `shaders`, `root_path_env`, `include`, `define`, `vulkan_version`, \
                    `spirv_version`, `generate_structs`, `custom_derives`, `linalg_type` or \
                    `dump` as a field, found `{field_name}`",
                ),
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        self.check_valid()
    }

    fn parse_shader_field(
        &mut self,
        input: ParseStream<'_>,
        field_ident: &Ident,
        field_name: &str,
    ) -> Result {
        if matches!(&self.shaders, Some(Shaders::Multiple(_))) {
            bail!(
                field_ident,
                "only one of `path`, `src`, `bytes` or `shaders` can be defined",
            );
        }

        if self.shaders.is_none() {
            self.shaders = Some(Shaders::Single(ShaderFields::default()));
        }

        let Some(Shaders::Single(shader_fields)) = &mut self.shaders else {
            unreachable!();
        };

        shader_fields.parse_shader_field(input, field_name)
    }

    fn parse_lang(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.global_source_language.is_some() {
            bail!(lit, "field `lang` is already defined");
        }

        self.global_source_language = Some(match lit.value().as_str() {
            "glsl" => SourceLanguage::Glsl,
            "hlsl" => SourceLanguage::Hlsl,
            "slang" => SourceLanguage::Slang,
            lang => bail!(lit, "expected `glsl`, `hlsl` or `slang`, found `{lang}`"),
        });

        Ok(())
    }

    fn parse_compiler(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.global_compiler.is_some() {
            bail!(lit, "field `compiler` is already defined");
        }

        self.global_compiler = Some(match lit.value().as_str() {
            "shaderc" => Compiler::Shaderc,
            "slangc" => Compiler::Slangc,
            compiler => bail!(lit, "expected `shaderc` or `slangc`, found `{compiler}`"),
        });

        Ok(())
    }

    fn parse_shader_entries(&mut self, input: ParseStream<'_>, field_ident: &Ident) -> Result {
        if !self.shaders.is_none() {
            bail!(
                field_ident,
                "only one of `path`, `src`, `bytes` or `shaders` can be defined",
            );
        }

        self.shaders = Some(Shaders::Multiple(HashMap::default()));

        let Some(Shaders::Multiple(shaders)) = &mut self.shaders else {
            unreachable!();
        };

        let in_braces;
        braced!(in_braces in input);

        while !in_braces.is_empty() {
            let shader_ident = in_braces.parse::<Ident>()?;

            let shader_fields = match shaders.entry(shader_ident) {
                Entry::Occupied(entry) => {
                    let shader_ident = entry.key();
                    let shader_name = shader_ident.to_string();

                    bail!(
                        shader_ident,
                        "shader entry `{shader_name}` is already defined",
                    );
                }
                Entry::Vacant(entry) => entry.insert(ShaderFields::default()),
            };

            in_braces.parse::<Token![:]>()?;

            let in_shader_definition;
            braced!(in_shader_definition in in_braces);

            while !in_shader_definition.is_empty() {
                let field_ident = in_shader_definition.parse::<Ident>()?;
                in_shader_definition.parse::<Token![:]>()?;
                let field_name = field_ident.to_string();

                match field_name.as_str() {
                    "ty" | "path" | "src" | "bytes" | "lang" | "entry_point" | "compiler"
                    | "define" => {
                        shader_fields.parse_shader_field(&in_shader_definition, &field_name)?;
                    }
                    _ => bail!(
                        field_ident,
                        "expected `ty`, `path`, `src`, `bytes`, `lang`, `entry_point`, `compiler` \
                        or `define` as a field, found `{field_name}`",
                    ),
                }

                if !in_shader_definition.is_empty() {
                    in_shader_definition.parse::<Token![,]>()?;
                }
            }

            if !in_braces.is_empty() {
                in_braces.parse::<Token![,]>()?;
            }
        }

        if shaders.is_empty() {
            bail!(
                field_ident,
                "please specify at least one shader entry (e.g, `shaders: {{ `ty: \"vertex\", \
                path: \"entry_point.glsl\" }}`)",
            );
        }

        Ok(())
    }

    fn parse_root_path_env(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.root_path_env.is_some() {
            bail!(lit, "field `root_path_env` is already defined");
        }

        self.root_path_env = Some(lit);

        Ok(())
    }

    fn parse_include(&mut self, input: ParseStream<'_>) -> Result {
        let in_brackets;
        bracketed!(in_brackets in input);

        while !in_brackets.is_empty() {
            let path = in_brackets.parse::<LitStr>()?;

            self.include_directories
                .push([&self.manifest_dir, &path.value()].into_iter().collect());

            if !in_brackets.is_empty() {
                in_brackets.parse::<Token![,]>()?;
            }
        }

        Ok(())
    }

    fn parse_define(&mut self, input: ParseStream<'_>) -> Result {
        let array_input;
        bracketed!(array_input in input);

        while !array_input.is_empty() {
            let tuple_input;
            parenthesized!(tuple_input in array_input);

            let name = tuple_input.parse::<LitStr>()?;
            tuple_input.parse::<Token![,]>()?;
            let value = tuple_input.parse::<LitStr>()?;
            self.global_macro_defines
                .push((name.value(), value.value()));

            if !array_input.is_empty() {
                array_input.parse::<Token![,]>()?;
            }
        }

        Ok(())
    }

    fn parse_vulkan_version(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.vulkan_version.is_some() {
            bail!(lit, "field `vulkan_version` is already defined");
        }

        self.vulkan_version = Some(match lit.value().as_str() {
            "1.0" => VulkanVersion::V1_0,
            "1.1" => VulkanVersion::V1_1,
            "1.2" => VulkanVersion::V1_2,
            "1.3" => VulkanVersion::V1_3,
            ver => bail!(lit, "expected `1.0`, `1.1`, `1.2` or `1.3`, found `{ver}`"),
        });

        Ok(())
    }

    fn parse_spirv_version(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.spirv_version.is_some() {
            bail!(lit, "field `spirv_version` is already defined");
        }

        self.spirv_version = Some(match lit.value().as_str() {
            "1.0" => SpirvVersion::V1_0,
            "1.1" => SpirvVersion::V1_1,
            "1.2" => SpirvVersion::V1_2,
            "1.3" => SpirvVersion::V1_3,
            "1.4" => SpirvVersion::V1_4,
            "1.5" => SpirvVersion::V1_5,
            "1.6" => SpirvVersion::V1_6,
            ver => bail!(
                lit,
                "expected `1.0`, `1.1`, `1.2`, `1.3`, `1.4`, `1.5` or `1.6`, found `{ver}`",
            ),
        });

        Ok(())
    }

    fn parse_generate_structs(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitBool>()?;

        if self.generate_structs.is_some() {
            bail!(lit, "field `generate_structs` is already defined");
        }

        self.generate_structs = Some(lit.value);

        Ok(())
    }

    fn parse_custom_derives(&mut self, input: ParseStream<'_>) -> Result {
        let in_brackets;
        bracketed!(in_brackets in input);

        while !in_brackets.is_empty() {
            if self.custom_derives.is_none() {
                self.custom_derives = Some(Vec::new());
            }

            self.custom_derives
                .as_mut()
                .unwrap()
                .push(in_brackets.parse::<SynPath>()?);

            if !in_brackets.is_empty() {
                in_brackets.parse::<Token![,]>()?;
            }
        }

        Ok(())
    }

    fn parse_linalg_type(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.linalg_type.is_some() {
            bail!(lit, "field `linalg_type` is already defined");
        }

        self.linalg_type = Some(match lit.value().as_str() {
            "std" => LinAlgType::Std,
            "cgmath" => LinAlgType::CgMath,
            "nalgebra" => LinAlgType::Nalgebra,
            ty => bail!(lit, "expected `std`, `cgmath` or `nalgebra`, found `{ty}`"),
        });

        Ok(())
    }

    fn parse_dump(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitBool>()?;

        if self.dump.is_some() {
            bail!(lit, "field `dump` is already defined");
        }

        self.dump = Some(lit);

        Ok(())
    }

    fn check_valid(&self) -> Result {
        match &self.shaders {
            Some(Shaders::Single(shader_fields)) => {
                shader_fields.check_valid(
                    Span::call_site(),
                    self.global_source_language,
                    self.global_compiler,
                )?;
            }
            Some(Shaders::Multiple(shaders)) if !shaders.is_empty() => {
                for (shader_ident, shader_fields) in shaders {
                    shader_fields.check_valid(
                        shader_ident.span(),
                        self.global_source_language,
                        self.global_compiler,
                    )?;
                }
            }
            Some(Shaders::Multiple(_)) | None => {
                bail!(
                    "please specify at least one shader (e.g., `ty: \"vertex\", path: \
                    \"entry_point.glsl\"`)",
                );
            }
        }

        Ok(())
    }
}

impl ShaderFields {
    fn parse_shader_field(&mut self, input: ParseStream<'_>, field_name: &str) -> Result {
        match field_name {
            "ty" => self.parse_ty(input)?,
            "path" => self.parse_path(input)?,
            "src" => self.parse_src(input)?,
            "bytes" => self.parse_bytes(input)?,
            "lang" => self.parse_lang(input)?,
            "entry_point" => self.parse_entry_point(input)?,
            "compiler" => self.parse_compiler(input)?,
            "define" => self.parse_define(input)?,
            _ => unreachable!(),
        }

        Ok(())
    }

    fn parse_ty(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.shader_kind.is_some() {
            bail!(lit, "field `ty` is already defined");
        }

        if matches!(self.source_kind, Some(SourceKind::Precompiled(_))) {
            bail!(
                lit,
                "fields `ty` and `bytes` cannot be defined in the same shader entry",
            );
        }

        self.shader_kind = Some(match lit.value().as_str() {
            "vertex" => ShaderKind::Vertex,
            "tess_ctrl" => ShaderKind::TessControl,
            "tess_eval" => ShaderKind::TessEvaluation,
            "geometry" => ShaderKind::Geometry,
            "task" => ShaderKind::Task,
            "mesh" => ShaderKind::Mesh,
            "fragment" => ShaderKind::Fragment,
            "compute" => ShaderKind::Compute,
            "raygen" => ShaderKind::RayGeneration,
            "anyhit" => ShaderKind::AnyHit,
            "closesthit" => ShaderKind::ClosestHit,
            "miss" => ShaderKind::Miss,
            "intersection" => ShaderKind::Intersection,
            "callable" => ShaderKind::Callable,
            kind => bail!(
                lit,
                "expected `vertex`, `tess_ctrl`, `tess_eval`, `geometry`, `task`, `mesh`, \
                `fragment`, `compute`, `raygen`, `anyhit`, `closesthit`, `miss`, `intersection` \
                or `callable`, found `{kind}`",
            ),
        });

        Ok(())
    }

    fn parse_path(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if let Some(source_kind) = &self.source_kind {
            let msg = match source_kind {
                SourceKind::Compiled(CompiledSourceKind::Path(_)) => {
                    "field `path` is already defined"
                }
                SourceKind::Compiled(CompiledSourceKind::Inline(_)) => {
                    "fields `path` and `src` cannot be defined in the same shader entry"
                }
                SourceKind::Precompiled(_) => {
                    "fields `path` and `bytes` cannot be defined in the same shader entry"
                }
            };

            bail!(lit, "{msg}");
        }

        self.source_kind = Some(SourceKind::Compiled(CompiledSourceKind::Path(lit)));

        Ok(())
    }

    fn parse_src(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if let Some(source_kind) = &self.source_kind {
            let msg = match source_kind {
                SourceKind::Compiled(CompiledSourceKind::Path(_)) => {
                    "fields `src` and `path` cannot be defined in the same shader entry"
                }
                SourceKind::Compiled(CompiledSourceKind::Inline(_)) => {
                    "field `src` is already defined"
                }
                SourceKind::Precompiled(_) => {
                    "fields `src` and `bytes` cannot be defined in the same shader entry"
                }
            };

            bail!(lit, "{msg}");
        }

        self.source_kind = Some(SourceKind::Compiled(CompiledSourceKind::Inline(lit)));

        Ok(())
    }

    fn parse_bytes(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if let Some(source_kind) = &self.source_kind {
            let msg = match source_kind {
                SourceKind::Compiled(CompiledSourceKind::Path(_)) => {
                    "fields `bytes` and `path` cannot be defined in the same shader entry"
                }
                SourceKind::Compiled(CompiledSourceKind::Inline(_)) => {
                    "fields `bytes` and `src` cannot be defined in the same shader entry"
                }
                SourceKind::Precompiled(_) => "field `bytes` is already defined",
            };

            bail!(lit, "{msg}");
        }

        if self.shader_kind.is_some() {
            bail!(
                lit,
                "fields `bytes` and `ty` cannot be defined in the same shader entry",
            );
        }

        if self.source_language.is_some() {
            bail!(
                lit,
                "fields `bytes` and `lang` cannot be defined in the same shader entry",
            );
        }

        if self.entry_point.is_some() {
            bail!(
                lit,
                "fields `bytes` and `entry_point` cannot be defined in the same shader entry",
            );
        }

        if self.compiler.is_some() {
            bail!(
                lit,
                "fields `bytes` and `compiler` cannot be defined in the same shader entry",
            );
        }

        self.source_kind = Some(SourceKind::Precompiled(lit));

        Ok(())
    }

    fn parse_lang(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.source_language.is_some() {
            bail!(lit, "field `lang` is already defined");
        }

        if matches!(self.source_kind, Some(SourceKind::Precompiled(_))) {
            bail!(
                lit,
                "fields `lang` and `bytes` cannot be defined in the same shader entry",
            );
        }

        self.source_language = Some(match lit.value().as_str() {
            "glsl" => SourceLanguage::Glsl,
            "hlsl" => SourceLanguage::Hlsl,
            "slang" => SourceLanguage::Slang,
            lang => bail!(lit, "expected `glsl`, `hlsl` or `slang`, found `{lang}`"),
        });

        Ok(())
    }

    fn parse_entry_point(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.entry_point.is_some() {
            bail!(lit, "field `entry_point` is already defined");
        }

        if matches!(self.source_kind, Some(SourceKind::Precompiled(_))) {
            bail!(
                lit,
                "fields `entry_point` and `bytes` cannot be defined in the same shader entry",
            );
        }

        self.entry_point = Some(lit.value());

        Ok(())
    }

    fn parse_compiler(&mut self, input: ParseStream<'_>) -> Result {
        let lit = input.parse::<LitStr>()?;

        if self.compiler.is_some() {
            bail!(lit, "field `compiler` is already defined");
        }

        if matches!(self.source_kind, Some(SourceKind::Precompiled(_))) {
            bail!(
                lit,
                "fields `compiler` and `bytes` cannot be defined in the same shader entry",
            );
        }

        self.compiler = Some(match lit.value().as_str() {
            "shaderc" => Compiler::Shaderc,
            "slangc" => Compiler::Slangc,
            compiler => bail!(lit, "expected `shaderc` or `slangc`, found `{compiler}`"),
        });

        Ok(())
    }

    fn parse_define(&mut self, input: ParseStream<'_>) -> Result {
        let array_input;
        bracketed!(array_input in input);

        while !array_input.is_empty() {
            let tuple_input;
            parenthesized!(tuple_input in array_input);

            let name = tuple_input.parse::<LitStr>()?;
            tuple_input.parse::<Token![,]>()?;
            let value = tuple_input.parse::<LitStr>()?;
            self.macro_defines.push((name.value(), value.value()));

            if !array_input.is_empty() {
                array_input.parse::<Token![,]>()?;
            }
        }

        Ok(())
    }

    fn check_valid(
        &self,
        span: Span,
        global_source_language: Option<SourceLanguage>,
        global_compiler: Option<Compiler>,
    ) -> Result {
        if self.source_kind.is_none() {
            return Err(Error::new(
                span,
                "please specify the shader source (e.g., `path: \"entry_point.glsl\"`)",
            ));
        }

        let source_language = self
            .source_language
            .or(global_source_language)
            .unwrap_or_default();
        let compiler = self
            .compiler
            .or(global_compiler)
            .unwrap_or(source_language.default_compiler());

        if self.entry_point.is_some() && compiler == Compiler::Shaderc {
            return Err(Error::new(
                span,
                "the `entry_point` option cannot be used when compiling with shaderc; please \
                remove it",
            ));
        }

        if !compiler.is_valid_for(source_language) {
            return Err(Error::new(
                span,
                format_args!(
                    "source language `{}` cannot be compiled with compiler `{}`",
                    source_language.as_str(),
                    compiler.as_str(),
                ),
            ));
        }

        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
enum ShaderKind {
    Vertex,
    TessControl,
    TessEvaluation,
    Geometry,
    Task,
    Mesh,
    Fragment,
    Compute,
    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
    Intersection,
    Callable,
}

impl ShaderKind {
    fn as_shaderc_stage(self) -> &'static str {
        match self {
            ShaderKind::Vertex => "vert",
            ShaderKind::TessControl => "tesc",
            ShaderKind::TessEvaluation => "tese",
            ShaderKind::Geometry => "geom",
            ShaderKind::Task => "task",
            ShaderKind::Mesh => "mesh",
            ShaderKind::Fragment => "frag",
            ShaderKind::Compute => "comp",
            ShaderKind::RayGeneration => "rgen",
            ShaderKind::AnyHit => "rahit",
            ShaderKind::ClosestHit => "rchit",
            ShaderKind::Miss => "rmiss",
            ShaderKind::Intersection => "rint",
            ShaderKind::Callable => "rcall",
        }
    }

    fn as_slangc_stage(self) -> &'static str {
        match self {
            ShaderKind::Vertex => "vertex",
            ShaderKind::TessControl => "hull",
            ShaderKind::TessEvaluation => "domain",
            ShaderKind::Geometry => "geometry",
            ShaderKind::Task => "amplification",
            ShaderKind::Mesh => "mesh",
            ShaderKind::Fragment => "fragment",
            ShaderKind::Compute => "compute",
            ShaderKind::RayGeneration => "raygeneration",
            ShaderKind::AnyHit => "anyhit",
            ShaderKind::ClosestHit => "closesthit",
            ShaderKind::Miss => "miss",
            ShaderKind::Intersection => "intersection",
            ShaderKind::Callable => "callable",
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
enum SourceLanguage {
    #[default]
    Glsl,
    Hlsl,
    Slang,
}

impl SourceLanguage {
    fn as_str(self) -> &'static str {
        match self {
            SourceLanguage::Glsl => "glsl",
            SourceLanguage::Hlsl => "hlsl",
            SourceLanguage::Slang => "slang",
        }
    }

    fn default_compiler(self) -> Compiler {
        match self {
            Self::Glsl => Compiler::Shaderc,
            Self::Hlsl | Self::Slang => Compiler::Slangc,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compiler {
    Shaderc,
    Slangc,
}

impl Compiler {
    fn as_str(self) -> &'static str {
        match self {
            Compiler::Shaderc => "shaderc",
            Compiler::Slangc => "slangc",
        }
    }

    fn as_command(self) -> &'static str {
        match self {
            Compiler::Shaderc => "glslc",
            Compiler::Slangc => "slangc",
        }
    }

    fn is_valid_for(self, source_language: SourceLanguage) -> bool {
        match self {
            Compiler::Shaderc => matches!(source_language, SourceLanguage::Glsl),
            Compiler::Slangc => {
                matches!(
                    source_language,
                    SourceLanguage::Glsl | SourceLanguage::Hlsl | SourceLanguage::Slang,
                )
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
enum VulkanVersion {
    #[default]
    V1_0,
    V1_1,
    V1_2,
    V1_3,
}

impl VulkanVersion {
    fn as_shaderc_target_env(self) -> &'static str {
        match self {
            VulkanVersion::V1_0 => "vulkan1.0",
            VulkanVersion::V1_1 => "vulkan1.1",
            VulkanVersion::V1_2 => "vulkan1.2",
            VulkanVersion::V1_3 => "vulkan1.3",
        }
    }

    fn to_spirv_version(self) -> SpirvVersion {
        match self {
            VulkanVersion::V1_0 => SpirvVersion::V1_0,
            VulkanVersion::V1_1 => SpirvVersion::V1_3,
            VulkanVersion::V1_2 => SpirvVersion::V1_5,
            VulkanVersion::V1_3 => SpirvVersion::V1_6,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum SpirvVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
    V1_4,
    V1_5,
    V1_6,
}

impl SpirvVersion {
    fn as_shaderc_target_spv(self) -> &'static str {
        match self {
            SpirvVersion::V1_0 => "spv1.0",
            SpirvVersion::V1_1 => "spv1.1",
            SpirvVersion::V1_2 => "spv1.2",
            SpirvVersion::V1_3 => "spv1.3",
            SpirvVersion::V1_4 => "spv1.4",
            SpirvVersion::V1_5 => "spv1.5",
            SpirvVersion::V1_6 => "spv1.6",
        }
    }

    fn as_slangc_profile(self) -> &'static str {
        match self {
            SpirvVersion::V1_0 => "spirv_1_0",
            SpirvVersion::V1_1 => "spirv_1_1",
            SpirvVersion::V1_2 => "spirv_1_2",
            SpirvVersion::V1_3 => "spirv_1_3",
            SpirvVersion::V1_4 => "spirv_1_4",
            SpirvVersion::V1_5 => "spirv_1_5",
            SpirvVersion::V1_6 => "spirv_1_6",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LinAlgType {
    #[default]
    Std,
    CgMath,
    Nalgebra,
}

type Result<T = (), E = syn::Error> = ::std::result::Result<T, E>;

macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format_args!($msg),
        ))
    };
    ($span:expr, $msg:literal $(,)?) => {
        return Err(syn::Error::new_spanned(&$span, format_args!($msg)))
    };
}
use bail;
