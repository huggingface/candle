//! Compiles the cuTile DSL into wax IR.
//!
//! DSL to Tile IR is done by NVIDIA's [`cutile_frontend`]. Name resolution, type inference,
//! and lowering the kernel to a `cutile_ir::Module`.
//! This crate is the wax adapter and API around it.

pub mod adapter;
pub mod builtin_core;

pub use adapter::CutileSource;
pub use cutile_frontend::ast;
pub use cutile_frontend::hints::CompileOptions;
pub use cutile_frontend::specialization::{DivHint, SpecializationBits};

use cutile_frontend::compiler::{CUDATileFunctionCompiler, CUDATileModules};

/// The compiled kernel.
///
/// Materialized wax `ir` to be lowered by a backend, and the original cutile_ir `module`,
/// which we keep as context in case lowering goes wrong.
pub struct Compiled {
    pub module: cutile_ir::ir::Module,
    pub ir: wax_ir::ir::WaxIr,
}

/// Compiler of one kernel entry point in a `#[wax::module]`.
///
/// Mirrors the `cutile-compiler` closely.
pub struct KernelCompiler<F: Fn() -> ast::Module> {
    module_ast_fn: F,
    module_name: String,
    function_name: String,
    generics: Vec<String>,
    stride_args: Vec<(String, Vec<i32>)>,
    spec_args: Vec<(String, SpecializationBits)>,
    scalar_hints: Vec<(String, DivHint)>,
    const_grid: Option<(u32, u32, u32)>,
    compile_options: CompileOptions,
}

impl<F: Fn() -> ast::Module> KernelCompiler<F> {
    pub fn new(module_ast_fn: F, module_name: &str, function_name: &str) -> Self {
        Self {
            module_ast_fn,
            module_name: module_name.to_string(),
            function_name: function_name.to_string(),
            generics: Vec::new(),
            stride_args: Vec::new(),
            spec_args: Vec::new(),
            scalar_hints: Vec::new(),
            const_grid: None,
            compile_options: CompileOptions::default(),
        }
    }

    pub fn generics(mut self, generics: Vec<String>) -> Self {
        self.generics = generics;
        self
    }

    pub fn strides(mut self, strides: &[(&str, &[i32])]) -> Self {
        self.stride_args = strides
            .iter()
            .map(|(n, s)| (n.to_string(), s.to_vec()))
            .collect();
        self
    }

    pub fn spec_args(mut self, specs: &[(&str, SpecializationBits)]) -> Self {
        self.spec_args = specs
            .iter()
            .map(|(n, s)| (n.to_string(), s.clone()))
            .collect();
        self
    }

    pub fn scalar_hints(mut self, hints: &[(&str, DivHint)]) -> Self {
        self.scalar_hints = hints
            .iter()
            .map(|(n, h)| (n.to_string(), h.clone()))
            .collect();
        self
    }

    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.const_grid = Some(grid);
        self
    }

    pub fn options(mut self, options: CompileOptions) -> Self {
        self.compile_options = options;
        self
    }

    pub fn compile(self) -> Result<Compiled, String> {
        let kernel_module = (self.module_ast_fn)();
        let core_module = builtin_core::core_module();
        let modules =
            CUDATileModules::new(vec![core_module, kernel_module]).map_err(|e| e.to_string())?;

        let stride_refs: Vec<(&str, &[i32])> = self
            .stride_args
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        let spec_refs: Vec<(&str, &SpecializationBits)> = self
            .spec_args
            .iter()
            .map(|(n, s)| (n.as_str(), s))
            .collect();
        let hint_refs: Vec<(&str, &DivHint)> = self
            .scalar_hints
            .iter()
            .map(|(n, h)| (n.as_str(), h))
            .collect();

        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            &self.module_name,
            &self.function_name,
            &self.generics,
            &stride_refs,
            &spec_refs,
            &hint_refs,
            self.const_grid,
            "wax-gpu-name".to_string(),
            &self.compile_options,
        )
        .map_err(|e| e.to_string())?;

        let module = compiler.compile().map_err(|e| e.to_string())?;
        let root = *module
            .functions
            .first()
            .ok_or("compiled module has no entry function")?;
        let ir = wax_ir::ir::from_source(&CutileSource::new(&module), root);
        Ok(Compiled { module, ir })
    }
}
