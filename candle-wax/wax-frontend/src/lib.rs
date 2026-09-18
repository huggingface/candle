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

impl Compiled {
    /// The same kernel as a pliron module in the `wax` dialect.
    ///
    /// Built on request rather than always, because a backend lowers from [`Self::ir`] and does
    /// not need this. What does need it is our IR-level work - `cross_loop_reuse`,
    /// `verify_operation`, and the passes plan section 8 has queued - all of which operate on
    /// the dialect. Without it that layer is bypassed entirely on this path.
    pub fn to_wax_module(&self, name: &str) -> Result<wax_ir::Module, String> {
        let root = *self
            .module
            .functions
            .first()
            .ok_or("compiled module has no entry function")?;
        let src = CutileSource::new(&self.module);
        let params: Vec<_> = {
            use wax_ir::source::WaxSource;
            let regions = src.regions(root);
            let first = regions.first().copied().ok_or("entry has no region")?;
            let blocks = src.region_blocks(first);
            let entry = blocks.first().copied().ok_or("entry has no block")?;
            src.block_args(entry).into_iter().map(|(_, t)| t).collect()
        };
        wax_ir::from_source::module_from_source(name, &src, root, &params)
            .map_err(|e| e.to_string())
    }
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
        // Route through the dialect rather than straight to the flat form: the IR-level
        // passes operate on the pliron module, and materialising from their output is what
        // carries their conclusions into what a backend lowers. Going direct - which this did
        // until now - silently skipped every one of them.
        let src = CutileSource::new(&module);
        let params: Vec<_> = {
            use wax_ir::source::WaxSource;
            let r = src.regions(root);
            let first = r.first().copied().ok_or("entry has no region")?;
            let b = src.region_blocks(first);
            let entry = b.first().copied().ok_or("entry has no block")?;
            src.block_args(entry).into_iter().map(|(_, t)| t).collect()
        };
        let mut pm =
            wax_ir::from_source::module_from_source(&self.function_name, &src, root, &params)
                .map_err(|e| e.to_string())?;
        let func = *pm.functions.first().ok_or("built module has no function")?;

        let mut analyses = wax_ir::passes::AnalysisManager::default();
        // A pass failing is an analysis result, not a compile error: the kernel is still
        // correct, it just misses an optimisation. Same handling the backend gives it.
        let _ = wax_ir::passes::cross_loop_reuse::annotate_with(&mut analyses, &pm.ctx, func);

        let ir = wax_ir::ir::from_source(&pm, func);
        Ok(Compiled { module, ir })
    }
}
