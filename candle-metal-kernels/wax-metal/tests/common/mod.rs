//! The shared corpus machinery for the parity drivers: one `Case` table per corpus and one
//! module builder, so the two drivers and the sweep all lower the SAME population. A case
//! that exists in three places drifts; a table that exists in one is the corpus.

use wax_frontend::{ast, KernelCompiler};

pub mod bench;
pub mod corpus;

/// One row: the entry, its generics, and the per-buffer strides, as the corpus's own tests
/// instantiate it. `n as i32`-style symbolic strides are filled in with the concrete value
/// the POC test ran.
pub struct Case {
    pub entry: &'static str,
    pub generics: &'static [&'static str],
    pub strides: &'static [(&'static str, &'static [i32])],
    /// On the allowlist: the conversion is expected to refuse this kernel.
    pub miss: bool,
}

impl Case {
    pub const fn same(
        entry: &'static str,
        generics: &'static [&'static str],
        strides: &'static [(&'static str, &'static [i32])],
    ) -> Self {
        Case { entry, generics, strides, miss: false }
    }
    /// The allowlist constructor. Unused while the allowlist is empty - the definition of
    /// done - but the mechanism stays: a kernel the conversion legitimately cannot take
    /// yet is a `miss` with a named opcode, never a silent DIFF or a panic.
    #[allow(dead_code)]
    pub const fn miss(
        entry: &'static str,
        generics: &'static [&'static str],
        strides: &'static [(&'static str, &'static [i32])],
    ) -> Self {
        Case { entry, generics, strides, miss: true }
    }
}

/// Compile `c` out of a corpus's source file into a dialect module.
fn module_for(
    c: &Case,
    src: &str,
    span_label: &str,
    module_name: &'static str,
    absolute_path: &str,
) -> wax_ir::Module {
    let item: syn::ItemMod = syn::parse_str(src).expect("kernels parse");
    let span_base = ast::SpanBase::new(span_label.into(), 1, 1);
    let make = move || {
        let mut m = ast::Module::with_span_base(module_name, item.clone(), span_base.clone());
        m.set_absolute_path(absolute_path.to_string());
        m
    };
    KernelCompiler::new(make, module_name, c.entry)
        .generics(c.generics.iter().map(|s| s.to_string()).collect())
        .strides(c.strides)
        .compile()
        .unwrap_or_else(|e| panic!("{}: compile: {e}", c.entry))
        .to_wax_module(c.entry)
        .unwrap_or_else(|e| panic!("{}: dialect module: {e}", c.entry))
}

/// The bench case's dialect module.
pub fn bench_module(c: &Case) -> wax_ir::Module {
    module_for(
        c,
        include_str!("../kernels/bench.rs"),
        "tests/common/bench.rs",
        "bench",
        "wax_metal::bench",
    )
}

/// The corpus case's dialect module.
pub fn corpus_module(c: &Case) -> wax_ir::Module {
    module_for(
        c,
        include_str!("../kernels/corpus.rs"),
        "tests/common/corpus.rs",
        "kernels",
        "wax_metal::kernels",
    )
}
