#![allow(dead_code)]

use wax_frontend::{KernelCompiler, ast};
use wax_ir::builder::{OpBuilder, append_op};
use wax_ir::{
    Attribute, Location, Module, Opcode, PointerType, ScalarType, TileElementType, TileType, Type,
};

pub fn specialise(
    module: &wax_ir::Module,
    m: i64,
    n: i64,
    sg: i32,
    splits: i32,
) -> Option<(wax_ir::Module, wax_metal::gemm::Rewritten)> {
    use wax_metal::gemm::{GemmSpec, realise};

    let spec = GemmSpec {
        trans_b: trans_b_of(module),
        m: m as usize,
        n: n as usize,
        k: 0,
        elem_bytes: 4,
        ldc: n as usize,
        batch: 1,
        a_batch: 0,
        b_batch: 0,
    };
    let r = realise(module, sg, splits, &spec).ok()?;
    r.rewritten.launch.as_ref()?;
    Some((r.module, r.rewritten))
}

fn trans_b_of(module: &wax_ir::Module) -> bool {
    use wax_ir::passes::{AnalysisManager, gemm_analysis::GemmAnalysis};

    let Some(&func) = module.entries().first() else {
        return false;
    };
    let mut analyses = AnalysisManager::default();
    match analyses.get_analysis::<GemmAnalysis>(func, &module.ctx) {
        Ok(a) => a.pattern.as_ref().map(|p| p.trans_b).unwrap_or(false),
        Err(_) => false,
    }
}

pub fn profile() -> wax_metal::TargetProfile {
    wax_metal::TargetProfile::new(wax_metal::Platform::MacOS, 26, 0)
}

pub fn module_from_src(
    src: &str,
    span_label: &str,
    module_name: &'static str,
    entry: &str,
    generics: &[&str],
    strides: &'static [(&'static str, &'static [i32])],
) -> wax_ir::Module {
    let item: syn::ItemMod = syn::parse_str(src).expect("kernel parses");
    let span_base = ast::SpanBase::new(span_label.into(), 1, 1);
    let absolute = format!("wax_metal::{module_name}");
    KernelCompiler::new(
        move || {
            let mut m = ast::Module::with_span_base(module_name, item.clone(), span_base.clone());
            m.set_absolute_path(absolute.clone());
            m
        },
        module_name,
        entry,
    )
    .generics(generics.iter().map(|g| g.to_string()).collect())
    .strides(strides)
    .compile()
    .unwrap_or_else(|e| panic!("{entry}: compile: {e}"))
    .to_wax_module(entry)
    .unwrap_or_else(|e| panic!("{entry}: dialect module: {e}"))
}

pub fn rmw_module(name: &str, elem: ScalarType, mode: i32) -> Module {
    let mut m = Module::new(name);
    let ptr_ty = Type::Pointer(PointerType {
        pointee: elem.clone(),
    });
    let scalar = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(elem.clone()),
    });
    let (entry, block, args) = m
        .begin_function(name, &[ptr_ty, scalar.clone()])
        .expect("the function builds");
    m.mark_entry(entry);
    m.set_op_attr(
        entry,
        &wax_ir::build::ATTR_SYM_NAME,
        Attribute::String(name.to_string()),
    );

    let (rmw_id, _res) = OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
        .operands([args[0], args[1]])
        .result(scalar.clone())
        .result(Type::Token)
        .attr("mode", Attribute::i32(mode as i64))
        .attr("memory_ordering_semantics", Attribute::i32(1))
        .attr("memory_scope", Attribute::i32(1))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut m)
        .expect("fixture op builds");
    append_op(&mut m, block, rmw_id);

    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .build(&mut m)
        .expect("fixture op builds");
    append_op(&mut m, block, ret);
    m
}
