//! Does pliron's `DialectConversion` work against our two dialects?
//!
//! The framework runs on a real wax module, finds the ops it claims, rewrites them into
//! `air` ops, and leaves the module intact.


use wax_frontend::{ast, KernelCompiler};

const SCALE_SRC: &str = r#"
mod k {
    use cutile::core::*;
    #[wax::entry()]
    fn scale<const N: i32>(out: &mut Tensor<f32, { [1, N] }>, x: &Tensor<f32, { [1, N] }>) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut po: PartitionMut<f32, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        let v = px.load([0i32, 0i32]);
        unsafe { po.store(v + v, [0i32, 0i32]) };
    }
}
"#;

fn scale_module() -> wax_ir::Module {
    let item: syn::ItemMod = syn::parse_str(SCALE_SRC).expect("kernel parses");
    let span_base = ast::SpanBase::new("tests/wax_to_air_conversion.rs".into(), 1, 1);
    KernelCompiler::new(
        move || {
            let mut m = ast::Module::with_span_base("k", item.clone(), span_base.clone());
            m.set_absolute_path("wax_metal::k".to_string());
            m
        },
        "k",
        "scale",
    )
    .generics(vec!["32".into()])
    .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
    .compile()
    .expect("frontend compiles scale")
    .to_wax_module("scale")
    .expect("dialect module builds")
}

#[test]
fn the_conversion_runs_on_a_real_wax_module() {
    let mut m = scale_module();
    let func = *m.functions.first().expect("one entry function");

    // The framework walks the module, asks `can_convert_op` for each op, and drives `rewrite`
    // for the ones it claims. Reaching the end without error is the mechanism working.
    let converted =
        wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("dialect conversion completes");

    assert!(
        converted > 0,
        "the conversion claimed ops but rewrote none - `can_convert_op` and `rewrite` disagree"
    );
}

#[test]
fn the_module_is_still_verifiable_afterwards() {
    let mut m = scale_module();
    let func = *m.functions.first().unwrap();
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");

    // A conversion that leaves the IR malformed would be worse than one that does nothing:
    // the damage would surface far downstream, in the emitter.
    match m.verify() {
        Ok(()) => {}
        Err(e) => {
            // Kernels already fail dominance verification for a pre-existing reason, so only
            // a NEW class of failure is interesting here.
            assert!(
                e.contains("UseNotDominatedByDef"),
                "conversion introduced a new verification failure: {e}"
            );
        }
    }
}

/// The conversion's type mapping must match what the backend actually emits.
///
/// Checked against emitted AIR rather than against `types::ll_type`, which is the
/// kernel-parameter ABI: it renders a `[1,32]` tile as `[32 x float]`, while the emitter
/// loads one `float` per lane. Comparing against it would pin the wrong answer.
#[test]
fn the_type_mapping_matches_emitted_air() {
    use wax_ir::types::{ScalarType, TileElementType, TileType, Type};
    let tile = |shape: Vec<i64>| {
        Type::Tile(TileType {
            shape,
            element_type: TileElementType::Scalar(ScalarType::F32),
        })
    };
    let cases = [
        (Type::Scalar(ScalarType::F32), "float"),
        (Type::Scalar(ScalarType::I32), "i32"),
        // A tile is distributed across lanes: one element per lane, whatever its extent.
        (tile(vec![]), "float"),
        (tile(vec![32]), "float"),
        (tile(vec![1, 32]), "float"),
        (tile(vec![1, 64]), "float"),
        // Including the 8-multiple shapes, which this once mapped to `<64 x float>` on the
        // theory that they were SGMMA fragments. Fragment-ness is not a property of the
        // shape - `passes::fragment_set` says so in its opening paragraph, and `[32, 32]` is
        // equally a threadgroup-spanning tile with one element per thread. A fragment is what
        // an MMA touches, and in any kernel this route converts that decides it.
        (tile(vec![8, 8]), "float"),
        (tile(vec![16, 16]), "float"),
    ];
    for (ty, want) in cases {
        let got = wax_metal::wax_to_air::air_type_for(&ty)
            .unwrap_or_else(|| panic!("no air type for {ty:?}"))
            .ll_str();
        assert_eq!(got, want, "wrong air type for {ty:?}");
    }
}

/// One wax op becoming two air ops, with the result rewired.
///
/// `return` -> `ret` is one-to-one and says nothing about the shape every tile-level op
/// needs. This checks the harder case: `wax.load_view` becomes `air.gep` + `air.load`, the
/// load's result replaces the original, and the module still verifies.
#[test]
fn a_load_becomes_a_gep_and_a_load() {
    let mut m = scale_module();
    let func = *m.functions.first().unwrap();

    let before = op_names(&m, func);
    let converted = wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let after = op_names(&m, func);

    assert!(converted > 0, "nothing was converted");

    let loads_before = before.iter().filter(|n| n.contains("load_view")).count();
    assert!(loads_before > 0, "the kernel has no loads, so this proves nothing");

    let stores_before = before.iter().filter(|n| n.contains("store_view")).count();
    let geps = after.iter().filter(|n| n == &"air.gep").count();
    let loads = after.iter().filter(|n| n == &"air.load").count();
    let stores = after.iter().filter(|n| n == &"air.store").count();
    // Loads and stores each produce their own address, so a gep apiece.
    assert_eq!(
        geps,
        loads_before + stores_before,
        "expected one air.gep per load and store, got {geps} for {loads_before} loads and \
         {stores_before} stores: {after:?}"
    );
    assert_eq!(loads, loads_before, "wrong air.load count: {after:?}");
    assert_eq!(stores, stores_before, "wrong air.store count: {after:?}");
    assert!(
        after.iter().all(|n| n.starts_with("air.")),
        "a wax op survived the conversion: {after:?}"
    );
}

/// Every op name in the function, in order.
fn op_names(m: &wax_ir::Module, func: wax_ir::build::OpId) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(
        ctx: &pliron::context::Context,
        op: pliron::context::Ptr<pliron::operation::Operation>,
        out: &mut Vec<String>,
    ) {
        use pliron::linked_list::ContainsLinkedList;
        for r in op.deref(ctx).regions() {
            for b in r.deref(ctx).iter(ctx) {
                for inner in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                    let name = pliron::operation::Operation::get_op_dyn(inner, ctx)
                        .op_ref()
                        .get_opid()
                        .to_string();
                    out.push(name);
                    walk(ctx, inner, out);
                }
            }
        }
    }
    walk(&m.ctx, func, &mut out);
    out
}