use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::r#type::{Typed, TypedHandle};
use pliron::value::DefiningEntity;
use pliron_llvm::types::VectorType;
use wax_frontend::{KernelCompiler, ast};

const SRC: &str = r#"
mod kernels {
    use cutile::core::*;

    #[wax::entry()]
    pub fn lane_vec(
        out: &mut Tensor<f32, { [1, 32] }>,
        k: &Tensor<f32, { [-1, 32, 4] }>,
        row: &Tensor<i32, { [1] }>,
    ) {
        let mp = row.partition(const_shape![1]);
        let row: i32 = tile_to_scalar(mp.load([0i32]).reshape(const_shape![]));
        let kp = k.partition(const_shape![1, 32, 1]);
        let a: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 0i32]).reshape(const_shape![1, 32]);
        let b: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 1i32]).reshape(const_shape![1, 32]);
        let c: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 2i32]).reshape(const_shape![1, 32]);
        let e: Tile<f32, { [1, 32] }> = kp.load([row, 0i32, 3i32]).reshape(const_shape![1, 32]);
        out.store(a + b + c + e);
    }

    #[wax::entry()]
    pub fn accumulate<E: ElementType, const N: i32, const K: i32>(
        out: &mut Tensor<E, { [1, N] }>,
        x: &Tensor<E, { [K, N] }>,
    ) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut acc = px.load([0i32, 0i32]);
        for i in 1i32..K {
            acc = acc + px.load([i, 0i32]);
        }
        let mut po: PartitionMut<E, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        unsafe { po.store(acc, [0i32, 0i32]) };
    }

    #[wax::entry()]
    pub fn row_sum<const N: i32>(out: &mut Tensor<f32, { [1, 1] }>, x: &Tensor<f32, { [1, N] }>) {
        let v = x.partition(const_shape![1, N]).load([0i32, 0i32]);
        let s: Tile<f32, { [1] }> = reduce_sum(v, 1i32);
        let mut po: PartitionMut<f32, { [1, 1] }> =
            unsafe { out.partition_full_mut(const_shape![1, 1]) };
        unsafe { po.store(s.reshape(const_shape![1, 1]), [0i32, 0i32]) };
    }

    #[wax::entry()]
    pub fn gemm_8x8(
        z: &mut Tensor<f32, { [8, 8] }>,
        x: &Tensor<f32, { [8, 8] }>,
        y: &Tensor<f32, { [8, 8] }>,
    ) {
        let tile_x = x.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let tile_y = y.partition(const_shape![8, 8]).load([0i32, 0i32]);
        let acc = constant(0.0f32, const_shape![8, 8]);
        z.store(mma(tile_x, tile_y, acc));
    }

    #[wax::entry()]
    pub fn shape_q(out: &mut Tensor<i32, { [1, 1] }>, t: &Tensor<i32, { [16, 8] }>) {
        let s: [i32; 2] = get_tensor_shape(t);
        out.store(broadcast_scalar(s[0] + s[1], const_shape![1, 1]));
    }

    #[wax::entry()]
    pub fn block_id(out: &mut Tensor<i32, { [1, 1] }>) {
        let b = get_tile_block_id().0;
        out.store(broadcast_scalar(b, const_shape![1, 1]));
    }

    #[wax::entry()]
    pub fn block_id_assumed(out: &mut Tensor<i32, { [1, 1] }>) {
        let b = unsafe { assume_div_by::<i32, 4>(get_tile_block_id().0) };
        out.store(broadcast_scalar(b, const_shape![1, 1]));
    }

    #[wax::entry()]
    pub fn lane_index(out: &mut Tensor<i32, { [1, 32] }>) {
        let lane: Tile<i32, { [32] }> = iota(const_shape![32]);
        out.store(lane.reshape(const_shape![1, 32]));
    }

    #[wax::entry()]
    pub fn lane_broadcast(out: &mut Tensor<i32, { [4, 32] }>) {
        let lane: Tile<i32, { [32] }> = iota(const_shape![32]);
        out.store(
            lane.reshape(const_shape![1, 32])
                .broadcast(const_shape![4, 32]),
        );
    }

    #[wax::entry()]
    pub fn gather(
        out: &mut Tensor<f32, { [32] }>,
        addr: &Tensor<i64, { [1] }>,
        idx: &Tensor<i32, { [32] }>,
    ) {
        let ts = const_shape![32];
        let o: Tile<i32, { [32] }> = idx.partition(ts).load([0i32]);
        let a: Tile<i64, { [1] }> = addr.partition(const_shape![1]).load([0i32]);
        let b: PointerTile<*mut f32, { [1] }> = int_to_ptr(a);
        let w: PointerTile<*mut f32, { [32] }> = broadcast_ptr(b, ts);
        let p: PointerTile<*mut f32, { [32] }> = addptr_tile(w, o);
        let (v, _t): (Tile<f32, { [32] }>, Token) = load_ptr_tko(
            p,
            ordering::Weak,
            None::<scope::Device>,
            None,
            None,
            None,
            Latency::<0>,
        );
        out.store(v);
    }
}
"#;

fn kernel(entry: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> wax_ir::Module {
    let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
    let span = ast::SpanBase::new("tests/feature_cells.rs".into(), 1, 1);
    let make = move || {
        let mut m = ast::Module::with_span_base("kernels", item.clone(), span.clone());
        m.set_absolute_path("wax_metal::feature_cells".to_string());
        m
    };
    KernelCompiler::new(make, "kernels", entry)
        .generics(generics.iter().map(|s| s.to_string()).collect())
        .strides(strides)
        .compile()
        .unwrap_or_else(|e| panic!("{entry}: frontend: {e}"))
        .to_wax_module(entry)
        .unwrap_or_else(|e| panic!("{entry}: wax module: {e}"))
}

struct Loads {
    vector: Vec<(u32, bool)>,
    scalar: usize,
}

fn loads_of(mut m: wax_ir::Module) -> Loads {
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut out = Loads {
        vector: Vec::new(),
        scalar: 0,
    };
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                let name = Operation::get_op_dyn(op, ctx)
                    .op_ref()
                    .get_opid()
                    .to_string();
                if name != "air.load" {
                    continue;
                }
                let o = op.deref(ctx);
                if o.get_num_results() != 1 {
                    continue;
                }
                let t = o.get_result(0).get_type(ctx);
                match TypedHandle::<VectorType>::from_handle(t, ctx) {
                    Ok(v) => {
                        let v = v.deref(ctx);
                        out.vector.push((
                            v.num_elements(),
                            v.elem_type() == wax_metal::air_dialect::types::f32_ty(ctx),
                        ));
                    }
                    Err(_) => out.scalar += 1,
                }
            }
        }
    }
    out
}

fn air_op_names(mut m: wax_ir::Module) -> Vec<String> {
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut out = Vec::new();
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                out.push(
                    Operation::get_op_dyn(op, ctx)
                        .op_ref()
                        .get_opid()
                        .to_string(),
                );
            }
        }
    }
    out
}

fn intrinsic_names(mut m: wax_ir::Module) -> Vec<String> {
    use pliron::op::Op;
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut out = Vec::new();
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                let Some(c) = wax_metal::air_dialect::ops::CallIntrinsicOp::wrap_operation(op)
                    .downcast::<wax_metal::air_dialect::ops::CallIntrinsicOp>()
                else {
                    continue;
                };
                if let Some(n) = c.get_attr_llvm_intrinsic_name(ctx) {
                    out.push(n.clone().into());
                }
            }
        }
    }
    out
}

fn block_count(mut m: wax_ir::Module) -> usize {
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    func.deref(ctx)
        .regions()
        .map(|r| r.deref(ctx).iter(ctx).count())
        .sum()
}

fn stored_value_origin(mut m: wax_ir::Module) -> (String, Vec<usize>) {
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut blocks = Vec::new();
    for r in func.deref(ctx).regions() {
        blocks.extend(r.deref(ctx).iter(ctx));
    }
    let args: Vec<_> = blocks[0].deref(ctx).arguments().collect();
    let mut store = None;
    for b in &blocks {
        for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if opid_of(ctx, op) == "air.store" {
                assert!(store.is_none(), "expected exactly one store");
                store = Some(op);
            }
        }
    }
    let value = store
        .expect("a store")
        .deref(ctx)
        .operands()
        .next()
        .expect("value operand");
    let root = match value.defining_entity() {
        DefiningEntity::Op(o) => opid_of(ctx, o),
        DefiningEntity::Block(_) => "<block argument>".to_string(),
    };
    let mut seen = std::collections::HashSet::new();
    let mut work = vec![value];
    let mut deps = std::collections::BTreeSet::new();
    while let Some(v) = work.pop() {
        if !seen.insert(v) {
            continue;
        }
        match v.defining_entity() {
            DefiningEntity::Op(o) => work.extend(o.deref(ctx).operands()),
            DefiningEntity::Block(_) => {
                if let Some(i) = args.iter().position(|a| *a == v) {
                    deps.insert(i);
                }
            }
        }
    }
    (root, deps.into_iter().collect())
}

fn opid_of(ctx: &Context, op: Ptr<Operation>) -> String {
    Operation::get_op_dyn(op, ctx)
        .op_ref()
        .get_opid()
        .to_string()
}

fn one_tensor(entry: &str) -> wax_ir::Module {
    kernel(entry, &[], &[("out", &[32, 1])])
}

#[test]
fn loads_fuse_correctly() {
    let loads = loads_of(kernel(
        "lane_vec",
        &[],
        &[("out", &[32, 1]), ("k", &[128, 4, 1]), ("row", &[1])],
    ));
    assert_eq!(
        loads.vector.len(),
        1,
        "expected exactly one fused load, got {:?}",
        loads.vector
    );
    assert_eq!(loads.vector[0].0, 4, "fused width");
    assert!(loads.vector[0].1, "fused element type should be f32");
    assert_eq!(
        loads.scalar, 1,
        "only the `row` scalar load should remain unfused, got {}",
        loads.scalar
    );
}

#[test]
fn views_and_access_lower_correctly() {
    let names = air_op_names(kernel(
        "lane_vec",
        &[],
        &[("out", &[32, 1]), ("k", &[128, 4, 1]), ("row", &[1])],
    ));
    for view_ish in ["view", "partition", "reshape"] {
        assert!(
            !names.iter().any(|n| n.contains(view_ish)),
            "`{view_ish}` survived into the emitted air: {names:?}"
        );
    }
    assert!(names.iter().any(|n| n == "air.gep"), "{names:?}");
    assert!(names.iter().any(|n| n == "air.load"), "{names:?}");
    assert!(names.iter().any(|n| n == "air.store"), "{names:?}");
}

#[test]
fn counted_loop_becomes_multiple_blocks_with_branch() {
    let k = kernel(
        "accumulate",
        &["f32", "32", "8"],
        &[("out", &[32, 1]), ("x", &[32, 1])],
    );
    assert!(block_count(k) > 1, "a loop must emit more than one block");
    let names = air_op_names(kernel(
        "accumulate",
        &["f32", "32", "8"],
        &[("out", &[32, 1]), ("x", &[32, 1])],
    ));
    assert!(
        names.iter().any(|n| n.contains("br")),
        "no branch emitted: {names:?}"
    );
}

#[test]
fn simd_reduce_lowers_correctly() {
    let got = intrinsic_names(kernel(
        "row_sum",
        &["32"],
        &[("out", &[1, 1]), ("x", &[32, 1])],
    ));
    assert!(
        got.iter().any(|n| n.contains("simd")),
        "expected a simd reduce intrinsic, got {got:?}"
    );
}

#[test]
fn mma_calls_the_simdgroup_matrix_intrinsics() {
    let got = intrinsic_names(kernel(
        "gemm_8x8",
        &[],
        &[("z", &[8, 1]), ("x", &[8, 1]), ("y", &[8, 1])],
    ));
    assert!(
        got.iter().any(|n| n.contains("simdgroup_matrix")),
        "expected an SGMMA intrinsic, got {got:?}"
    );
    assert!(
        got.iter().any(|n| n.contains("multiply_accumulate")),
        "expected the mma itself, got {got:?}"
    );
}

#[test]
fn static_tensor_shape_query_folds_into_the_code() {
    let (root, deps) =
        stored_value_origin(kernel("shape_q", &[], &[("out", &[1, 1]), ("t", &[8, 1])]));
    assert_eq!(root, "llvm.add", "16 + 8 of the two static extents");
    assert!(
        deps.is_empty(),
        "a static shape query must not depend on a kernel argument, got {deps:?}"
    );
    let names = air_op_names(kernel("shape_q", &[], &[("out", &[1, 1]), ("t", &[8, 1])]));
    assert!(
        !names.iter().any(|n| n == "air.load"),
        "the extents were read from memory: {names:?}"
    );
}

#[test]
fn assumption_is_lowered_away() {
    assert_eq!(
        air_op_names(one_tensor("block_id")),
        air_op_names(one_tensor("block_id_assumed")),
        "`assume_div_by` changed the emitted code"
    );
}

#[test]
fn iota_and_the_block_id_read_different_launch_positions() {
    let (_, lane) = stored_value_origin(one_tensor("lane_index"));
    let (_, block) = stored_value_origin(one_tensor("block_id"));
    assert_eq!(lane.len(), 1, "lane index deps {lane:?}");
    assert_eq!(block.len(), 1, "block id deps {block:?}");
    assert_ne!(
        lane, block,
        "the lane index and the block id read the same parameter"
    );
}

#[test]
fn tile_broadcast_adds_no_work_to_the_stored_value() {
    assert_eq!(
        stored_value_origin(one_tensor("lane_index")),
        stored_value_origin(one_tensor("lane_broadcast")),
        "the broadcast changed what is stored"
    );
}

#[test]
fn gather_through_raw_address_lowers_correctly() {
    use pliron::op::Op;
    let mut m = kernel(
        "gather",
        &[],
        &[("out", &[1]), ("addr", &[1]), ("idx", &[1])],
    );
    let func = *m.entries().first().expect("one entry");
    wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
    let ctx = &m.ctx;
    let mut casts = Vec::new();
    for r in func.deref(ctx).regions() {
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
                if opid_of(ctx, op) != "air.cast" {
                    continue;
                }
                if let Some(c) = wax_metal::air_dialect::ops::CastOp::wrap_operation(op)
                    .downcast::<wax_metal::air_dialect::ops::CastOp>()
                {
                    casts.push(c.get_attr_air_op(ctx).map(|a| *a));
                }
            }
        }
    }
    assert_eq!(
        casts,
        vec![Some(wax_metal::air_dialect::attrs::CastOpAttr::IntToPtr)],
        "expected exactly one int-to-pointer cast"
    );
    let names = air_op_names(kernel(
        "gather",
        &[],
        &[("out", &[1]), ("addr", &[1]), ("idx", &[1])],
    ));
    let cast = names.iter().position(|n| n == "air.cast").expect("a cast");
    assert!(
        names[cast..].iter().any(|n| n == "air.gep"),
        "no offset after the cast: {names:?}"
    );
    assert!(
        names[cast..].iter().any(|n| n == "air.load"),
        "no load after the cast: {names:?}"
    );
}
