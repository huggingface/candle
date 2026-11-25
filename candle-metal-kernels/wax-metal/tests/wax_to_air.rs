mod common;

mod atomic {
    use super::common::rmw_module;
    use wax_ir::ScalarType;

    #[test]
    fn float_atomics_air_lacks_are_rejected() {
        for (name, mut m) in [
            (
                "rmw_f32_maxu",
                rmw_module("rmw_f32_maxu", ScalarType::F32, 7),
            ),
            ("rmw_f16_add", rmw_module("rmw_f16_add", ScalarType::F16, 4)),
            (
                "rmw_bf16_add",
                rmw_module("rmw_bf16_add", ScalarType::BF16, 4),
            ),
            (
                "rmw_i32_xchg",
                rmw_module("rmw_i32_xchg", ScalarType::I32, 9),
            ),
            (
                "rmw_i32_addf",
                rmw_module("rmw_i32_addf", ScalarType::I32, 4),
            ),
        ] {
            let profile = wax_metal::profile::TargetProfile::host();
            assert!(
                wax_metal::lower_to_metallib_for(&mut m, &profile).is_err(),
                "AIR does not support `{name}`, but lowering did not fail"
            );
        }
    }
}

mod broadcast {
    use wax_frontend::{KernelCompiler, ast};

    const SRC: &str = r#"
    mod k {
        use cutile::core::*;
        #[wax::entry()]
        fn bcast(out: &mut Tensor<f32, { [32, 32] }>, x: &Tensor<f32, { [1, 32] }>) {
            let v: Tile<f32, { [1, 32] }> = x.partition(const_shape![1, 32]).load([0i32, 0i32]);
            let w: Tile<f32, { [32, 32] }> = v.broadcast(const_shape![32, 32]);
            out.store(w);
        }
        #[wax::entry()]
        fn bcast_redist(out: &mut Tensor<f32, { [4, 8] }>, x: &Tensor<f32, { [4, 1] }>) {
            let v: Tile<f32, { [4, 1] }> = x.partition(const_shape![4, 1]).load([0i32, 0i32]);
            let w: Tile<f32, { [4, 8] }> = v.broadcast(const_shape![4, 8]);
            out.store(w);
        }
    }
    "#;

    fn module(entry: &str, strides: &[(&str, &[i32])]) -> wax_ir::Module {
        let item: syn::ItemMod = syn::parse_str(SRC).expect("kernels parse");
        let span_base = ast::SpanBase::new("tests/wax_to_air_broadcast.rs".into(), 1, 1);
        KernelCompiler::new(
            move || {
                let mut m = ast::Module::with_span_base("k", item.clone(), span_base.clone());
                m.set_absolute_path("wax_metal::k".to_string());
                m
            },
            "k",
            entry,
        )
        .strides(strides)
        .compile()
        .unwrap_or_else(|e| panic!("{entry}: frontend: {e}"))
        .to_wax_module(entry)
        .unwrap_or_else(|e| panic!("{entry}: wax module: {e}"))
    }

    #[test]
    fn broadcast_along_last_dim_is_a_forward() {
        let mut m = module("bcast", &[("out", &[32, 1]), ("x", &[32, 1])]);
        let lib = wax_metal::lower_to_metallib(&mut m)
            .expect("every lane already holds its element")
            .metallib;
        let needle = b"_tg_user_scratch";
        assert!(
            !lib.windows(needle.len()).any(|w| w == needle),
            "forwardable broadcast redistributed"
        );
    }

    #[test]
    fn broadcast_with_redistribution_uses_scratch() {
        let mut m = module("bcast_redist", &[("out", &[8, 1]), ("x", &[1, 1])]);
        let lib = wax_metal::lower_to_metallib(&mut m)
            .expect("redistribution is a lowering")
            .metallib;
        let needle = b"_tg_user_scratch";
        assert!(
            lib.windows(needle.len()).any(|w| w == needle),
            "no threadgroup scratch"
        );
    }
}

mod conversion {
    const SRC: &str = r#"
    mod kernels {
        use cutile::core::*;

        #[wax::entry()]
        pub fn scale<E: ElementType, const N: i32>(
            out: &mut Tensor<E, { [1, N] }>,
            x: &Tensor<E, { [1, N] }>,
        ) {
            let ts = const_shape![1, N];
            let px = x.partition(ts);
            let mut po: PartitionMut<E, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
            let v = px.load([0i32, 0i32]);
            unsafe { po.store(v + v, [0i32, 0i32]) };
        }
    }
    "#;

    fn scale_module() -> wax_ir::Module {
        super::common::module_from_src(
            SRC,
            "tests/wax_to_air_conversion.rs",
            "kernels",
            "scale",
            &["f32", "32"],
            &[("out", &[32, 1]), ("x", &[32, 1])],
        )
    }

    #[test]
    fn conversion_runs_on_a_real_wax_module() {
        let mut m = scale_module();
        let func = *m.entries().first().expect("one entry function");

        let converted =
            wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("dialect conversion completes");

        assert!(converted > 0, "conversion rewrote no ops");
    }

    #[test]
    fn converted_function_gains_implicit_parameters() {
        use pliron::r#type::Typed;
        use wax_metal::air_dialect::types as t;

        let mut m = scale_module();
        let func = *m.entries().first().expect("one entry function");

        let _ =
            wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("dialect conversion completes");

        let b = wax_metal::wax_to_air::entry_block(&m.ctx, func)
            .expect("entry block")
            .deref(&m.ctx);
        let tys: Vec<_> = (0..b.get_num_arguments())
            .map(|i| b.get_argument(i).get_type(&m.ctx))
            .collect();

        let vec3 = t::vec_ty(&m.ctx, 3, t::i32_ty(&m.ctx));
        assert!(tys.len() >= 2, "signature is too short: {tys:?}");
        assert_eq!(
            &tys[tys.len() - 2..],
            &[vec3, vec3],
            "expected `%_tp` and `%_tg` last: {tys:?}"
        );
        assert!(
            !tys[..tys.len() - 2].contains(&vec3),
            "position vector appears before trailing pair: {tys:?}"
        );
    }

    #[test]
    fn load_view_becomes_gep_and_load() {
        let mut m = scale_module();
        let func = *m.entries().first().unwrap();

        let before = op_names(&m, func);
        let converted =
            wax_metal::wax_to_air::convert(&mut m.ctx, func).expect("conversion completes");
        let after = op_names(&m, func);

        assert!(converted > 0, "nothing was converted");

        let loads_before = before.iter().filter(|n| n.contains("load_view")).count();
        assert!(loads_before > 0);

        let stores_before = before.iter().filter(|n| n.contains("store_view")).count();
        let geps = after.iter().filter(|n| n == &"air.gep").count();
        let loads = after.iter().filter(|n| n == &"air.load").count();
        let stores = after.iter().filter(|n| n == &"air.store").count();
        assert_eq!(geps, loads_before + stores_before);
        assert_eq!(loads, loads_before, "wrong air.load count: {after:?}");
        assert_eq!(stores, stores_before, "wrong air.store count: {after:?}");
        assert!(
            !after.iter().any(|n| n.starts_with("wax.")),
            "wax op survived the conversion: {after:?}"
        );
    }

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
}

mod nested_if {

    use wax_frontend::{KernelCompiler, ast};

    const SRC: &str = r#"
    mod kernels {
        use cutile::core::*;

        #[wax::entry()]
        pub fn nested_guard<const N: i32, const K: i32, const J: i32>(
            output: &mut Tensor<f32, { [N] }>,
        ) {
            let ts = const_shape![1];
            let mut out_part = unsafe { output.partition_full_mut(ts) };
            for i in 0i32..N {
                if i < K {
                    if i < J {
                        let one: Tile<f32, { [1] }> = constant(1.0f32, ts);
                        unsafe { out_part.store(one, [i]) };
                    }
                }
            }
        }

        #[wax::entry()]
        pub fn nested_select<const N: i32, const K: i32, const J: i32>(
            output: &mut Tensor<f32, { [N] }>,
        ) {
            let ts = const_shape![1];
            let mut out_part = unsafe { output.partition_full_mut(ts) };
            for i in 0i32..N {
                let v: Tile<f32, { [1] }> = if i < K {
                    if i < J {
                        constant(1.0f32, ts)
                    } else {
                        constant(2.0f32, ts)
                    }
                } else {
                    constant(3.0f32, ts)
                };
                unsafe { out_part.store(v, [i]) };
            }
        }
    }
    "#;

    fn nested_module(entry: &str) -> wax_ir::Module {
        let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
        let span_base = ast::SpanBase::new("tests/wax_to_air_nested_if.rs".into(), 1, 1);
        let make = move || {
            let mut m = ast::Module::with_span_base("kernels", item.clone(), span_base.clone());
            m.set_absolute_path("wax_metal::nested_if".to_string());
            m
        };
        KernelCompiler::new(make, "kernels", entry)
            .generics(vec!["16".into(), "8".into(), "4".into()])
            .strides(&[("output", &[1])])
            .compile()
            .unwrap_or_else(|e| panic!("frontend: {e}"))
            .to_wax_module(entry)
            .unwrap_or_else(|e| panic!("dialect module: {e}"))
    }

    #[test]
    fn nested_if_converts_fully() {
        let mut m = nested_module("nested_guard");
        let func = *m.entries().first().expect("one entry function");
        let (_, left) =
            wax_metal::wax_to_air::run_conversion(&mut m.ctx, func).expect("conversion runs");
        assert!(
            left.0.is_empty(),
            "nested `if` did not fully convert: {:?}",
            left.0.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn nested_if_lowers() {
        let mut m = nested_module("nested_guard");
        let bytes = wax_metal::lower_to_metallib(&mut m)
            .expect("nested `if` lowers")
            .metallib;
        assert!(!bytes.is_empty(), "empty metallib");
    }

    #[test]
    fn nested_value_producing_if_converts_fully() {
        let mut m = nested_module("nested_select");
        let func = *m.entries().first().expect("one entry function");
        let (_, left) =
            wax_metal::wax_to_air::run_conversion(&mut m.ctx, func).expect("conversion runs");
        assert!(
            left.0.is_empty(),
            "nested value-producing `if` did not fully convert: {:?}",
            left.0.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn nested_value_producing_if_lowers() {
        let mut m = nested_module("nested_select");
        let bytes = wax_metal::lower_to_metallib(&mut m)
            .expect("nested value-producing `if` lowers")
            .metallib;
        assert!(!bytes.is_empty(), "empty metallib");
    }
}

mod params {
    const PADDED_STORE_SRC: &str = r#"
    mod kernels {
        use cutile::core::*;
        #[wax::entry()]
        fn padded_store<const M: i32>(x: &Tensor<f32, { [-1] }>, out: &mut Tensor<f32, { [M] }>) {
            let ts = const_shape![32];
            let blk = get_tile_block_id().0;
            let t: Tile<f32, { [32] }> = load_tile(x, ts, [blk]);
            let tok: Token = get_tensor_token(out);
            let mut op: PartitionMut<f32, { [32] }> =
                unsafe { make_partition_view_mut(out, ts, padding::Zero, tok) };
            unsafe { op.store(t, [blk]) };
        }
    }
    "#;

    fn module() -> wax_ir::Module {
        super::common::module_from_src(
            PADDED_STORE_SRC,
            "tests/wax_to_air_params.rs",
            "kernels",
            "padded_store",
            &["64"],
            &[("x", &[1i32]), ("out", &[1i32])],
        )
    }

    #[test]
    fn padded_store_promotes_runtime_extents() {
        use wax_metal::bitcode::Ty;

        let m = module();
        let params = wax_metal::wax_to_air::converted_param_types(&m).expect("signature");

        let vec3 = Ty::vec(3, Ty::Int(32));
        let expected = [Ty::ptr(Ty::Int(32), 2), vec3.clone(), vec3];
        assert!(
            params.len() >= expected.len(),
            "signature too short: {params:?}"
        );
        let tail = &params[params.len() - expected.len()..];
        assert_eq!(
            tail,
            &expected[..],
            "expected promoted extent then _tp/_tg. Got {params:?}"
        );
        for (i, p) in params[..params.len() - expected.len()].iter().enumerate() {
            assert!(
                matches!(p, Ty::Ptr(..)),
                "parameter {i} should be a buffer pointer, got {p:?}"
            );
        }
    }
}

mod refusals {
    use super::common::rmw_module;
    use wax_frontend::{KernelCompiler, ast};
    use wax_ir::ScalarType;

    const SRC: &str = r#"
    mod kernels {
        use cutile::core::*;

        #[wax::entry()]
        pub fn scale<E: ElementType, const N: i32>(
            out: &mut Tensor<E, { [1, N] }>,
            x: &Tensor<E, { [1, N] }>,
        ) {
            let ts = const_shape![1, N];
            let px = x.partition(ts);
            let mut po: PartitionMut<E, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
            let v = px.load([0i32, 0i32]);
            unsafe { po.store(v + v, [0i32, 0i32]) };
        }
    }
    "#;

    fn scale_at(elem: &str) -> wax_ir::Module {
        let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
        let span = ast::SpanBase::new("tests/wax_to_air_refusals.rs".into(), 1, 1);
        let make = move || {
            let mut m = ast::Module::with_span_base("kernels", item.clone(), span.clone());
            m.set_absolute_path("wax_metal::refusals".to_string());
            m
        };
        KernelCompiler::new(make, "kernels", "scale")
            .generics(vec![elem.to_string(), "32".to_string()])
            .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
            .compile()
            .unwrap_or_else(|e| panic!("{elem}: frontend: {e}"))
            .to_wax_module("scale")
            .unwrap_or_else(|e| panic!("{elem}: wax module: {e}"))
    }

    fn rejected(m: &mut wax_ir::Module) -> Vec<String> {
        let profile = wax_metal::profile::TargetProfile::host();
        assert!(
            wax_metal::lower_to_metallib_for(m, &profile).is_err(),
            "expected module to not be lowered"
        );
        let f = *m.entries().first().expect("one entry function");
        let (_, left) =
            wax_metal::wax_to_air::run_conversion(&mut m.ctx, f).expect("conversion runs");
        left.0.keys().cloned().collect()
    }

    const SUB_REDUCE_SRC: &str = r#"
    mod kernels {
        use cutile::core::*;

        #[wax::entry()]
        pub fn sub_reduce(out: &mut Tensor<f32, { [1, 1] }>, x: &Tensor<f32, { [1, 32] }>) {
            let v = x.partition(const_shape![1, 32]).load([0i32, 0i32]);
            let r: Tile<f32, { [1] }> = reduce(v, 1i32, 0.0f32, |a: f32, b: f32| a - b);
            let mut po: PartitionMut<f32, { [1, 1] }> =
                unsafe { out.partition_full_mut(const_shape![1, 1]) };
            unsafe { po.store(r.reshape(const_shape![1, 1]), [0i32, 0i32]) };
        }
    }
    "#;

    #[test]
    fn unknown_reduction_is_rejected() {
        let item: syn::ItemMod = syn::parse_str(SUB_REDUCE_SRC).expect("kernel parses");
        let span = ast::SpanBase::new("tests/wax_to_air.rs".into(), 1, 1);
        let make = move || {
            let mut m = ast::Module::with_span_base("kernels", item.clone(), span.clone());
            m.set_absolute_path("wax_metal::refusals".to_string());
            m
        };
        let mut m = KernelCompiler::new(make, "kernels", "sub_reduce")
            .strides(&[("out", &[1, 1]), ("x", &[32, 1])])
            .compile()
            .expect("frontend")
            .to_wax_module("sub_reduce")
            .expect("wax module");
        assert!(rejected(&mut m).iter().any(|b| b.contains("Reduce")));
    }

    #[test]
    fn unsupported_atomic_widths_are_rejected() {
        let mut m = rmw_module("rmw_i64_add", ScalarType::I64, 3);
        let blamed = rejected(&mut m);
        assert!(blamed.iter().any(|k| k.contains("Atomic")));
    }

    #[test]
    fn element_types_air_cannot_spell_are_rejected() {
        for ty in ["f64", "i4"] {
            let mut m = scale_at(ty);
            let blamed = rejected(&mut m);
            assert!(blamed.iter().any(|k| k == "LoadViewTko"));
        }
    }

    fn mixed_half_add() -> wax_ir::Module {
        use wax_ir::builder::{OpBuilder, append_op};
        use wax_ir::{Attribute, Location, Module, Opcode, TileElementType, TileType, Type};
        let tile = |s: ScalarType| {
            Type::Tile(TileType {
                shape: vec![],
                element_type: TileElementType::Scalar(s),
            })
        };
        let mut m = Module::new("mixed_half_add");
        let (entry, block, args) = m
            .begin_function(
                "mixed_half_add",
                &[tile(ScalarType::F16), tile(ScalarType::BF16)],
            )
            .expect("the function builds");
        m.mark_entry(entry);
        m.set_op_attr(
            entry,
            &wax_ir::build::ATTR_SYM_NAME,
            Attribute::String("mixed_half_add".to_string()),
        );
        let (add_id, _res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operands([args[0], args[1]])
            .result(tile(ScalarType::F16))
            .build(&mut m)
            .expect("fixture op builds");
        append_op(&mut m, block, add_id);
        let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
            .build(&mut m)
            .expect("fixture op builds");
        append_op(&mut m, block, ret);
        m
    }

    #[test]
    fn float_cast_between_same_width_types_is_rejected() {
        let mut m = mixed_half_add();
        let blamed = rejected(&mut m);
        assert!(blamed.iter().any(|k| k == "AddF"));
    }
}

mod barriers {
    const SRC: &str = r#"
    mod k {
        use cutile::core::*;
        #[wax::entry()]
        fn one_colsum(out: &mut Tensor<f32, { [16, 1] }>, inp: &Tensor<f32, { [-1, 32] }>) {
            let w: Tile<f32, { [16, 32] }> = inp.partition(const_shape![16, 32]).load([0i32, 0i32]);
            let cs: Tile<f32, { [16] }> = reduce_sum(w, 0i32);
            out.store(cs.reshape(const_shape![16, 1]));
        }
    }
    "#;

    fn module() -> wax_ir::Module {
        super::common::module_from_src(
            SRC,
            "tests/wax_to_air_barriers.rs",
            "k",
            "one_colsum",
            &[],
            &[("out", &[1, 1]), ("inp", &[32, 1])],
        )
    }

    #[test]
    fn conversion_adds_required_barriers_correctly() {
        use pliron::printable::Printable;

        let mut m = module();
        let f = *m.entries().first().expect("one entry");
        wax_metal::wax_to_air::convert(&mut m.ctx, f).expect("one_colsum converts");
        let text = f.deref(&m.ctx).disp(&m.ctx).to_string();
        let barriers = text.matches("air.wg.barrier").count();
        assert_eq!(barriers, 3, "load, sum, store");
    }
}
