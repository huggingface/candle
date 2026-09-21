//! The conversion route against the CFG route: same kernel, same bytes?
//!
//! Two questions, and the first decides whether the second is worth asking.
//!
//! 1. **Functional equality.** The CFG path's metallib is the reference. Byte-identity is the
//!    strongest check available and the one the backend already holds itself to elsewhere.
//! 2. **Cost.** What the conversion route costs per kernel, end to end.


use wax_frontend::{ast, KernelCompiler};

fn parse(src: &'static str, file: &str) -> impl Fn() -> ast::Module {
    let item: syn::ItemMod = syn::parse_str(src).expect("kernel parses");
    let span_base = ast::SpanBase::new(file.into(), 1, 1);
    move || {
        let mut m = ast::Module::with_span_base("kernels", item.clone(), span_base.clone());
        m.set_absolute_path("wax_metal::kernels".to_string());
        m
    }
}

const SCALE_SRC: &str = r#"
mod kernels {
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

const COND_SCALE_SRC: &str = r#"
mod kernels {
    use cutile::core::*;
    #[wax::entry()]
    fn cond_scale<const N: i32>(out: &mut Tensor<f32, { [1, N] }>, x: &Tensor<f32, { [1, N] }>) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut po: PartitionMut<f32, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        let v = px.load([0i32, 0i32]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        if pid.0 == 0 {
            unsafe { po.store(v + v, [0i32, 0i32]) };
        } else {
            unsafe { po.store(v, [0i32, 0i32]) };
        }
    }
}
"#;

fn scale_module() -> wax_ir::Module {
    KernelCompiler::new(parse(SCALE_SRC, "tests/wax_to_air_e2e.rs"), "kernels", "scale")
        .generics(vec!["32".into()])
        .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
        .compile()
        .expect("frontend compiles scale")
        .to_wax_module("scale")
        .expect("dialect module builds")
}

#[test]
fn the_conversion_route_produces_the_same_metallib() {
    let m = scale_module();
    let profile = wax_metal::TargetProfile::host();

    let via_cfg = wax_metal::lower_to_metallib(&m).expect("CFG path lowers");

    // Both routes must agree exactly on how Metal binds the kernel's buffers, so the
    // descriptors come from one shared derivation over the signature rather than each
    // route rolling its own. `scale` has no threadgroup scratch, so tg_globals is empty
    // for both.
    let func = *m.functions.first().expect("module has a function");
    let args = wax_metal::lower::kernel_args_for_module(&m, func);

    let via_conv =
        wax_metal::wax_to_air::lower_to_metallib_via_conversion(&m, &args, &[], &profile);

    match via_conv {
        Err(e) => panic!("conversion route failed: {e}"),
        Ok(bytes) => {
            assert_eq!(&bytes[..4], b"MTLB", "not a metallib container");
            assert_eq!(
                bytes.len(),
                via_cfg.len(),
                "conversion metallib is {} bytes against the CFG route's {}; with the same \
                 argument descriptors the two should now agree",
                bytes.len(),
                via_cfg.len()
            );
            assert_eq!(
                bytes, via_cfg,
                "same length but different bytes - the function body differs between the two \
                 routes"
            );
        }
    }
}

/// What the conversion route costs, now that its output is identical.
///
/// Both timings start from the same frontend module and end at metallib bytes, so the
/// frontend is excluded from both and the difference is the lowering itself. Reported
/// rather than asserted: a threshold here would be a flaky test on a shared machine.
#[test]
fn the_two_routes_cost_about_the_same() {
    use std::time::Instant;

    let m = scale_module();
    let profile = wax_metal::TargetProfile::host();
    let func = *m.functions.first().expect("module has a function");

    // Warm both paths before timing: first touch pays for lazily-built dialect registries.
    let _ = wax_metal::lower_to_metallib(&m).expect("CFG warmup");
    let warm_args = wax_metal::lower::kernel_args_for_module(&m, func);
    let _ = wax_metal::wax_to_air::lower_to_metallib_via_conversion(&m, &warm_args, &[], &profile)
        .expect("conversion warmup");

    const N: u32 = 200;

    let t0 = Instant::now();
    for _ in 0..N {
        let bytes = wax_metal::lower_to_metallib(&m).expect("CFG path lowers");
        std::hint::black_box(bytes);
    }
    let cfg = t0.elapsed();

    let t1 = Instant::now();
    for _ in 0..N {
            let args = wax_metal::lower::kernel_args_for_module(&m, func);
        let bytes =
            wax_metal::wax_to_air::lower_to_metallib_via_conversion(&m, &args, &[], &profile)
                .expect("conversion path lowers");
        std::hint::black_box(bytes);
    }
    let conv = t1.elapsed();

    let cfg_us = cfg.as_secs_f64() * 1e6 / f64::from(N);
    let conv_us = conv.as_secs_f64() * 1e6 / f64::from(N);
    println!(
        "lowering `scale`, {N} iterations: CFG {cfg_us:.1} us, conversion {conv_us:.1} us \
         ({:.2}x)",
        conv_us / cfg_us
    );
}

/// The `if` lowering, on its own, as bytes.
///
/// `wax.if` becomes a `cond_br` over two arms rejoining at a merge block, with results as
/// merge-block arguments. The CFG route's loop kernels cannot tell a working `if` lowering
/// from a working `for` one, so this one exists on its own.
#[test]
fn a_conditional_kernel_is_byte_identical() {
    let m = KernelCompiler::new(
        parse(COND_SCALE_SRC, "tests/wax_to_air_e2e.rs"),
        "kernels",
        "cond_scale",
    )
    .generics(vec!["32".into()])
    .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
    .compile()
    .expect("frontend compiles cond_scale")
    .to_wax_module("cond_scale")
    .expect("dialect module builds");
    let profile = wax_metal::TargetProfile::host();
    let via_cfg = wax_metal::lower_to_metallib(&m).expect("CFG path lowers cond_scale");
    let func = *m.functions.first().expect("module has a function");
    let args = wax_metal::lower::kernel_args_for_module(&m, func);
    let bytes =
        wax_metal::wax_to_air::lower_to_metallib_via_conversion(&m, &args, &[], &profile)
            .expect("conversion route lowers cond_scale");
    assert_eq!(
        bytes.len(),
        via_cfg.len(),
        "conversion {} bytes vs CFG {}",
        bytes.len(),
        via_cfg.len()
    );
    assert_eq!(bytes, via_cfg, "same length, different bytes");
}