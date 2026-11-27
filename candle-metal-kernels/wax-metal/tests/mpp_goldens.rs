use common::profile;
use wax_metal::mpp::{MppDtype, MppGemm};
mod common;

fn cfg(name: &str) -> MppGemm {
    MppGemm {
        batch: 1,
        a_batch: 0,
        b_batch: 0,
        entry: name.to_string(),
        dtype: MppDtype::F32,
        m: 256,
        n: 256,
        k: 256,
        lda: 256,
        ldb: 256,
        ldc: 256,
        bm: 32,
        bn: 32,
        out_dtype: MppDtype::F32,
        splits: 1,
        trans_b: false,
        trans_a: false,
        simdgroups: 1,
    }
}

fn corpus() -> Vec<(&'static str, MppGemm)> {
    let mut v: Vec<(&'static str, MppGemm)> = Vec::new();

    v.push(("f32_plain", cfg("mpp_f32_plain")));

    let mut c = cfg("mpp_f16");
    c.dtype = MppDtype::F16;
    c.out_dtype = MppDtype::F16;
    v.push(("f16", c));

    let mut c = cfg("mpp_bf16");
    c.dtype = MppDtype::BF16;
    c.out_dtype = MppDtype::BF16;
    v.push(("bf16", c));

    let mut c = cfg("mpp_bf16_out_f32");
    c.dtype = MppDtype::BF16;
    c.out_dtype = MppDtype::F32;
    v.push(("bf16_out_f32", c));

    let mut c = cfg("mpp_trans_b");
    c.trans_b = true;
    v.push(("trans_b", c));

    let mut c = cfg("mpp_splitk");
    c.dtype = MppDtype::BF16;
    c.out_dtype = MppDtype::F32;
    c.splits = 4;
    v.push(("splitk", c));

    let mut c = cfg("mpp_tile_64x32");
    c.bm = 64;
    c.bn = 32;
    v.push(("tile_64x32", c));

    let mut c = cfg("mpp_sg4");
    c.simdgroups = 4;
    v.push(("sg4", c));

    v
}

fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens/mpp")
        .join(format!("{name}.bc"))
}

#[test]
fn air_realisation_matches_goldens() {
    use pliron::context::Context;
    use std::collections::HashMap;
    use wax_metal::air_dialect::to_bitcode::{AirFunc, body_from_air};
    use wax_metal::lower::to_bitcode::{BuiltKernel, bitcode_with_kernels};
    use wax_metal::mpp::{mpp_abi, mpp_air};

    let profile = profile();
    let mut failed = Vec::new();
    for (name, cfg) in corpus() {
        let expected = match std::fs::read(golden_path(name)) {
            Ok(w) => w,
            Err(_) => {
                failed.push(format!(
                    "{name}: no golden at {}",
                    golden_path(name).display()
                ));
                continue;
            }
        };
        let abi = mpp_abi(&cfg, &profile);
        let mut ctx = Context::new();
        let func = mpp_air(&cfg, &mut ctx);
        let air = AirFunc {
            func,
            blocks: vec![func.get_entry_block(&ctx)],
        };
        let callees: HashMap<String, u32> = abi
            .callees
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), i as u32))
            .collect();
        let res_body = match body_from_air(&ctx, &air, 4, &|n| callees.get(n).copied(), &|_| None) {
            Ok(b) => b,
            Err(e) => {
                failed.push(format!("{name}: air lowering failed: {e:?}"));
                continue;
            }
        };
        let res = match bitcode_with_kernels(
            &abi.callees,
            vec![BuiltKernel {
                name: cfg.entry.as_str(),
                param_types: abi.entry_params.clone(),
                args: &abi.args,
                body: res_body,
                pinned_types: abi.pinned_types.clone(),
                threads: None,
            }],
            &[],
            &profile,
        ) {
            Ok(b) => b,
            Err(e) => {
                failed.push(format!("{name}: assembly failed: {e:?}"));
                continue;
            }
        };
        if res != expected {
            failed.push(format!(
                "{name}: {} bytes vs golden {}",
                res.len(),
                expected.len()
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "MPP air realisation changed:\n  {}",
        failed.join("\n  ")
    );
}
