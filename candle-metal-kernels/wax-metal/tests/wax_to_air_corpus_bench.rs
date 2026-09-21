//! Byte parity of the conversion route on the POC's real kernel corpus.
//!
//! `kernels/bench.rs` holds the kernels from the metal-tile-rs project's
//! metal-tile-bench (tests/src/benches), import-swapped for candle3 with the
//! bodies otherwise unedited. Every case runs BOTH routes - the CFG walk and
//! the pliron-air DialectConversion - and compares the resulting `.metallib`
//! bytes. The instantiations (generics + strides) mirror the shapes the POC's
//! own tests dispatch.
//!
//! Out of scope, with reasons:
//! - `bitops` / `shr_u32`: integer tile shifts. The POC's vendored DSL exposes
//!   `<<`/`>>` on integer tiles; candle3's cutile frontend does not, so the
//!   kernels cannot even be written here. A frontend gap, not a conversion gap.
//! - `gemv__gemm_kernels` / `gemv__gemv_verify`: the SGMMA `mma()` at M=1 is
//!   invalid; the POC's own test (`gemv_m1_sgmma_carry_is_rejected_cleanly`)
//!   asserts that the backend rejects them, and this route does too. The
//!   working reduce-based replacement is `gemv` / `gemv_test` below.


mod common;

use common::Case;

/// Run one case through both routes and classify it (same mechanism as
/// `wax_to_air_corpus.rs`).
enum Outcome {
    Same,
    Miss,
    Diff(String),
}

fn run_case(c: &Case) -> Outcome {
    let m = common::bench_module(c);
    let profile = wax_metal::TargetProfile::host();

    // The CFG route runs FIRST, on purpose: it annotates the module in place
    // (the `cross_loop_reuse` staging decisions ride along as attributes), and
    // the conversion route reads those decisions to refuse rather than
    // silently discard a plan.
    let via_cfg = wax_metal::lower_to_metallib(&m)
        .unwrap_or_else(|e| panic!("{}: cfg path: {e}", c.entry));

    let func = *m.functions.first().expect("one entry function");
    let args = wax_metal::lower::kernel_args_for_module(&m, func);

    match wax_metal::wax_to_air::lower_to_metallib_via_conversion(&m, &args, &[], &profile) {
        Ok(bytes) if bytes == via_cfg => Outcome::Same,
        Ok(bytes) => {
            if let Some(dir) = std::env::var_os("WAX_CONVERSION_DUMP") {
                let d = std::path::PathBuf::from(dir);
                let _ = std::fs::write(d.join(format!("{}.conv.metallib", c.entry)), &bytes);
                let _ = std::fs::write(d.join(format!("{}.cfg.metallib", c.entry)), &via_cfg);
            }
            let at = bytes
                .iter()
                .zip(via_cfg.iter())
                .position(|(a, b)| a != b)
                .map(|i| i.to_string())
                .unwrap_or_else(|| "len".to_string());
            Outcome::Diff(format!(
                "{}: DIFF - conversion {} bytes vs CFG {}, first difference at {at}",
                c.entry,
                bytes.len(),
                via_cfg.len()
            ))
        }
        Err(e) => {
            eprintln!("{}: MISS {e}", c.entry);
            Outcome::Miss
        }
    }
}

#[test]
fn the_poc_corpus_routes_byte_identical_or_refused() {
    let mut same = 0;
    let mut miss = 0;
    for c in common::bench::CORPUS {
        match run_case(c) {
            Outcome::Same => {
                if c.miss {
                    panic!(
                        "{}: converted byte-identical but is on the allowlist - remove it \
                         from the allowlist",
                        c.entry
                    );
                }
                same += 1;
                println!("{:<34} SAME", c.entry);
            }
            Outcome::Miss => {
                if !c.miss {
                    panic!(
                        "{}: refused by the conversion but is NOT on the allowlist", c.entry
                    );
                }
                miss += 1;
                println!("{:<34} MISS (allowlisted)", c.entry);
            }
            Outcome::Diff(diff) => panic!("{diff}"),
        }
    }
    println!("bench corpus: {same} SAME, 0 DIFF, {miss} MISS (allowlist)");
}