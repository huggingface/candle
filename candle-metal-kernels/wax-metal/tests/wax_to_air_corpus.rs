//! The corpus driver: every kernel below through BOTH routes, compared as `.metallib` bytes.
//!
//! This is the plan's acceptance mechanism: the conversion route has to route each kernel
//! byte-identically through the CFG path, or refuse it with a reason. The three outcomes:
//!
//! - `SAME` - both routes produced identical containers.
//! - `DIFF` - both produced a metallib and they DISAGREE. This is the failure: a conversion
//!   that claims success while emitting a different lowering is a wrong answer, not a
//!   missing feature. The test panics on the first one.
//! - `MISS` - the conversion refused the kernel, naming the opcodes it choked on. Expected
//!   for the cases on the allowlist below; shrinking that list to empty is the definition
//!   of done.


mod common;

use common::Case;

/// Run one case through both routes and classify it.
enum Outcome {
    Same,
    Miss,
    Diff(String),
}

fn run_case(c: &Case) -> Outcome {
    let m = common::corpus_module(c);
    let profile = wax_metal::TargetProfile::host();

    // The CFG route runs FIRST, on purpose: it annotates the module in place (the
    // `cross_loop_reuse` staging decisions ride along as attributes), and the conversion
    // route reads those decisions to refuse rather than silently discard a plan.
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

/// The whole corpus: no DIFF, allowlist honoured, and the counts reported.
#[test]
fn the_corpus_routes_byte_identical_or_refused() {
    let mut same = 0;
    let mut miss = 0;
    for c in common::corpus::CORPUS {
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
                println!("{:<12} SAME", c.entry);
            }
            Outcome::Miss => {
                if !c.miss {
                    panic!(
                        "{}: refused by the conversion but is NOT on the allowlist", c.entry
                    );
                }
                miss += 1;
                println!("{:<12} MISS (allowlisted)", c.entry);
            }
            Outcome::Diff(diff) => panic!("{diff}"),
        }
    }
    println!("corpus: {same} SAME, 0 DIFF, {miss} MISS (allowlist)");
}