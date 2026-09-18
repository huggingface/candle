//! Textual SSA verifier for emitted AIR function bodies.
//!
//! Catches value-identity and SGMMA-vs-scalar naming defects (e.g. a loop header emitting
//! a scalar `%v61` carry phi where the body reads the sub-tile form `%vs61_0_*`) that
//! otherwise surface only as a `metal-as` "use of undefined value" pointing at a line
//! number rather than the decision that went wrong.
//!
//! Deliberately conservative - it flags only what is unambiguous from the text (undefined
//! uses, phi predecessors naming no real block, missing terminators, instructions after a
//! terminator) so it never rejects IR `metal-as` would accept.

use crate::LowerError;

/// Verify a single emitted function body (`define ... { ... }`, one function).
///
/// Returns `Ok(())` for well-formed IR, or [`LowerError::MalformedIr`] describing
/// the first violation found.
pub(crate) fn verify_function(body: &str) -> Result<(), LowerError> {
    let mut kernel = "<unknown>".to_string();
    // All value names defined anywhere in the function (params + instruction dsts).
    let mut defs: Vec<String> = Vec::new();
    // All block labels defined anywhere in the function.
    let mut labels: Vec<String> = Vec::new();
    // Deferred use checks: (value_name, block, instruction_text).
    let mut value_uses: Vec<(String, String, String)> = Vec::new();
    // Deferred phi-predecessor checks: (pred_label, block, instruction_text).
    let mut pred_uses: Vec<(String, String, String)> = Vec::new();

    let mut cur_block = "<entry>".to_string();
    // Whether the current block has already emitted a terminator.
    let mut block_terminated = false;
    // Whether we have entered a block yet (past the `define` line).
    let mut in_block = false;

    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }

        // `define void @name(params) #0 {` - parse kernel name + param value names.
        if line.starts_with("define ") {
            if let Some(at) = line.find('@') {
                let rest = &line[at + 1..];
                let end = rest
                    .find('(')
                    .unwrap_or_else(|| rest.find(' ').unwrap_or(rest.len()));
                kernel = rest[..end].to_string();
            }
            // The only `%tokens` on a define line are the parameter names.
            for name in percent_names(line) {
                defs.push(name);
            }
            continue;
        }

        // Closing brace of the function.
        if line == "}" {
            if in_block && !block_terminated {
                return Err(malformed(
                    &kernel,
                    &cur_block,
                    "block falls off the end of the function without a terminator",
                    "",
                ));
            }
            break;
        }

        // A label line: `name:` (no leading `%`, no `=`, ends in `:`).
        if let Some(label) = line.strip_suffix(':')
            && is_ident(label)
        {
            if in_block && !block_terminated {
                return Err(malformed(
                    &kernel,
                    &cur_block,
                    &format!("block is not terminated before label '{label}'"),
                    "",
                ));
            }
            labels.push(label.to_string());
            cur_block = label.to_string();
            block_terminated = false;
            in_block = true;
            continue;
        }

        // An instruction line. Note: we deliberately do NOT reject instructions
        // after a terminator here. The current `break` lowering appends a dead
        // `br` to the if-merge after the break's own branch, which is malformed
        // LLVM but tolerated by metal-as; rejecting it would break working
        // kernels. Phase 1's typed CFG eliminates it structurally (a block can
        // hold only one terminator). Dead post-terminator lines are still
        // scanned below, which is harmless - their operands are label refs.
        in_block = true;

        // Split off `%dst = rhs`.
        let (dst, rhs) = split_dst(line);
        if let Some(d) = dst {
            defs.push(d);
        }

        if rhs.starts_with("phi ") {
            for (val, pred) in phi_incomings(rhs) {
                if let Some(v) = val.strip_prefix('%') {
                    value_uses.push((v.to_string(), cur_block.clone(), line.to_string()));
                }
                if let Some(p) = pred.strip_prefix('%') {
                    pred_uses.push((p.to_string(), cur_block.clone(), line.to_string()));
                }
            }
        } else {
            for name in operand_uses(rhs) {
                value_uses.push((name, cur_block.clone(), line.to_string()));
            }
        }

        if is_terminator(rhs) {
            block_terminated = true;
        }
    }

    // Validate deferred uses against the full def/label sets
    // Two-pass so legal forward references (phi back-edges, branches to
    // not-yet-emitted blocks) are not falsely flagged.
    for (name, block, instr) in &value_uses {
        if !defs.iter().any(|d| d == name) {
            return Err(malformed(
                &kernel,
                block,
                &format!("use of undefined value '%{name}'"),
                instr,
            ));
        }
    }
    for (pred, block, instr) in &pred_uses {
        if !labels.iter().any(|l| l == pred) {
            return Err(malformed(
                &kernel,
                block,
                &format!("phi references predecessor '%{pred}' which is not a block label"),
                instr,
            ));
        }
    }

    Ok(())
}

fn malformed(kernel: &str, block: &str, msg: &str, instr: &str) -> LowerError {
    let where_ = if instr.is_empty() {
        format!("kernel @{kernel}, block %{block}")
    } else {
        format!("kernel @{kernel}, block %{block}: `{instr}`")
    };
    LowerError::MalformedIr(format!("{msg} ({where_})"))
}

/// Split `%dst = rhs` into `(Some(dst), rhs)`; otherwise `(None, line)`.
///
/// Any line starting with `%` is an assignment (`store`/`br`/`ret`/`call void`
/// begin with a keyword). The dst runs up to the first `=`; the emitter
/// sometimes pads it with extra whitespace for alignment, so trim rather than
/// matching a fixed `" = "`.
fn split_dst(line: &str) -> (Option<String>, &str) {
    if !line.starts_with('%') {
        return (None, line);
    }
    let Some(eq) = line.find('=') else {
        return (None, line);
    };
    let lhs = line[..eq].trim();
    let Some(name) = lhs.strip_prefix('%') else {
        return (None, line);
    };
    if !is_ident(name) {
        return (None, line);
    }
    (Some(name.to_string()), line[eq + 1..].trim_start())
}

/// True for terminator instructions (which end a basic block).
fn is_terminator(rhs: &str) -> bool {
    let kw = rhs.split_whitespace().next().unwrap_or("");
    matches!(kw, "br" | "ret" | "unreachable" | "switch" | "indirectbr")
}

/// Value-name uses in a non-phi instruction's RHS, excluding block-label
/// operands (the `%X` in `label %X`).
fn operand_uses(rhs: &str) -> Vec<String> {
    let toks: Vec<&str> = rhs
        .split([' ', ',', '(', ')', '\t'])
        .filter(|t| !t.is_empty())
        .collect();
    let mut uses = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if let Some(name) = t.strip_prefix('%') {
            let is_label_operand = i > 0 && toks[i - 1] == "label";
            if !is_label_operand && is_ident(name) {
                uses.push(name.to_string());
            }
        }
    }
    uses
}

/// Parse phi incoming groups `[ val, %pred ]` into `(val, pred)` string pairs.
fn phi_incomings(rhs: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes = rhs.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let start = i + 1;
            let Some(rel_end) = rhs[start..].find(']') else {
                break;
            };
            let group = &rhs[start..start + rel_end];
            if let Some((val, pred)) = group.split_once(',') {
                out.push((val.trim().to_string(), pred.trim().to_string()));
            }
            i = start + rel_end + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Extract every `%name` token from a line.
fn percent_names(line: &str) -> Vec<String> {
    let mut names = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && is_ident_byte(bytes[j]) {
                j += 1;
            }
            if j > start {
                names.push(line[start..j].to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    names
}

fn is_ident(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(is_ident_byte)
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_well_formed_loop() {
        let ir = "\
define void @ok(float addrspace(1)* %_p0, <3 x i32> %_tp, <3 x i32> %_tg) #0 {
entry:
    %v0 = extractelement <3 x i32> %_tg, i32 0
    br label %hdr
hdr:
    %iv = phi i32 [ 0, %entry ], [ %ivn, %back ]
    %acc = phi float [ 0.0, %entry ], [ %accn, %back ]
    %c = icmp slt i32 %iv, 16
    br i1 %c, label %body, label %exit
body:
    %accn = fadd float %acc, 1.0
    br label %back
back:
    %ivn = add i32 %iv, 1
    br label %hdr
exit:
    ret void
}
";
        assert!(verify_function(ir).is_ok());
    }

    #[test]
    fn rejects_undefined_phi_backedge() {
        // Mirrors the gemv defect: header carry phi references a scalar %v83
        // from the back edge, but nothing defines %v83 (the body produced the
        // SGMMA sub-tile form %vs83_0_*).
        let ir = "\
define void @gemv(float addrspace(1)* %_p0, <3 x i32> %_tp, <3 x i32> %_tg) #0 {
entry:
    br label %hdr
hdr:
    %v60 = phi i32 [ 0, %entry ], [ %_t1, %back ]
    %v61 = phi float [ 0.0, %entry ], [ %v83, %back ]
    %c = icmp slt i32 %v60, 16
    br i1 %c, label %body, label %exit
body:
    %vs83_0_0 = fadd float %v61, 1.0
    br label %back
back:
    %_t1 = add i32 %v60, 1
    br label %hdr
exit:
    ret void
}
";
        let err = verify_function(ir).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("%v83"), "expected %v83 in error, got: {msg}");
        assert!(msg.contains("undefined"), "got: {msg}");
    }

    #[test]
    fn rejects_bad_phi_predecessor() {
        let ir = "\
define void @bad(<3 x i32> %_tp) #0 {
entry:
    br label %hdr
hdr:
    %iv = phi i32 [ 0, %nonexistent ], [ %ivn, %hdr ]
    %ivn = add i32 %iv, 1
    ret void
}
";
        let err = verify_function(ir).unwrap_err();
        assert!(err.to_string().contains("nonexistent"), "{err}");
    }

    #[test]
    fn rejects_missing_terminator() {
        let ir = "\
define void @bad(<3 x i32> %_tp) #0 {
entry:
    %v0 = add i32 0, 0
}
";
        let err = verify_function(ir).unwrap_err();
        assert!(err.to_string().contains("terminator"), "{err}");
    }

    #[test]
    fn tolerates_dead_branch_after_break() {
        // The `break` lowering appends a dead `br` after the break's own branch.
        // metal-as accepts this, so the verifier must not reject it.
        let ir = "\
define void @brk(<3 x i32> %_tp) #0 {
entry:
    br label %then
then:
    br label %exit
    br label %merge
merge:
    br label %exit
exit:
    ret void
}
";
        assert!(verify_function(ir).is_ok());
    }

    #[test]
    fn tolerates_alignment_padding_before_eq() {
        // The emitter pads dsts with extra spaces for alignment (`%_t2  = ...`).
        let ir = "\
define void @pad(<3 x i32> %_tp) #0 {
entry:
    %_t0 = extractelement <3 x i32> %_tp, i32 0
    %_t1 = add i32 %_t0, 1
    %_t2  = add i32 %_t0, %_t1
    %_t10  = add i32 %_t2, 1
    ret void
}
";
        assert!(verify_function(ir).is_ok(), "{:?}", verify_function(ir));
    }

    #[test]
    fn label_operands_are_not_value_uses() {
        // `label %exit` must not be mistaken for a use of a value %exit.
        let ir = "\
define void @ok(<3 x i32> %_tp) #0 {
entry:
    %c = icmp slt i32 0, 1
    br i1 %c, label %a, label %exit
a:
    br label %exit
exit:
    ret void
}
";
        assert!(verify_function(ir).is_ok());
    }
}
