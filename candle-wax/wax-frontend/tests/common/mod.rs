/// Every op inside `m`'s entries.
///
/// Entry functions are excluded. Walk starts at their regions.
pub fn all_ops(m: &wax_ir::Module) -> Vec<pliron::context::Ptr<pliron::operation::Operation>> {
    use pliron::graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_region};
    use pliron::operation::Operation;

    let mut out = Vec::new();
    for f in m.entries() {
        for r in f.deref(&m.ctx).regions() {
            walk_region(
                &m.ctx,
                &mut out,
                &WalkConfig::default(),
                r,
                |_, out, node| {
                    if let IRNode::Operation(op) = node {
                        out.push(op);
                    }
                },
            );
        }
    }
    out
}
