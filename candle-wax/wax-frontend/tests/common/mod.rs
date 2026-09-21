/// Every op inside `m`'s entries.
///
/// Entry functions are excluded. Walk starts at their regions.
pub fn all_ops(m: &wax_ir::Module) -> Vec<pliron::context::Ptr<pliron::operation::Operation>> {
    use pliron::graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_region};
    use wax_ir::source::WaxSource;

    let mut out = Vec::new();
    for f in m.entries() {
        for r in m.regions(f) {
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
