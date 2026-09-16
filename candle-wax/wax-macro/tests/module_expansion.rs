//! What `#[wax::module]` emits around the kernel items.
//!
//! The items themselves are `cutile-expand`'s and are covered by its own tests; these are the
//! three things the driver adds, each of which used to be missing and each of which fails
//! quietly rather than loudly when it is.

/// A kernel module whose contents would trip host-Rust lints if they applied.
///
/// `row_sum` is never called from Rust - the JIT is its only caller - and its parameters go
/// unread because the body is re-parsed rather than executed. Without the lint allows the
/// macro emits, this module warns on every build.
#[wax_macro::module]
mod kernels {
    use cutile_dsl::core::*;

    #[wax_macro::entry(unchecked_accesses = true)]
    fn row_sum(x: &Tensor<f32, { [-1, 32] }>, out: &mut Tensor<f32, { [-1, 32] }>) {
        let ts = const_shape![1, 32];
        let row = get_tile_block_id().0;
        let t = x.partition(ts).load([row, 0i32]);
        let mut o: PartitionMut<f32, { [1, 32] }> = unsafe { out.partition_full_mut(ts) };
        unsafe { o.store(t, [row, 0i32]) };
    }
}

#[test]
fn the_module_carries_a_source_hash() {
    let h = kernels::SOURCE_HASH;
    assert_eq!(h.len(), 64, "SHA-256 renders as 64 hex chars, got {h:?}");
    assert!(
        h.chars().all(|c| c.is_ascii_hexdigit()),
        "not hex: {h:?}"
    );
}

#[test]
fn the_hash_is_of_the_source_the_frontend_re_parses() {
    // Not an independent reimplementation - that would only test that two copies of the same
    // expression agree. The point is that the constant tracks the captured text, so a module
    // whose source differs must hash differently. `other` below differs by one literal.
    assert_ne!(
        kernels::SOURCE_HASH,
        other::SOURCE_HASH,
        "two different modules hashed the same"
    );
}

#[wax_macro::module]
mod other {
    use cutile_dsl::core::*;

    #[wax_macro::entry()]
    fn row_sum(x: &Tensor<f32, { [-1, 64] }>, out: &mut Tensor<f32, { [-1, 64] }>) {
        let ts = const_shape![1, 64];
        let row = get_tile_block_id().0;
        let t = x.partition(ts).load([row, 0i32]);
        let mut o: PartitionMut<f32, { [1, 64] }> = unsafe { out.partition_full_mut(ts) };
        unsafe { o.store(t, [row, 0i32]) };
    }
}

#[test]
fn both_modules_still_expose_their_ast() {
    assert_eq!(kernels::__module_ast_self().name(), "kernels");
    assert_eq!(other::__module_ast_self().name(), "other");
}
