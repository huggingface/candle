use wax_ir::{ScalarType, Type};

#[test]
fn only_marked_functions_are_exposed_as_entries() {
    let mut m = wax_ir::Module::new("my_module");
    let (entry, _, _) = m.begin_function("k", &[Type::Scalar(ScalarType::F32)]);
    let (_helper, _, _) = m.begin_function("h", &[Type::Scalar(ScalarType::F32)]);
    // Mark the entry fn as an antry
    m.mark_entry(entry);

    assert_eq!(m.functions().len(), 2, "both fns were built");
    assert_eq!(m.entries(), vec![entry], "only `entry` is marked");

    // Marking is idempotent
    m.mark_entry(entry);
    assert_eq!(m.entries(), vec![entry]);
}
