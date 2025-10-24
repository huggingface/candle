// Compile-time tests using trybuild
//
// These tests verify that:
// 1. Valid macro usage compiles successfully
// 2. Invalid macro usage produces helpful error messages

#[test]
fn ui_tests() {
    let t = trybuild::TestCases::new();
    
    // Test cases that should compile successfully
    t.pass("tests/ui/pass/*.rs");
    
    // Test cases that should fail to compile with specific error messages
    t.compile_fail("tests/ui/fail/*.rs");
}
