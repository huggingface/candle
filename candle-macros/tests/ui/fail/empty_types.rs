// Test: Empty macro invocation should fail
// This should produce a compile error

use candle_macros::register_quantized_types;

// This should fail - no types provided
register_quantized_types! {}

fn main() {}
