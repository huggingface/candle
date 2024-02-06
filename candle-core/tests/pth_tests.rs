/// Regression test for pth files not loading on Windows.
#[test]
fn test_pth() {   
    let tensors = candle_core::pickle::PthTensors::new("tests/test.pt").unwrap();
    tensors.get("test").unwrap().unwrap();
}
