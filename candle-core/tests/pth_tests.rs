/// Regression test for pth files not loading on Windows.
#[test]
fn test_pth() {
    let tensors = candle_core::pickle::PthTensors::new("tests/test.pt", None).unwrap();
    tensors.get("test").unwrap().unwrap();
}

#[test]
fn test_pth_with_key() {
    let tensors =
        candle_core::pickle::PthTensors::new("tests/test_with_key.pt", Some("model_state_dict"))
            .unwrap();
    tensors.get("test").unwrap().unwrap();
}
