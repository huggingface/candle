use candle_core::pickle::PthTensors;

/// Regression test for pth files not loading on Windows.
#[test]
fn test_pth() {
    let tensors = PthTensors::new("tests/test.pt", None).unwrap();
    tensors.get("test").unwrap().unwrap();
}

#[test]
fn test_pth_with_key() {
    let tensors = PthTensors::new("tests/test_with_key.pt", Some("model_state_dict")).unwrap();
    tensors.get("test").unwrap().unwrap();
}

#[test]
fn test_pth_fortran_contiguous() {
    let tensors = PthTensors::new("tests/fortran_tensor_3d.pth", None).unwrap();
    let tensor = tensors.get("tensor_fortran").unwrap().unwrap();

    assert_eq!(tensor.dims3().unwrap(), (2, 3, 4));

    assert_eq!(
        tensor.to_vec3::<i64>().unwrap(),
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
        ]
    );
}

#[test]
fn test_pth_with_tuple_index() -> candle_core::Result<()> {
    let tensors = PthTensors::new("tests/test_with_tuple.pt", Some("1"))?;
    assert_eq!(tensors.tensor_infos().len(), 1);
    let tensor = tensors.get("test")?.unwrap();
    assert_eq!(tensor.dims2()?, (2, 4));
    assert_eq!(tensor.to_vec2::<i64>()?, [[1, 2, 3, 4], [5, 6, 7, 8]]);
    assert!(tensors.get("missing")?.is_none());
    Ok(())
}

#[test]
fn test_pth_tuple_selection_errors() {
    let cases = [
        ("invalid", "invalid tuple index"),
        ("-1", "invalid tuple index"),
        ("2", "out of bounds"),
    ];
    for (index, message) in cases {
        let error = PthTensors::new("tests/test_with_tuple.pt", Some(index))
            .err()
            .unwrap();
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[test]
fn test_pth_tuple_without_state_dict() -> candle_core::Result<()> {
    for key in [None, Some("0")] {
        let tensors = PthTensors::new("tests/test_with_tuple.pt", key)?;
        assert!(tensors.tensor_infos().is_empty());
    }
    Ok(())
}
