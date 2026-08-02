use candle_transformers::generation::{compile_schema, FsmLogitProcessor};
use std::sync::Arc;

#[test]
fn allows_text_prefixes_and_eos_only_for_a_valid_value() {
    let fsm = Arc::new(compile_schema(r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}"#).unwrap());
    let mut processor = FsmLogitProcessor::new(fsm);
    let vocab = ["{", "\"answer\"", ":", "\"ok\"", "}", "x", "@"];
    let allowed = processor.allowed_token_ids(vocab.len(), &[5], |id| vocab[id as usize].into());
    assert!(allowed.contains(&0));
    assert!(!allowed.contains(&5));
    assert!(!allowed.contains(&6));
    for text in ["{", "\"answer\"", ":", "\"ok\"", "}"] {
        processor.commit(text);
    }
    assert!(processor.is_complete());
    let allowed = processor.allowed_token_ids(vocab.len(), &[5], |id| vocab[id as usize].into());
    assert!(allowed.contains(&5));
    assert!(!allowed.contains(&0));
}

#[test]
fn rejects_unsupported_schema_constructs() {
    assert!(compile_schema(r#"{"oneOf":[]}"#).is_err());
}

#[test]
fn rejects_an_unknown_object_key_before_the_object_is_complete() {
    let fsm = Arc::new(compile_schema(r#"{"type":"object","properties":{"a":{"type":"number"}},"required":["a"],"additionalProperties":false}"#).unwrap());
    let mut processor = FsmLogitProcessor::new(fsm);
    processor.commit("{\"");
    let vocab = ["a", "zz"];
    let allowed = processor.allowed_token_ids(vocab.len(), &[], |id| vocab[id as usize].into());
    assert_eq!(allowed, vec![0]);
}

#[test]
fn transitions_through_nested_arrays_and_number_phases() {
    let schema = r#"{
        "type":"object",
        "properties":{"items":{"type":"array","items":{"type":"integer"}}},
        "required":["items"],
        "additionalProperties":false
    }"#;
    let mut processor = FsmLogitProcessor::new(Arc::new(compile_schema(schema).unwrap()));
    for text in [
        "{", "\"", "i", "t", "e", "m", "s", "\"", ":", "[", "1", ",", "2", "]", "}",
    ] {
        assert_eq!(
            processor.allowed_token_ids(1, &[], |_| text.into()),
            vec![0],
            "{text}"
        );
        processor.commit(text);
    }
    assert!(processor.is_complete());

    let mut processor = FsmLogitProcessor::new(Arc::new(compile_schema(schema).unwrap()));
    processor.commit("{\"items\":[");
    let vocab = ["1", "1.", "\"bad\"", "]"];
    let allowed = processor.allowed_token_ids(vocab.len(), &[], |id| vocab[id as usize].into());
    assert_eq!(allowed, vec![0, 1, 3]);
}

#[test]
fn restricts_enum_prefixes_without_reparsing_the_document() {
    let fsm = Arc::new(compile_schema(r#"{"enum":["yes","no"]}"#).unwrap());
    let mut processor = FsmLogitProcessor::new(fsm);
    processor.commit("\"");
    let vocab = ["y", "n", "x"];
    assert_eq!(
        processor.allowed_token_ids(vocab.len(), &[], |id| vocab[id as usize].into()),
        vec![0, 1]
    );
    processor.commit("yes\"");
    assert!(processor.is_complete());
}

#[test]
fn rejects_numeric_and_composite_enum_forms_at_compile_time() {
    for schema in [
        r#"{"enum":[12]}"#,
        r#"{"const":1.0}"#,
        r#"{"enum":[{"a":1}]}"#,
        r#"{"const":[1]}"#,
    ] {
        assert!(compile_schema(schema).is_err(), "{schema}");
    }
}

#[test]
fn tracks_escapes_literals_and_required_fields() {
    let schema = Arc::new(
        compile_schema(
            r#"{"type":"object","properties":{"a":{"type":"string"},"ok":{"type":"boolean"}},"required":["a","ok"],"additionalProperties":false}"#,
        )
        .unwrap(),
    );
    let mut processor = FsmLogitProcessor::new(schema);
    processor.commit("{\"\\u0061\":\"line\\n\",");
    let vocab = ["\"ok\":true}", "}", "\"wrong\""];
    assert_eq!(
        processor.allowed_token_ids(vocab.len(), &[], |id| vocab[id as usize].into()),
        vec![0]
    );
    processor.commit("\"ok\":true}");
    assert!(processor.is_complete());
}

#[test]
fn rejects_trailing_commas_and_non_json_whitespace() {
    let object = Arc::new(
        compile_schema(
            r#"{"type":"object","properties":{"a":{"type":"number"}},"required":["a"],"additionalProperties":false}"#,
        )
        .unwrap(),
    );
    let mut processor = FsmLogitProcessor::new(object);
    processor.commit("{\"a\":1,");
    assert!(processor
        .allowed_token_ids(1, &[], |_| "}".into())
        .is_empty());

    let array = Arc::new(compile_schema(r#"{"type":"array","items":{"type":"number"}}"#).unwrap());
    let mut processor = FsmLogitProcessor::new(array);
    processor.commit("[1,");
    assert!(processor
        .allowed_token_ids(1, &[], |_| "]".into())
        .is_empty());

    let processor =
        FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"number"}"#).unwrap()));
    assert!(processor
        .allowed_token_ids(1, &[], |_| "1\u{00a0}".into())
        .is_empty());
}

#[test]
fn accepts_surrogate_pairs_and_integral_decimal_forms() {
    let mut string =
        FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"string"}"#).unwrap()));
    string.commit("\"\\uD83D\\uDE00\"");
    assert!(string.is_complete());

    for number in ["1.0", "1e0", "-0.0"] {
        let mut integer =
            FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"integer"}"#).unwrap()));
        integer.commit(number);
        assert!(integer.is_complete(), "{number}");
    }
}

#[test]
fn uses_decimal_arithmetic_for_integers_and_rejects_out_of_range_numbers() {
    let integer_schema = Arc::new(compile_schema(r#"{"type":"integer"}"#).unwrap());
    let mut decimal_prefix = FsmLogitProcessor::new(Arc::clone(&integer_schema));
    decimal_prefix.commit("9007199254740992.5");
    assert!(!decimal_prefix.is_complete());
    assert!(decimal_prefix
        .allowed_token_ids(1, &[0], |_| "ignored".into())
        .is_empty());

    let processor = FsmLogitProcessor::new(Arc::clone(&integer_schema));
    assert!(processor
        .allowed_token_ids(1, &[], |_| "1e-400".into())
        .is_empty());

    let mut exponent_prefix = FsmLogitProcessor::new(integer_schema);
    exponent_prefix.commit("1.5e0");
    assert!(!exponent_prefix.is_complete());
    assert_eq!(
        exponent_prefix.allowed_token_ids(1, &[], |_| "1".into()),
        vec![0]
    );
    exponent_prefix.commit("1");
    assert!(exponent_prefix.is_complete());
    let number_schema = Arc::new(compile_schema(r#"{"type":"number"}"#).unwrap());
    let processor = FsmLogitProcessor::new(number_schema);
    assert!(processor
        .allowed_token_ids(1, &[], |_| "1e400".into())
        .is_empty());
    let mut underflow =
        FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"number"}"#).unwrap()));
    underflow.commit("1e-400");
    assert!(underflow.is_complete());
}

#[test]
fn permits_compatible_scalar_type_and_enum_constraints() {
    let mut processor = FsmLogitProcessor::new(Arc::new(
        compile_schema(r#"{"type":"string","enum":["yes","no"]}"#).unwrap(),
    ));
    processor.commit("\"yes\"");
    assert!(processor.is_complete());
}

#[test]
fn rejects_schema_forms_the_compiler_does_not_implement() {
    for schema in [
        r#"{"type":["string","null"]}"#,
        r#"{"type":"object","properties":{},"additionalProperties":{"type":"string"}}"#,
        r#"{"type":"object","properties":{},"required":["missing"],"additionalProperties":false}"#,
        r#"{"enum":[]}"#,
        r#"{"enum":["x"],"type":"number"}"#,
        r#"{"enum":["a"],"const":"b"}"#,
        r#"{"properties":{"a":{"type":"string"}},"additionalProperties":false}"#,
        r#"{"items":{"type":"string"}}"#,
        r#"{"type":"any"}"#,
    ] {
        assert!(compile_schema(schema).is_err(), "{schema}");
    }
}

#[test]
fn never_advances_after_a_complete_root_value_or_into_an_impossible_key() {
    let mut processor =
        FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"string"}"#).unwrap()));
    processor.commit("\"done\"");
    assert!(processor.is_complete());
    assert!(processor
        .allowed_token_ids(1, &[], |_| " ".into())
        .is_empty());

    let processor = FsmLogitProcessor::new(Arc::new(
        compile_schema(r#"{"type":"object","properties":{},"additionalProperties":false}"#)
            .unwrap(),
    ));
    let mut processor = processor;
    processor.commit("{");
    let vocab = ["{", "\""];
    let allowed = processor.allowed_token_ids(vocab.len(), &[], |id| vocab[id as usize].into());
    assert!(allowed.is_empty());
}

#[test]
fn ignores_empty_decoded_tokens() {
    let processor =
        FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"string"}"#).unwrap()));
    let vocab = ["", "\""];
    assert_eq!(
        processor.allowed_token_ids(vocab.len(), &[], |id| vocab[id as usize].into()),
        vec![1]
    );
}

#[test]
fn uses_json_schema_defaults_for_object_and_array_contents() {
    let mut object =
        FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"object"}"#).unwrap()));
    object.commit("{\"anything\":[true]}");
    assert!(object.is_complete());

    let mut array =
        FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"type":"array"}"#).unwrap()));
    array.commit("[null,{\"anything\":false}]");
    assert!(array.is_complete());
}

#[test]
fn accepts_the_unconstrained_json_schema() {
    let mut processor = FsmLogitProcessor::new(Arc::new(compile_schema("{}").unwrap()));
    processor.commit("[true,{\"nested\":null}]");
    assert!(processor.is_complete());
}

#[test]
fn permits_trailing_json_whitespace_in_the_finishing_token_only() {
    for (schema, token) in [
        (r#"{"type":"number"}"#, "1 "),
        (r#"{"type":"string"}"#, "\"x\"\n"),
        (
            r#"{"type":"object","properties":{},"additionalProperties":false}"#,
            "{}\r",
        ),
    ] {
        let mut processor = FsmLogitProcessor::new(Arc::new(compile_schema(schema).unwrap()));
        processor.commit(token);
        assert!(processor.is_complete(), "{token:?}");
        assert!(processor
            .allowed_token_ids(1, &[], |_| " ".into())
            .is_empty());
    }
}
