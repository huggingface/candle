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
    processor.commit("{\"items\":[1,2]}");
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
fn rejects_the_wrong_scalar_kind_for_an_enum_at_the_first_transition() {
    let processor = FsmLogitProcessor::new(Arc::new(compile_schema(r#"{"enum":[12]}"#).unwrap()));
    let vocab = ["1", "\"", "t"];
    assert_eq!(
        processor.allowed_token_ids(vocab.len(), &[], |id| vocab[id as usize].into()),
        vec![0]
    );
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
