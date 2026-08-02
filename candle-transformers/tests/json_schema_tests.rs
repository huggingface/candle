use candle_transformers::generation::{compile_schema, FsmLogitProcessor};
use std::sync::Arc;

#[test]
fn allows_text_prefixes_and_eos_only_for_a_valid_value() {
    let fsm = Arc::new(compile_schema(r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}"#).unwrap());
    let mut processor = FsmLogitProcessor::new(fsm);
    let vocab = ["{", "\"answer\"", ":", "\"ok\"", "}", "x", "@"];
    let allowed = processor.allowed_token_ids(vocab.len(), &[5], |id| vocab[id as usize].into());
    assert!(allowed.contains(&0)); assert!(!allowed.contains(&5)); assert!(!allowed.contains(&6));
    for text in ["{", "\"answer\"", ":", "\"ok\"", "}"] { processor.commit(text); }
    assert!(processor.is_complete());
    let allowed = processor.allowed_token_ids(vocab.len(), &[5], |id| vocab[id as usize].into());
    assert!(allowed.contains(&5));
    assert!(!allowed.contains(&0));
}

#[test]
fn rejects_unsupported_schema_constructs() {
    assert!(compile_schema(r#"{"oneOf":[]}"#).is_err());
}
