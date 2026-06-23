use candle_core::quantized::gguf_file::{self, Value, VersionedMagic};
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use std::collections::HashMap;
use tokenizers::Tokenizer;

fn make_content(tokens: Vec<&str>, merges: Vec<&str>, added_tokens: Vec<&str>) -> gguf_file::Content {
    let tokens: Vec<Value> = tokens
        .into_iter()
        .map(|t| Value::String(t.to_string()))
        .collect();
    let merges: Vec<Value> = merges
        .into_iter()
        .map(|m| Value::String(m.to_string()))
        .collect();
    let added: Vec<Value> = added_tokens
        .into_iter()
        .map(|t| Value::String(t.to_string()))
        .collect();

    let mut metadata: HashMap<String, Value> = HashMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        Value::String("gpt2".to_string()),
    );
    metadata.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
    metadata.insert("tokenizer.ggml.merges".to_string(), Value::Array(merges));
    metadata.insert(
        "tokenizer.ggml.pre".to_string(),
        Value::String("gpt2".to_string()),
    );
    if !added.is_empty() {
        metadata.insert(
            "tokenizer.ggml.added_tokens".to_string(),
            Value::Array(added),
        );
    }

    gguf_file::Content {
        magic: VersionedMagic::GgufV3,
        metadata,
        tensor_infos: HashMap::new(),
        tensor_data_offset: 0,
    }
}

#[test]
fn added_tokens_are_loaded_as_special() {
    let content = make_content(
        vec!["a", "b"],
        vec![],
        vec!["<think>", "</think>"],
    );
    let tokenizer = Tokenizer::from_gguf(&content).expect("from_gguf should succeed");

    assert!(
        tokenizer.token_to_id("<think>").is_some(),
        "<think> should be in vocab"
    );
    assert!(
        tokenizer.token_to_id("</think>").is_some(),
        "</think> should be in vocab"
    );

    let think_id = tokenizer.token_to_id("<think>").unwrap();
    let close_id = tokenizer.token_to_id("</think>").unwrap();
    assert_ne!(think_id, close_id, "tokens should have different ids");
    assert!(
        think_id >= 2,
        "<think> should get id >= base vocab size (2)"
    );
}

#[test]
fn added_tokens_encode_as_single_token() {
    let content = make_content(
        vec!["a", "b"],
        vec![],
        vec!["<think>", "</think>"],
    );
    let tokenizer = Tokenizer::from_gguf(&content).expect("from_gguf should succeed");

    let encoding = tokenizer
        .encode("<think>", false)
        .expect("encode should succeed");
    let ids = encoding.get_ids();
    assert_eq!(ids.len(), 1, "<think> should be a single token");
    assert_eq!(
        *ids.first().unwrap(),
        tokenizer.token_to_id("<think>").unwrap()
    );

    let encoding = tokenizer
        .encode("</think>", false)
        .expect("encode should succeed");
    let ids = encoding.get_ids();
    assert_eq!(ids.len(), 1, "</think> should be a single token");

    let encoding = tokenizer
        .encode("a <think> b </think>", false)
        .expect("encode should succeed");
    let ids = encoding.get_ids();
    assert!(
        ids.contains(&tokenizer.token_to_id("<think>").unwrap()),
        "should contain <think> token"
    );
    assert!(
        ids.contains(&tokenizer.token_to_id("</think>").unwrap()),
        "should contain </think> token"
    );
}

#[test]
fn no_added_tokens_does_not_fail() {
    let content = make_content(vec!["a", "b"], vec![], vec![]);
    let tokenizer = Tokenizer::from_gguf(&content).expect("from_gguf should succeed without added_tokens");
    assert_eq!(tokenizer.get_vocab_size(true), 2);
}
