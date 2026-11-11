#![cfg(not(target_arch = "wasm32"))]

use candle_core::quantized::gguf_file::{Content, Value, VersionedMagic};
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use std::collections::HashMap;
use tokenizers::Tokenizer;

fn content_with_metadata(metadata: HashMap<String, Value>) -> Content {
    Content {
        magic: VersionedMagic::GgufV3,
        metadata,
        tensor_infos: HashMap::new(),
        tensor_data_offset: 0,
    }
}

fn value_string_array(values: &[&str]) -> Value {
    Value::Array(
        values
            .iter()
            .map(|value| Value::String(value.to_string()))
            .collect(),
    )
}

fn value_f32_array(values: &[f32]) -> Value {
    Value::Array(values.iter().copied().map(Value::F32).collect())
}

#[test]
fn llama_tokenizer_without_merges_is_reconstructed_from_scores() {
    let tokens = [
        "<unk>", "<s>", "</s>", "▁", "H", "e", "l", "o", "C", "a", "n", "d", "!", "el", "lo",
        "Hel", "Hello", "▁Hello", "nd", "and", "Cand", "▁Cand", "le",
    ];
    let scores = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 20.0, 30.0,
        40.0, 10.0, 20.0, 30.0, 40.0, 10.0,
    ];

    let mut metadata = HashMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        Value::String("llama".to_string()),
    );
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        value_string_array(&tokens),
    );
    metadata.insert(
        "tokenizer.ggml.scores".to_string(),
        value_f32_array(&scores),
    );
    metadata.insert(
        "tokenizer.ggml.add_bos_token".to_string(),
        Value::Bool(true),
    );
    metadata.insert(
        "tokenizer.ggml.add_eos_token".to_string(),
        Value::Bool(false),
    );
    metadata.insert("tokenizer.ggml.bos_token_id".to_string(), Value::U32(1));
    metadata.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U32(2));
    metadata.insert("tokenizer.ggml.unknown_token_id".to_string(), Value::U32(0));

    let content = content_with_metadata(metadata);
    let tokenizer = Tokenizer::from_gguf(&content).expect("llama tokenizer should load");

    let encoding = tokenizer
        .encode("Hello Candle!", true)
        .expect("prompt should encode");
    assert_eq!(encoding.get_ids(), &[1, 17, 21, 22, 12]);
    assert_eq!(
        encoding.get_tokens(),
        &["<s>", "▁Hello", "▁Cand", "le", "!"]
    );

    let decoded = tokenizer
        .decode(encoding.get_ids(), true)
        .expect("tokens should decode");
    assert_eq!(decoded, "Hello Candle!");
}
