use candle_transformers::generation::IncrementalDecoder;
use serde_json::{json, Value};
use tokenizers::Tokenizer;

/// Build a tokenizer from an explicit vocabulary and decoder, so the tests exercise decoder
/// behaviour without downloading anything.
fn tokenizer(vocab: &[&str], decoder: Value) -> Tokenizer {
    tokenizer_with_added(vocab, decoder, json!([]))
}

fn tokenizer_with_added(vocab: &[&str], decoder: Value, added_tokens: Value) -> Tokenizer {
    let vocab: serde_json::Map<String, Value> = vocab
        .iter()
        .enumerate()
        .map(|(i, token)| ((*token).to_string(), json!(i)))
        .collect();
    let spec = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": decoder,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": [],
        },
    });
    Tokenizer::from_bytes(serde_json::to_vec(&spec).unwrap()).unwrap()
}

fn byte_level() -> Value {
    json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true})
}

/// The invariant from the issue: the incremental text equals a whole-sequence decode at every
/// step, and the concatenation of the emitted deltas reproduces it exactly.
fn check_stream(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    let mut decoder = IncrementalDecoder::new(tokenizer);
    let mut streamed = String::new();
    for (i, &id) in ids.iter().enumerate() {
        let delta = decoder.push(id).unwrap().to_string();
        assert_eq!(decoder.retracted(), 0, "unexpected retraction at step {i}");
        streamed.push_str(&delta);
        let expected = tokenizer.decode(&ids[..=i], true).unwrap();
        assert_eq!(decoder.text(), expected, "diverged at step {i}");
        assert!(
            expected.starts_with(&streamed),
            "emitted text is not a prefix of the decode at step {i}: {streamed:?} vs {expected:?}"
        );
    }
    streamed.push_str(decoder.finish());
    assert_eq!(streamed, decoder.text());
    streamed
}

#[test]
fn multi_byte_char_spanning_tokens() {
    // "Ã" and "©" are the byte-level spellings of 0xC3 and 0xA9, the two bytes of "é".
    let tokenizer = tokenizer(&["Hello", "Ġworld", "Ã", "©"], byte_level());
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    assert!(decoder.is_windowed());

    assert_eq!(decoder.push(0).unwrap(), "Hello");
    assert_eq!(decoder.push(1).unwrap(), " world");
    // The lone 0xC3 decodes to U+FFFD, which the next token replaces, so nothing is emitted.
    assert_eq!(decoder.push(2).unwrap(), "");
    assert_eq!(decoder.text(), "Hello world\u{FFFD}");
    assert_eq!(decoder.push(3).unwrap(), "é");
    assert_eq!(decoder.text(), "Hello worldé");

    check_stream(&tokenizer, &[0, 1, 2, 3]);
}

#[test]
fn undecodable_bytes_are_released_by_finish() {
    let tokenizer = tokenizer(&["Hello", "Ã"], byte_level());
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    assert_eq!(decoder.push(0).unwrap(), "Hello");
    assert_eq!(decoder.push(1).unwrap(), "");
    assert_eq!(decoder.finish(), "\u{FFFD}");
    assert_eq!(decoder.finish(), "");
}

#[test]
fn sentencepiece_leading_space() {
    // The decoder shipped by the Llama-style tokenizers: a per-token replace, byte fallback,
    // a fuse, then a strip of the leading space introduced by the first "▁".
    let decoder = json!({"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ]});
    let tokenizer = tokenizer(&["▁Hello", "▁world", "<0xE5>", "<0x8f>", "<0xab>"], decoder);
    let mut incremental = IncrementalDecoder::new(&tokenizer);
    assert!(incremental.is_windowed());

    // The leading space is stripped from the first token but not from the second, so a naive
    // per-token decode would produce "HelloWorld".
    assert_eq!(incremental.push(0).unwrap(), "Hello");
    assert_eq!(incremental.push(1).unwrap(), " world");
    // 叫 is three byte-fallback tokens.
    assert_eq!(incremental.push(2).unwrap(), "");
    assert_eq!(incremental.push(3).unwrap(), "");
    assert_eq!(incremental.push(4).unwrap(), "叫");
    assert_eq!(incremental.text(), "Hello world叫");

    check_stream(&tokenizer, &[0, 1, 2, 3, 4]);
}

#[test]
fn wordpiece_decoder() {
    let decoder = json!({"type": "WordPiece", "prefix": "##", "cleanup": true});
    let tokenizer = tokenizer(&["Ara", "##új", "##o", "No", "##guera"], decoder);
    let streamed = check_stream(&tokenizer, &[0, 1, 2, 3, 4]);
    assert_eq!(streamed, "Araújo Noguera");
}

#[test]
fn no_decoder_joins_with_spaces() {
    let tokenizer = tokenizer(&["a", "b", "c"], Value::Null);
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    assert!(decoder.is_windowed());
    assert_eq!(decoder.push(0).unwrap(), "a");
    assert_eq!(decoder.push(1).unwrap(), " b");
    assert_eq!(check_stream(&tokenizer, &[0, 1, 2]), "a b c");
}

#[test]
fn bpe_decoder_defers_the_last_token() {
    // `BPEDecoder` drops the end-of-word suffix of the last token instead of turning it into a
    // space, so the tail of the text is rewritten by the next push and must not be emitted yet.
    let decoder = json!({"type": "BPEDecoder", "suffix": "</w>"});
    let tokenizer = tokenizer(&["ab</w>", "cd</w>"], decoder);
    let mut incremental = IncrementalDecoder::new(&tokenizer);
    assert_eq!(incremental.push(0).unwrap(), "");
    assert_eq!(incremental.text(), "ab");
    assert_eq!(incremental.push(1).unwrap(), "ab");
    assert_eq!(incremental.text(), "ab cd");
    assert_eq!(incremental.finish(), " cd");

    check_stream(&tokenizer, &[0, 1, 0, 1, 0]);
}

#[test]
fn window_stays_bounded_over_a_long_run() {
    let tokenizer = tokenizer(
        &["Hello", "Ġworld", "Ã", "©", "Ġthe", "Ġquick", "."],
        byte_level(),
    );
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    let mut ids = Vec::new();
    // A cheap deterministic shuffle, so the run mixes plain tokens with byte pairs.
    let mut state = 12345u64;
    for _ in 0..2000 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let id = (state >> 33) as u32 % 7;
        ids.push(id);
        decoder.push(id).unwrap();
        assert_eq!(decoder.text(), tokenizer.decode(&ids, true).unwrap());
        // A window that fails to re-anchor keeps growing until the next attempt, so the bound
        // is on the order of the re-anchoring threshold rather than exactly it.
        assert!(
            decoder.window_len() <= 128,
            "window grew to {} tokens",
            decoder.window_len()
        );
    }
}

#[test]
fn unbounded_decoder_falls_back_to_whole_sequence() {
    // A `Replace` with a regex pattern, applied after the tokens have been fused, can rewrite
    // text arbitrarily far back: windowing has to be disabled for it.
    let decoder = json!({"type": "Sequence", "decoders": [
        byte_level(),
        {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
    ]});
    let tokenizer = tokenizer(&["Hello", "Ġworld", "Ġ", "Ġthe"], decoder);
    let mut incremental = IncrementalDecoder::new(&tokenizer);
    assert!(!incremental.is_windowed());

    let mut ids = Vec::new();
    for id in [0, 1, 2, 2, 3, 2, 0] {
        ids.push(id);
        incremental.push(id).unwrap();
        assert_eq!(incremental.text(), tokenizer.decode(&ids, true).unwrap());
    }
    assert_eq!(incremental.window_len(), ids.len());
}

#[test]
fn fused_bpe_decoder_accounts_for_its_suffix() {
    // After a `Fuse` the decoder sees one string, so every occurrence of the suffix is dropped
    // wherever it lands - including one assembled across several pushed tokens, which rewrites
    // text that a per-token reading would have considered settled.
    let decoder = json!({"type": "Sequence", "decoders": [
        {"type": "Fuse"},
        {"type": "BPEDecoder", "suffix": "abc"},
    ]});
    let tokenizer = tokenizer(&["a", "b", "c", "x"], decoder);
    let mut incremental = IncrementalDecoder::new(&tokenizer);
    assert!(incremental.is_windowed());

    // "a", then "ab", then the suffix completes and the text collapses to "".
    for (i, id) in [0, 1, 2].into_iter().enumerate() {
        assert_eq!(incremental.push(id).unwrap(), "");
        assert_eq!(incremental.retracted(), 0, "retraction at step {i}");
    }
    assert_eq!(incremental.text(), "");
    check_stream(&tokenizer, &[3, 0, 1, 2, 3]);
}

#[test]
fn fused_ctc_accounts_for_its_pad_token() {
    // The pad token is stripped from the fused string whether or not `cleanup` is set.
    let decoder = json!({"type": "Sequence", "decoders": [
        {"type": "Fuse"},
        {"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": false},
    ]});
    let tokenizer = tokenizer(&["<", "p", "a", "d", ">"], decoder);
    let mut incremental = IncrementalDecoder::new(&tokenizer);
    assert!(incremental.is_windowed());

    for (i, id) in [0, 1, 2, 3, 4].into_iter().enumerate() {
        assert_eq!(incremental.push(id).unwrap(), "");
        assert_eq!(incremental.retracted(), 0, "retraction at step {i}");
    }
    assert_eq!(incremental.text(), "");
}

#[test]
fn byte_decoders_after_a_fuse_are_unbounded() {
    // Both match a shape against the whole accumulated string, so appending one token can
    // rewrite the entire prefix; neither may be windowed once the tokens have been fused.
    for inner in [byte_level(), json!({"type": "ByteFallback"})] {
        let fused = json!({"type": "Sequence", "decoders": [{"type": "Fuse"}, inner.clone()]});
        let after_fuse = tokenizer(&["<0x61>", "Hello", "Ã"], fused);
        assert!(
            !IncrementalDecoder::new(&after_fuse).is_windowed(),
            "{inner} after a Fuse must not be windowed"
        );
        // On its own it stays windowed.
        let alone = tokenizer(&["<0x61>", "Hello", "Ã"], inner);
        assert!(IncrementalDecoder::new(&alone).is_windowed());
    }
}

#[test]
fn a_run_of_special_tokens_keeps_the_window_bounded() {
    // Skipped special tokens decode to nothing, so no candidate tail is ever long enough to
    // re-anchor on. They never reach the decoder either, so the window can simply skip them.
    let added = json!([{
        "id": 2, "content": "<eos>", "single_word": false,
        "lstrip": false, "rstrip": false, "normalized": false, "special": true,
    }]);
    let tokenizer = tokenizer_with_added(&["Hello", "Ġworld", "<eos>"], byte_level(), added);
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    decoder.push(0).unwrap();

    let mut ids = vec![0];
    for _ in 0..500 {
        ids.push(2);
        decoder.push(2).unwrap();
        assert_eq!(decoder.text(), tokenizer.decode(&ids, true).unwrap());
        // The skipped tokens never enter the window at all.
        assert_eq!(decoder.window_len(), 1);
    }
    ids.push(1);
    assert_eq!(decoder.push(1).unwrap(), " world");
    assert_eq!(decoder.text(), tokenizer.decode(&ids, true).unwrap());
}

#[test]
fn clear_resets_the_stream() {
    let tokenizer = tokenizer(&["Hello", "Ġworld"], byte_level());
    let mut decoder = IncrementalDecoder::new(&tokenizer);
    decoder.push(0).unwrap();
    decoder.clear();
    assert_eq!(decoder.text(), "");
    assert!(decoder.tokens().is_empty());
    assert_eq!(decoder.push(1).unwrap(), " world");
}
