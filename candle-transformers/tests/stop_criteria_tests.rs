use candle_transformers::generation::{FinishReason, StopCriteria};

#[test]
fn holds_back_the_longest_stop_sequence_on_a_char_boundary() {
    let criteria = StopCriteria::new(["<end>".to_owned(), "éx".to_owned()], vec![]);
    assert_eq!(criteria.hold(), 4);

    // The byte cut is in the middle of `é`; retain that whole character instead.
    assert_eq!(criteria.safe_emit_end("aéxyz"), 1);
    assert_eq!(criteria.safe_emit_end("hello"), 1);
}

#[test]
fn finds_the_earliest_stop_regardless_of_configuration_order() {
    let criteria = StopCriteria::new(["later".to_owned(), "first".to_owned()], vec![]);
    let text = "first, then later";
    assert_eq!(criteria.matched(text), Some(0));
    assert_eq!(criteria.safe_emit_end(text), 0);

    // A stop that completes across token boundaries must not be streamed.
    let split = StopCriteria::new(["<stop>".to_owned()], vec![]);
    assert_eq!(split.safe_emit_end("answer<st"), 4);
    assert_eq!(split.matched("answer<stop>"), Some(6));
    assert_eq!(split.safe_emit_end("answer<stop>"), 6);
}

#[test]
fn reports_model_stop_before_budget_exhaustion() {
    let criteria = StopCriteria::new(["<stop>".to_owned()], vec![2]);
    assert!(criteria.is_eos(2));
    assert!(!criteria.is_eos(3));

    assert_eq!(
        criteria.finish_reason(2, "answer", true),
        Some(FinishReason::Stop)
    );
    assert_eq!(
        criteria.finish_reason(3, "answer<stop>", true),
        Some(FinishReason::Stop)
    );
    assert_eq!(
        criteria.finish_reason(3, "answer", true),
        Some(FinishReason::Length)
    );
    assert_eq!(criteria.finish_reason(3, "answer", false), None);
}

#[test]
fn ignores_empty_stop_sequences() {
    let criteria = StopCriteria::new([String::new()], vec![]);
    assert_eq!(criteria.hold(), 0);
    assert_eq!(criteria.matched("answer"), None);
    assert_eq!(criteria.safe_emit_end("answer"), 6);
}
