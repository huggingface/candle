pub fn decode(tokens: &[usize], vocabulary: &[String]) -> String {
    tokens
        .iter()
        .filter_map(|&id| vocabulary.get(id))
        .map(|s| s.replace('▁', " "))
        .collect::<String>()
}
