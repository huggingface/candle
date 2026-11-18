use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct AlignedToken {
    pub id: usize,
    pub text: String,
    pub start: f64,
    pub duration: f64,
    pub confidence: f64,
    pub end: f64,
}

impl AlignedToken {
    pub fn new(id: usize, text: String, start: f64, duration: f64, confidence: f64) -> Self {
        let end = start + duration;
        Self {
            id,
            text,
            start,
            duration,
            confidence,
            end,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlignedSentence {
    pub text: String,
    pub tokens: Vec<AlignedToken>,
    pub start: f64,
    pub end: f64,
    pub duration: f64,
    pub confidence: f64,
}

impl AlignedSentence {
    pub fn new(text: String, mut tokens: Vec<AlignedToken>) -> Self {
        tokens.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(Ordering::Equal));
        let start = tokens.first().map(|t| t.start).unwrap_or(0.0);
        let end = tokens.last().map(|t| t.end).unwrap_or(0.0);
        let duration = end - start;
        let confidence = if tokens.is_empty() {
            1.0
        } else {
            let log_sum = tokens
                .iter()
                .map(|t| (t.confidence + 1e-10).ln())
                .sum::<f64>();
            (log_sum / tokens.len() as f64).exp()
        };
        Self {
            text,
            tokens,
            start,
            end,
            duration,
            confidence,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlignedResult {
    pub text: String,
    pub sentences: Vec<AlignedSentence>,
}

impl AlignedResult {
    pub fn new(text: String, sentences: Vec<AlignedSentence>) -> Self {
        Self {
            text: text.trim().to_string(),
            sentences,
        }
    }

    pub fn tokens(&self) -> Vec<AlignedToken> {
        self.sentences
            .iter()
            .flat_map(|s| s.tokens.clone())
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SentenceConfig {
    pub max_words: Option<usize>,
    pub silence_gap: Option<f64>,
    pub max_duration: Option<f64>,
}

pub fn tokens_to_sentences(
    tokens: &[AlignedToken],
    config: &SentenceConfig,
) -> Vec<AlignedSentence> {
    let mut sentences = Vec::new();
    let mut current_tokens: Vec<AlignedToken> = Vec::new();

    for (idx, token) in tokens.iter().enumerate() {
        current_tokens.push(token.clone());

        let is_punctuation = token.text.contains('!')
            || token.text.contains('?')
            || token.text.contains('。')
            || token.text.contains('？')
            || token.text.contains('！')
            || (token.text.contains('.')
                && (idx == tokens.len() - 1
                    || tokens
                        .get(idx + 1)
                        .map(|t| t.text.contains(' '))
                        .unwrap_or(false)));

        let is_word_limit = if let Some(max_words) = config.max_words {
            if idx != tokens.len() - 1 {
                let words_in_current = current_tokens
                    .iter()
                    .filter(|t| t.text.contains(' '))
                    .count();
                let next_is_word = tokens
                    .get(idx + 1)
                    .map(|t| t.text.contains(' '))
                    .unwrap_or(false);
                words_in_current + if next_is_word { 1 } else { 0 } > max_words
            } else {
                false
            }
        } else {
            false
        };

        let is_long_silence = if let Some(gap) = config.silence_gap {
            if idx != tokens.len() - 1 {
                tokens[idx + 1].start - token.end >= gap
            } else {
                false
            }
        } else {
            false
        };

        let is_over_duration = if let Some(max_dur) = config.max_duration {
            let start_time = current_tokens
                .first()
                .map(|t| t.start)
                .unwrap_or(token.start);
            token.end - start_time >= max_dur
        } else {
            false
        };

        if is_punctuation || is_word_limit || is_long_silence || is_over_duration {
            let sentence_text = current_tokens.iter().map(|t| t.text.clone()).collect();
            sentences.push(AlignedSentence::new(sentence_text, current_tokens));
            current_tokens = Vec::new();
        }
    }

    if !current_tokens.is_empty() {
        let sentence_text = current_tokens.iter().map(|t| t.text.clone()).collect();
        sentences.push(AlignedSentence::new(sentence_text, current_tokens));
    }

    sentences
}

pub fn sentences_to_result(sentences: &[AlignedSentence]) -> AlignedResult {
    let text = sentences.iter().map(|s| s.text.clone()).collect::<String>();
    AlignedResult::new(text, sentences.to_vec())
}

pub fn merge_longest_contiguous(
    a: &[AlignedToken],
    b: &[AlignedToken],
    overlap_duration: f64,
) -> Result<Vec<AlignedToken>, String> {
    if a.is_empty() {
        return Ok(b.to_vec());
    }
    if b.is_empty() {
        return Ok(a.to_vec());
    }

    let a_end_time = a.last().unwrap().end;
    let b_start_time = b.first().unwrap().start;

    if a_end_time <= b_start_time {
        let mut out = a.to_vec();
        out.extend_from_slice(b);
        return Ok(out);
    }

    let overlap_a: Vec<_> = a
        .iter()
        .cloned()
        .filter(|t| t.end > b_start_time - overlap_duration)
        .collect();
    let overlap_b: Vec<_> = b
        .iter()
        .cloned()
        .filter(|t| t.start < a_end_time + overlap_duration)
        .collect();

    let enough_pairs = overlap_a.len() / 2;

    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        let cutoff_time = (a_end_time + b_start_time) / 2.0;
        let mut out: Vec<AlignedToken> =
            a.iter().cloned().filter(|t| t.end <= cutoff_time).collect();
        out.extend(b.iter().cloned().filter(|t| t.start >= cutoff_time));
        return Ok(out);
    }

    let mut best_contiguous: Vec<(usize, usize)> = Vec::new();
    for i in 0..overlap_a.len() {
        for j in 0..overlap_b.len() {
            if overlap_a[i].id == overlap_b[j].id
                && (overlap_a[i].start - overlap_b[j].start).abs() < overlap_duration / 2.0
            {
                let mut current: Vec<(usize, usize)> = Vec::new();
                let mut k = i;
                let mut l = j;
                while k < overlap_a.len()
                    && l < overlap_b.len()
                    && overlap_a[k].id == overlap_b[l].id
                    && (overlap_a[k].start - overlap_b[l].start).abs() < overlap_duration / 2.0
                {
                    current.push((k, l));
                    k += 1;
                    l += 1;
                }
                if current.len() > best_contiguous.len() {
                    best_contiguous = current;
                }
            }
        }
    }

    if best_contiguous.len() < enough_pairs {
        return Err(format!("no pairs exceeding {enough_pairs}"));
    }

    let a_start_idx = a.len() - overlap_a.len();
    let lcs_indices_a: Vec<usize> = best_contiguous
        .iter()
        .map(|(i, _)| a_start_idx + i)
        .collect();
    let lcs_indices_b: Vec<usize> = best_contiguous.iter().map(|(_, j)| *j).collect();

    let mut result = Vec::new();
    result.extend_from_slice(&a[..lcs_indices_a[0]]);

    for i in 0..best_contiguous.len() {
        let idx_a = lcs_indices_a[i];
        let idx_b = lcs_indices_b[i];

        result.push(a[idx_a].clone());

        if i < best_contiguous.len() - 1 {
            let next_idx_a = lcs_indices_a[i + 1];
            let next_idx_b = lcs_indices_b[i + 1];
            let gap_tokens_a = &a[idx_a + 1..next_idx_a];
            let gap_tokens_b = &b[idx_b + 1..next_idx_b];
            if gap_tokens_b.len() > gap_tokens_a.len() {
                result.extend_from_slice(gap_tokens_b);
            } else {
                result.extend_from_slice(gap_tokens_a);
            }
        }
    }

    result.extend_from_slice(&b[lcs_indices_b.last().unwrap() + 1..]);
    Ok(result)
}

pub fn merge_longest_common_subsequence(
    a: &[AlignedToken],
    b: &[AlignedToken],
    overlap_duration: f64,
) -> Vec<AlignedToken> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let a_end_time = a.last().unwrap().end;
    let b_start_time = b.first().unwrap().start;

    if a_end_time <= b_start_time {
        let mut out = a.to_vec();
        out.extend_from_slice(b);
        return out;
    }

    let overlap_a: Vec<_> = a
        .iter()
        .cloned()
        .filter(|t| t.end > b_start_time - overlap_duration)
        .collect();
    let overlap_b: Vec<_> = b
        .iter()
        .cloned()
        .filter(|t| t.start < a_end_time + overlap_duration)
        .collect();

    if overlap_a.len() < 2 || overlap_b.len() < 2 {
        let cutoff_time = (a_end_time + b_start_time) / 2.0;
        let mut out: Vec<AlignedToken> =
            a.iter().cloned().filter(|t| t.end <= cutoff_time).collect();
        out.extend(b.iter().cloned().filter(|t| t.start >= cutoff_time));
        return out;
    }

    let mut dp = vec![vec![0usize; overlap_b.len() + 1]; overlap_a.len() + 1];
    for i in 1..=overlap_a.len() {
        for j in 1..=overlap_b.len() {
            if overlap_a[i - 1].id == overlap_b[j - 1].id
                && (overlap_a[i - 1].start - overlap_b[j - 1].start).abs() < overlap_duration / 2.0
            {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let mut lcs_pairs = Vec::new();
    let mut i = overlap_a.len();
    let mut j = overlap_b.len();
    while i > 0 && j > 0 {
        if overlap_a[i - 1].id == overlap_b[j - 1].id
            && (overlap_a[i - 1].start - overlap_b[j - 1].start).abs() < overlap_duration / 2.0
        {
            lcs_pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    lcs_pairs.reverse();

    if lcs_pairs.is_empty() {
        let cutoff_time = (a_end_time + b_start_time) / 2.0;
        let mut out: Vec<AlignedToken> =
            a.iter().cloned().filter(|t| t.end <= cutoff_time).collect();
        out.extend(b.iter().cloned().filter(|t| t.start >= cutoff_time));
        return out;
    }

    let a_start_idx = a.len() - overlap_a.len();
    let lcs_indices_a: Vec<usize> = lcs_pairs.iter().map(|(i, _)| a_start_idx + i).collect();
    let lcs_indices_b: Vec<usize> = lcs_pairs.iter().map(|(_, j)| *j).collect();

    let mut result = Vec::new();
    result.extend_from_slice(&a[..lcs_indices_a[0]]);
    for i in 0..lcs_pairs.len() {
        let idx_a = lcs_indices_a[i];
        let idx_b = lcs_indices_b[i];
        result.push(a[idx_a].clone());

        if i < lcs_pairs.len() - 1 {
            let next_idx_a = lcs_indices_a[i + 1];
            let next_idx_b = lcs_indices_b[i + 1];
            let gap_tokens_a = &a[idx_a + 1..next_idx_a];
            let gap_tokens_b = &b[idx_b + 1..next_idx_b];
            if gap_tokens_b.len() > gap_tokens_a.len() {
                result.extend_from_slice(gap_tokens_b);
            } else {
                result.extend_from_slice(gap_tokens_a);
            }
        }
    }

    result.extend_from_slice(&b[lcs_indices_b.last().unwrap() + 1..]);
    result
}
