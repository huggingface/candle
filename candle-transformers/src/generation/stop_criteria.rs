//! Stop-sequence matching and finish reasons for autoregressive decode loops.
//!
//! Stop sequences are strings, rather than token ids: a sequence can span token boundaries or
//! use a tokenization different from the one that produced the completion.  [`StopCriteria`]
//! therefore operates on the text maintained by [`super::IncrementalDecoder`].

/// Why a generation finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model produced an EOS token or a configured stop sequence.
    Stop,
    /// The configured token budget was spent before the model stopped.
    Length,
}

/// Text and token conditions that end an autoregressive generation.
///
/// A decode loop should retain [`Self::hold`] bytes of its decoded text before streaming it.
/// [`Self::safe_emit_end`] returns the corresponding character-boundary offset, and accounts
/// for an already matched stop sequence.  This lets a stop sequence spanning two tokens be
/// detected without sending any part of it to the consumer.
#[derive(Clone, Debug)]
pub struct StopCriteria {
    stop: Vec<String>,
    eos: Vec<u32>,
    hold: usize,
}

impl StopCriteria {
    /// Creates criteria from text stop sequences and EOS token ids.
    ///
    /// Empty stop sequences are ignored: they would match before any generated text and do not
    /// describe a useful stopping condition.
    pub fn new(stop: impl IntoIterator<Item = String>, eos: Vec<u32>) -> Self {
        let stop: Vec<_> = stop.into_iter().filter(|stop| !stop.is_empty()).collect();
        let hold = stop
            .iter()
            .map(|stop| stop.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        Self { stop, eos, hold }
    }

    /// Number of trailing bytes that must be retained while looking for a stop sequence.
    pub fn hold(&self) -> usize {
        self.hold
    }

    /// Returns whether `token` is an EOS token configured for this generation.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }

    /// Returns the byte offset of the earliest configured stop sequence in `text`.
    ///
    /// When several sequences match, position wins over configuration order.  This avoids
    /// truncating at a later stop merely because it appeared earlier in the configuration.
    pub fn matched(&self, text: &str) -> Option<usize> {
        self.stop.iter().filter_map(|stop| text.find(stop)).min()
    }

    /// Returns the largest character-boundary offset that may be streamed from `text` now.
    ///
    /// If a stop sequence has matched, its beginning is returned so that the sequence itself is
    /// not emitted.  Otherwise this retains [`Self::hold`] trailing bytes, rounded down to a
    /// UTF-8 character boundary.
    pub fn safe_emit_end(&self, text: &str) -> usize {
        if let Some(matched) = self.matched(text) {
            return matched;
        }
        let mut end = text.len().saturating_sub(self.hold);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        end
    }

    /// Determines a known finish reason after a token has been decoded.
    ///
    /// Model-controlled endings win when they land on the last allowed token: an EOS token or a
    /// stop sequence at that boundary is [`FinishReason::Stop`], not `Length`.  If the loop
    /// ended for a reason outside these criteria, do not call this method and report no reason.
    pub fn finish_reason(
        &self,
        token: u32,
        text: &str,
        budget_exhausted: bool,
    ) -> Option<FinishReason> {
        if self.is_eos(token) || self.matched(text).is_some() {
            Some(FinishReason::Stop)
        } else if budget_exhausted {
            Some(FinishReason::Length)
        } else {
            None
        }
    }
}
