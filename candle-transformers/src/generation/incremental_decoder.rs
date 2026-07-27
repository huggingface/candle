//! Incremental detokenization for autoregressive decode loops.
//!
//! An autoregressive loop needs the text produced so far on every step, both to match stop
//! sequences and to stream deltas back to the caller. Re-decoding the whole token sequence on
//! every step is correct but quadratic in the generated length. Appending the decode of each
//! token is linear but wrong: a BPE/SentencePiece decode is *not* the concatenation of its
//! tokens' decodes. A multi-byte character can span several tokens (the prefix decodes to
//! U+FFFD, and a later token retroactively replaces it), and decoding a sub-sequence can gain
//! or lose a leading space.
//!
//! [`IncrementalDecoder`] keeps a bounded trailing window of tokens, re-decodes only that
//! window, and appends the difference against the *previous* window decode, so any artifact
//! caused by decoding a sub-sequence appears in both decodes and cancels out. The window is
//! only moved forward when the shorter decode is provably a suffix of the current one, and
//! only when the tokenizer's decoder has been shown to have a bounded context; otherwise the
//! decoder degrades to whole-sequence decoding rather than corrupting the output.
//!
//! The invariant is that [`IncrementalDecoder::text`] equals a whole-sequence decode of every
//! token pushed so far, at every step.
//!
//! ```no_run
//! use candle_transformers::generation::IncrementalDecoder;
//!
//! # fn main() -> candle::Result<()> {
//! # let tokenizer: tokenizers::Tokenizer = unimplemented!();
//! # let next_token = |_: &[u32]| -> candle::Result<u32> { unimplemented!() };
//! let mut decoder = IncrementalDecoder::new(&tokenizer);
//! let mut tokens = vec![1u32];
//! for _ in 0..128 {
//!     let token = next_token(&tokens)?;
//!     tokens.push(token);
//!     let delta = decoder.push(token)?;
//!     print!("{delta}");
//!     if decoder.text().ends_with("</s>") {
//!         break;
//!     }
//! }
//! print!("{}", decoder.finish());
//! # Ok(())
//! # }
//! ```

use candle::{Error, Result};
use serde_json::Value;
use std::collections::HashSet;
use tokenizers::Tokenizer;

/// The character `String::from_utf8_lossy` (and `ByteFallback`) emits for bytes that do not
/// form a valid UTF-8 sequence yet. A trailing run of these is never emitted, as a later token
/// may complete the sequence and replace them.
const REPLACEMENT: char = '\u{FFFD}';

/// Number of trailing tokens kept in the window after a successful re-anchor.
const MIN_WINDOW_TOKENS: usize = 8;

/// Window length that triggers a re-anchoring attempt. Re-anchoring costs one extra decode of
/// `MIN_WINDOW_TOKENS` tokens and happens at most once every
/// `MAX_WINDOW_TOKENS - MIN_WINDOW_TOKENS` pushes, so the amortized cost per token is constant.
const MAX_WINDOW_TOKENS: usize = 64;

/// Longest pattern rewritten by `tokenizers`' wordpiece `cleanup` helper (`" do not"`).
const CLEANUP_PATTERN_LEN: usize = 7;

/// What could be established about the tokenizer's decoder by inspecting its serialized form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecoderContext {
    /// Whether the decode of a token depends only on a bounded amount of surrounding context.
    /// When false, windowing is unsound and the whole sequence is re-decoded on every step.
    bounded: bool,
    /// Number of trailing bytes of the decoded text that a future token may still rewrite.
    unstable_tail: usize,
    /// Whether the decode of the last token depends on it being last, so that text only
    /// becomes stable once a further token has been pushed.
    lag_one_step: bool,
    /// Whether a `ByteFallback` renders the tokens. It groups a run of `<0xXX>` markers and
    /// converts the run as a whole, so an open run's text is not settled until a non-marker
    /// token closes it - one invalid byte turns the entire run into replacement characters,
    /// including the part that already decoded cleanly.
    byte_fallback: bool,
}

impl DecoderContext {
    const UNBOUNDED: Self = Self {
        bounded: false,
        unstable_tail: 0,
        lag_one_step: false,
        byte_fallback: false,
    };

    /// Resolve the bounded-context property of a tokenizer's decoder.
    ///
    /// `tokenizers` keeps the decoder fields private, so the decoder is inspected through its
    /// serialized form. Anything unrecognized is treated as unbounded.
    fn resolve(tokenizer: &Tokenizer) -> Self {
        let decoder = match tokenizer.get_decoder() {
            // Without a decoder `Tokenizer::decode` joins the tokens with a space, which is
            // strictly per-token.
            None => {
                return Self {
                    bounded: true,
                    unstable_tail: 0,
                    lag_one_step: false,
                    byte_fallback: false,
                }
            }
            Some(decoder) => decoder,
        };
        let value = match serde_json::to_value(decoder) {
            Ok(value) => value,
            Err(_) => return Self::UNBOUNDED,
        };
        let mut ctx = Self {
            bounded: true,
            unstable_tail: 0,
            lag_one_step: false,
            byte_fallback: false,
        };
        // `fused` tracks whether the decoders seen so far have collapsed the token list into a
        // single string. Before that point a decoder only ever sees one token at a time and
        // cannot rewrite across token boundaries; after it, string rewrites apply to the whole
        // decoded text and their reach has to be accounted for.
        let mut fused = false;
        if !scan_decoder(&value, &mut fused, &mut ctx) {
            return Self::UNBOUNDED;
        }
        ctx
    }
}

/// Walk a serialized decoder, accumulating how far a future token can reach back into the text
/// already produced. Returns false as soon as the reach cannot be bounded.
fn scan_decoder(value: &Value, fused: &mut bool, ctx: &mut DecoderContext) -> bool {
    let ty = match value.get("type").and_then(Value::as_str) {
        Some(ty) => ty,
        None => return false,
    };
    match ty {
        "Sequence" => match value.get("decoders").and_then(Value::as_array) {
            Some(decoders) => decoders.iter().all(|d| scan_decoder(d, fused, ctx)),
            None => false,
        },
        // Joins the token list into a single string.
        "Fuse" => {
            *fused = true;
            true
        }
        // Concatenates every token's bytes before a single lossy UTF-8 conversion, which is what
        // lets a multi-byte character span tokens and be fixed up retroactively. It also
        // collapses the list into one string.
        //
        // Applied to an already fused string it is unbounded: a token whose characters are not
        // all in the byte-level alphabet falls back to that token's raw UTF-8 bytes, so one
        // appended character can flip the whole accumulated string onto the fallback path and
        // rewrite the entire prefix.
        "ByteLevel" => {
            if *fused {
                return false;
            }
            *fused = true;
            true
        }
        // Groups runs of `<0xXX>` tokens before converting them to text, so an incomplete run
        // shows up as a trailing run of U+FFFD which a later token replaces.
        //
        // Applied to an already fused string it is unbounded: the `<0xXX>` shape is matched
        // against the whole accumulated string, so appending anything can retroactively turn a
        // decoded byte back into its literal marker.
        "ByteFallback" => {
            if *fused {
                return false;
            }
            ctx.byte_fallback = true;
            true
        }
        // Per-token, apart from the prepend scheme which only affects the first token, i.e. the
        // start of the text.
        "Metaspace" => true,
        "WordPiece" => {
            if *fused
                && value
                    .get("cleanup")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
            {
                ctx.unstable_tail += CLEANUP_PATTERN_LEN;
            }
            true
        }
        "CTC" => {
            if *fused {
                // The pad token is stripped from the string whatever `cleanup` says, so its
                // reach counts even when the cleanup patterns do not.
                let pad = value.get("pad_token").and_then(Value::as_str).unwrap_or("");
                ctx.unstable_tail += pad.len();
                if value
                    .get("cleanup")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
                {
                    ctx.unstable_tail += CLEANUP_PATTERN_LEN;
                }
                let delimiter = value
                    .get("word_delimiter_token")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                ctx.unstable_tail += delimiter.len();
            }
            true
        }
        "BPEDecoder" => {
            if *fused {
                // On a fused string every occurrence of the suffix is dropped, wherever it is,
                // so a suffix completed across pushes rewrites up to its own length of text.
                let suffix = value.get("suffix").and_then(Value::as_str).unwrap_or("");
                ctx.unstable_tail += suffix.len();
            } else {
                // The end-of-word suffix of the *last* token is dropped rather than turned into
                // a space, so the tail of the text changes as soon as another token is pushed.
                ctx.lag_one_step = true;
            }
            true
        }
        "Strip" => {
            if *fused {
                let stop = value.get("stop").and_then(Value::as_u64).unwrap_or(0) as usize;
                let content = value.get("content").and_then(Value::as_str).unwrap_or("");
                ctx.unstable_tail += stop * content.len();
            }
            true
        }
        "Replace" => {
            if !*fused {
                // Applied to each token independently, so it cannot reach across tokens.
                return true;
            }
            // Applied to the whole text: appending bytes can create a match straddling the
            // boundary, rewriting up to `pattern.len()` bytes already emitted. A regex pattern
            // can match arbitrarily far back, which is exactly the unbounded case.
            match value.get("pattern").and_then(|p| p.get("String")) {
                Some(Value::String(pattern)) => {
                    ctx.unstable_tail += pattern.len();
                    true
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// A detokenizer that turns a stream of token ids into a stream of text deltas in amortized
/// constant time per token.
///
/// See the [module documentation](self) for the approach.
pub struct IncrementalDecoder {
    tokenizer: Tokenizer,
    skip_special_tokens: bool,
    ctx: DecoderContext,
    /// Ids that `Tokenizer::decode` filters out before the decoder ever sees them.
    dropped: HashSet<u32>,
    /// Every token pushed so far.
    tokens: Vec<u32>,
    /// The tokens the next decode runs over: a trailing slice of `tokens`, with the ids that
    /// `decode` filters out left out. Dropping those is a no-op on the decode - the decoder is
    /// handed the exact same input either way - and it keeps a run of skipped special tokens
    /// from growing the window, since such a run offers no tail long enough to re-anchor on.
    window: Vec<u32>,
    /// Decode of `window`; always a suffix of `text`.
    window_text: String,
    /// Decode of the whole token sequence.
    text: String,
    /// Number of bytes of `text` already returned by `push`.
    emitted: usize,
    /// Where in `text` the currently open run of `<0xXX>` byte markers starts, if one is open.
    /// Everything from there on is still subject to being re-rendered by the next marker.
    open_byte_run: Option<usize>,
    /// Bytes of previously returned text invalidated by the last `push`.
    retracted: usize,
}

impl IncrementalDecoder {
    /// Create a decoder for `tokenizer`, resolving the bounded-context property of its decoder.
    ///
    /// The tokenizer is cloned; use [`IncrementalDecoder::from_tokenizer`] to avoid the clone.
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self::from_tokenizer(tokenizer.clone())
    }

    /// Create a decoder taking ownership of `tokenizer`.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        let ctx = DecoderContext::resolve(&tokenizer);
        let dropped = dropped_ids(&tokenizer, true);
        Self {
            tokenizer,
            skip_special_tokens: true,
            ctx,
            dropped,
            tokens: Vec::new(),
            window: Vec::new(),
            window_text: String::new(),
            text: String::new(),
            emitted: 0,
            open_byte_run: None,
            retracted: 0,
        }
    }

    /// Whether special tokens are dropped from the decoded text, `true` by default.
    pub fn with_skip_special_tokens(mut self, skip_special_tokens: bool) -> Self {
        self.skip_special_tokens = skip_special_tokens;
        self.dropped = dropped_ids(&self.tokenizer, skip_special_tokens);
        self.clear();
        self
    }

    /// Whether the tokenizer's decoder was shown to have a bounded context.
    ///
    /// When false the decoder is still correct, but it re-decodes the whole sequence on every
    /// push - quadratic in the number of generated tokens - and [`IncrementalDecoder::push`]
    /// streams nothing, since no part of the text can be shown to be settled. Use
    /// [`IncrementalDecoder::text`] to observe it as it grows, and
    /// [`IncrementalDecoder::finish`] to collect it at the end.
    pub fn is_windowed(&self) -> bool {
        self.ctx.bounded
    }

    /// Push a token and return the text that became available because of it.
    ///
    /// The returned slice may be empty: text that a further token could still rewrite is held
    /// back until it is settled - a trailing run of U+FFFD from an incomplete multi-byte
    /// character, an open run of `ByteFallback` byte markers, the reach of a rewriting decoder,
    /// or, for a decoder that is not [windowed](IncrementalDecoder::is_windowed), all of it.
    ///
    /// Emission is append-only: concatenating every value returned by `push`, followed by
    /// [`IncrementalDecoder::finish`], reproduces [`IncrementalDecoder::text`] exactly.
    pub fn push(&mut self, token: u32) -> Result<&str> {
        self.retracted = 0;
        self.tokens.push(token);
        if !self.dropped.contains(&token) {
            self.window.push(token);
        }

        let new_window = self.decode(&self.window)?;
        let committed = self.text.len() - self.window_text.len();
        // The delta is taken against the *previous* decode of the same window, so a leading
        // space gained or lost by decoding a sub-sequence is present in both and cancels.
        let common = committed + common_prefix_len(&self.window_text, &new_window);
        self.text.truncate(committed);
        self.text.push_str(&new_window);
        self.window_text = new_window;

        if self.ctx.bounded && self.window.len() >= MAX_WINDOW_TOKENS {
            self.reanchor()?;
        }
        self.track_byte_run(token, common);

        if common < self.emitted {
            self.retracted = self.emitted - common;
            self.emitted = common;
        }
        let start = self.emitted;
        self.emitted = self.emission_limit(common);
        Ok(&self.text[start..self.emitted])
    }

    /// The whole text decoded so far.
    ///
    /// This is always equal to decoding every pushed token in one go.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The text held back by [`IncrementalDecoder::push`] because a future token could still
    /// rewrite it.
    pub fn pending(&self) -> &str {
        &self.text[self.emitted..]
    }

    /// Release the text held back by [`IncrementalDecoder::push`], for use once generation is
    /// over. Any trailing U+FFFD is genuinely undecodable at that point.
    pub fn finish(&mut self) -> &str {
        let start = self.emitted;
        self.emitted = self.text.len();
        &self.text[start..]
    }

    /// Number of bytes of previously returned text invalidated by the last
    /// [`IncrementalDecoder::push`].
    ///
    /// This is a diagnostic, and is always zero: text is only ever emitted once it has been
    /// shown to be settled. A non-zero value means the rewrite reach of some decoder was
    /// under-estimated, i.e. a bug here rather than something a caller has to handle.
    pub fn retracted(&self) -> usize {
        self.retracted
    }

    /// The tokens pushed so far.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of tokens re-decoded by the next [`IncrementalDecoder::push`].
    ///
    /// This stays bounded for a windowed decoder, and grows with the sequence otherwise.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// The underlying tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Consume the decoder, returning the underlying tokenizer.
    pub fn into_tokenizer(self) -> Tokenizer {
        self.tokenizer
    }

    /// Reset the decoder, keeping the tokenizer and its resolved decoder context.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.window.clear();
        self.window_text.clear();
        self.text.clear();
        self.emitted = 0;
        self.open_byte_run = None;
        self.retracted = 0;
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        if tokens.is_empty() {
            // Some decoders (`BPEDecoder`) index the last token unconditionally.
            return Ok(String::new());
        }
        self.tokenizer
            .decode(tokens, self.skip_special_tokens)
            .map_err(|err| Error::Msg(format!("cannot decode: {err}")))
    }

    /// Try to move the window forward, keeping the text unchanged.
    ///
    /// The window is only moved when the shorter decode is provably a suffix of the current
    /// window decode, and long enough that a rewriting decoder cannot reach past it. When no
    /// split is acceptable the window simply keeps growing, degrading towards whole-sequence
    /// decoding rather than dropping or duplicating text.
    fn reanchor(&mut self) -> Result<()> {
        let min_tail = self.ctx.unstable_tail.max(1);
        let mut keep = MIN_WINDOW_TOKENS;
        while keep < self.window.len() {
            let start = self.window.len() - keep;
            let tail = self.decode(&self.window[start..])?;
            if tail.len() >= min_tail && self.window_text.ends_with(&tail) {
                self.window.drain(..start);
                self.window_text = tail;
                return Ok(());
            }
            keep *= 2;
        }
        Ok(())
    }

    /// Note which tokens are still inside an open `ByteFallback` run, so their text can be held
    /// back: the run is converted as a whole, so a later marker can re-render all of it.
    fn track_byte_run(&mut self, token: u32, common: usize) {
        // A dropped id never reaches the decoder, so it neither opens nor closes a run.
        if !self.ctx.byte_fallback || self.dropped.contains(&token) {
            return;
        }
        if !self.is_byte_marker(token) {
            self.open_byte_run = None;
        } else if self.open_byte_run.is_none() {
            self.open_byte_run = Some(common);
        }
    }

    /// Whether an id decodes to a `<0xXX>` marker, matching `ByteFallback`'s own test.
    fn is_byte_marker(&self, token: u32) -> bool {
        match self.tokenizer.id_to_token(token) {
            Some(t) => {
                t.len() == 6
                    && t.starts_with("<0x")
                    && t.ends_with('>')
                    && t.get(3..5)
                        .is_some_and(|hex| u8::from_str_radix(hex, 16).is_ok())
            }
            None => false,
        }
    }

    /// Largest prefix of `text` that no future token can rewrite.
    fn emission_limit(&self, common: usize) -> usize {
        // Nothing can be proven about an unbounded decoder, so nothing is emitted until
        // `finish`: this type never hands out text it may have to take back.
        if !self.ctx.bounded {
            return self.emitted;
        }
        let mut limit = self.text.len();
        // An incomplete multi-byte character decodes to U+FFFD, which a later token replaces.
        while limit > 0 && self.text[..limit].ends_with(REPLACEMENT) {
            limit -= REPLACEMENT.len_utf8();
        }
        limit = limit.saturating_sub(self.ctx.unstable_tail);
        while limit > 0 && !self.text.is_char_boundary(limit) {
            limit -= 1;
        }
        if self.ctx.lag_one_step {
            limit = limit.min(common);
        }
        if let Some(run_start) = self.open_byte_run {
            limit = limit.min(run_start);
        }
        limit.max(self.emitted)
    }
}

/// Ids that `Tokenizer::decode` drops before running the decoder: special tokens when they are
/// being skipped, and ids that map to no token at all.
///
/// Conservative on purpose - an id missing from this set only means the window advances less
/// eagerly, never that it advances when it should not.
fn dropped_ids(tokenizer: &Tokenizer, skip_special_tokens: bool) -> HashSet<u32> {
    if !skip_special_tokens {
        return HashSet::new();
    }
    tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, token)| token.special)
        .map(|(id, _)| id)
        .collect()
}

/// Length in bytes of the longest common prefix of `a` and `b`, on a character boundary.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut len = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        len += ca.len_utf8();
    }
    len
}
