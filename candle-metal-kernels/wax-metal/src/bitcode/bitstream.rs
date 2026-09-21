//! LLVM bitstream writer.
//!
//! Little endian bits packed into 32-bit words. Structure comes from abbreviated ids at the
//! current width (2 bits at top, wider in blocks).
//! 0 END_BLOCK
//! 1 ENTER_SUBBLOCK
//! 2 DEFINE_ABBREV
//! 3 UNABBREV_RECORD
//! 4.. application-defined
//!
//! Records are UNABBREV_RECORD wherever possible.
//! Exception is blobs, which can not be unabbreviated ([`BitWriter::define_blob_abbr`]).
//!
//! Reference: LLVM "Bitcode File Format", bitstream container sections.

/// Abbreviation IDs.
pub const END_BLOCK: u32 = 0;
pub const ENTER_SUBBLOCK: u32 = 1;
pub const UNABBREV_RECORD: u32 = 3;

/// Bit-level writer. Bits fill each 32-bit word from the least significant end.
pub struct BitWriter {
    words: Vec<u32>,
    /// Bits not yet flushed into `words`, held in low `cur_bits` positions.
    cur: u32,
    cur_bits: u32,
    /// Width in bits of an abbreviation id at the current nested level.
    abbr_width: u32,
    /// Block (`abbr_width`, `words.len()`, `next_abbr`).
    block_stack: Vec<(u32, usize, u32)>,
    /// Next `DEFINE_ABBREV` id. `END_BLOCK`, `ENTER_SUBBLOCK`, `DEFINE_ABBREV`, and
    /// `UNABBREV_RECORD` precede it.
    next_abbr: u32,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            cur: 0,
            cur_bits: 0,
            // Outermost level uses 2-bit abbreviation ids.
            abbr_width: 2,
            block_stack: Vec::new(),
            next_abbr: 4,
        }
    }

    /// Emit `val` in exactly `width` bits (width <= 32).
    pub fn fixed(&mut self, val: u32, width: u32) {
        debug_assert!(width <= 32);
        if width == 0 {
            return;
        }
        let val = if width == 32 {
            val
        } else {
            val & ((1u32 << width) - 1)
        };
        let room = 32 - self.cur_bits;
        if width < room {
            self.cur |= val << self.cur_bits;
            self.cur_bits += width;
        } else {
            // Fill current word, flush, keep remainder.
            self.cur |= val << self.cur_bits;
            self.words.push(self.cur);
            let consumed = room;
            self.cur = if consumed == 32 { 0 } else { val >> consumed };
            self.cur_bits = width - consumed;
        }
    }

    pub fn fixed64(&mut self, val: u64, width: u32) {
        if width <= 32 {
            self.fixed(val as u32, width);
        } else {
            self.fixed(val as u32, 32);
            self.fixed((val >> 32) as u32, width - 32);
        }
    }

    pub fn vbr(&mut self, mut val: u64, width: u32) {
        debug_assert!(width > 1);
        let payload = width - 1;
        let cont = 1u32 << payload;
        let mask = cont as u64 - 1;
        loop {
            let chunk = (val & mask) as u32;
            val >>= payload;
            if val == 0 {
                self.fixed(chunk, width);
                return;
            }
            self.fixed(chunk | cont, width);
        }
    }

    /// Pad with zero bits up to the next 32-bit boundary.
    pub fn align32(&mut self) {
        if self.cur_bits > 0 {
            self.words.push(self.cur);
            self.cur = 0;
            self.cur_bits = 0;
        }
    }

    /// Open a subblock. `abbr_width` is the ID width used inside it.
    pub fn enter_block(&mut self, block_id: u32, abbr_width: u32) {
        self.fixed(ENTER_SUBBLOCK, self.abbr_width);
        self.vbr(block_id as u64, 8);
        self.vbr(abbr_width as u64, 4);
        self.align32();
        // Block length in 32-bit words, patched by `end_block`.
        let len_pos = self.words.len();
        self.words.push(0);
        self.block_stack
            .push((self.abbr_width, len_pos, self.next_abbr));
        self.abbr_width = abbr_width;
        self.next_abbr = 4;
    }

    pub fn end_block(&mut self) {
        self.fixed(END_BLOCK, self.abbr_width);
        self.align32();
        let (outer, len_pos, outer_abbr) = self
            .block_stack
            .pop()
            .expect("end_block without enter_block");
        self.abbr_width = outer;
        self.next_abbr = outer_abbr;
        // Length excludes the placeholder word itself.
        self.words[len_pos] = (self.words.len() - len_pos - 1) as u32;
    }

    /// Emit `code` with `ops` as an UNABBREV_RECORD.
    pub fn record(&mut self, code: u32, ops: &[u64]) {
        self.fixed(UNABBREV_RECORD, self.abbr_width);
        self.vbr(code as u64, 6);
        self.vbr(ops.len() as u64, 6);
        for &op in ops {
            self.vbr(op, 6);
        }
    }

    /// Emit an LLVM string record.
    pub fn record_str(&mut self, code: u32, s: &str) {
        let ops: Vec<u64> = s.bytes().map(|b| b as u64).collect();
        self.record(code, &ops);
    }

    /// Declare abbreviation for `[literal(code), blob]` and return id.
    pub fn define_blob_abbr(&mut self, code: u32) -> u32 {
        const DEFINE_ABBREV: u32 = 2;
        self.fixed(DEFINE_ABBREV, self.abbr_width);
        self.vbr(2, 5);
        // literal(code)
        self.fixed(1, 1);
        self.vbr(code as u64, 8);
        // blob
        self.fixed(0, 1);
        self.fixed(5, 3);

        let id = self.next_abbr;
        self.next_abbr += 1;
        id
    }

    /// Declare abbreviation for `[literal(code), vbr6, vbr6, blob]`.
    /// The `METADATA_STRINGS` shape.
    pub fn define_strings_abbr(&mut self, code: u32) -> u32 {
        const DEFINE_ABBREV: u32 = 2;
        self.fixed(DEFINE_ABBREV, self.abbr_width);
        self.vbr(4, 5);
        // literal(code)
        self.fixed(1, 1);
        self.vbr(code as u64, 8);
        // vbr6, vbr6
        for _ in 0..2 {
            self.fixed(0, 1);
            self.fixed(2, 3);
            self.vbr(6, 5);
        }
        // blob
        self.fixed(0, 1);
        self.fixed(5, 3);

        let id = self.next_abbr;
        self.next_abbr += 1;
        id
    }

    /// Emit a record declared by [`BitWriter::define_strings_abbr`].
    pub fn strings_record(&mut self, abbr_id: u32, count: u64, offset: u64, blob: &[u8]) {
        self.fixed(abbr_id, self.abbr_width);
        self.vbr(count, 6);
        self.vbr(offset, 6);
        self.vbr(blob.len() as u64, 6);
        self.align32();
        for &b in blob {
            self.fixed(b as u32, 8);
        }
        self.align32();
    }

    /// Emit a record using the abbreviation [`BitWriter::define_blob_abbr`] declares.
    pub fn blob_record(&mut self, abbr_id: u32, blob: &[u8]) {
        self.fixed(abbr_id, self.abbr_width);
        self.vbr(blob.len() as u64, 6);
        self.align32();
        for &b in blob {
            self.fixed(b as u32, 8);
        }
        // Blob payloads are padded to word boundary.
        self.align32();
    }

    /// Finish and return the bitstream bytes. All blocks must be closed.
    pub fn finish(mut self) -> Vec<u8> {
        assert!(self.block_stack.is_empty(), "unclosed block");
        self.align32();
        let mut out = Vec::with_capacity(self.words.len() * 4);
        for w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Current bit position.
    pub fn bit_pos(&self) -> usize {
        self.words.len() * 32 + self.cur_bits as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bits fill words from the low end, so two 4-bit values pack into one byte with
    /// the first value in the low nibble.
    #[test]
    fn fixed_packs_little_endian_lsb_first() {
        let mut w = BitWriter::new();
        w.fixed(0xA, 4);
        w.fixed(0x3, 4);
        assert_eq!(w.finish()[0], 0x3A);
    }

    #[test]
    fn fixed_spans_word_boundary() {
        let mut w = BitWriter::new();
        w.fixed(0xFFFF_FFFF, 32);
        w.fixed(0b101, 3);
        let b = w.finish();
        assert_eq!(&b[0..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(b[4], 0b101);
    }

    /// VBR6: values under 32 fit one chunk. Larger values set continuation bit.
    #[test]
    fn vbr_encodes_continuation() {
        let mut w = BitWriter::new();
        w.vbr(5, 6);
        assert_eq!(w.finish()[0] & 0x3F, 5);

        let mut w = BitWriter::new();
        w.vbr(32, 6);
        let b = w.finish();
        let first = b[0] & 0x3F;
        assert_eq!(
            first, 0b100000,
            "low chunk is 0 with the continuation bit set"
        );
    }

    #[test]
    fn block_length_is_patched_in_words() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        w.record(1, &[2]);
        w.end_block();
        let bytes = w.finish();
        // Word 0: header. Word 1: length. Then block body.
        let len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        assert_eq!(len, bytes.len() / 4 - 2, "length counts the body words");
    }

    #[test]
    #[should_panic(expected = "unclosed block")]
    fn finish_rejects_unclosed_block() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        let _ = w.finish();
    }
}
