//! Writes Apple `.metallib` (MTLB) containers.
//!
//! Replaces `xcrun metallib`, which cannot be shelled out to on iOS or in a sandboxed app.
//! A metallib is a tag/length/value container around AIR bitcode. The loader accepts any
//! AIR bitcode the compiler service can read, so the container is the only Apple-specific
//! artifact needed. Output is byte-identical to `xcrun metallib` for a single-function
//! library.
//!
//! Layout, little-endian, offsets from file start. The 12 bytes from 0x04 come from
//! [`crate::profile::TargetProfile`].
//!
//! ```text
//!   0x00  'MTLB'
//!   0x04  u16 format word         (0x8001 on macOS, 0x0001 elsewhere)
//!   0x06  u16 (0x0002)
//!   0x08  u16 format code         (9 for AIR 2.8)
//!   0x0a  u8  (0)
//!   0x0b  u8  platform tag        (0x81 macOS, 0x82 iOS, 0x87 iOS simulator, ...)
//!   0x0c  u16 deployment target major, 0x0e  u16 minor
//!   0x10  u64 total file size
//!   0x18  u64 function-list offset      0x20  u64 function-list size
//!   0x28  u64 public-metadata offset    0x30  u64 public-metadata size
//!   0x38  u64 private-metadata offset   0x40  u64 private-metadata size
//!   0x48  u64 bitcode offset            0x50  u64 bitcode size
//!   0x58  u32 function count, then one entry per function, then b"ENDT"
//! ```
//!
//! Each function entry is `u32 entry_size` (counting itself) followed by
//! `TAG | u16 len | payload` records terminated by `ENDT`:
//! `NAME` (NUL-terminated), `TYPE` (u8 function kind), `HASH` (SHA-256 of the
//! bitcode), `MDSZ` (u64 bitcode size), `OFFT` (3 x u64 section offsets, zero for a
//! single-function library) and `VERS` (AIR then language version, as u16 pairs).
//!
//! Both metadata sections are the empty form: `u32 4` followed by `ENDT`.

use crate::profile::TargetProfile;

/// Header size, and therefore the offset of the function list.
const HEADER_LEN: usize = 0x58;

/// What a function in the library is. The value is the `TYPE` tag's payload.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FunctionKind {
    Vertex = 0,
    Fragment = 1,
    Kernel = 2,
}

/// A function to record in the library's function list.
#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub kind: FunctionKind,
}

impl Function {
    pub fn kernel(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: FunctionKind::Kernel,
        }
    }
}

fn tag(out: &mut Vec<u8>, name: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(name);
    out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    out.extend_from_slice(payload);
}

/// Wrap AIR `bitcode` (as produced for the `air64` target, with or without the
/// 0x0b17c0de wrapper) into a metallib exposing `functions`.
///
/// `profile` must be the one the bitcode was emitted for: the header and the `VERS`
/// tag repeat the target's platform and AIR version, and the loader checks them
/// against the module.
pub fn wrap(bitcode: &[u8], functions: &[Function], profile: &TargetProfile) -> Vec<u8> {
    let digest = sha256(bitcode);

    let mut entries = Vec::new();
    for f in functions {
        let mut e = Vec::new();
        let mut name = f.name.clone().into_bytes();
        name.push(0);
        tag(&mut e, b"NAME", &name);
        tag(&mut e, b"TYPE", &[f.kind as u8]);
        tag(&mut e, b"HASH", &digest);
        tag(&mut e, b"MDSZ", &(bitcode.len() as u64).to_le_bytes());
        tag(&mut e, b"OFFT", &[0u8; 24]);
        let mut vers = Vec::with_capacity(8);
        for v in [
            profile.air_version.0,
            profile.air_version.1,
            profile.language_version.0,
            profile.language_version.1,
        ] {
            vers.extend_from_slice(&(v as u16).to_le_bytes());
        }
        tag(&mut e, b"VERS", &vers);
        e.extend_from_slice(b"ENDT");
        // entry_size counts the size field itself.
        entries.extend_from_slice(&((e.len() + 4) as u32).to_le_bytes());
        entries.extend_from_slice(&e);
    }

    let mut func_list = Vec::new();
    func_list.extend_from_slice(&(functions.len() as u32).to_le_bytes());
    func_list.extend_from_slice(&entries);
    func_list.extend_from_slice(b"ENDT");
    // The recorded size covers only the entries: not the count prefix, not the
    // list terminator.
    let func_list_size = func_list.len() - 8;

    // Empty metadata section: a size word followed by the terminator.
    let empty_md = {
        let mut m = Vec::with_capacity(8);
        m.extend_from_slice(&4u32.to_le_bytes());
        m.extend_from_slice(b"ENDT");
        m
    };

    let func_off = HEADER_LEN;
    let pub_off = func_off + func_list.len();
    let priv_off = pub_off + empty_md.len();
    let bc_off = priv_off + empty_md.len();
    let total = bc_off + bitcode.len();

    let mut out = vec![0u8; HEADER_LEN];
    out[0..4].copy_from_slice(b"MTLB");
    out[0x04..0x10].copy_from_slice(&profile.container_header());
    let mut put =
        |off: usize, v: usize| out[off..off + 8].copy_from_slice(&(v as u64).to_le_bytes());
    put(0x10, total);
    put(0x18, func_off);
    put(0x20, func_list_size);
    put(0x28, pub_off);
    put(0x30, empty_md.len());
    put(0x38, priv_off);
    put(0x40, empty_md.len());
    put(0x48, bc_off);
    put(0x50, bitcode.len());

    out.extend_from_slice(&func_list);
    out.extend_from_slice(&empty_md);
    out.extend_from_slice(&empty_md);
    out.extend_from_slice(bitcode);
    debug_assert_eq!(out.len(), total);
    out
}

/// SHA-256, so the container writer carries no dependency of its own.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (dst, src) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *dst = dst.wrapping_add(src);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Spans more than one block, to exercise the padding/length path.
        assert_eq!(
            hex(&sha256(b"The quick brown fox jumps over the lazy dog")),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// The header must describe a self-consistent file: every section inside it,
    /// laid end to end, with the bitcode last.
    #[test]
    fn header_is_self_consistent() {
        let bc = vec![0xABu8; 777];
        let lib = wrap(
            &bc,
            &[Function::kernel("my_kernel")],
            &TargetProfile::host(),
        );
        let g = |o: usize| u64::from_le_bytes(lib[o..o + 8].try_into().unwrap()) as usize;
        assert_eq!(&lib[0..4], b"MTLB");
        assert_eq!(g(0x10), lib.len(), "file size");
        assert_eq!(g(0x18), 0x58, "function list starts after the header");
        assert_eq!(g(0x48) + g(0x50), lib.len(), "bitcode runs to EOF");
        assert_eq!(g(0x50), bc.len(), "bitcode size");
        assert_eq!(&lib[g(0x48)..], &bc[..], "bitcode stored verbatim");
        // The hash must cover the bitcode payload.
        let hpos = lib.windows(4).position(|w| w == b"HASH").unwrap();
        assert_eq!(&lib[hpos + 6..hpos + 38], &sha256(&bc)[..]);
        // The name is recorded NUL-terminated.
        let npos = lib.windows(4).position(|w| w == b"NAME").unwrap();
        assert_eq!(&lib[npos + 6..npos + 6 + 10], b"my_kernel\0");
    }

    #[test]
    fn multiple_functions_are_listed() {
        let bc = vec![1u8; 16];
        let lib = wrap(
            &bc,
            &[Function::kernel("a"), Function::kernel("b")],
            &TargetProfile::host(),
        );
        assert_eq!(
            u32::from_le_bytes(lib[0x58..0x5C].try_into().unwrap()),
            2,
            "function count"
        );
        let g = |o: usize| u64::from_le_bytes(lib[o..o + 8].try_into().unwrap()) as usize;
        assert_eq!(g(0x48) + g(0x50), lib.len());
    }
}
