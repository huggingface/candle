//! Writing Apple `.metallib` (MTLB) files.
//!
//! Feature subset of `xcrun metallib`, but works on iOS and in sandboxed environments.
//! Metallib is a `tag | length | value` wrapper around AIR bitcode. Our output is byte-identical
//! to `xcrun metallib` for a single function library.
//!
//! Every section but the header is built from `tag | u16 length | payload`, where a tag is
//! four ASCII bytes, and is closed by `ENDT`.
//!
//! Metallib layout
//! ```text
//! 0x00  b"MTLB"
//! 0x04  u16 format word                 (0x8001 on macOS, 0x0001 elsewhere)
//! 0x06  u16 (0x0002)
//! 0x08  u16 container era code          (6, 7, 8 or 9. Its own column in `ERAS`)
//! 0x0a  u8  (0)
//! 0x0b  u8  platform tag                (0x81 macOS, 0x82 iOS, 0x87 iOS simulator, etc)
//! 0x0c  u16 deployment target major
//! 0x0e  u16 deployment target minor
//! 0x10  u64 total file size
//! 0x18  u64 function list offset        (always 0x58. header is fixed width)
//! 0x20  u64 function list size          (entries only - see below)
//! 0x28  u64 public metadata offset
//! 0x30  u64 public metadata size
//! 0x38  u64 private metadata offset
//! 0x40  u64 private metadata size
//! 0x48  u64 bitcode offset
//! 0x50  u64 bitcode size
//! 0x58  function list
//! ...   public metadata section
//! ...   private metadata section
//! ...   bitcode
//! ```
//!
//! Function list layout
//! ```text
//! 0x00  u32 function count
//! 0x04  one entry per function
//! ...   b"ENDT"
//! ```
//!
//! Function entry layout
//! ```text
//! 0x00  u32 entry size        (counts this field)
//! 0x04  b"NAME"  u16  len     (entry point name. NUL-terminated)
//! ....  b"TYPE"  u16    1     (u8 function kind)
//! ....  b"HASH"  u16   32     (SHA-256 of the bitcode)
//! ....  b"MDSZ"  u16    8     (u64 bitcode size)
//! ....  b"OFFT"  u16   24     (3 x u64 section offsets)
//! ....  b"VERS"  u16    8     (u16 AIR major, minor, then language major, minor)
//! ....  b"ENDT"
//! ```
//!
//! Metadata section layout
//! ```text
//! 0x00  u32           (section size)
//! 0x04  tags, if any
//! ...   b"ENDT"
//! ```
//! We emit two empty metadata sections.

use crate::profile::TargetProfile;
use sha2::{Digest, Sha256};

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
/// `profile` must be the one the bitcode was emitted for. The header and the `VERS`
/// tag repeat the target's platform and AIR version, and the loader checks them
/// against the module.
pub fn wrap(bitcode: &[u8], functions: &[Function], profile: &TargetProfile) -> Vec<u8> {
    let digest: [u8; 32] = Sha256::digest(bitcode).into();

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
    // Recorded size is for entries only. Not count prefix, not list terminator.
    let func_list_size = func_list.len() - 8;

    // Empty metadata section.
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

/// Extract bitcode from a metallib container.
pub fn bitcode_of(metallib: &[u8]) -> Result<&[u8], String> {
    if metallib.get(..4) != Some(b"MTLB") {
        return Err("not a metallib".into());
    }
    let g = |o: usize| -> Result<usize, String> {
        metallib
            .get(o..o + 8)
            .and_then(|b| b.try_into().ok())
            .map(|b| u64::from_le_bytes(b) as usize)
            .ok_or_else(|| format!("truncated at {o:#x}"))
    };
    let (off, size) = (g(0x48)?, g(0x50)?);
    let mut blob = metallib
        .get(off..off + size)
        .ok_or("bitcode range outside the container")?;
    if blob.get(..4) == Some(&[0xde, 0xc0, 0x17, 0x0b]) {
        blob = &blob[20..];
    }
    if blob.get(..4) != Some(b"BC\xc0\xde") {
        return Err(format!("no bitcode magic at {off:#x}"));
    }
    Ok(blob)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(&lib[hpos + 6..hpos + 38], &Sha256::digest(&bc)[..]);
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
