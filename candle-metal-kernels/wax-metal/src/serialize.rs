//! Byte encoding for [`LoweredKernel`].

use crate::launch::{Launch, LoweredKernel, SecondaryDispatch};

impl LoweredKernel {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.metallib.len() + 256);
        put_bytes(&mut v, self.entry.as_bytes());
        put_launch(&mut v, &self.launch);
        put_u64(&mut v, self.scratch_bytes as u64);
        put_scalars(&mut v, &self.scalar_args);
        match &self.second {
            None => v.push(0),
            Some(s) => {
                v.push(1);
                put_bytes(&mut v, s.entry.as_bytes());
                put_launch(&mut v, &s.launch);
                put_scalars(&mut v, &s.scalar_args);
            }
        }
        put_bytes(&mut v, &self.metallib);
        v
    }

    pub fn from_bytes(raw: &[u8]) -> Option<Self> {
        let (entry, rest) = get_string(raw)?;
        let (launch, rest) = get_launch(rest)?;
        let (scratch_bytes, rest) = get_u64(rest)?;
        let (scalar_args, rest) = get_scalars(rest)?;
        let (&tag, rest) = rest.split_first()?;
        let (second, rest) = match tag {
            0 => (None, rest),
            1 => {
                let (entry, rest) = get_string(rest)?;
                let (launch, rest) = get_launch(rest)?;
                let (scalar_args, rest) = get_scalars(rest)?;
                (
                    Some(SecondaryDispatch {
                        entry,
                        launch,
                        scalar_args,
                    }),
                    rest,
                )
            }
            _ => return None,
        };
        Some(LoweredKernel {
            metallib: get_bytes(rest)?.0.to_vec(),
            entry,
            launch,
            scratch_bytes: scratch_bytes as usize,
            scalar_args,
            second,
        })
    }
}

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_bytes(v: &mut Vec<u8>, b: &[u8]) {
    put_u64(v, b.len() as u64);
    v.extend_from_slice(b);
}
/// Widened to `u64` for compatibility.
fn put_launch(v: &mut Vec<u8>, l: &Launch) {
    for x in [
        l.grid.0,
        l.grid.1,
        l.grid.2,
        l.threadgroup.0,
        l.threadgroup.1,
        l.threadgroup.2,
    ] {
        put_u64(v, x as u64);
    }
}
fn put_scalars(v: &mut Vec<u8>, s: &[i32]) {
    put_u32(v, s.len() as u32);
    for &x in s {
        put_u32(v, x as u32);
    }
}

fn get_bytes(raw: &[u8]) -> Option<(&[u8], &[u8])> {
    let (len, rest) = raw.split_first_chunk::<8>()?;
    rest.split_at_checked(u64::from_le_bytes(*len) as usize)
}

fn get_u32(raw: &[u8]) -> Option<(u32, &[u8])> {
    let (b, rest) = raw.split_first_chunk::<4>()?;
    Some((u32::from_le_bytes(*b), rest))
}

fn get_u64(raw: &[u8]) -> Option<(u64, &[u8])> {
    let (b, rest) = raw.split_first_chunk::<8>()?;
    Some((u64::from_le_bytes(*b), rest))
}

fn get_string(raw: &[u8]) -> Option<(String, &[u8])> {
    let (b, rest) = get_bytes(raw)?;
    Some((String::from_utf8(b.to_vec()).ok()?, rest))
}

fn get_launch(raw: &[u8]) -> Option<(Launch, &[u8])> {
    let mut v = [0usize; 6];
    let mut rest = raw;
    for slot in &mut v {
        let (x, r) = get_u64(rest)?;
        *slot = x as usize;
        rest = r;
    }
    Some((Launch::new((v[0], v[1], v[2]), (v[3], v[4], v[5])), rest))
}

fn get_scalars(raw: &[u8]) -> Option<(Vec<i32>, &[u8])> {
    let (n, mut rest) = get_u32(raw)?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let (v, r) = get_u32(rest)?;
        out.push(v as i32);
        rest = r;
    }
    Some((out, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(tag: u8) -> LoweredKernel {
        LoweredKernel {
            metallib: vec![tag; 64],
            entry: "gemm".into(),
            launch: Launch::new((7, 1, 1), (256, 1, 1)),
            scratch_bytes: 1024,
            scalar_args: vec![3, -4],
            second: Some(SecondaryDispatch {
                entry: "reduce".into(),
                launch: Launch::new((2, 3, 4), (32, 1, 1)),
                scalar_args: vec![9],
            }),
        }
    }

    #[test]
    fn kernel_round_trips_correctly() {
        let k = kernel(0xAB);
        let got = LoweredKernel::from_bytes(&k.to_bytes()).expect("decodes");
        assert_eq!(got.metallib, k.metallib);
        assert_eq!(got.entry, k.entry);
        assert_eq!(got.launch, k.launch);
        assert_eq!(got.scratch_bytes, k.scratch_bytes);
        assert_eq!(got.scalar_args, k.scalar_args);
        let (a, b) = (got.second.unwrap(), k.second.unwrap());
        assert_eq!(
            (a.entry, a.launch, a.scalar_args),
            (b.entry, b.launch, b.scalar_args)
        );
    }

    #[test]
    fn kernel_with_empty_fields_round_trips_correctly() {
        let mut k = kernel(3);
        k.second = None;
        k.scalar_args.clear();
        let got = LoweredKernel::from_bytes(&k.to_bytes()).expect("decodes");
        assert!(got.second.is_none());
        assert!(got.scalar_args.is_empty());
    }

    #[test]
    fn truncation_is_none_not_panic() {
        let blob = kernel(2).to_bytes();
        for n in 0..blob.len() {
            assert!(
                LoweredKernel::from_bytes(&blob[..n]).is_none(),
                "truncated to {n} decoded"
            );
        }
    }

    #[test]
    fn corrupt_length_prefix_is_none_not_large_allocation() {
        let mut blob = kernel(4).to_bytes();
        blob[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(LoweredKernel::from_bytes(&blob).is_none());
    }
}
