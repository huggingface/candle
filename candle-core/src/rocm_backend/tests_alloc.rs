//! GPU-level behaviour of the caching allocator in [`super::alloc`].
//!
//! Split from that module's own unit tests, which cover the pure bucket
//! arithmetic and need no device.

use super::RocmDevice;
use crate::Result;

/// Returns early when the machine has no ROCm GPU, like the other ROCm test
/// modules here.
macro_rules! device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        }
    };
}

/// A dropped buffer must come back rather than go to the driver — that is
/// the whole reason this allocator exists.
#[test]
fn a_dropped_block_is_reused() -> Result<()> {
    let dev = device!();
    let first = dev.alloc::<f32>(1024)?.as_ptr();
    let second = dev.alloc::<f32>(1024)?;
    assert_eq!(first, second.as_ptr());
    Ok(())
}

/// Two live buffers of the same size must never alias.
#[test]
fn live_blocks_do_not_alias() -> Result<()> {
    let dev = device!();
    let a = dev.alloc::<f32>(1024)?;
    let b = dev.alloc::<f32>(1024)?;
    assert_ne!(a.as_ptr(), b.as_ptr());
    Ok(())
}

/// Rounding is what lets near-miss sizes share a block, but a *smaller*
/// request must never be handed a block that cannot hold it.
#[test]
fn a_reused_block_is_large_enough() -> Result<()> {
    let dev = device!();
    let big = dev.alloc::<u8>(4096)?.as_ptr();
    // 4000 rounds to the same 4096-byte bucket, so it reuses the block.
    let small = dev.alloc::<u8>(4000)?;
    assert_eq!(big, small.as_ptr());
    assert_eq!(small.count(), 4000);
    // 5000 does not, so it gets a fresh one.
    let other = dev.alloc::<u8>(5000)?;
    assert_ne!(big, other.as_ptr());
    Ok(())
}

/// `count` reports what the caller asked for, not the bucket it landed in;
/// `clone_dtoh` sizes its host `Vec` from it.
#[test]
fn count_reflects_the_request_not_the_bucket() -> Result<()> {
    let dev = device!();
    let mem = dev.alloc::<f32>(7)?;
    assert_eq!(mem.count(), 7);
    assert_eq!(mem.size(), 28);
    Ok(())
}

/// A zero-length allocation is a null pointer, and recycling it must not
/// put a null on the free list for a later request to be handed.
#[test]
fn an_empty_allocation_is_null_and_is_not_recycled() -> Result<()> {
    let dev = device!();
    assert!(dev.alloc::<f32>(0)?.as_ptr().is_null());
    assert!(!dev.alloc::<f32>(1)?.as_ptr().is_null());
    Ok(())
}
