use super::{
    are_blocks_on_same_page, AllocationType, Region, Suballocation, SuballocationNode,
    SuballocationType, Suballocator, SuballocatorError,
};
use crate::{
    memory::{
        allocator::{align_up, AllocationHandle, DeviceLayout},
        DeviceAlignment,
    },
    DeviceSize,
};
use std::iter::FusedIterator;

/// A [suballocator] which can allocate dynamically, but can only free all allocations at once.
///
/// With bump allocation, the used up space increases linearly as allocations are made, and
/// allocations can never be freed individually, which is why this algorithm is also called *linear
/// allocation*.
///
/// `BumpAllocator`s are best suited for very short-lived (say a few frames at best) resources that
/// need to be allocated often (say each frame) to really take advantage of the performance gains.
/// For creating long-lived allocations, [`TlsfAllocator`] is best suited. The way you would
/// typically use this allocator is to have one for each frame in flight. At the start of a frame,
/// you reset it and allocate your resources with it. You then write to the resources and render
/// that frame with them.
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// What happens is that every time you make an allocation, you receive one with an offset
/// corresponding to the *free start* within the [region], and then the free start is *bumped* so
/// that following allocations wouldn't alias it. As you can imagine, this is **extremely fast**
/// because it doesn't need to keep a [free-list]. It only needs to do a few arithmetic and bitwise
/// operations and comparisons. But beware, **fast is about all this is**. It is horribly memory
/// inefficient when used wrong, and is very susceptible to [memory leaks].
///
/// Once you know that you are done with the allocations, meaning you know they are all no longer
/// used, you can safely reset the allocator using the [`reset`] method as long as the allocator
/// is not shared between threads. This is one of the reasons you are generally advised to use (at
/// least) one `BumpAllocator` per thread if you can.
///
/// # Efficiency
///
/// Allocation is *O*(1), and so is resetting the allocator (freeing all allocations).
///
/// [suballocator]: Suballocator
/// [`TlsfAllocator`]: super::TlsfAllocator
/// [the `Suballocator` implementation]: Self#impl-Suballocator-for-BumpAllocator
/// [region]: Suballocator#regions
/// [free-list]: Suballocator#free-lists
/// [memory leaks]: super::super#leakage
/// [`reset`]: Self::reset
/// [hierarchy]: Suballocator#memory-hierarchies
#[derive(Debug)]
pub struct BumpAllocator {
    region: Region,
    free_start: DeviceSize,
    free_end: DeviceSize,
    prev_allocation_type: AllocationType,
}

unsafe impl Suballocator for BumpAllocator {
    type Suballocations<'a> = Suballocations<'a>;

    /// Creates a new `BumpAllocator` for the given [region].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        BumpAllocator {
            region,
            free_start: region.offset(),
            free_end: region.offset() + region.size(),
            prev_allocation_type: AllocationType::Unknown,
        }
    }

    #[inline]
    fn allocate_buffer(
        &mut self,
        layout: DeviceLayout,
    ) -> Result<Suballocation, SuballocatorError> {
        self.allocate_inner(layout, AllocationType::Linear, DeviceAlignment::MIN)
    }

    #[inline]
    fn allocate(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        self.allocate_inner(layout, allocation_type, buffer_image_granularity)
    }

    #[inline]
    unsafe fn deallocate(&mut self, _suballocation: Suballocation) {
        // such complex, very wow
    }

    /// Resets the free-start back to the beginning of the [region].
    ///
    /// [region]: Suballocator#regions
    #[inline]
    fn reset(&mut self) {
        self.free_start = self.region.offset();
        self.prev_allocation_type = AllocationType::Unknown;
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_end - self.free_start
    }

    #[inline]
    fn suballocations(&self) -> Self::Suballocations<'_> {
        let start = if self.free_start == self.region.offset() {
            1
        } else {
            0
        };
        let end = if self.free_start == self.free_end {
            1
        } else {
            2
        };

        Suballocations {
            allocator: self,
            start,
            end,
        }
    }
}

impl BumpAllocator {
    #[inline(always)]
    fn allocate_inner(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        fn has_granularity_conflict(prev_ty: AllocationType, ty: AllocationType) -> bool {
            prev_ty == AllocationType::Unknown || prev_ty != ty
        }

        let size = layout.size();
        let alignment = layout.alignment();

        // This can't overflow because suballocation offsets are bounded by the region, whose end
        // can itself not exceed `DeviceLayout::MAX_SIZE`.
        let mut offset = align_up(self.free_start, alignment);

        if buffer_image_granularity != DeviceAlignment::MIN
            && are_blocks_on_same_page(0, self.free_start, offset, buffer_image_granularity)
            && has_granularity_conflict(self.prev_allocation_type, allocation_type)
        {
            // This can't overflow for the same reason as above.
            offset = align_up(offset, buffer_image_granularity);
        }

        // `offset`, no matter the alignment, can't end up as more than `DeviceAlignment::MAX` for
        // the same reason as above. `DeviceLayout` guarantees that `size` doesn't exceed
        // `DeviceLayout::MAX_SIZE`. `DeviceAlignment::MAX.as_devicesize() +
        // DeviceLayout::MAX_SIZE` is equal to `DeviceSize::MAX`. Therefore, `offset + size` can't
        // overflow.
        let free_start = offset + size;

        if free_start > self.free_end {
            return Err(SuballocatorError::OutOfRegionMemory);
        }

        self.free_start = free_start;
        self.prev_allocation_type = allocation_type;

        Ok(Suballocation {
            offset,
            size,
            allocation_type,
            handle: AllocationHandle::null(),
        })
    }

    fn suballocation_node(&self, part: usize) -> SuballocationNode {
        if part == 0 {
            SuballocationNode {
                offset: self.region.offset(),
                size: self.free_start - self.region.offset(),
                allocation_type: self.prev_allocation_type.into(),
            }
        } else {
            debug_assert_eq!(part, 1);

            SuballocationNode {
                offset: self.free_start,
                size: self.free_size(),
                allocation_type: SuballocationType::Free,
            }
        }
    }
}

#[derive(Clone)]
pub struct Suballocations<'a> {
    allocator: &'a BumpAllocator,
    start: usize,
    end: usize,
}

impl Iterator for Suballocations<'_> {
    type Item = SuballocationNode;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() != 0 {
            let node = self.allocator.suballocation_node(self.start);
            self.start += 1;

            Some(node)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();

        (len, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for Suballocations<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len() != 0 {
            self.end -= 1;
            let node = self.allocator.suballocation_node(self.end);

            Some(node)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Suballocations<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl FusedIterator for Suballocations<'_> {}
