use super::{
    are_blocks_on_same_page, AllocationType, Region, Suballocation, SuballocationNode,
    SuballocationType, Suballocator, SuballocatorError,
};
use crate::{
    memory::{
        allocator::{align_up, AllocationHandle, DeviceLayout},
        is_aligned, DeviceAlignment,
    },
    DeviceSize,
};
use std::{cmp, hint, iter::FusedIterator, marker::PhantomData, ptr::NonNull};

/// A [suballocator] that uses the most generic [free-list].
///
/// You generally shouldn't use this allocator as [`TlsfAllocator`] is better in almost every way.
/// However, unlike `TlsfAllocator`, this allocator employs a best-fit strategy, which can lead to
/// slightly better [fragmentation] at the cost of considerably worse performance.
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// The free-list stores suballocations which can have any offset and size. When an allocation
/// request is made, the list is searched using the best-fit strategy, meaning that the smallest
/// suballocation that fits the request is chosen. If required, the chosen suballocation is trimmed
/// at the ends, and the ends are returned to the free-list. As such, no [internal fragmentation]
/// occurs. The front might need to be trimmed because of [alignment requirements] and the end
/// because of a larger than required size. When an allocation is freed, the allocator checks if
/// the adjacent suballocations are free, and if so, it coalesces them into a bigger one before
/// putting it in the free-list.
///
/// # Efficiency
///
/// The free-list is sorted by size, which means that when allocating, finding a best-fit is always
/// possible in *O*(log(*n*)) time in the worst case. When deallocating, the coalescing requires us
/// to remove the adjacent free suballocations from the free-list, which is *O*(log(*n*)), and
/// insert the possibly coalesced suballocation into the free-list, which has the same time
/// complexity, so in total deallocation is *O*(log(*n*)). Resetting is *O*(1).
///
/// There is one notable edge case: after the allocator finds a best-fit, it is possible that it
/// needs to align the suballocation's offset to a higher value, after which the requested size
/// might no longer fit. In such a case, the next free suballocation in sorted order is tried until
/// a fit is successful. If this issue is encountered with all candidates, then the time complexity
/// would be *O*(*n*). However, this scenario is extremely unlikely, which is why we are not
/// considering it in the above analysis. Additionally, if your free-list is filled with
/// allocations that all have the same size, then that seems pretty sus. Sounds like you're in dire
/// need of an allocation pool.
///
/// [suballocator]: Suballocator
/// [free-list]: Suballocator#free-lists
/// [`TlsfAllocator`]: super::TlsfAllocator
/// [fragmentation]: super::super#fragmentation
/// [the `Suballocator` implementation]: Self#impl-Suballocator-for-FreeListAllocator
/// [internal fragmentation]: super::super#internal-fragmentation
/// [alignment requirements]: super::super#alignment
#[derive(Debug)]
pub struct FreeListAllocator {
    // Total memory remaining in the region.
    free_size: DeviceSize,
    suballocations: SuballocationList,
}

unsafe impl Suballocator for FreeListAllocator {
    type Suballocations<'a> = Suballocations<'a>;

    /// Creates a new `FreeListAllocator` for the given [region].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        let mut suballocations = SuballocationList {
            region,
            head_ptr: NonNull::dangling(),
            tail_ptr: NonNull::dangling(),
            len: 0,
            free_list: Vec::with_capacity(32),
            node_allocator: slabbin::SlabAllocator::new(32),
        };

        unsafe { suballocations.init() };

        FreeListAllocator {
            free_size: region.size(),
            suballocations,
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
    unsafe fn deallocate(&mut self, suballocation: Suballocation) {
        let node_ptr = suballocation
            .handle
            .as_ptr()
            .cast::<SuballocationListNode>();

        // SAFETY: The caller must guarantee that `suballocation` refers to a currently allocated
        // allocation of `self`, which means that `node_ptr` is the same one we gave out on
        // allocation, making it a valid pointer.
        let mut node_ptr = unsafe { NonNull::new_unchecked(node_ptr) };

        debug_assert!(self.suballocations.node_allocator.contains(node_ptr));

        let node = unsafe { node_ptr.as_mut() };

        debug_assert_ne!(node.allocation_type, SuballocationType::Free);

        // Suballocation sizes are constrained by the size of the region, so they can't possibly
        // overflow when added up.
        self.free_size += node.size;

        node.allocation_type = SuballocationType::Free;

        unsafe { self.suballocations.coalesce(node_ptr) };
        unsafe { self.suballocations.add_to_free_list(node_ptr) };
    }

    fn reset(&mut self) {
        self.suballocations.reset();
        self.free_size = self.region().size();
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size
    }

    #[inline]
    fn suballocations(&self) -> Self::Suballocations<'_> {
        self.suballocations.iter()
    }
}

impl FreeListAllocator {
    #[inline(always)]
    fn allocate_inner(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        fn has_granularity_conflict(prev_ty: SuballocationType, ty: AllocationType) -> bool {
            if prev_ty == SuballocationType::Free {
                false
            } else if prev_ty == SuballocationType::Unknown {
                true
            } else {
                prev_ty != ty.into()
            }
        }

        let size = layout.size();
        let alignment = layout.alignment();

        match self.suballocations.free_list.last() {
            Some(&last) if unsafe { last.as_ref() }.size >= size => {
                // We create a dummy node to compare against in the below binary search. The only
                // fields of importance are `offset` and `size`. It is paramount that we set
                // `offset` to zero, so that in the case where there are multiple free
                // suballocations with the same size, we get the first one of them, that is, the
                // one with the lowest offset.
                let dummy_node = SuballocationListNode {
                    prev_ptr: NonNull::dangling(),
                    next_ptr: NonNull::dangling(),
                    offset: 0,
                    size,
                    allocation_type: SuballocationType::Unknown,
                };

                // This is almost exclusively going to return `Err`, but that's expected: we are
                // first comparing the size, looking for an allocation of the given `size`, however
                // the next-best will do as well (that is, a size somewhat larger). In that case we
                // get `Err`. If we do find a suballocation with the exact size however, we are
                // then comparing the offsets to make sure we get the suballocation with the lowest
                // offset, in case there are multiple with the same size. In that case we also
                // exclusively get `Err` except when the offset is zero.
                //
                // Note that `index == free_list.len()` can't be because we checked that the
                // free-list contains a suballocation that is big enough.
                let (Ok(index) | Err(index)) = self
                    .suballocations
                    .free_list
                    .binary_search_by_key(&&dummy_node, |&ptr| unsafe { ptr.as_ref() });

                for (index, &(mut node_ptr)) in
                    self.suballocations.free_list.iter().enumerate().skip(index)
                {
                    let node = unsafe { node_ptr.as_ref() };

                    // This can't overflow because suballocation offsets are bounded by the region,
                    // whose end can itself not exceed `DeviceLayout::MAX_SIZE`.
                    let mut offset = align_up(node.offset, alignment);

                    if buffer_image_granularity != DeviceAlignment::MIN {
                        debug_assert!(is_aligned(self.region().offset(), buffer_image_granularity));

                        let prev = unsafe { node.prev_ptr.as_ref() };

                        if are_blocks_on_same_page(
                            prev.offset,
                            prev.size,
                            offset,
                            buffer_image_granularity,
                        ) && has_granularity_conflict(prev.allocation_type, allocation_type)
                        {
                            // This is overflow-safe for the same reason as above.
                            offset = align_up(offset, buffer_image_granularity);
                        }
                    }

                    // `offset`, no matter the alignment, can't end up as more than
                    // `DeviceAlignment::MAX` for the same reason as above. `DeviceLayout`
                    // guarantees that `size` doesn't exceed `DeviceLayout::MAX_SIZE`.
                    // `DeviceAlignment::MAX.as_devicesize() + DeviceLayout::MAX_SIZE` is equal to
                    // `DeviceSize::MAX`. Therefore, `offset + size` can't overflow.
                    //
                    // `node.offset + node.size` can't overflow for the same reason as above.
                    if offset + size <= node.offset + node.size {
                        if buffer_image_granularity != DeviceAlignment::MIN {
                            let next = unsafe { node.next_ptr.as_ref() };

                            if are_blocks_on_same_page(
                                offset,
                                size,
                                next.offset,
                                buffer_image_granularity,
                            ) && has_granularity_conflict(next.allocation_type, allocation_type)
                            {
                                continue;
                            }
                        }

                        self.suballocations.free_list.remove(index);

                        // SAFETY:
                        // - `node` is free.
                        // - We have removed `node` from the free-list.
                        // - `offset` is that of `node`, possibly rounded up.
                        // - We checked that `offset + size` falls within `node`.
                        unsafe { self.suballocations.split(node_ptr, offset, size) };

                        let node = unsafe { node_ptr.as_mut() };
                        node.allocation_type = allocation_type.into();

                        // This can't overflow because suballocation sizes in the free-list are
                        // constrained by the remaining size of the region.
                        self.free_size -= size;

                        return Ok(Suballocation {
                            offset,
                            size,
                            allocation_type,
                            handle: AllocationHandle::from_ptr(node_ptr.as_ptr().cast()),
                        });
                    }
                }

                // There is not enough space due to alignment requirements.
                Err(SuballocatorError::OutOfRegionMemory)
            }
            // There would be enough space if the region wasn't so fragmented. :(
            Some(_) if self.free_size() >= size => Err(SuballocatorError::FragmentedRegion),
            // There is not enough space.
            Some(_) => Err(SuballocatorError::OutOfRegionMemory),
            // There is no space at all.
            None => Err(SuballocatorError::OutOfRegionMemory),
        }
    }

    #[inline]
    fn region(&self) -> &Region {
        &self.suballocations.region
    }
}

#[derive(Debug)]
struct SuballocationList {
    region: Region,
    head_ptr: NonNull<SuballocationListNode>,
    tail_ptr: NonNull<SuballocationListNode>,
    len: usize,
    // Free suballocations sorted by size in ascending order. This means we can always find a
    // best-fit in *O*(log(*n*)) time in the worst case, and iterating in order is very efficient.
    free_list: Vec<NonNull<SuballocationListNode>>,
    node_allocator: slabbin::SlabAllocator<SuballocationListNode>,
}

unsafe impl Send for SuballocationList {}
unsafe impl Sync for SuballocationList {}

#[derive(Clone, Copy, Debug)]
struct SuballocationListNode {
    prev_ptr: NonNull<Self>,
    next_ptr: NonNull<Self>,
    offset: DeviceSize,
    size: DeviceSize,
    allocation_type: SuballocationType,
}

impl SuballocationList {
    unsafe fn init(&mut self) {
        let head_ptr = self.node_allocator.allocate();
        let root_ptr = self.node_allocator.allocate();
        let tail_ptr = self.node_allocator.allocate();

        let head = SuballocationListNode {
            prev_ptr: NonNull::dangling(),
            next_ptr: root_ptr,
            offset: self.region.offset(),
            size: 0,
            allocation_type: SuballocationType::Unknown,
        };
        unsafe { head_ptr.write(head) };

        let root = SuballocationListNode {
            prev_ptr: head_ptr,
            next_ptr: tail_ptr,
            offset: self.region.offset(),
            size: self.region.size(),
            allocation_type: SuballocationType::Free,
        };
        unsafe { root_ptr.write(root) };

        let tail = SuballocationListNode {
            prev_ptr: root_ptr,
            next_ptr: NonNull::dangling(),
            offset: self.region.offset() + self.region.size(),
            size: 0,
            allocation_type: SuballocationType::Unknown,
        };
        unsafe { tail_ptr.write(tail) };

        self.head_ptr = head_ptr;
        self.tail_ptr = tail_ptr;
        self.len = 1;
        self.free_list.push(root_ptr);
    }

    /// Fits a suballocation inside the target one, splitting the target at the ends if required.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently free suballocation of `self`.
    /// - `node_ptr` must have been removed from the free-list.
    /// - `offset` and `size` must refer to a subregion of the given suballocation.
    #[inline]
    unsafe fn split(
        &mut self,
        mut node_ptr: NonNull<SuballocationListNode>,
        offset: DeviceSize,
        size: DeviceSize,
    ) {
        let node = unsafe { node_ptr.as_mut() };

        debug_assert_eq!(node.allocation_type, SuballocationType::Free);
        debug_assert!(!self.free_list.contains(&node_ptr));
        debug_assert!(offset >= node.offset);
        debug_assert!(offset + size <= node.offset + node.size);

        // These are guaranteed to not overflow because the caller must uphold that the given
        // region is contained within that of `node`.
        let padding_front = offset - node.offset;
        let padding_back = node.offset + node.size - offset - size;

        if padding_front > 0 {
            let padding_ptr = self.node_allocator.allocate();
            let padding = SuballocationListNode {
                prev_ptr: node.prev_ptr,
                next_ptr: node_ptr,
                offset: node.offset,
                size: padding_front,
                allocation_type: SuballocationType::Free,
            };
            unsafe { padding_ptr.write(padding) };

            let prev = unsafe { node.prev_ptr.as_mut() };
            prev.next_ptr = padding_ptr;

            node.prev_ptr = padding_ptr;
            node.offset = offset;
            // The caller must uphold that the given region is contained within that of `node`, and
            // it follows that if there is padding, the size of the node must be larger than that
            // of the padding, so this can't overflow.
            node.size -= padding.size;

            self.len += 1;

            // SAFETY: We just created this suballocation, so there's no way that it was
            // deallocated already.
            unsafe { self.add_to_free_list(padding_ptr) };
        }

        if padding_back > 0 {
            let padding_ptr = self.node_allocator.allocate();
            let padding = SuballocationListNode {
                prev_ptr: node_ptr,
                next_ptr: node.next_ptr,
                offset: offset + size,
                size: padding_back,
                allocation_type: SuballocationType::Free,
            };
            unsafe { padding_ptr.write(padding) };

            let next = unsafe { node.next_ptr.as_mut() };
            next.prev_ptr = padding_ptr;

            node.next_ptr = padding_ptr;
            // This is overflow-safe for the same reason as above.
            node.size -= padding.size;

            self.len += 1;

            // SAFETY: Same as above.
            unsafe { self.add_to_free_list(padding_ptr) };
        }
    }

    /// Inserts the target suballocation into the free-list.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently allocated suballocation of `self`.
    #[inline]
    unsafe fn add_to_free_list(&mut self, node_ptr: NonNull<SuballocationListNode>) {
        debug_assert!(!self.free_list.contains(&node_ptr));

        let node = unsafe { node_ptr.as_ref() };
        let (Ok(index) | Err(index)) = self
            .free_list
            .binary_search_by_key(&node, |&ptr| unsafe { ptr.as_ref() });
        self.free_list.insert(index, node_ptr);
    }

    /// Coalesces the target (free) suballocation with adjacent ones that are also free.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently free suballocation `self`.
    /// - `node_ptr` must not be in the free-list yet.
    #[inline]
    unsafe fn coalesce(&mut self, mut node_ptr: NonNull<SuballocationListNode>) {
        let node = unsafe { node_ptr.as_mut() };

        debug_assert_eq!(node.allocation_type, SuballocationType::Free);
        debug_assert!(!self.free_list.contains(&node_ptr));

        let prev_ptr = node.prev_ptr;

        if unsafe { prev_ptr.as_ref() }.allocation_type == SuballocationType::Free {
            // SAFETY: We checked that the suballocation is free, which means that it must be in
            // the free-list.
            unsafe { self.remove_from_free_list(prev_ptr) };

            let prev = unsafe { prev_ptr.as_ref() };

            node.prev_ptr = prev.prev_ptr;
            node.offset = prev.offset;
            // The sizes of suballocations are constrained by that of the parent allocation, so
            // they can't possibly overflow when added up.
            node.size += prev.size;

            let mut prev_prev_ptr = prev.prev_ptr;
            let prev_prev = unsafe { prev_prev_ptr.as_mut() };
            prev_prev.next_ptr = node_ptr;

            self.len -= 1;

            // SAFETY:
            // - The suballocation is free.
            // - The suballocation was removed from the free-list.
            // - The next suballocation and possibly a previous suballocation have been updated
            //   such that they no longer reference the suballocation.
            //
            // All of these conditions combined guarantee that `prev_ptr` cannot be used again.
            unsafe { self.node_allocator.deallocate(prev_ptr) };
        }

        let next_ptr = node.next_ptr;

        if unsafe { next_ptr.as_ref() }.allocation_type == SuballocationType::Free {
            // SAFETY: Same as above.
            unsafe { self.remove_from_free_list(next_ptr) };

            let next = unsafe { next_ptr.as_ref() };

            node.next_ptr = next.next_ptr;
            // This is overflow-safe for the same reason as above.
            node.size += next.size;

            let mut next_next_ptr = next.next_ptr;
            let next_next = unsafe { next_next_ptr.as_mut() };
            next_next.prev_ptr = node_ptr;

            self.len -= 1;

            // SAFETY: Same as above.
            unsafe { self.node_allocator.deallocate(next_ptr) };
        }
    }

    /// Removes the target suballocation from the free-list.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently free suballocation of `self`.
    /// - `node_ptr` must be in the free-list.
    #[inline]
    unsafe fn remove_from_free_list(&mut self, node_ptr: NonNull<SuballocationListNode>) {
        debug_assert!(self.free_list.contains(&node_ptr));

        let node = unsafe { node_ptr.as_ref() };

        let Ok(index) = self
            .free_list
            .binary_search_by_key(&node, |&ptr| unsafe { ptr.as_ref() })
        else {
            // SAFETY: The caller must ensure that `node_ptr` is in the free-list.
            unsafe { hint::unreachable_unchecked() };
        };

        self.free_list.remove(index);
    }

    fn reset(&mut self) {
        self.free_list.clear();
        unsafe { self.node_allocator.reset() };
        unsafe { self.init() };
    }

    #[inline]
    fn iter(&self) -> Suballocations<'_> {
        let head_ptr = unsafe { self.head_ptr.as_ref() }.next_ptr;
        let tail_ptr = unsafe { self.tail_ptr.as_ref() }.prev_ptr;

        Suballocations {
            head_ptr,
            tail_ptr,
            len: self.len,
            marker: PhantomData,
        }
    }
}

impl PartialEq for SuballocationListNode {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.offset == other.offset
    }
}

impl Eq for SuballocationListNode {}

impl PartialOrd for SuballocationListNode {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SuballocationListNode {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // We want to sort the free-list by size.
        self.size
            .cmp(&other.size)
            // However there might be multiple free suballocations with the same size, so we need
            // to compare the offset as well to differentiate.
            .then(self.offset.cmp(&other.offset))
    }
}

#[derive(Clone)]
pub struct Suballocations<'a> {
    head_ptr: NonNull<SuballocationListNode>,
    tail_ptr: NonNull<SuballocationListNode>,
    len: usize,
    marker: PhantomData<&'a SuballocationList>,
}

unsafe impl Send for Suballocations<'_> {}
unsafe impl Sync for Suballocations<'_> {}

impl Iterator for Suballocations<'_> {
    type Item = SuballocationNode;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            let head = unsafe { self.head_ptr.as_ref() };
            self.head_ptr = head.next_ptr;
            self.len -= 1;

            Some(SuballocationNode {
                offset: head.offset,
                size: head.size,
                allocation_type: head.allocation_type,
            })
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for Suballocations<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            let tail = unsafe { self.tail_ptr.as_ref() };
            self.tail_ptr = tail.prev_ptr;
            self.len -= 1;

            Some(SuballocationNode {
                offset: tail.offset,
                size: tail.size,
                allocation_type: tail.allocation_type,
            })
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Suballocations<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl FusedIterator for Suballocations<'_> {}
