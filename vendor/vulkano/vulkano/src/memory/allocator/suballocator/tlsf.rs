use super::{
    AllocationType, Region, Suballocation, SuballocationNode, SuballocationType, Suballocator,
    SuballocatorError,
};
use crate::{
    memory::{
        allocator::{align_up, AllocationHandle, DeviceLayout},
        DeviceAlignment,
    },
    DeviceSize,
};
use slabbin::SlabAllocator;
use std::{
    cmp,
    fmt::{Debug, Error as FmtError, Formatter},
    hint,
    iter::FusedIterator,
    marker::PhantomData,
    num::NonZero,
    ptr::NonNull,
};

const FIRST_LEVEL_BINS: usize = 32;
const SECOND_LEVEL_INDEX_BITS: u32 = 3;
const SECOND_LEVEL_INDEX_MASK: DeviceSize = SECOND_LEVEL_BINS as DeviceSize - 1;
const SECOND_LEVEL_BINS: usize = 1 << SECOND_LEVEL_INDEX_BITS;

const MIN_NODE_BITS: u32 = 4;
pub(super) const MIN_NODE_SIZE: DeviceSize = 1 << MIN_NODE_BITS;
const MAX_NODE_SIZE: DeviceSize = (MIN_NODE_SIZE << FIRST_LEVEL_BINS) - 1;

/// A two-level segregated-fit [suballocator].
///
/// This should be your first choice of allocator because it's the best overall. Its performance is
/// better than the performance of both [`FreeListAllocator`] and [`BuddyAllocator`] at the cost of
/// only marginally worse [fragmentation] than `FreeListAllocator`'s. Its fragmentation is still
/// much better than `BuddyAllocator`'s.
///
/// This allocator can be thought of as something between `FreeListAllocator` and `BuddyAllocator`:
/// `BuddyAllocator` is a special case of an allocator employing segregated lists in that there is
/// only the power-of-two level of segregation and that it only coalesces buddies; the segregation
/// is *physical*. On the other hand, this *two-level* segregated-fit allocator has a power-of-two
/// level of segregation which is further segregated linearly. This means that it can find a free
/// suballocation that fits the requested size much better. Also, it can split and coalesce
/// suballocations willy-nilly like `FreeListAllocator`; the segregation is purely *logical*.
/// However, it doesn't use a best-fit strategy like `FreeListAllocator`, only a good-fit one,
/// making it much faster.
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// There is one unsorted free-list for each size class, with two levels of segregation. The first
/// level segregates in increasing powers of two (e.g., \[16,&nbsp;32), \[32,&nbsp;64),
/// \[64,&nbsp;128), etc.), and the second level further segregates each first level linearly
/// (e.g., \[64,&nbsp;72), \[72&nbsp;80), \[80,&nbsp;88), etc.). When allocating, then, the first-
/// and second-level indices of the smallest size class where all suballocations fit the allocation
/// request (since the lists are unsorted) must be calculated, and a suballocation is removed from
/// this list. This is what gives the allocator its good-fit strategy. The suballocation is then
/// potentially trimmed at the ends, and the ends are added to their corresponding free-lists. The
/// front might need to be trimmed because of [alignment requirements] and the end because of a
/// larger than required size. When deallocating, the allocation is coalesced with the adjacent
/// suballocations if they are free, and the resulting suballocation is added to its corresponding
/// free-list.
///
/// # Efficiency
///
/// Calculating the first- and second-level indices for a suitable free-list to remove from for
/// allocation is done with only bitwise, arithmetic and conditional move instructions. Calculating
/// the first- and second-level indices for the free-list corresponding to a suballocation size is
/// even cheaper. This is what gives the allocator its excellent performance, and why both
/// allocation and deallocation is *O*(1). Resetting is also *O*(1).
///
/// [suballocator]: Suballocator
/// [`FreeListAllocator`]: super::FreeListAllocator
/// [fragmentation]: super::super#fragmentation
/// [`BuddyAllocator`]: super::BuddyAllocator
/// [the `Suballocator` implementation]: Self#impl-Suballocator-for-TlsfAllocator
/// [alignment requirements]: super::super#alignment
#[derive(Debug)]
pub struct TlsfAllocator {
    first_level: FirstLevel,
    free_size: DeviceSize,
}

unsafe impl Suballocator for TlsfAllocator {
    type Suballocations<'a> = Suballocations<'a>;

    /// Creates a new `TlsfAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region.size` is not in the range \[16B,&nbsp;32GiB).
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        assert!(region.size() >= MIN_NODE_SIZE);
        assert!(region.size() <= MAX_NODE_SIZE);

        let mut first_level = FirstLevel {
            occupied_bins: 0,
            bins: [None; FIRST_LEVEL_BINS + 1],
            node_allocator: SlabAllocator::new(32),
            len: 0,
            head_ptr: NonNull::dangling(),
            tail_ptr: NonNull::dangling(),
            bin_allocator: SlabAllocator::new(8),
            region,
        };

        unsafe { first_level.init() };

        TlsfAllocator {
            first_level,
            free_size: region.size(),
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
        let node_ptr = suballocation.handle.as_ptr().cast::<Node>();

        // SAFETY: The caller must guarantee that `suballocation` refers to a currently allocated
        // allocation of `self`, which means that `node_ptr` is the same one we gave out on
        // allocation, making it a valid pointer.
        let mut node_ptr = unsafe { NonNull::new_unchecked(node_ptr) };

        debug_assert!(self.first_level.node_allocator.contains(node_ptr));

        let node = unsafe { node_ptr.as_mut() };

        debug_assert_ne!(node.allocation_type, SuballocationType::Free);

        // Suballocation sizes are constrained by the size of the region, so they can't possibly
        // overflow when added up.
        self.free_size += node.size;

        node.allocation_type = SuballocationType::Free;

        unsafe { self.first_level.coalesce(node_ptr) };
        unsafe { self.first_level.add_to_free_list(node_ptr) };
    }

    fn reset(&mut self) {
        self.first_level.reset();
        self.free_size = self.region().size();
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size
    }

    #[inline]
    fn suballocations(&self) -> Self::Suballocations<'_> {
        self.first_level.iter()
    }
}

impl TlsfAllocator {
    #[inline(always)]
    fn allocate_inner(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        let size = layout.size();
        let mut alignment = layout.alignment();

        if buffer_image_granularity != DeviceAlignment::MIN
            && allocation_type != AllocationType::Linear
        {
            alignment = cmp::max(alignment, buffer_image_granularity);
        }

        // SAFETY: `DeviceLayout` guarantees that `size` doesn't exceed `DeviceSize::MAX_SIZE`.
        if let Some((node_ptr, offset, size)) =
            unsafe { self.first_level.allocate(size, alignment, allocation_type) }
        {
            self.free_size -= size;

            Ok(Suballocation {
                offset,
                size: layout.size(),
                allocation_type,
                handle: AllocationHandle::from_ptr(node_ptr.as_ptr().cast()),
            })
        } else {
            if self.free_size() > layout.size() {
                Err(SuballocatorError::FragmentedRegion)
            } else {
                Err(SuballocatorError::OutOfRegionMemory)
            }
        }
    }

    #[inline]
    fn region(&self) -> &Region {
        &self.first_level.region
    }
}

#[repr(C)]
struct FirstLevel {
    /// A bitfield where each index corresponds to the index of a bin and the bit at that index
    /// corresponds to whether the bin is occupied or not (whether it is not `None`).
    occupied_bins: u64,
    /// The first-level bins used for segregation, each corresponding to the size class
    ///
    /// \[`MIN_NODE_SIZE` \* 2<sup>*i*</sup>, `MIN_NODE_SIZE` \* 2<sup>*i* + 1</sup>)
    ///
    /// where *i* is the index of the bin.
    ///
    /// Besides the `FIRST_LEVEL_BINS` "real" bins, there is also an additional "fake" bin at the
    /// end. This bin is always initialized but never has any initialized second-level bins. This
    /// is useful as a stopgap for what would have been overflow when calculating the indices of
    /// the free-list to remove from while allocating. There always being a next bin removes the
    /// need for some branches. The fake bin is not counted in `occupied_bins`.
    ///
    /// We use pointers to the second level bins because there would be too much inline memory
    /// wasted otherwise as not all bins are going to be used at all times.
    bins: [Option<NonNull<SecondLevel>>; FIRST_LEVEL_BINS + 1],
    node_allocator: SlabAllocator<Node>,
    /// The total number of suballocations (free and allocated).
    len: usize,
    /// The head of the list of all suballocations (sorted by offset).
    head_ptr: NonNull<Node>,
    /// The tail of the list of all suballocations (sorted by offset).
    tail_ptr: NonNull<Node>,
    bin_allocator: SlabAllocator<SecondLevel>,
    region: Region,
}

#[repr(C)]
struct SecondLevel {
    /// A bitfield where each index corresponds to the index of a bin and the bit at that index
    /// corresponds to whether the bin is occupied or not (whether it is not `None`).
    occupied_bins: u64,
    /// The second-level bins used for segregation, each corresponding to the size class
    ///
    /// \[*m* + *m* / `SECOND_LEVEL_BINS` \* *i*, *m* + \* *m* / `SECOND_LEVEL_BINS` \* (*i* + 1))
    ///
    /// where *m* is the minimum size of the first-level bin and *i* is the index of the bin.
    ///
    /// Each bin refers to the sentinel node of an unsorted circular doubly-linked list. "Sentinel"
    /// and "circular" meaning that the node being referred to doesn't hold data (an allocation in
    /// this case) and that it's both the head and tail of the list, respectively.
    bins: [Option<NonNull<Node>>; SECOND_LEVEL_BINS],
}

struct Node {
    /// The previous node in the list of all suballocations (sorted by offset).
    prev_ptr: NonNull<Self>,
    /// The next node in the list of all suballocations (sorted by offset).
    next_ptr: NonNull<Self>,
    /// The previous free node in the same size class (unsorted). Only set if the node is free.
    prev_free_ptr: NonNull<Self>,
    /// The next free node in the same size class (unsorted). Only set if the node is free.
    next_free_ptr: NonNull<Self>,
    offset: DeviceSize,
    size: DeviceSize,
    allocation_type: SuballocationType,
}

unsafe impl Send for FirstLevel {}
unsafe impl Sync for FirstLevel {}

impl FirstLevel {
    unsafe fn init(&mut self) {
        let head_ptr = self.node_allocator.allocate();
        let root_ptr = self.node_allocator.allocate();
        let tail_ptr = self.node_allocator.allocate();

        let head = Node {
            prev_ptr: NonNull::dangling(),
            next_ptr: root_ptr,
            prev_free_ptr: NonNull::dangling(),
            next_free_ptr: NonNull::dangling(),
            offset: self.region.offset(),
            size: 0,
            allocation_type: SuballocationType::Unknown,
        };
        unsafe { head_ptr.write(head) };

        let root = Node {
            prev_ptr: head_ptr,
            next_ptr: tail_ptr,
            prev_free_ptr: NonNull::dangling(),
            next_free_ptr: NonNull::dangling(),
            offset: self.region.offset(),
            size: self.region.size(),
            allocation_type: SuballocationType::Free,
        };
        unsafe { root_ptr.write(root) };

        let tail = Node {
            prev_ptr: root_ptr,
            next_ptr: NonNull::dangling(),
            prev_free_ptr: NonNull::dangling(),
            next_free_ptr: NonNull::dangling(),
            offset: self.region.offset() + self.region.size(),
            size: 0,
            allocation_type: SuballocationType::Unknown,
        };
        unsafe { tail_ptr.write(tail) };

        self.len = 1;
        self.head_ptr = head_ptr;
        self.tail_ptr = tail_ptr;
        unsafe { self.add_to_free_list(root_ptr) };

        unsafe { self.init_bin(FIRST_LEVEL_BINS) };
    }

    #[inline]
    unsafe fn allocate(
        &mut self,
        size: DeviceSize,
        alignment: DeviceAlignment,
        allocation_type: AllocationType,
    ) -> Option<(NonNull<Node>, DeviceSize, DeviceSize)> {
        // This can't overflow because the caller must ensure that `size` doesn't exceed
        // `DeviceLayout::MAX_SIZE`.
        let size = align_up(cmp::max(size, MIN_NODE_SIZE), alignment);

        // We need to guarantee that the free-list we pop from has a node that fits the allocation
        // request, which includes the alignment. We do this by aligning the size to the alignment
        // and adding the maximum size that could be needed to align a free suballocation's offset
        // while having enough size left over for the request. The former is standard practice;
        // the latter, however, is unfortunate, but it doesn't cause much more fragmentation than
        // the TLSF algorithm already causes in practice because the average ratio of alignment to
        // size is very small for Vulkan allocations. Vulkan allocations are dominated by big ones
        // that have small alignments in comparison.
        //
        // `size`, no matter the alignment, can't end up as more than `DeviceAlignment::MAX` for
        // the same reason as above. Therefore, `size + (alignment.as_devicesize() - 1)` can't
        // overflow (the parentheses are load-bearing).
        let alignment_padded_size = size + (alignment.as_devicesize() - 1);

        // Make sure `segregate_up` returns a first-level index that's at most 62 so that we have
        // room for what would have been overflow.
        if alignment_padded_size > (DeviceAlignment::MAX.as_devicesize() >> 1) {
            crate::cold_path();
            return None;
        }

        // SAFETY: `alignment_padded_size` is at least `MIN_NODE_SIZE`, and we checked that the
        // aligning up can't overflow above.
        let (first_level_index, second_level_index) =
            unsafe { segregate_up(alignment_padded_size) };

        let mut node_ptr = unsafe {
            self.find_and_remove_suitable_free_node(first_level_index, second_level_index)
        }?;

        let offset = unsafe { self.split(node_ptr, size, alignment) };

        let node = unsafe { node_ptr.as_mut() };

        node.allocation_type = allocation_type.into();

        Some((node_ptr, offset, node.size))
    }

    #[inline]
    unsafe fn find_and_remove_suitable_free_node(
        &mut self,
        first_level_index: usize,
        second_level_index: usize,
    ) -> Option<NonNull<Node>> {
        let suitable_first_level_index = self.find_suitable_bin_index(first_level_index);

        // It's possible that `first_level_index` is an occupied first-level bin, but one where the
        // only occupied second-level bins are unsuitable. In that case, we have to check the next
        // occupied first-level bin. We do this by searching the occupied bins, but starting after
        // `first_level_index`.
        //
        // The addition still results in a 64-bit bit index because we make sure that
        // `first_level_index` is at most 62.
        let next_suitable_first_level_index = self.find_suitable_bin_index(first_level_index + 1);

        // SAFETY: `suitable_first_level_index` is the index of an occupied bin.
        let second_level = unsafe { self.second_level_unchecked_mut(suitable_first_level_index) };

        // If `first_level_index` is occupied, this checks whether that first-level bin has a
        // suitable second-level bin, and if not, we select the next suitable first-level bin. If
        // `first_level_index` is not occupied, it doesn't matter what the result of the check is
        // because `suitable_first_level_index` is already `next_suitable_first_level_index`.
        let suitable_first_level_index = hint::select_unpredictable(
            second_level.has_suitable_bin(second_level_index),
            suitable_first_level_index,
            next_suitable_first_level_index,
        );

        // SAFETY: `suitable_first_level_index` is the index of an occupied bin.
        let second_level = unsafe { self.second_level_unchecked_mut(suitable_first_level_index) };

        // If `first_level_index` is occupied and has a suitable second-level bin, this simply
        // reruns the calculation of that bin. Otherwise, we are using the next suitable
        // first-level bin, which means that all second-level bins are suitable by definition, so
        // we search starting at the first second-level bin. If the next suitable first-level bin
        // is the "fake" bin, this will simply return `None` as none of its second-level bins are
        // ever initialized.
        let suitable_second_level_index =
            second_level.find_suitable_bin_index(hint::select_unpredictable(
                suitable_first_level_index == first_level_index,
                second_level_index,
                0,
            ));

        let Some(suitable_second_level_index) = suitable_second_level_index else {
            crate::cold_path();
            return None;
        };

        // SAFETY: We checked that a suitable bin exists above.
        let mut sentinel_ptr =
            unsafe { second_level.free_list_unchecked_mut(suitable_second_level_index) };

        let sentinel = unsafe { sentinel_ptr.as_mut() };
        let node_ptr = sentinel.next_free_ptr;
        let mut next_free_ptr = unsafe { node_ptr.as_ref() }.next_free_ptr;
        sentinel.next_free_ptr = next_free_ptr;

        let next_free = unsafe { next_free_ptr.as_mut() };
        next_free.prev_free_ptr = sentinel_ptr;

        let is_last_node = next_free_ptr == sentinel_ptr;
        second_level.occupied_bins &= !(u64::from(is_last_node) << suitable_second_level_index);
        let is_last_bin = second_level.occupied_bins == 0;
        self.occupied_bins &= !(u64::from(is_last_bin) << suitable_first_level_index);

        Some(node_ptr)
    }

    #[inline(always)]
    fn find_suitable_bin_index(&mut self, first_level_index: usize) -> usize {
        // These can't overflow because `first_level_index` is a 64-bit bit index.
        let suitable_bins = (self.occupied_bins >> first_level_index) << first_level_index;

        // We add the index of the "fake" bin so that a suitable bin index is always found.
        let suitable_bins = suitable_bins | (1 << FIRST_LEVEL_BINS);

        // SAFETY: `suitable_bins` is nonzero.
        let suitable_bins = unsafe { NonZero::new_unchecked(suitable_bins) };

        suitable_bins.trailing_zeros() as usize
    }

    #[inline]
    unsafe fn split(
        &mut self,
        mut node_ptr: NonNull<Node>,
        size: DeviceSize,
        alignment: DeviceAlignment,
    ) -> DeviceSize {
        let node = unsafe { node_ptr.as_mut() };

        debug_assert_eq!(node.allocation_type, SuballocationType::Free);

        let offset = align_up(node.offset, alignment);

        debug_assert!(offset + size <= node.offset + node.size);

        // These can't overflow because the caller must ensure that `node_ptr` can fit an
        // allocation of `size` and `alignment`.
        let padding_front = offset - node.offset;
        let padding_back = node.offset + node.size - offset - size;

        if padding_front >= MIN_NODE_SIZE {
            let padding_ptr = self.node_allocator.allocate();
            let padding = Node {
                prev_ptr: node.prev_ptr,
                next_ptr: node_ptr,
                prev_free_ptr: NonNull::dangling(),
                next_free_ptr: NonNull::dangling(),
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
            node.size -= padding_front;

            self.len += 1;

            // SAFETY: We just created this suballocation, so there's no way that it was
            // deallocated already.
            unsafe { self.add_to_free_list(padding_ptr) };
        }

        if padding_back >= MIN_NODE_SIZE {
            let padding_ptr = self.node_allocator.allocate();
            let padding = Node {
                prev_ptr: node_ptr,
                next_ptr: node.next_ptr,
                prev_free_ptr: NonNull::dangling(),
                next_free_ptr: NonNull::dangling(),
                offset: offset + size,
                size: padding_back,
                allocation_type: SuballocationType::Free,
            };
            unsafe { padding_ptr.write(padding) };

            let next = unsafe { node.next_ptr.as_mut() };
            next.prev_ptr = padding_ptr;

            node.next_ptr = padding_ptr;
            // This is overflow-safe for the same reason as above.
            node.size -= padding_back;

            self.len += 1;

            // SAFETY: Same as above.
            unsafe { self.add_to_free_list(padding_ptr) };
        }

        offset
    }

    #[inline]
    unsafe fn add_to_free_list(&mut self, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        // SAFETY: The caller must ensure that `node_ptr` refers to a suballocation, and
        // suballocations always have a size of at least `MIN_NODE_SIZE`.
        let (first_level_index, second_level_index) = unsafe { segregate_down(node.size) };

        // SAFETY: The caller must ensure that `node_ptr` refers to a suballocation.
        let second_level_ptr = *unsafe { self.bins.get_unchecked_mut(first_level_index) };

        let mut second_level_ptr = if let Some(second_level_ptr) = second_level_ptr {
            second_level_ptr
        } else {
            crate::cold_path();

            // SAFETY: Same as the previous.
            unsafe { self.init_bin(first_level_index) }
        };

        // SAFETY: Same as the previous.
        let second_level = unsafe { second_level_ptr.as_mut() };

        // SAFETY: Same as the previous.
        let free_list = *unsafe { second_level.bins.get_unchecked_mut(second_level_index) };

        let mut sentinel_ptr = if let Some(sentinel_ptr) = free_list {
            sentinel_ptr
        } else {
            crate::cold_path();

            // SAFETY: Same as the previous.
            unsafe { second_level.init_bin(second_level_index, &self.node_allocator) }
        };

        let sentinel = unsafe { sentinel_ptr.as_mut() };
        let mut next_free_ptr = sentinel.next_free_ptr;
        sentinel.next_free_ptr = node_ptr;
        node.prev_free_ptr = sentinel_ptr;

        node.next_free_ptr = next_free_ptr;
        let next_free = unsafe { next_free_ptr.as_mut() };
        next_free.prev_free_ptr = node_ptr;

        second_level.occupied_bins |= 1 << second_level_index;
        self.occupied_bins |= 1 << first_level_index;
    }

    unsafe fn init_bin(&mut self, first_level_index: usize) -> NonNull<SecondLevel> {
        let second_level_ptr = self.bin_allocator.allocate();

        let second_level = SecondLevel {
            occupied_bins: 0,
            bins: [None; SECOND_LEVEL_BINS],
        };
        unsafe { second_level_ptr.write(second_level) };

        // SAFETY: Enforced by the caller.
        *unsafe { self.bins.get_unchecked_mut(first_level_index) } = Some(second_level_ptr);

        second_level_ptr
    }

    #[inline]
    unsafe fn coalesce(&mut self, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        debug_assert_eq!(node.allocation_type, SuballocationType::Free);

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

    #[inline]
    unsafe fn remove_from_free_list(&mut self, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        // SAFETY: The caller must ensure that `node_ptr` refers to a suballocation, and
        // suballocations always have a size of at least `MIN_NODE_SIZE`.
        let (first_level_index, second_level_index) = unsafe { segregate_down(node.size) };

        // SAFETY: The caller must ensure that `node_ptr` refers to a free suballocation, which
        // means that there must be a free-list with it in it, and this list is given by
        // `segregate_down`.
        let second_level = unsafe { self.second_level_unchecked_mut(first_level_index) };

        let prev_free = unsafe { node.prev_free_ptr.as_mut() };
        prev_free.next_free_ptr = node.next_free_ptr;

        let next_free = unsafe { node.next_free_ptr.as_mut() };
        next_free.prev_free_ptr = node.prev_free_ptr;

        let is_last_node = node.next_free_ptr == node.prev_free_ptr;
        second_level.occupied_bins &= !(u64::from(is_last_node) << second_level_index);
        let is_last_bin = second_level.occupied_bins == 0;
        self.occupied_bins &= !(u64::from(is_last_bin) << first_level_index);
    }

    #[inline(always)]
    unsafe fn second_level_unchecked_mut(&mut self, first_level_index: usize) -> &mut SecondLevel {
        // SAFETY: Enforced by the caller.
        let second_level = *unsafe { self.bins.get_unchecked_mut(first_level_index) };

        // SAFETY: Enforced by the caller.
        let mut second_level = unsafe { second_level.unwrap_unchecked() };

        // SAFETY: Enforced by the caller.
        unsafe { second_level.as_mut() }
    }

    fn reset(&mut self) {
        self.occupied_bins = 0;
        self.bins = [None; FIRST_LEVEL_BINS + 1];
        unsafe { self.node_allocator.reset() };
        unsafe { self.bin_allocator.reset() };
        unsafe { self.init() };
    }

    #[inline]
    fn iter(&self) -> Suballocations<'_> {
        let head_ptr = unsafe { self.head_ptr.as_ref() }.next_ptr;
        let tail_ptr = unsafe { self.tail_ptr.as_ref() }.prev_ptr;

        Suballocations {
            len: self.len,
            head_ptr,
            tail_ptr,
            marker: PhantomData,
        }
    }
}

impl SecondLevel {
    #[inline(always)]
    fn has_suitable_bin(&self, second_level_index: usize) -> bool {
        // This can't overflow because `second_level_index` is a 64-bit bit index.
        self.occupied_bins >> second_level_index != 0
    }

    #[inline(always)]
    fn find_suitable_bin_index(&self, second_level_index: usize) -> Option<usize> {
        // These can't overflow because `second_level_index` is a 64-bit bit index.
        let suitable_bins = (self.occupied_bins >> second_level_index) << second_level_index;

        let suitable_bins = NonZero::new(suitable_bins)?;

        Some(suitable_bins.trailing_zeros() as usize)
    }

    #[inline(always)]
    unsafe fn free_list_unchecked_mut(&mut self, second_level_index: usize) -> NonNull<Node> {
        // SAFETY: Enforced by the caller.
        let free_list = *unsafe { self.bins.get_unchecked_mut(second_level_index) };

        // SAFETY: Enforced by the caller.
        unsafe { free_list.unwrap_unchecked() }
    }

    unsafe fn init_bin(
        &mut self,
        second_level_index: usize,
        node_allocator: &SlabAllocator<Node>,
    ) -> NonNull<Node> {
        let sentinel_ptr = node_allocator.allocate();

        let sentinel = Node {
            prev_ptr: NonNull::dangling(),
            next_ptr: NonNull::dangling(),
            prev_free_ptr: sentinel_ptr,
            next_free_ptr: sentinel_ptr,
            offset: 0,
            size: 0,
            allocation_type: SuballocationType::Free,
        };
        unsafe { sentinel_ptr.write(sentinel) };

        // SAFETY: Enforced by the caller.
        *unsafe { self.bins.get_unchecked_mut(second_level_index) } = Some(sentinel_ptr);

        sentinel_ptr
    }
}

/// Returns the segregated list indices of the size class that has the lowest minimum node size
/// greater than or equal to `size`.
///
/// This amounts to the same exact calculation as [`segregate_down`] except that the size needs to
/// be aligned up to that size's mantissa first:
///
/// ```plain
/// 1YYYZZZ...ZZZ ────────────┐
///                 needs to be aligned to
///    1000000000 ◀───────────┘
/// ```
///
/// Some examples:
///
/// ```plain
/// size (in binary) | aligned up       | exp   mts
/// -----------------+------------------+----------
/// 0000101000000000 | 0000101000000000 | 00111 010
/// 0000101001000110 | 0000101100000000 | 00111 011
/// 0000101100000000 | 0000101100000000 | 00111 011
/// ```
///
/// See [`segregate_down`] for a detailed explanation of how the segregation works.
#[inline(always)]
unsafe fn segregate_up(size: DeviceSize) -> (usize, usize) {
    // This lowers to just 1 instruction on x86 (`bsr`) and 2 on Arm (`clz; eor`).
    //
    // SAFETY: The caller must ensure that `size` is at least `MIN_NODE_SIZE`.
    let highest_one_index = unsafe { NonZero::new_unchecked(size) }.ilog2();

    // This can't overflow because `size` is at least `MIN_NODE_SIZE` and `SECOND_LEVEL_INDEX_BITS`
    // doesn't exceed `MIN_NODE_BITS`.
    let second_level_index_index = highest_one_index - SECOND_LEVEL_INDEX_BITS;

    // The shift can't overflow because `second_level_index_index` is the index of a bit.
    //
    // SAFETY: `1 << second_level_index_index` is a power of two.
    let mantissa_one = unsafe { DeviceAlignment::new_unchecked(1 << second_level_index_index) };

    let size = align_up(size, mantissa_one);

    // SAFETY: The caller must ensure that the `align_up` above doesn't overflow, and since `size`
    // started out as at least `MIN_NODE_SIZE`, it must still be.
    unsafe { segregate_down(size) }
}

/// Returns the segregated list indices of the size class that has the highest minimum node size
/// less than or equal to `size`.
///
/// Given that the TLSF algorithm uses power-of-two segregation at the first level and linear
/// segregation at the second level, the overall bin distribution equates to a floating point
/// number with no sign bit, [`FIRST_LEVEL_BINS`] exponent values (0-31, also no sign, no special
/// values, no subnormals), and [`SECOND_LEVEL_BINS`] mantissa values (0-7). This makes it very
/// easy to reason about and implement in an efficient manner. The size classes correspond to this
/// float multiplied by [`MIN_NODE_SIZE`] (16).
///
/// Let's look at some examples because words are hard to understand:
///
/// ```plain
/// exp   mts | * 16 (in binary)                     | * 16 (in decimal)
/// ----------+--------------------------------------+------------------
/// 00000 000 | 000000000000000000000000000000010000 | 16
/// 00000 001 | 000000000000000000000000000000010010 | 18
/// ...
/// 00000 110 | 000000000000000000000000000000011100 | 28
/// 00000 111 | 000000000000000000000000000000011110 | 30
/// 00001 000 | 000000000000000000000000000000100000 | 32
/// 00001 001 | 000000000000000000000000000000100100 | 36
/// ...
/// 00001 110 | 000000000000000000000000000000111000 | 56
/// 00001 111 | 000000000000000000000000000000111100 | 60
/// 00010 000 | 000000000000000000000000000001000000 | 64
/// 00010 001 | 000000000000000000000000000001001000 | 72
/// ...
/// 00010 110 | 000000000000000000000000000001110000 | 96
/// 00010 111 | 000000000000000000000000000001111000 | 112
/// ...
/// ...
/// ...
/// 11111 000 | 100000000000000000000000000000000000 | 32G
/// ...
/// 11111 111 | 111100000000000000000000000000000000 | 60G
/// ```
///
/// Quick refresher on how floats work:
///
/// ```plain
/// exp   mts                    * 16
/// XXXXX YYY = 1.YYY << XXXXX    =>    1YYY0 << XXXXX
/// ```
///
/// From the above table, we can conclude a few things:
/// - the number of values of this float we've introduced is the number of bins;
/// - the float multiplied by `MIN_NODE_SIZE` gives us the minimum node size for every bin;
/// - the exponent and mantissa correspond to our `FIRST_LEVEL_BINS` and `SECOND_LEVEL_BINS`.
///
/// As such, it perfectly maps all of TLSF's parameters to the corresponding bins.
///
/// We want to determine the size class with the highest minimum node size that doesn't exceed
/// `size`, which is very easy to do by looking at `size`'s bits:
///
/// ```plain
/// 1YYYZZZ...ZZZ
/// │^^^───────── YYY just needs to be extracted
/// └──────────── the index of this bit is XXXXX + MIN_NODE_BITS
/// ```
///
/// It doesn't matter what the `ZZZ...ZZZ` bits are: this size is always going to be in the range
/// of the size class where `XXXXX` is the first-level index and `YYY` is the second-level index
/// because it's the top bits that define the size class; the low bits are all in that range by
/// definition.
#[inline(always)]
unsafe fn segregate_down(size: DeviceSize) -> (usize, usize) {
    // This lowers to just 1 instruction on x86 (`bsr`) and 2 on Arm (`clz; eor`).
    //
    // SAFETY: The caller must ensure that `size` is at least `MIN_NODE_SIZE`.
    let highest_one_index = unsafe { NonZero::new_unchecked(size) }.ilog2();

    // This can't overflow because `size` is at least `MIN_NODE_SIZE`.
    let first_level_index = highest_one_index - MIN_NODE_BITS;

    // This can't overflow because `size` is at least `MIN_NODE_SIZE` and `SECOND_LEVEL_INDEX_BITS`
    // doesn't exceed `MIN_NODE_BITS`.
    let second_level_index_index = highest_one_index - SECOND_LEVEL_INDEX_BITS;

    // The shift can't overflow because `first_level_index` is the index of a bit, which means
    // the smaller quantity `second_level_index_index` is too.
    let second_level_index = (size >> second_level_index_index) & SECOND_LEVEL_INDEX_MASK;

    (first_level_index as usize, second_level_index as usize)
}

impl Debug for FirstLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        struct Nodes<'a>(&'a FirstLevel);

        impl Debug for Nodes<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
                let mut debug = f.debug_list();
                let mut head_ptr = unsafe { self.0.head_ptr.as_ref() }.next_ptr;
                let mut len = self.0.len;

                while len != 0 {
                    let head = unsafe { head_ptr.as_ref() };

                    debug.entry(head);

                    head_ptr = head.next_ptr;
                    len -= 1;
                }

                debug.finish()
            }
        }

        struct OccupiedBins(u64);

        impl Debug for OccupiedBins {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
                write!(f, "0b{:032b}", self.0)
            }
        }

        struct Bins<'a>(&'a FirstLevel);

        impl Debug for Bins<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
                let bins = self.0.bins.iter();
                let entries = bins.enumerate().filter_map(|(index, second_level)| {
                    second_level.and_then(|second_level| {
                        (self.0.occupied_bins & (1 << index) != 0)
                            .then_some((index, unsafe { second_level.as_ref() }))
                    })
                });

                f.debug_map().entries(entries).finish()
            }
        }

        f.debug_struct("FirstLevel")
            .field("region", &self.region)
            .field("nodes", &Nodes(self))
            .field("len", &self.len)
            .field("occupied_bins", &OccupiedBins(self.occupied_bins))
            .field("bins", &Bins(self))
            .finish()
    }
}

impl Debug for SecondLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        struct OccupiedBins(u64);

        impl Debug for OccupiedBins {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
                write!(f, "0b{:08b}", self.0)
            }
        }

        struct Bins<'a>(&'a SecondLevel);

        impl Debug for Bins<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
                let bins = self.0.bins.iter();
                let entries = bins.enumerate().filter_map(|(index, free_list)| {
                    free_list.and_then(|sentinel_ptr| {
                        (self.0.occupied_bins & (1 << index) != 0)
                            .then_some((index, FreeList(sentinel_ptr)))
                    })
                });

                f.debug_map().entries(entries).finish()
            }
        }

        struct FreeList(NonNull<Node>);

        impl Debug for FreeList {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
                let mut debug = f.debug_list();
                let sentinel_ptr = self.0;
                let mut head_free_ptr = unsafe { sentinel_ptr.as_ref() }.next_free_ptr;

                loop {
                    if head_free_ptr == sentinel_ptr {
                        break;
                    }

                    let head_free = unsafe { head_free_ptr.as_ref() };

                    debug.entry(head_free);

                    head_free_ptr = head_free.next_free_ptr;
                }

                debug.finish()
            }
        }

        f.debug_struct("SecondLevel")
            .field("occupied_bins", &OccupiedBins(self.occupied_bins))
            .field("bins", &Bins(self))
            .finish()
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("Node")
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("allocation_type", &self.allocation_type)
            .finish()
    }
}

pub struct Suballocations<'a> {
    len: usize,
    head_ptr: NonNull<Node>,
    tail_ptr: NonNull<Node>,
    marker: PhantomData<&'a FirstLevel>,
}

impl<'a> Iterator for Suballocations<'a> {
    type Item = SuballocationNode;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            let head = unsafe { self.head_ptr.as_ref() };
            self.len -= 1;
            self.head_ptr = head.next_ptr;

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
    fn count(self) -> usize {
        self.len
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
            self.len -= 1;
            self.tail_ptr = tail.prev_ptr;

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
